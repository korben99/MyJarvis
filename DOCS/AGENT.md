# The agentic loop

> Give Jarvis a task and let it work until it is done: search, read, execute, write —
> in the background, never in front of the chat.

`AGENT_ENABLED` is **false** by default. The shell tool has a second switch,
`AGENT_SHELL_ENABLED`, also false by default.

---

## The loop

```
objective ──► tool ──► observation ──► tool ──► … ──► finish ──► deliverable
                 ▲                        │
                 └────────────────────────┘
                        max_steps · timeout · no-progress
```

Each step is one inference call that renders the tool schemas, the plan, the previous
steps and the accumulated observations. The loop uses `stream_local()` — the only
inference path that makes `tools` available — at **background GPU priority**: an agent step
only takes the GPU when no chat call is waiting. A long task therefore never delays a
conversation.

Concurrency is deliberately **1**. Two tasks in parallel would gain nothing — they would
fight over the same GPU, already serialised by the inference lock — and would double the
pressure on the prompt LRU cache, where an agent already occupies one continuously growing
entry. The queue itself is unbounded.

Nothing is lost to a restart: the context is written to disk after every step, and
interrupted tasks are requeued at the next boot.

---

## The tools

Nine, and the number is a design decision. Every extra tool is one more opportunity to pick
the wrong one, and the cost is paid at **every** step since the schemas are rendered at the
top of the prompt. Every output is truncated, because the whole context is re-injected each
step.

| Tool | Role | Cap |
|---|---|---|
| `web_search` | web search through Jarvis's own search pipeline | `AGENT_MAX_TOOL_OUTPUT` |
| `fetch_url` | fetch and read one page | `AGENT_PAGE_MAX_CHARS` |
| `search_docs` | the document RAG, with a stricter score floor than chat | `AGENT_DOCS_MIN_SCORE` |
| `threat_intel` | CTI sources | `AGENT_MAX_TOOL_OUTPUT` |
| `list_dir` | list a directory | — |
| `read_file` | read a file — sized so a source file fits in **one** read | `AGENT_READ_MAX_CHARS` |
| `write_file` | write into the task workspace | `AGENT_WRITE_MAX_CHARS` |
| `plan` | the only tool allowed alongside an action in the same turn | — |
| `shell` | command execution — off by default | `AGENT_SHELL_TIMEOUT` |

`finish` is not dispatched as a tool: it is the exit from the loop.

`read_file` gets a higher cap than the general one on purpose. Pagination is what the model
handles worst — given a file split across two reads, it ignores the "resume at offset=318"
hint and replays the same read until the task budget is exhausted. A single read removes the
failure mode entirely.

Loop service files (`transcript.jsonl`, `messages.json`) are hidden from `list_dir`: left
visible, the agent spends a step reading its own transcript, which teaches it nothing it
does not already have in context.

---

## The budgets

Three, independent, each bounding a different failure:

| Budget | Variable | Default | Bounds |
|---|---|---|---|
| Steps | `AGENT_MAX_STEPS` | 20 | reasoning in circles |
| Wall clock | `AGENT_TASK_TIMEOUT_MINUTES` | 45 | how long chat waits behind it |
| No progress | — | 2 identical calls | tight loops on a failing tool |

Two token budgets sit underneath:

- `AGENT_STEP_MAX_TOKENS` (2200) and `AGENT_THINKING_BUDGET` (1000) share the same
  allowance: reasoning + visible output + tool call.
- `AGENT_WRITE_MAX_TOKENS` (6000) applies to the **writing** step only. A deliverable
  travels through `write_file`'s `content` parameter, so it is generated *inside* the
  `<tool_call>` block; the normal step cap would cut it mid-block, the block would never
  close, no call would be detected and the step would be lost. This larger budget is only
  ever spent when truncation actually happened, and without reasoning.

---

## The sandbox

When `shell` is enabled, three independent layers confine it. Jarvis runs under the user's
account with full rights: a 35 GB quantised model handed a shell under that account is one
hallucination away from `rm -rf ~`. Confinement is not a configuration option, it is the
condition of the feature existing.

1. **seatbelt (`sandbox-exec`)** — the only real barrier, because the kernel refuses, not us.
   Writes limited to the task workspace and `/tmp`; reads denied on `.env`, `keys/`,
   `~/.ssh` and the keychain; network cut.
2. **Pattern blacklist** — a guard rail against honest mistakes (`sudo`, `rm -rf /`,
   `curl … | sh`, machine shutdown, raw disk writes). This is **not** a security boundary:
   a blacklist can be worked around. The barrier is seatbelt.
3. **Budgets** — per-command timeout, per-task call quota, truncated output.

The profile is `(allow default)` with targeted restrictions rather than `(deny default)`:
a deny-default profile breaks half the Unix tooling on macOS (mach-lookup, sysctl, dyld) and
would have produced an unusable shell. The two paths that matter — writing outside the zone,
and the network — are closed; the rest stays open.

Reads stay broad on purpose: the agent has to be able to inspect the system to be useful,
and everything it reads ends up in a context the user re-reads anyway. Writing and the
network are the two ways a mistake leaves the machine, and those are the ones that are shut.

The network is cut inside the shell even though the agent has `web_search` and `fetch_url`:
those two go through Jarvis's own code, logged and bounded. A `curl` in a shell is not, and
it is the shortest exfiltration path there is.

---

## Delivering the result

An iOS push announces that a task has finished; it cannot carry the deliverable, capped at
500 characters and read on a lock screen. The **email** carries the whole document, can be
kept, forwarded and re-read on a real screen.

It is sent from the requester's own Google account, to their own address — never to a third
party. Gmail sending accepts no attachment here (text + HTML alternative only), so the
document travels in the **body**, which has the side benefit of being readable without
opening anything. Controlled by `AGENT_EMAIL_REPORT` and `AGENT_EMAIL_MAX_CHARS`.

---

## API

Task creation is **restricted to administrators** (`admin: true` in `users_list.json`).

| Endpoint | Role |
|---|---|
| `POST /agent/tasks` | queue a task — returns `202` immediately, execution is asynchronous |
| `GET /agent/tasks` | list tasks; without `user_code`, all users (operations view) |
| `GET /agent/tasks/{id}` | state of one task |
| `POST /agent/tasks/{id}/cancel` | request cancellation — taken between two steps, never mid-step |
| `GET /agent/tasks/{id}/transcript` | the last *n* events — this is where you see what the agent actually did |

```bash
curl -X POST http://localhost:8000/agent/tasks \
  -H "Content-Type: application/json" \
  -d '{"user_code": "ALICE1", "objective": "…"}'
```

An objective shorter than 10 characters is rejected as not executable. Tasks expire after
`AGENT_TASK_TTL_DAYS` (30).

---

## Configuration

Every variable, with its default:

| Variable | Default | Role |
|---|---|---|
| `AGENT_ENABLED` | `false` | master switch |
| `AGENT_WORKSPACE` | `/opt/jarvis/agent_workspace` | one directory per task |
| `AGENT_MAX_STEPS` | `20` | step budget |
| `AGENT_TASK_TIMEOUT_MINUTES` | `45` | wall-clock budget |
| `AGENT_STEP_MAX_TOKENS` | `2200` | reasoning + output + tool call per step |
| `AGENT_THINKING_BUDGET` | `1000` | reasoning share of the above |
| `AGENT_WRITE_MAX_TOKENS` | `6000` | writing step only |
| `AGENT_WRITE_MAX_CHARS` | derived | cap on one written file |
| `AGENT_MAX_TOOL_OUTPUT` | `15000` | truncation of tool output injected into context |
| `AGENT_READ_MAX_CHARS` | `32000` | cap on one `read_file` |
| `AGENT_PAGE_MAX_CHARS` | `14000` | cap on one fetched page |
| `AGENT_DOCS_MIN_SCORE` | `0.35` | RAG score floor for the agent |
| `AGENT_QUIET_SECONDS` | `45` | quiet window before yielding the GPU |
| `AGENT_TASK_TTL_DAYS` | `30` | task retention |
| `AGENT_EMAIL_REPORT` | `true` | email the deliverable |
| `AGENT_EMAIL_MAX_CHARS` | `120000` | cap on the emailed body |
| `AGENT_SHELL_ENABLED` | `false` | shell tool |
| `AGENT_SHELL_TIMEOUT` | `60` | per-command timeout |
| `AGENT_SHELL_MAX_CALLS` | `25` | shell calls per task |
| `AGENT_SHELL_NETWORK` | `false` | network inside the sandbox |
| `AGENT_READONLY_ROOTS` | `src`, `scripts`, `DOCS` | paths the agent may read outside its workspace |

See **[SECURITY.md](SECURITY.md)** for the threat model.

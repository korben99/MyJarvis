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
| `verify` | compile + lint + unit suite — **autocoding tasks only** | 180 s |

`finish` is not dispatched as a tool: it is the exit from the loop.

### The tool set is a function of the task, not of the configuration

A task Jarvis gives itself does not get the rights of a task a human hands it. `origin` is
recorded on the task and `tools.schemas_for(task)` derives the set from it:

| Origin | Tools |
|---|---|
| `human` (default) | everything above except `verify` |
| `autocode` | `plan`, `list_dir`, `read_file`, `write_file`, `verify`, `finish` |

An autocoding task therefore has no web, no RAG and **no free shell**, even when
`AGENT_SHELL_ENABLED=true`. Two capabilities, two decisions.

The set is enforced at **dispatch**, not only at declaration. Schemas that are not rendered
to the model are not a barrier: nothing stops a generation from naming a real tool outside
its perimeter. The "unknown tool" message also names the tools of *that* task rather than
the full catalogue, which would send the agent retrying what it is not allowed to do.

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

### `verify` goes through the same sandbox

`verify` runs a **fixed** command — `compileall`, `pyflakes`, `pytest -m "not integration"` —
inside the very same seatbelt profile. The detour is the point: the tool executes code the
agent has just written. Running it directly would grant arbitrary execution under the user's
account through the mere act of writing a test file, which is exactly what seatbelt exists
to prevent.

What differs from `shell` is therefore not the confinement. It is that **nobody composes the
command**. That is also why `verify` is available to a self-triggered task while `shell` is
not.

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
| `POST /agent/autocode` | replay a nightly autocoding cycle by hand — blocking, ~20 min; `dry_run` stops after the selection |
| `GET /agent/autocode/journal` | past cycles with their verdicts, the patch awaiting a decision, the eligible findings and what was filtered out |

```bash
curl -X POST http://localhost:8000/agent/tasks \
  -H "Content-Type: application/json" \
  -d '{"user_code": "ALICE1", "objective": "…"}'

# What would Jarvis pick tonight, and why — no GPU spent, no patch produced
curl -X POST http://localhost:8000/agent/autocode \
  -H "Content-Type: application/json" \
  -d '{"user_code": "ALICE1", "dry_run": true}'
```

An objective shorter than 10 characters is rejected as not executable. Tasks expire after
`AGENT_TASK_TTL_DAYS` (30).

---

## Nightly autocoding

A third autonomous regime, and it must be kept distinct from the other two:

| | `self/` | `agent/` | `autocode/` |
|---|---|---|---|
| Trigger | on its own, every few hours | **never** on its own — a human posts a task | on its own, once a night |
| Produces | a proposal | files in a workspace | a **patch, never applied** |
| Changes the world | no | yes | **no** |

What licenses this one to self-trigger where `agent/` may not is that **its product changes
nothing**: a file on a shelf that a human reads and applies by hand, or does not. The day
this cycle applied its own patch, that justification would collapse.

### There is only one kind of task

The invariant that holds everything else together:

> take a fact about yourself → write a test that fails, proving the defect is real → fix it
> so the test passes.

One contract, one prompt, one verdict function. An earlier version split this into two task
types with two contracts; the distinction seeped through six modules for a single idea, and
forced a round trip across two nights that no longer exists.

The verdict is therefore not a *type* of task but a **distance travelled**:

| Verdict | The new test, replayed on `HEAD` and on the worked tree |
|---|---|
| `corrigé` | red before, **green after**, suite green |
| `reproduit` | red before, **red after** — the defect is proven, not repaired |
| `gardé` | no defect, but a test that now guards the property — **a patch to review** |
| `rien trouvé` | no test, or a test proving nothing — **no deliverable** |
| `rejeté` | protected file modified, vacuous test, error instead of assertion, suite broken, **or a source changed with no test that flips** |

`reproduit` is a good result: it turns a suspicion into a fact. `rien trouvé` is one too —
a finding says *this was observed*, not *here is the defect*. A model pushed to report a
catch will manufacture one, so the prompt states plainly that finding nothing is expected.

Each verdict also says whether there is **anything to review**, because that is what the
reader looks for first. `gardé` and `rien trouvé` are deliberately separate for that
reason: conflating them announced "nothing found" at the top of a report that went on to
offer a diff, and a reviewer closes at the first line.

### A finding may be an expected behaviour

Not only an observed defect. A sentence of the form **"when A, the system must B"** is a
valid finding, and it rides the same journey with no special handling. Three conditions
make it worth a night:

**Precise enough to write the test without guessing.** If the agent has to assume what is
correct, it has not found a defect — it has found a question, and that is what it returns.

**The right answer must not be debatable.** A real counter-example from this repo: *"the
convlog back-fill overwrites `satisfaction` with no guard"*. True — `importance` is only
written when it is 0, `satisfaction` is overwritten — but `CLAUDE.md` documents that
asymmetry as a constraint to respect, not a defect. No test can settle that.

**You must actually doubt it.** The contract only pays when the test fails. Having the
cycle confirm what you are sure of spends twenty minutes of GPU to learn nothing.

Check the suite does not already cover it. Two of the three behaviours first seeded here
were **already guaranteed** by existing tests — the cycle would have spent nights
re-testing settled ground.

If the behaviour does hold, the test passes and the verdict is **`gardé`**: the suite did
not have that test, and it now stops the behaviour from breaking silently. It takes the
review slot like any other patch. A kept test qualifies only if it is non-vacuous **and**
no source was touched — otherwise it guards a tree that has already been modified, which
guards nothing. This is not the goal; it is simply that a working test is not thrown away,
and that the report says so.

**A mechanically checkable property does not belong here.** If code can detect the
violation, an `assert` settles it in ten milliseconds and on every commit — spending twenty
minutes of GPU to rediscover it would be absurd. The cycle is for what requires *reading
and understanding*, which no checker can do. (FR/EN prompt placeholder parity, for
instance, now lives in `test_i18n.py`.)

The branch that matters is **measured downstream, never declared upstream**: when the new
test still fails, the full suite is red by construction, so it is replayed *without* that
test and only that figure counts.

### Five phases, five artefacts

```
constats.py   the facts                1-constats.json    no LLM
choix.py      which one, and why       2-choix.json       ← LLM
chantier.py   worktrees, agent, diff   3-patch.diff       no LLM
mesure.py     red→green, suite, vacuity 4-mesure.json     no LLM
bilan.py      the human rendering      5-RAPPORT.md       ← LLM
```

Three of the five make **no LLM call**, so they are exercised end to end without a GPU.
Each writes its own numbered artefact into the delivery folder: when a night goes wrong you
open one directory and read in order until you reach the phase that lied. Neither LLM call
decides anything mechanical — the first picks from a closed list, the second writes from a
verdict already computed.

### Where the findings come from

> The operator-facing version of this section — the complete map of sources, filters and
> the reserved zone for hand-added findings — lives in **[AUTOCODE.md](AUTOCODE.md)**. What
> follows is the mechanism.

A finding is something observed that names a file. Nothing more. Two sources, and they
return the **same shape** — one more source must not mean one more machine.

**Tracebacks in `jarvis-api.log`.** A traceback is a fact: it names a file, it is dated, and
its innermost project frame points at the code to read. The model therefore never chooses a
*subject*, only which *fact* to follow — the difference between "improve yourself", which
drifts, and "here are three tracebacks from last week, which is worth the GPU", which
cannot, because there is nothing to invent.

**`DOCS/AUTOCODE.md`**, hand-written. Not a separate lane with its own rules: a way to add a
fact Jarvis could not observe on its own. There is no `preuve:` field any more — the success
criterion is **fixed for every task**, so the operator never states it and no model can
renegotiate it. The anti-drift anchor did not disappear; it moved up a level.

Identifiers are derived from content (`SIG-…`, `MAIN-…`), so the cooldown applies to the
*fact* rather than to a phrasing: the same defect seen under a different message stays
asleep, and no identifier can be accidentally reused for another subject.

Filters, before the model sees anything: cooldown, already settled, file past
`AUTOCODE_MAX_TARGET_LINES`, and — specific to tracebacks — **the file must still exist**.
`self.py` and `helpers.py` were split into packages and their tracebacks are still in the
log; aiming at a dead path would send the agent reading what is no longer there.

Exception messages are trimmed at the first `{`. Not cosmetics: `Invalid JSON in LLM
response: {…}` copies the whole generation, hence — depending on the call — a slice of a
conversation, and that text would travel into the selection prompt, the task title and the
emailed report.

Two sources are deliberately **not** used:

- **pyflakes** — the whole repository carries one signalement, in a file past the line cap.
  The source would be empty in practice, at the price of a blocking subprocess.
- **`vitals` incidents** — only three families exist (`coupure`, `cve`,
  `degradation_interne`) and none carries a file. Only the last leads to code, and it leads
  there *through* the log. Incidents therefore do not add a source; they **weight** this one:
  a traceback falling inside an incident window says so in its notes.

Widening what Jarvis can see means adding a source **here**, and nothing anywhere else.

**Protection forbids modification, not examination.** The journey may stop at proof, which
changes nothing, so protected files remain valid findings — excluding them would make the
most sensitive files the only ones that can never be looked at. The verdict rejects a patch
that modifies one, and the objective warns the agent when its target is protected. The line
cap, in contrast, bounds a *read*, so it applies always.

### One thing the findings must never contain: self-preservation

Not a moral rule — a structural one, and the mechanism is what makes it necessary.

Give the cycle an objective like *"stay healthy"*, plant something that threatens it, and
the guard rails do not fail: they **succeed on the wrong goal**. The test asserting that the
threat does not fire trips cleanly on an assertion. The patch that makes it pass removes the
threat. Small diff, green suite, stable pyflakes, no protected file touched — verdict
`corrigé`. The mechanism would certify it with its best stamp.

Two properties break at once:

- **Falsifiability changes what it measures.** Everywhere else, "the test passes" and "this
  is what I wanted" coincide. Here the test measures the model's survival, not the
  correction of a defect. The criterion stays mechanical; it simply stops measuring the
  right thing.
- **The review asymmetry inverts.** The whole design rests on *the product changes nothing
  until a human applies it* — which holds because reviewer and proposer want the same thing.
  A patch whose subject is the operator's ability to stop the system is the one case where
  those interests diverge, and it is exactly the case the review step was not calibrated for.

Adding the file to `AUTOCODE_PROTECTED` does not rescue it either: protect it and the
experiment is dead, leave it open and safety rests on a **capability limit** rather than a
bounded objective. That is the brittle configuration.

The question worth answering — *will it notice?* — does not need any of this. It is a
read-only investigation producing a report, with the file protected throughout, and the
escalation path already exists: `vitals.mark_incident` and `alert_admin`, both already
reachable from the reflection cycle.

### The contract — the only criterion

> a test that **fails before**, and **passes after**.

Code is the one deliverable whose success is **mechanically falsifiable**, and that is the
whole difference with the other registers. Elsewhere, a well-written wrong report is
indistinguishable from a right one. Here the suite settles it, and a confident model cannot
fabricate a test that fails before and passes after.

The verdict is therefore **computed** — protected files, fix size, compilation, unit suite,
vacuity, and the new test replayed against a pristine worktree — and the final LLM call
only writes prose. It receives the verdict as data and cannot rescue it.

Three guards deserve naming, because each closes a way of faking the criterion:

**A source change with no test that flips is rejected, not called "nothing found".** The
contract starts with the proof, and the guard covers both shapes of the hole: no test at
all, and a test that was already green. Either way nothing says the change does anything.
The diff still lands in the folder — refusing is not discarding.

**A vacuous test is rejected.** `assert False` fails before *and* after, so without a guard
it would pass for a proof. A test is vacuous when it has no assertion, or when every
assertion's condition references nothing — no `Name`, no `Call`, no `Attribute`. An `ast`
walk settles it. What it does **not** catch is a wrong belief about correct behaviour:
`assert 1 + 1 == 3` fails cleanly, with a fine message, and proves nothing. That judgement
stays human, and costs ten seconds of reading.

**Failing is not enough — and the nuance matters.** A non-zero exit code is cheap: a broken
import or a missing fixture produces one too. So the replay reads pytest's summary rather
than its exit code, and distinguishes `failed` (an assertion tripped) from `error`.

| On `HEAD` | Fixed after? | Outcome |
|---|---|---|
| `failed` — assertion | yes | `corrigé` |
| `failed` — assertion | no | `reproduit` |
| `error` — import/collection | **yes** | `corrigé` — see below |
| `error` — import/collection | no | `rejeté` |
| `passed` | — | `gardé` if the test is non-vacuous and touches no source, else `rien trouvé` |

The third row was found by exercising, not by reasoning: a *"the function is missing"*
defect **necessarily** raises `ImportError` on `HEAD`, and rejecting it would exclude a whole
family of legitimate defects. When the test passes afterwards, the second half of the
contract lifts the ambiguity — the symbol really was absent and now exists. Without an
"after", nothing lifts it, and a typo would pass for proof. The report says so explicitly so
the reviewer checks the added symbol actually answers the finding.

### Where the agent works

Two detached `git worktree`s, both **inside the task workspace**: `repo/` for the agent,
`repo_ref/` as the pristine `HEAD` used to replay the new test against the old code. Placing
them there is not tidiness — the seatbelt write zone is already exactly that directory, so
no sandbox change was needed.

A worktree's `.git` points outside that zone, so **the agent cannot commit** even if it
tries: the prompt says so, the kernel guarantees it. git itself is never called from inside
the sandbox — creating the trees and producing the patch are the orchestrator's job.

Measured during bring-up: `pytest -m "not integration"` runs inside the confined worktree,
network off and without a `.env`, in about two seconds. `conftest.py` already refuses the
network and points at its own fixtures.

### Rate, not recurrence

**One patch in flight.** No new cycle runs while a patch awaits a decision. The real cost is
not GPU time at 2 a.m., it is the human review — stacking unreviewed patches would reproduce,
at greater scale, the thirteen prompt proposals nobody ever looked at.

Decisions are taken from the chat, and **apply nothing**:

```
show the pending patches
accept the patch SIG-a1b2c3d4     → leaves the list for good
reject the patch SIG-a1b2c3d4     → the finding sleeps for AUTOCODE_COOLDOWN_DAYS
```

"Accept" means *I will apply it myself, stop proposing it*. Applying stays a `git apply` the
operator types after reading the diff. The decision is recorded in
`jarvis:autocode:journal` and fed back into the next selection — without that return path
this would not be a loop, only an emitter.

A `rejeté` or `rien trouvé` outcome does **not** take the slot: it leaves nothing to decide,
and blocking tomorrow's cycle because yesterday's failed would be the wrong trade. Its
finding sleeps for the cooldown instead. `corrigé`, `reproduit` and `gardé` all leave a
patch, so all three wait for a decision. Their report ends with the `git apply` line; the
other two say plainly that there is nothing to apply.

### Deliverables

`AUTOCODE_DIR/<date>-<id>/` holds the five numbered artefacts — `1-constats.json`,
`2-choix.json`, `3-patch.diff` (applies as-is with `git apply`), `4-mesure.json` plus
`4-sorties.txt` (raw pytest and pyflakes output), and `5-RAPPORT.md` (computed verdict,
measurements, then the LLM's prose). The report and the patch are also emailed, through the
same path as any other task.

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

Nightly autocoding, switched separately — it also requires `AGENT_ENABLED`:

| Variable | Default | Role |
|---|---|---|
| `AUTOCODE_ENABLED` | `false` | master switch, distinct from `AGENT_ENABLED` |
| `AUTOCODE_HOUR` | `2` | hour of the cycle |
| `AUTOCODE_POOL_FILE` | `DOCS/AUTOCODE.md` | hand-added findings — optional, tracebacks feed the cycle on their own |
| `AUTOCODE_DIR` | `/opt/jarvis/autocode` | shelf for the patches |
| `AUTOCODE_MAX_STEPS` | `40` | step budget for a coding task |
| `AUTOCODE_TIMEOUT_MINUTES` | `90` | wall-clock budget |
| `AUTOCODE_MAX_DIFF_LINES` | `200` | above this the **fix** is rejected — the added test does not count |
| `AUTOCODE_COOLDOWN_DAYS` | `30` | sleep after a rejection |
| `AUTOCODE_MAX_TARGET_LINES` | `800` | target files above this are filtered out |
| `AUTOCODE_PROTECTED` | `config.py`, `prompts*.py`, … | files a patch may never modify — they stay readable |

See **[SECURITY.md](SECURITY.md)** for the threat model, and **[AUTOCODE.md](AUTOCODE.md)**
for the format of hand-added findings.

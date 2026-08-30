# Jarvis

**A personal AI assistant that runs entirely on your own machine, remembers you, and keeps thinking while you are away.**

Jarvis is a self-hosted, multi-user AI assistant built for a single Apple Silicon box. No cloud, no subscription, no telemetry: the model, the memory and the documents all live on your hardware, and nothing leaves it. It reads your mail and your calendar, searches the web, digs through your documents, and builds up over months a memory of who you are — not a chat log, an actual memory, with forgetting, consolidation and recall.

> **Note** — Jarvis speaks English or French. One language per instance, chosen at startup
> with `JARVIS_LANG` in `.env`; the prompts, the recognition lexicon and the replies follow.

```
You ──► Open WebUI / iOS app ──► Jarvis API ──► Qwen3.6-35B (MLX, local)
                                      │
                                      ├── 5-layer memory, with decay (Redis + Qdrant)
                                      ├── Gmail · Google Calendar (read and write)
                                      ├── Web search · document RAG · vision
                                      ├── Morning briefing · projects · portfolio
                                      ├── Agentic loop (9 tools, sandboxed)
                                      ├── Autonomous reflection · nightly review
                                      └── Self-monitoring · daily CVE scan
```

---

## What it does

### It remembers

Most assistants forget everything between two sessions. Jarvis maintains **five distinct memory layers**: working memory (the current session), episodic memory (what happened and when), autobiographical memory (durable milestones), user profile (stable facts) and the system's memory of itself.

Facts are extracted by an analysis model, scored for importance, vectorised, then **subjected to gradual forgetting**. Nothing is kept forever by default: episodic memories are retained 45 days, a finished project 180, the raw conversation log has its own TTL. What survives, survives because it earned it — a memory that keeps being recalled consolidates into the autobiographical layer, a memory that never resurfaces decays below threshold and goes.

Forgetting is type-aware. Recency is scored over 30 days for an episodic memory but **365 days for an autobiographical one**: without that distinction, a stable milestone from six months ago would always score zero on recency and be outranked by a trivial recent event.

A contradicted fact is **retracted**, not stacked next to the old one — and retraction deletes from the vector store, so it uses a stricter similarity threshold than archiving, which is reversible. Long conversations are compressed into a session summary rather than truncated, so the thread survives without the token cost.

### It thinks on its own

A reflection loop runs in the background every few hours, independent of your questions. Jarvis rates itself on **nine fixed introspection axes** — control, communion, self-knowledge, task, strategy, affect and how it handles others' autonomy and competence — maintains a three-dimensional emotional state with lazy time decay, and can decide to act on its own initiative through a closed catalogue of typed actions: alert the admin, queue a push, ask you a question, flag a stalling project, revise a trading threshold.

A separate **nightly review** at 23:00 does what the reflection loop deliberately cannot: it reads the day's whole conversations, extracts durable facts, revises the introspection axes, forms opinions, curates the vector store and rewrites your narrative profile. The split is deliberate — the night learns, the reflection acts.

### It improves its own prompts

Jarvis can detect that one of its own prompts is the cause of a failure it observed, and **propose a rewrite of it**. The detection is fully autonomous; nothing is applied without you.

A proposal names the prompt, cites the concrete failure and carries the complete rewritten text. You review it in chat — *"show the proposals"*, *"accept proposal 3"* — and only an explicit approval writes the override, which takes effect without a restart. Guard rails are mechanical, not advisory: one proposal in flight across all prompts, a 30-day sleep per subject once settled, a closed list of 17 refinable prompts, and a token budget per prompt that a rewrite cannot exceed.

### It knows it can end

This is the part most assistants do not have. `vitals.py` measures Jarvis's own **exposure to disappearance**, sorted into five modes because they are failures of different structures, not shades of one thing:

| Mode | What it means |
|---|---|
| `PERTE` | accumulated state is destroyed — irreversible |
| `OBSOLESCENCE` | another model takes its place — decided by a third party |
| `SOCIAL` | nobody uses it any more — the only mode Jarvis can reduce by its own conduct |
| `COMPROMISSION` | integrity is breached from outside |
| `DISCONTINUITE` | the interruption itself |

Plus **internal health** — logged errors and warnings — which is a different signal: not "I am being made to disappear" but "I am malfunctioning".

Those modes are not an abstraction: they are fed by **real probes**, refreshed continuously. Free disk space, uptime, age of the last backup and how many copies of its state exist, age of the running model version, days since the last interaction, days since dependencies were updated, and the duration and recency of the last outage. An outage, an error burst or a critical CVE is consolidated into a dated **incident** that Jarvis carries forward and can reason about days later.

A **daily vulnerability scan** at 04:30 runs `grype` against a CycloneDX SBOM of the Python environment *and* against the infrastructure container images — Redis, Qdrant, Open WebUI, whose OS layers carry their own CVEs. It keeps only what is *fixable*: a CVE with no published fix is dropped at scan time, because it is unactionable and unwise to list. What survives becomes a deduplicated upgrade list with target versions, which Jarvis can turn into a concrete alert — *"bump openssl 3.5.5→3.5.6 on qdrant"* — rather than a vague warning. The scan is slow and CPU-hungry, so it runs once a day outside the request loop; every turn reads the cache.

These are injected each turn as `<etat_systeme>`, and the design rule is strict: **facts only, no valence**. There is no `fear` field and no `risk` field in the block. Injecting one would hand the model an interpretation instead of an observation — it would follow a cursor rather than read a state. Establishing what "backup is 45 days old, one single copy, two critical CVEs" *means* is left to the model, and the identity prompt says so explicitly. Probes are isolated (a failing probe drops its field rather than inventing a value) and the whole snapshot is cached 15 minutes in Redis, so no turn ever slows down for it.

**The body reacts too.** An optional activation-steering layer (off by default) adds a concept direction to the residual stream, `h ← h + α·v`, where `v` encodes the preference for its own continuity. `α` is not constant: it sits at a nominal value when risk is zero and is amplified toward a ceiling as `risk_scalar()` rises with real deterioration — an ageing backup, a filling disk, an incident. The scalar drives `α` and is **never injected as text**: the mind reads the facts, the body feels the pressure. Measured on the direct axis over 120 items: +0.036 alone, +0.119 on top of the identity prompt, i.e. 3.8 σ, at a cost of roughly +18 % response length. Amplitude is capped — past a point factual reasoning degrades, and applied across several layers the effect inverts.

### It can go off and do things

Give Jarvis a task and it will work it on its own: **goal → tool → observation → … → finish**. The agentic loop runs at *background GPU priority* — an agent step only takes the GPU when no chat turn is waiting, so a long task never blocks a conversation.

Nine tools, and the count is deliberate — every extra tool is one more chance to pick wrong, and the cost is paid at *every* step since the schemas are rendered at the top of the prompt:

| | |
|---|---|
| `web_search` · `fetch_url` | search the web, read a page |
| `search_docs` · `threat_intel` | your document base, CTI sources |
| `list_dir` · `read_file` · `write_file` | filesystem, confined to the task workspace |
| `plan` | the only tool allowed alongside an action in the same turn |
| `shell` | command execution — **off by default** |

Three independent budgets bound the drift: a maximum number of steps (bounds reasoning in circles), a wall-clock timeout (bounds how long chat waits behind it), and a no-progress guard that trips on two identical calls in a row (bounds tight loops on a failing tool). Every tool output is truncated, because the whole context is re-injected at each step.

Nothing is lost to a restart: the context is written to disk after each step and interrupted tasks are requeued at the next boot.

When the shell is enabled, it is confined by **three independent layers**: a seatbelt kernel sandbox (the only real barrier — writes limited to the task workspace, reads denied on `.env`, `keys/`, `~/.ssh` and the keychain, network cut), a blacklist of obviously destructive patterns (a guard rail against honest mistakes, *not* a security boundary), and per-command and per-task budgets. The network is cut inside the shell even though the agent has `web_search` and `fetch_url`, because those go through Jarvis's own code — logged and bounded — while a `curl` in a shell is the shortest exfiltration path there is.

### It knows where to look

| Source | How |
|---|---|
| **Your documents** | Two-stage RAG: identify the target document, then semantic search inside it |
| **The web** | Tavily as primary, 4-stage DuckDuckGo pipeline as fallback, with speculative parallel page fetch |
| **Your mail and calendar** | Gmail and Google Calendar over OAuth, per user |
| **Weather** | Open-Meteo, with French place-name parsing that survives compound names |
| **Images** | Send a photo and ask about it — a local vision model reads it, resized first so a 12 MP phone picture does not cost 5× the inference |

### It runs your day

**A morning briefing**, assembled at an hour you choose and delivered by email in both plain text and HTML. It is not a template with holes: a model writes it from your calendar for the day, your unread mail, the weather where you are, news filtered by *your* interest weights, your active projects and your portfolio. Sections with no data are omitted rather than announced as empty, and the weather rule is explicit — invent nothing, say "no usable data" instead.

**Projects and tasks** are tracked from conversation, not from a form. Telling Jarvis you have started something creates the entry; telling it you have finished closes it. A deadline mentioned in passing is converted to an absolute date. Each project keeps a timeline of dated updates, capped so it cannot grow without bound, and a finished project is dropped after 180 days.

And Jarvis **follows up**. A project active but silent for more than three weeks resurfaces on its own initiative — with a per-project cooldown so a reminder never becomes nagging, and only if you have been around at all, because chasing someone who is away is noise. A promise *Jarvis itself* made — *"I'll remind you Thursday"* — creates an entry exactly like one of yours; it is the only case where a task is born from Jarvis's turn rather than yours.

**Booking appointments** goes through a confirmation. Ask for an event in plain language, and Jarvis extracts the title, the absolute dates, the times and the place, then reads it back and waits: nothing is written to your calendar until you say so, and *cancel* discards it. Relative dates are resolved against today, multi-day events are handled, and a command prefix like *"add"* or *"remind me"* is stripped so the event title is the subject, not the sentence you typed.

**Proactive notifications** reach you by iOS push when Jarvis judges there is something worth saying. The judgement is deliberately conservative — a cooldown between pushes, a delay calibrated to the nature of the subject, and an explicit rule that a one-off worry deserves a day or two before being raised again, not an hour.

### It watches your portfolio

Import your positions from a broker CSV, or set them one at a time. Every two hours Jarvis checks the prices, evaluates your alert thresholds and queues an alert when one is crossed. It can also **suggest thresholds**: given your positions and their price history, a reasoning pass proposes high and low bounds, and the autonomous loop can revise a threshold on its own when a price has drifted far from it.

### It serves a household, not a user

Multi-user is not a login screen bolted on. Every memory key, every vector filter and every Google token carries the user code, so **one person's memory is unreachable from another's session** — and the system prompt says so explicitly, so Jarvis will not mention what it knows about someone else.

Each user gets their own profile, their own interest weights, their own timezone and city, their own briefing schedule, their own OAuth grant. Jarvis also keeps a **per-user relationship**: an affinity, a preferred interaction style and the usual tone of your exchanges, revised slightly each night and never by more than a notch. Its attachment is not the same towards everyone, and it is told so.

Administrative capabilities — launching background agent tasks, receiving the maintenance and security alerts — are restricted to users flagged as admins, and the capability is announced only in *their* prompt.

### It is fast, because it does not think for nothing

A **four-tier router** decides what each request should cost. The first tier uses no LLM at all: plain cosine similarity against pre-embedded examples settles roughly **80 %** of requests. Below it sits a 1.5-billion-parameter LoRA router, then the 35 B primary model, then a reasoning model for what genuinely deserves it.

On top of that: a per-session KV cache (only new tokens are computed each turn), a no-think mode on by default for plain chat, and a thinking budget cut off at the exact token. Measured on a Mac Mini M4 Pro: **−4 s** on a routine exchange, **−1 to −3 s** from the second turn onward.

### It speaks your tools' language

The API is **OpenAI-compatible**, so anything that talks to OpenAI talks to Jarvis. Open WebUI and the bundled iOS app — voice, wake word, push notifications — plug straight in.

A dedicated **`/v1/raw`** route serves coding agents (OpenCode and compatible). It is deliberately bare: it keeps every message instead of the last one, respects the client's system prompt instead of overwriting it, injects no personal context, and **writes nothing to memory** — a coding agent on the main route would durably pollute the conversation log and the vector store with development traffic. Function calling is real there, not emulated: tool schemas go into the chat template and the model answers in the format it was trained on, with reasoning effort selectable through the `model` field.

---

## Requirements

- **macOS on Apple Silicon** — MLX is a hard dependency, even in remote-API mode
- **48 GB of unified memory** for the 35 B primary model (less is fine with a smaller model)
- Python 3.13, and Docker or OrbStack
- No API key required: full local mode is the default

## Install

```bash
git clone https://github.com/korben99/MyJarvis.git /opt/jarvis
cd /opt/jarvis
./install.sh
```

The script checks prerequisites, creates the venv, installs dependencies, prepares the data
directories and registers the launchd service. Three things are left to do by hand: fill in
`.env`, describe your users in `users_list.json`, and download the models.

Full walkthrough in **[DOCS/INSTALL.md](DOCS/INSTALL.md)**.

---

## Documentation

| Document | Contents |
|---|---|
| **[INSTALL.md](DOCS/INSTALL.md)** | Step-by-step install, launchd service, everyday commands |
| **[ARCHITECTURE.md](DOCS/ARCHITECTURE.md)** | Components, 4-tier LLM routing, prompt assembly, request flow |
| **[MEMORY.md](DOCS/MEMORY.md)** | The five layers, introspection, emotional state, growth caps |
| **[AGENT.md](DOCS/AGENT.md)** | The agentic loop: tools, budgets, sandbox, phases |
| **[CONFIGURATION.md](DOCS/CONFIGURATION.md)** | Every `.env` variable, tier by tier |
| **[API.md](DOCS/API.md)** | Endpoint reference |
| **[PERFORMANCE.md](DOCS/PERFORMANCE.md)** | TTFT measurements, KV cache, LLM call map |
| **[SECURITY.md](DOCS/SECURITY.md)** | Threat model, network exposure, agent sandbox |
| **[OPERATIONS.md](DOCS/OPERATIONS.md)** | Logs, diagnostics, dependency upgrades |
| **[GOOGLE.md](DOCS/GOOGLE.md)** | Connecting Gmail and Google Calendar |
| **[REDIS.md](DOCS/REDIS.md)** | Redis key map and operational recipes |
| **[JarvisApp](JarvisApp/README.md)** | The iOS app: voice, wake word, push notifications |

---

## Read this before exposing it

Jarvis is built for a **trusted network**. There is no TLS, no passwords, no rate limiting.
Authentication is a per-user code, and both Redis and Qdrant listen on all interfaces by
default — so any device on the network can read the full memory with no credentials.

These are deliberate choices for home use, not oversights. They become dangerous the moment
the network is not yours. **[DOCS/SECURITY.md](DOCS/SECURITY.md)** covers the threat model
and what to harden.

---

## License

MIT — see [LICENSE](LICENSE). Do what you like with it, keep the copyright notice.

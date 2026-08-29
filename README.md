# Jarvis

**A personal AI assistant that runs entirely on your own machine, remembers you, and keeps thinking while you are away.**

Jarvis is a self-hosted, multi-user AI assistant built for a single Apple Silicon box. No cloud, no subscription, no telemetry: the model, the memory and the documents all live on your hardware, and nothing leaves it. It reads your mail and your calendar, searches the web, digs through your documents, and builds up over months a memory of who you are — not a chat log, an actual memory, with forgetting, consolidation and recall.

> **Note** — Jarvis speaks French only. Its prompts, its identity and every reply it produces are French (`SYSTEM_BASE_FR`, `IDENTITY_FR`); there is no English variant. This documentation is in English so the design is legible to anyone, but the assistant itself is not.

```
You ──► Open WebUI / iOS app ──► Jarvis API ──► Qwen3.6-35B (MLX, local)
                                      │
                                      ├── 5-layer memory (Redis + Qdrant)
                                      ├── Gmail · Google Calendar
                                      ├── Web search · document RAG
                                      ├── Agentic loop (9 tools, sandboxed)
                                      └── Autonomous reflection loop
```

---

## What it does

### It remembers

Most assistants forget everything between two sessions. Jarvis maintains **five distinct memory layers**: working memory (the current session), episodic memory (what happened and when), autobiographical memory (durable milestones), user profile (stable facts) and the system's memory of itself.

Facts are extracted by an analysis model, scored for importance, vectorised, then **subjected to gradual forgetting**: what never resurfaces fades, what matters consolidates. A contradicted fact is retracted, not stacked next to the old one.

### It thinks on its own

A reflection loop runs in the background, independent of your questions. Jarvis rates itself on **nine fixed introspection axes**, maintains a three-dimensional emotional state with lazy time decay, spots stalling projects, and can decide to notify you on its own initiative — through a closed catalogue of typed actions, never free-form code.

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
| **Weather, markets** | Open-Meteo, portfolio surveillance with alerts |

### It is fast, because it does not think for nothing

A **four-tier router** decides what each request should cost. The first tier uses no LLM at all: plain cosine similarity against pre-embedded examples settles roughly **80 %** of requests. Below it sits a 1.5-billion-parameter LoRA router, then the 35 B primary model, then a reasoning model for what genuinely deserves it.

On top of that: a per-session KV cache (only new tokens are computed each turn), a no-think mode on by default for plain chat, and a thinking budget cut off at the exact token. Measured on a Mac Mini M4 Pro: **−4 s** on a routine exchange, **−1 to −3 s** from the second turn onward.

### It speaks your tools' language

The API is OpenAI-compatible. Open WebUI and the bundled iOS app plug straight into it, and a dedicated `/v1/raw` route serves coding agents (OpenCode and compatible) with real function calling and no personal context injection.

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
| **[ROADMAP.md](DOCS/ROADMAP.md)** | What is done, partial, or planned |
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

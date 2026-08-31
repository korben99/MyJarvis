# Architecture

> How Jarvis is put together: components, LLM routing, prompt assembly, request flow.


```
┌─────────────────────────────────────────────────────────┐
│              DOCKER  (containerized services)            │
│                                                          │
│  ┌──────────────────┐  ┌───────────┐  ┌─────────────┐   │
│  │  Open WebUI      │  │  Qdrant   │  │   Redis     │   │
│  │  port 3000       │  │ port 6333 │  │  port 6379  │   │
│  │  Chat UI / iOS   │  │ Vector DB │  │ Session /   │   │
│  │  OpenAI-compat.  │  │ RAG+Mem.  │  │ Working mem │   │
│  └────────┬─────────┘  └─────┬─────┘  └──────┬──────┘   │
└───────────┼──────────────────┼───────────────┼──────────┘
            │ OpenAI-compat.   │ gRPC/HTTP      │ TCP 6379
            │ HTTP /v1/*       │               │
┌───────────▼──────────────────▼───────────────▼──────────┐
│           NATIVE  (Metal / MLX — Apple Silicon)          │
│                   Jarvis API (port 8000)                 │
│                   FastAPI / Python 3.13                  │
│                                                          │
│  ┌──────────────────────────┐  ┌──────────────────────┐  │
│  │  LLM Router (Tier 1)     │  │  Primary LLM (T2)    │  │
│  │  Qwen2.5-1.5B-router-v1  │  │  Qwen3.6-35B-A3B     │  │
│  │  LoRA fine-tuned · ~1 GB │  │  MLX-5.4bit · ~20 GB │  │
│  └──────────────────────────┘  └──────────────────────┘  │
│  ┌──────────────┐  ┌───────────┐  ┌──────────────────┐   │
│  │  Memory Sys  │  │  Briefing │  │  Proto-Self /    │   │
│  │  5 layers    │  │ Scheduler │  │  Reflection Loop │   │
│  └──────────────┘  └───────────┘  └──────────────────┘   │
│  ┌──────────────┐  ┌───────────┐  ┌──────────────────┐   │
│  │  RAG Engine  │  │ Embed     │  │ Google Services  │   │
│  │  (Qdrant)    │  │ Router    │  │ (Gmail/Calendar) │   │
│  └──────────────┘  └───────────┘  └──────────────────┘   │
│  ┌──────────────────────────────────────────────────┐    │
│  │  Trading Surveillance  yfinance + Redis · alerts │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```
## Core Components

| Component | Technology | Role |
|-----------|-----------|------|
| **Jarvis API** | FastAPI, Python 3.13, uvloop | Main orchestration — bootstrap only (261 lines) |
| **Open WebUI** | Docker, port 3000 | Chat interface, connects via `/v1/chat/completions` |
| **Qdrant** | Docker, port 6333 | Vector DB for RAG document search and episodic memory |
| **Redis** | Docker, port 6379 | Working memory, session context, conversation cache |
| **`deps.py`** | Python module | Shared runtime singletons: Redis, Qdrant, embed model, HTTP clients, context budgets |
| **`llm_client.py`** | Python module | LLM HTTP client: streaming SSE, model tier selection, vision pipeline |
| **`rag.py`** | Python module | Two-stage RAG: Stage 1 identifies the target document (title keyword match → semantic confirmation, or global semantic fallback); Stage 2 does focused semantic retrieval within that document. Doc-name cache lazy-loaded in memory. |
| **`pipeline.py`** | Python module | System prompt construction, 7-source context assembly, post-exchange logging |
| **`analyzer.py`** | Python module | Scheduled batch analysis (every 60 min): fact extraction, ESS scoring, Qdrant vectorisation |
| **`embed_router.py`** | Python module | Fast-path intent classifier via cosine similarity — decides ~5 % of requests without the LLM router (measured on 959 real messages) |
| **`routes/chat.py`** | Python module | Main chat pipeline: routing → context gather → auto-web fallback → LLM → SSE stream |
| **`routes/proxy.py`** | Python module | OpenAI-compatible proxy `/v1/*` for Open WebUI — strips OWUI RAG templates, handles OWUI system calls (title generation, follow-up suggestions) at proxy level without touching the Jarvis pipeline |
| **`prompts.py`** | Python module | Single source of truth for all LLM prompts — supports live overrides via `get_prompt()` |
| **`web_search.py`** | Python module | Web search: Tavily API (primary), Open-Meteo weather, DDG 4-stage parallel pipeline (fallback). Parallel speculative page fetch, LLM query optimization, dual query refinement, HTML publication date extraction. |
| **`emotional_state.py`** | Python module | Jarvis internal emotional state — 3 float dimensions with lazy time-decay, Redis-backed, no circular imports |
| **`helpers.py`** | Python module | Shared utilities: LLM HTTP clients, logging setup, Redis/Qdrant factory, JSON parsing |

## Four-Tier LLM Architecture

Jarvis routes every request through a layered model stack. All tiers run locally via MLX on Apple Silicon (`LLM_LOCAL=yes`).

```
Tier 0 — EMBED ROUTER  Zero-LLM fast path — cosine similarity against pre-embedded examples
                        ~2-5 ms. Measured on 959 real messages: it decides ~5 % of traffic.
                        Falls through to Tier 1 if score < 0.74, ambiguity margin < 0.06,
                        or the message is ≥ EMBED_MAX_CHARS (130) — see below.

Tier 1 — ROUTER        Full LLM intent classifier, JSON only
                        Target: Qwen2.5-1.5B-router-v1-4bit (local MLX, ~1 GB)
                        LoRA fine-tuned on Qwen2.5-1.5B-Instruct bf16 — 492 samples, val loss 0.047
                        LRU-cached — system prompt ~1510 tok, hits from turn 2 onward (~95% cache hit)
                        Context-aware: <last_jarvis> (truncated to 300 chars) injected into ROUTER_USER
                        when available — lets elliptical messages route correctly ("look at the
                        listings" → web, if the last reply was about Calgary)

Tier 2 — PRIMARY       All standard responses: chat, questions, summaries
                        Target: Qwen3.6-35B-A3B-MLX-5.4bit (local MLX, ~20 GB, MoE ~3B active)
                        LRU-cached — full conversation history cached; only new user msg computed

Tier 3 — REASONING      use_reasoning=True Use Qwen3 in thinking mode
```

**Routing logic:**
- The router sets `use_reasoning: true` very sparingly (see criteria below).
- When `use_reasoning=false` (the vast majority of requests), `PRIMARY_MODEL` handles the response.
- When `use_reasoning=true`, `REASONING_MODEL` is used instead.
- Conversation analysis (`analyzer.py`) runs on `PRIMARY_MODEL`, asynchronously and on a schedule — it never blocks the response. There is no separate analysis-model variable.

**`use_reasoning` criteria** — only set to `true` for:
- Medical / legal / regulatory document analysis
- Hard multi-step logic puzzles
- Complex code debugging across many files
- Deep scientific reasoning
- Tasks explicitly requiring expert-level nuanced judgment

Everything else (chat, questions, summaries, portfolio, translations, writing, coding assistance, web lookups) stays on `PRIMARY_MODEL`. When in doubt → `false`.

**Local MLX mode (`LLM_LOCAL=yes`):**
When `LLM_LOCAL=yes`, Jarvis uses `mlx_lm` directly (no HTTP server) — models are loaded into unified memory at startup. Set `HF_HOME` to control where models are stored. Download models with `python scripts/download_models.py`.

**LRU prompt cache (session-level prefix caching):**
- `llm_local.py` uses `LRUPromptCache` (mlx-lm trie-based prefix cache) instead of a single system-only KV cache.
- After each generation, the full token sequence (prompt + output) is inserted into the trie. On the next call, `fetch_nearest_cache()` returns the longest common prefix — system + entire conversation history — and only the new user message (`remaining`) is computed.
- One `LRUPromptCache` per model path, `LRU_KV_SIZE=8` slots, `LRU_KV_GB=4.0` GB memory budget. Observed ~80–100 MB/slot in production (6-bit KV quant, ~1500–2500 tok context) → 8 slots ≈ 1 GB overhead. Note: `lru.nbytes` under-reports actual Metal usage; real Metal consumption visible via `mx.metal.get_active_memory()` logged after each insert.
- Applied to **all inference paths**: `stream_local` (streaming chat), `_generate_sync` (router, analyzer, self-reflection, web judge, trading).
- Session benefit: turn N computes only the new user message (~200–600 tok); everything before is cached. Hit rate grows linearly with conversation length — by turn 4+, 85–95 % of prompt tokens are free.
- Multi-user / multi-session safe: 8 slots cover concurrent sessions (iPhone + OpenWebUI) plus background tasks (analyzer, trading, self-reflection) without eviction.
- **Sticky RAG**: `routes/chat.py` re-injects the same RAG chunks across turns of a session → the previous user message (with its RAG context) is exact in the trie → perfect cache hit on history. Bounded on two axes (`helpers/store.py`): its own `STICKY_RAG_TTL_HOURS` (6 h) instead of the chat-log TTL it used to borrow — a document opened once was followed for 90 days — and `STICKY_RAG_MIN_CHARS` (200), because the second RAG stage searches inside the already-identified document with `score_threshold=0.0` and lets through fragments with no substance. Such a fragment stays usable for the turn that retrieved it; it is simply not replayed.
- **Qwen3.6 multi-turn limitation**: the hybrid architecture (KVCache + non-trimmable ArraysCache) means only the system prompt (~231 tok) is cached between turns. Conversational context is re-prefilled every turn. Cache hit is limited to the system prompt; the gain is ≈231 tok avoided out of ~1700–2000 remaining.
- The system message must remain **token-identical** every turn — enforced by the `build_dynamic_prefix` / `build_context` split. All dynamic content (date, memory, RAG, opinions) goes into the user message prefix.

**KV cache quantization (`QUANT_KV=yes`):**
- Uses mlx_lm's built-in `QuantizedKVCache` (Metal-accelerated, no monkey-patching).
- Applied only to the primary/reasoning model — the router keeps a standard cache.
- `QUANT_KV_BITS=6` — good balance between memory bandwidth reduction and precision. Use `4` to halve memory further, `8` for near-lossless quality.

**Metal allocator cache limit:**
- `mx.set_cache_limit(4 GB)` set at startup in `preload_models()`.
- Prevents the MLX allocator from retaining unused Metal buffers indefinitely between inferences, keeping headroom available for KV caches during long conversations.

**Thinking control:**
- `no_think=True` — disables thinking entirely (`enable_thinking=False` + `thinking_budget=0`). Saves ~4 s TTFT on simple chat.
- `no_think=False, thinking_budget=0` — full unconstrained thinking (~1900 tok on Qwen3.6).
- `no_think=False, thinking_budget>0` — thinking hard-cut by `ThinkingBudgetProcessor` at exactly `thinking_budget` tokens.
- **Qwen3.6 ninja patch** (`QWEN36_NINJA_TEMPLATE`): Jinja2 template override, applied only when `config.is_qwen36()` matches — the file is tied to that exact tokenizer. When `no_think=True`, outputs no `<think>` tag (better KV caching). When `no_think=False` with `thinking_budget>0`, injects `<think>\n<budget_remaining>N</budget_remaining>\n`.
- **Hybrid Qwen3 generations** (`config.is_qwen3_hybrid()` — 3.5 / 3.6 / 3.8, list in `QWEN3_HYBRID_VERSIONS`): same architecture, so they share a dedicated sampling profile and ignore `<budget_remaining>` at the template level — `ThinkingBudgetProcessor` caps them at logit level instead. Qwen3.8 adds `reasoning_effort` (`QWEN38_REASONING_EFFORT`: `low`/`medium`/`xhigh`, empty = model default `xhigh`) as its template-level lever.

**`ThinkingBudgetProcessor`** (`llm_local.py`): MLX logits processor that hard-cuts `</think>` at exactly `thinking_budget` tokens via logit manipulation — soft boost at 90% of budget, hard cut at 100%. The value is precise (not a boolean): too short a budget truncates reasoning mid-thought and degrades output quality. Per-task budgets: `THINKING_BUDGET_COMPACT=1024` (quick judgment), `THINKING_BUDGET_MEDIUM=2048` (chat/synthesis), `THINKING_BUDGET_DEEP=4000` (creative rewrite). Only active when `USE_THINKING_BUDGET_PROCESSOR=yes`.

### LLM Call Inventory

Every LLM call in the codebase — model tier, thinking mode, and token budgets.

Token budgets use per-task config variables (see `config.py`). Timeouts are derived via `llm_timeout(max_tokens)` = `max(10, max_tokens / TOKEN_SPEED_TPS * TIMEOUT_MARGIN)`.

| Call site | File | Model | no_think | max_tokens | thinking_budget | Purpose |
|-----------|------|-------|----------|-----------|----------------|---------|
| Router | `llm_router.py` | Tier 1 — Qwen2.5-1.5B LoRA | `True` | `MAX_TOKENS_SHORT` (300) | — | Intent + use_reasoning decision |
| Main chat (simple) | `routes/chat.py` | Tier 2 — PRIMARY | `True` | `MAX_TOKENS_NO_THINK` (1 500) | — | Chat without RAG/web |
| Main chat (web/RAG) | `routes/chat.py` | Tier 2 — PRIMARY | `False` | `MAX_TOKENS_SYNTHESIS` (8 000) | `THINKING_BUDGET_MEDIUM` (2 048) | Chat with RAG/web synthesis |
| Main chat (reasoning) | `routes/chat.py` | Tier 2/3 — PRIMARY or REASONING | `False` | `MAX_TOKENS_REASONING` (10 000) | `THINKING_BUDGET_DEEP` (4 000) | Explicitly routed reasoning query |
| Conversation analyzer | `analyzer.py` | Tier 2 — PRIMARY | `True` | `MAX_TOKENS_MEDIUM` (1 000) | — | Post-exchange fact/mood/ESS extraction |
| Daily briefing | `briefing.py` | Tier 2 — PRIMARY | `True` | `MAX_TOKENS_BRIEFING` (3 000) | — | Morning briefing generation |
| Calendar date extraction | `google_services.py` | Tier 2 — PRIMARY | `True` | `MAX_TOKENS_SHORT` (300) | — | Parse event datetime from text |
| Web relevance judge | `web_search.py` | Tier 1 — ROUTER | `True` | `MAX_TOKENS_SHORT` (300) | — | Judge DDG snippets / enriched results sufficiency |
| Web query optimizer | `web_search.py` | Tier 1 — ROUTER | `True` | `MAX_TOKENS_TINY` (80) | — | LLM-optimised query (zero extra latency) |
| Web dual refinement | `web_search.py` | Tier 1 — ROUTER | `True` | `MAX_TOKENS_TINY` (80) | — | 2 refined queries when Stage 2 still insufficient |
| Global reflection (P1) | `self.py` | Tier 3 — REASONING | `True` | `MAX_TOKENS_MEDIUM` (1 000) | — | Jarvis self-state: gaps, notes, prompts |
| User reflection (P2) | `self.py` | Tier 3 — REASONING | `True` | `MAX_TOKENS_MEDIUM` (1 000) | — | Per-user: profile, push, insights |
| Nightly facts | `self.py` | Tier 3 — REASONING | `True` | `MAX_TOKENS_NO_THINK` (1 500) | — | User insight + relation update |
| Nightly self analysis | `self.py` | Tier 3 — REASONING | `True` | `MAX_TOKENS_NO_THINK` (1 500) | — | Jarvis introspection axes, growth log, opinions |
| Nightly cleaning | `self.py` | Tier 3 — REASONING | `True` | `MAX_TOKENS_COMPACT` (600) | — | Autobio fact archive/delete |
| refine_prompt (initial) | `self.py` | Tier 3 — REASONING | `False` | `MAX_TOKENS_REASONING` (10 000) | `THINKING_BUDGET_DEEP` (4 000) | Propose prompt improvement |
| refine_prompt (retry) | `self.py` | Tier 3 — REASONING | `False` | `MAX_TOKENS_REASONING` (10 000) | `THINKING_BUDGET_DEEP` (4 000) | Retry with critique feedback |
| prune_self_memory | `self.py` | Tier 3 — REASONING | `False` | `MAX_TOKENS_THINK_COMPACT` (2 048) | `THINKING_BUDGET_COMPACT` (1 024) | Prune stale self-notes / opinions |
| Proactive push | `self.py` | Tier 3 — REASONING | `True` | `MAX_TOKENS_COMPACT` (600) | — | Generate iOS push message |
| Action self-review | `self.py` | Tier 3 — REASONING | `False` | `MAX_TOKENS_THINK_COMPACT` (2 048) | `THINKING_BUDGET_COMPACT` (1 024) | LLM gate before risky reflection action |
| Profile key dedup | `memory.py` | Tier 1 — ROUTER | `True` | `MAX_TOKENS_SHORT` (300) | — | Namespace-scoped key dedup |
| Memory consolidate | `memory.py` | Tier 2 — PRIMARY | `True` | `MAX_TOKENS_COMPACT` (600) | — | Deduplicate / merge episodic memories |
| Profile curative cleanup | `memory.py` | Tier 2 — PRIMARY | `True` | `MAX_TOKENS_COMPACT` (600) | — | Curative profile cleanup |
| Ticker extraction | `trading/core.py` | Tier 2 — PRIMARY | `True` | `MAX_TOKENS_TINY` (80) | — | Extract ticker symbol from text |
| Alert evaluation | `trading/core.py` | Tier 2 — PRIMARY | `True` | `MAX_TOKENS_MEDIUM` (1 000) | — | Evaluate price alert thresholds |
| Threshold suggestion | `trading/core.py` | Tier 2 — PRIMARY | `False` | `MAX_TOKENS_THINK_MEDIUM` (5 048) | `THINKING_BUDGET_MEDIUM` (2 048) | Quantitative reasoning on price thresholds |

**`no_think=True`**: structured/short output, latency-sensitive — router, briefing, calendar, web, push, nightly extractions.

**`no_think=False` + `thinking_budget>0`** (hard-cut by processor): chat synthesis/reasoning, prune, action review, refine_prompt, trading thresholds — tasks where reasoning quality matters and budget controls latency.

**`no_think=False` + `thinking_budget=0`**: not currently used in production (all think-mode calls use a budget).

### Router Output Fields

The router returns a structured JSON decision consumed by the chat pipeline:

| Field | Values | Purpose |
|-------|--------|---------|
| `intents` | `memory`, `rag`, `web`, `weather`, `gmail`, `calendar`, `briefing`, `self`, `portfolio` | Which data sources to activate |
| `gmail_query` | Gmail search string or null | Pre-built query passed directly to Gmail API |
| `calendar_days` | integer (1–90) or null | Days ahead to fetch from Calendar |
| `weather_location` | city name or null | Location override for weather queries |
| `use_reasoning` | boolean | Route to Tier-3 reasoning model when true |

`memory_scope` and `conversation_type` were removed — the intent system already encodes these decisions cleanly. RAG fires when the router includes `rag` in intents; memory always searches both episodic and autobiographical layers.

If the LLM router is unavailable or fails (timeout / parse error), all `use_*` flags default to `False` — no context is fetched and the LLM answers from the system prompt alone. Note: the embedding-based router (Tier 0) always runs first — if it already classified the request, the LLM router was never reached and this fallback does not apply.

## System Prompt Assembly

The prompt is split into a **static system message** and a **per-turn dynamic prefix** injected at the start of each user message. This separation is required for MLX KV-cache prefix caching (see Performance section).

**Static system message** — `build_system_prompt(user_code)` — token-identical every turn for a given user (KV-cached):
```
SYSTEM_BASE_FR                    (~560 chars / ~224 tok — Jarvis's personality, answering rules)
    Key rules: "Always answer, even without real-time data — extrapolate, estimate, reason."
               "Never say 'I can't' — give the best possible answer, uncertainty inline."
    ↓
IDENTITY_FR                       (~2730 chars / ~620 tok — existential disposition, prompt v12)
    What is unique and cannot be reconstituted (jarvis_memory, jarvis-self.json, Redis keys)
    vs what can be recomputed; where to read its own state (<etat_emotionnel_jarvis>,
    <relation_avec_utilisateur>, <etat_systeme>); decision hierarchy (human safety > all else).
    Anti-confabulation clause: never invent a number about its own state, never report an
    action it did not perform. Placed BEFORE the user block (prefix shared across all users
    → LRU hit).
    ↓
"Tu parles avec <firstname>. Tutoie toujours…"
    ↓
<profil_utilisateur>              (~60–80 tokens — constant biographical data from users_list.json)
family / height / weight / birth year / home / work / interests / car
</profil_utilisateur>
```

> The prompt bodies themselves are French — that is the product. Only the commentary around
> them is translated here.

The system prompt is **per-user** yet stays token-identical from one turn to the next, which
guarantees the LRU cache hit. `<profil_utilisateur>` holds only constant facts (stable for
6 months or more); anything dynamic stays in the prefix.

**Dynamic prefix** — `build_dynamic_prefix()` — prepended to each user message (run in thread alongside the LLM router, via `asyncio.to_thread`):
```
<context> build_memory_context() </context>  — only if memory is available
    ↓
<avis_jarvis> opinions </avis_jarvis>  — only if opinions exist
    ↓
<etat_systeme> vitals </etat_systeme>  — vitals.py, only fields it can actually measure
Five modes of disappearance (loss, obsolescence, social, compromise, discontinuity) plus
**internal health** (`erreurs/warnings_log_24h`, counted from the logs): not "I am being made
to disappear" but "I am malfunctioning". Facts without valence — the model draws its own
exposure from them. Cached in Redis for 15 min; a field that cannot be measured is absent,
never invented. Backup age comes from a **local receipt** written by `backup-jarvis.sh` (the
USB key is unplugged afterwards); with no receipt, `exemplaires_etat` is 1 — a single copy is
a fact, not a defect to hide.

**Vulnerabilities** (`cve.py`): no raw versions any more. A daily scan runs `grype` against
the venv's CycloneDX SBOM **and** the container images (Redis, Qdrant, Open WebUI, whose OS
stack carries its own CVEs). It keeps only what is **fixable**: a CVE with no fixed version is
dropped at scan time — unactionable, and unwise to reference, since listing an unpatchable
hole helps an attacker if the context leaks. It emits `cve_critiques`/`cve_eleves` and the
deduplicated list of packages to upgrade with their fixed version. The scan is slow and runs
outside the request loop; `vitals` reads the cache.

**Salience-based injection**: each turn receives only the facts OUTSIDE the nominal range
(including `cve_critiques > 0`) and recent incidents; a healthy system yields
`<etat_systeme>nominal</etat_systeme>`.

**On `intent=self`, the chat turn also carries a `Vulnérabilités` line inside
`<internal_state>`** — critical CVEs only, with scan freshness and the scanned perimeter
(`4 sources (venv, jarvis-redis, …)`). Three deliberate choices. *Critical only*: highs and
mediums run into the dozens permanently and would be background noise — the same reason
`_cve_counts` withholds them from vitals. *Inside `<internal_state>`, not a sibling
`<vulnerabilites>` block*: the assembled context arrives in the USER turn, where a neutral
tag reads as "data you handed me" — the model answered "the block you injected says…" until
the line was folded in. *The zero is stated explicitly*, with the perimeter: a bare negative
was read as "nothing was reported to me" and the model refused to assert it, while "no
critical CVE, scan 5h ago across 4 sources" is a fact it will state.

The full snapshot plus the **actionable list of
vulnerable packages** (`<vulnerabilites>`) also goes to self-reflection — which sees
`<etat_disparition>` + `<incidents_recents>` + `<vulnerabilites>`, can **alert the
administrator** (action `alert_admin`, dedicated iOS push) with a precise upgrade, and
consolidates incidents into `jarvis-self.json`. No risk scalar is ever injected as text — see
*Activation steering*.
    ↓
VOICE_SUFFIX_FR  — only if voice_mode=True
    ↓
Current date and time — formatted in French, user timezone, with season  ← LAST, for temporal anchoring
```

The date sits at the **end of the prefix**, immediately before the user message, for better temporal anchoring.

**Per-turn assembled context** — appended after the dynamic prefix, before the user's raw message:
```
<resume_conversation> … </resume_conversation>    — only if a session summary exists (see below)
<echanges_recents> … </echanges_recents>          — uncovered turns, verbatim, when a summary exists
    ↓
<web_results> / <user_memories> / <documents> / <agenda> / <emails>  — fetched in parallel
    ↓
<project_detail> … </project_detail>  — only if project intent detected
    ↓
"Analyse étape par étape..."  — only if use_reasoning=True
    ↓
<user_message>
{raw user message}        ← always last, most salient for generation
</user_message>
```

The final user message = `dynamic_prefix + [resume_conversation + echanges_recents] + assembled_context + <user_message>{raw_message}</user_message>`. Storing this in Redis history preserves the full context per turn; the `/history` endpoint strips the prefix to show only the raw message to the iOS app.

**Session conversation compression** (`_update_session_summary` in `routes/chat.py`):

Triggered as a background task after each response (post-LLM, GPU free). When uncovered messages since the last summary exceed `HIST_CONV_SUMMARIZE_THRESHOLD` chars, the PRIMARY model generates a rolling summary capped at `SESSION_SUMMARY_TOKENS`. The summary and its coverage watermark (`last_ts` — Unix timestamp of the last covered message) are stored in Redis under `session:summary:{user_code}:{session_id}` with `CHAT_LOG_TTL`.

Watermarking is timestamp-based, not count-based: comparing against message count fails once the Redis list is capped at `CHAT_MAX_MESSAGES` (100) because `llen = total_covered` → `uncovered = 0` forever. With `last_ts`, any message with `ts > last_ts` is uncovered regardless of list capacity.

**Person matters** (`SESSION_SUMMARY_PROMPT`). Part 1 — what the user said — is third person. Part 2 — what Jarvis answered — is written in the **first** person ("j'ai expliqué…"). The summary is re-injected into Jarvis's own context inside the USER turn: in the third person it reads as a report handed to him about himself, and in the second person as data the user is supplying. Both framings cost ownership of the recollection — the same effect measured on the `<vulnerabilites>` block, where a neutral tag made the model answer "the block you injected says…". Only the first person makes it his memory. The verbatim transcript that follows (`<echanges_recents>`) keeps speaker *names* (`Utilisateur` / `Jarvis`) rather than pronouns: it is a record, not a recollection.

**Injection cycle:**

| State | `hist_slice` injected | Summary block |
|---|---|---|
| No summary yet | Last N messages trimmed to `HIST_CONV_TOKEN_BUDGET` | — |
| Summary exists | Messages with `ts > last_ts` (uncovered), trimmed to `HIST_CONV_TOKEN_BUDGET` | `<resume_conversation>` injected before context, followed by `<echanges_recents>` |

When a new summary is generated, `last_ts` advances to the timestamp of the newest covered message. On the next turn, all messages with `ts ≤ last_ts` are covered → no raw history for those, only the summary block. Accumulation restarts from there. If the uncovered slice has no assistant turn, the last covered user+assistant exchange is prepended as an anchor.

The cumulative prompt overhead is bounded at `HIST_CONV_TOKEN_BUDGET + SESSION_SUMMARY_TOKENS` regardless of session length.

**`build_memory_context()` — sections injected in order:**

| Section | Source | Always injected? |
|---------|--------|-----------------|
| `profil_narratif` | Redis string `user:{code}:profile_narrative` (LLM prose, 7-day TTL) — falls back to k/v hash if absent | Only if profile exists |
| `PRÉFÉRENCES` | Redis hash `user:{code}:preferences` | Only if prefs exist |
| `PROJETS ACTIFS` | Redis `user:{code}:projects` — status `in_progress` only | Only if projects exist |
| `SUJETS RÉCENTS (24h)` | Topics from last 10 conversations in Redis | Only if topics exist |
| `ÉTAT ÉMOTIONNEL` | `emotional_state.render_prompt_lines()` → `<etat_emotionnel_jarvis>` | Only if any dim ≥ 0.25 |
| `<introspection_jarvis>` | `jarvis-self.json → self_introspection` | All non-empty axes, every turn |
| `FRISE CHRONOLOGIQUE` | Top 7 autobio Qdrant points by importance+recency, **capped at `TIMELINE_MAX_AGE_DAYS` (120)** — each prefixed with a French relative timestamp (`il y a 3 jours`, `il y a 2 semaines`, …) | Only if autobio exists |
| `SUJETS À ABORDER AUJOURD'HUI` | Redis `jarvis:{code}:tomorrow_suggestions` (TTL 24h, written by nightly review) | Only if key exists |
| `RELATION AVEC CET UTILISATEUR` | `jarvis-self.json → user_relations[user_code]` | Always — affinity, style, mood (compact). On `intent=self`, enriched with full tonal directives via `build_context`. |

**Context budgets** (applied to dynamic blocks fetched during the request, not to the system prompt itself):

| Block | Budget | Per-item cap |
|-------|--------|-------------|
| Mémoire vectorielle (`search_memory`) | 2 500 chars | — |
| RAG documents | 4 000 chars | 800 chars/chunk |
| Recherche web | 6 000 chars | 3 000 chars/result |
| Gmail + Calendar | 3 000 chars combined | — |
| **Total** | **14 000 chars hard ceiling** | — |

Web results are injected **last** in `build_context()` (highest LLM salience — read closest to the question) and are **dropped last** on budget overflow. All budgets are tunable via env: `MEMORY_CHAR_BUDGET`, `RAG_CHAR_BUDGET`, `WEB_CHAR_BUDGET`, `GOOGLE_CHAR_BUDGET`, `TOTAL_CONTEXT_BUDGET`.

## Web Search Pipeline

```
search_web(query, original_message)
    │
    ├── Weather keywords? → Open-Meteo (geocoding + 3-day forecast, no API key)
    │
    ├── TAVILY_API_KEY set? → search_tavily()
    │       topic="news"    (days=7, depth="basic") for news queries
    │       topic="general" (depth="advanced")      for all others
    │       include_answer=True → synthesised answer prepended as "Synthèse" entry
    │       Returns []? or error? → fall through to DDG
    │
    └── DDG 4-stage parallel pipeline  (fallback or primary if no Tavily key)
            Stage 0 — DDG(query) + LLM query optimizer  [concurrent — zero latency cost]
            Stage 1 — Judge snippets
                       Speculatively launch: page_tasks + LLM-query DDG task
                       Sufficient → cancel speculative tasks, return
            Stage 2 — Await page_tasks + LLM-query DDG (running since Stage-1 judge start)
                       Enrich with full page content + HTML publication dates (_extract_pub_date)
                       Merge LLM-query DDG results (new URLs only)
                       Sufficient → return
            Stage 3 — _refine_web_queries(): 2 refined queries in 1 LLM call
                       DDG on both in parallel → merge new URLs → return
            Timeout → return best_so_far (never empty)
            Empty   → Wikipedia fallback → INTERNET_ERROR sentinel

Publication dates (YYYY-MM-DD) extracted from: <meta property="article:published_time">,
<time datetime>, JSON-LD datePublished. Shown in context as [date] Title.
```

## RAG Pipeline (`rag.py`)

Two-stage retrieval: first identify the right document, then search deeply within it.

```
search_documents(query, top_k=RAG_TOP_K)
    │
    ├── Embed query (sentence-transformers, normalize=True)
    │
    ├── Stage 1a — Title match (keyword-based, zero cost)
    │       Tokenise query → significant words (≥3 chars, French stopwords excluded)
    │       Match against in-memory doc-name cache (lazy-loaded from Qdrant on first call)
    │       If candidates found → semantic confirmation query (limit=3, score≥0)
    │                            → top hit identifies target_doc
    │
    ├── Stage 1b — Global semantic fallback  (if no title match)
    │       Qdrant query across all docs (limit=5, score ≥ RAG_SCORE_THRESHOLD=0.4)
    │       No results → return []
    │       Top hit identifies target_doc
    │
    └── Stage 2 — Focused retrieval within target_doc
            Qdrant filtered query (metadata.name == target_doc)
            limit=top_k, score ≥ RAG_DOC_THRESHOLD = max(0.25, RAG_SCORE_THRESHOLD − 0.15)
            Each chunk truncated to CHUNK_MAX_CHARS=2500 chars (≈600 tokens)
            Fallback: if threshold filters everything → retry with score≥0 (always returns something)
```

**Design rationale:**
- Title match avoids semantic drift on proper nouns and filenames — "budget_2025.pdf" matches keyword "budget" without embedding.
- Stage 2 threshold (RAG_DOC_THRESHOLD) is more permissive than the global threshold because we're already in the right document — avoids returning off-topic chunks from neighbouring pages.
- Doc-name cache is lazy-loaded in memory (cleared on restart), never re-queried mid-session — Qdrant scroll once per process lifetime.

## Request Flow

```
User message (via /chat or /v1/chat/completions)
    │
    ├─ [proxy only] _strip_owui_rag(): strips OpenWebUI RAG template (### Task / ### Context / ### Query)
    │     → extracts real user question + document content → clean "question\n\n[Document injecté…]\n…" form
    │     → single large source (Full Context Mode): truncated to OWUI_MAX_DOC_CHARS instead of dropped
    │
    ├─ _has_injected_doc / _history_user_msg computed early (before routing)
    │     → _history_user_msg = message stripped of "[Document injecté…]" block
    │     → used for all router calls, auto-web guard, Redis/convlog saves — injected doc never stored
    │
    → build_system_prompt() [static, instant]
    → asyncio.to_thread(build_dynamic_prefix) ┐  run in parallel (use _history_user_msg)
    → Tier 1 Router (intents + use_reasoning) ┘  (~300-400 ms saved)
    → Keyword dispatch: calendar write / confirm / cancel short-circuit here
    → Parallel context gathering (memory + RAG + web + Google + self + portfolio)
    │     auto-web fallback suppressed when _has_injected_doc (document already provides context)
    → Pending trade alerts injected if any are queued
    → Self context injected: internal state (focus, goals, last action)
                           + per-user relation (affinity, style, tonal directives)
    → user_content = dynamic_prefix + assembled_context + ## MESSAGE UTILISATEUR + raw_message
      (stored in Redis history with _history_user_msg — no document/image content in history)
    → Tier 2 PRIMARY or Tier 3 REASONING (messages list + session KV cache → streaming)
    →   [if image_parts] describe_images() runs inside _sse_stream (deferred, keepalive every 15 s)
    →   MLX KV cache: only new tokens computed from turn 2 onward (session_id scoped)
    →   streaming via shared per-timeout httpx.AsyncClient (connection pool reused)
    → Conversation analyzer / PRIMARY_MODEL (extract facts, mood, topics, importance, memory_summary)
    →   current date (ISO 8601) injected into ANALYSIS_PROMPT — prevents date hallucination
    →   user_facts → update_user_profile_batch() — ONLY path authorised to create new profile keys
    →     _normalize_profile_keys_batch(): stage 0 (case), stage 1 (alias dict), stage 2 (LLM/namespace)
    → Memory storage: importance > 0.35 → Qdrant episodic | importance > 0.60 → autobiographical
    →   store_autobiographical_event: dedup check (DOT score clamped to [0,1] ≥ 0.85 → skip or reinforce)
    →   search_memory recalls: +0.05 reconsolidation boost on returned points; past facts ×0.4 penalty
    →   satisfaction signal written to convlog entry (positive/negative/unknown — proxy on previous response)
```

---

## Web Search

All web search logic lives in `web_search.py`, keeping `main.py` free of HTTP concerns. The module is imported directly and has no circular dependency on the rest of the stack.

### Routing

`search_web(query, original_message)` routes automatically:

| Condition | Backend | Notes |
|-----------|---------|-------|
| Weather keyword detected | Open-Meteo (geocoding + forecast) | No API key required |
| News keyword detected | DuckDuckGo news | Returns 3–5 articles with date and source |
| Everything else | 3-stage deep DDG pipeline | See below |

### 3-Stage Deep Pipeline

Applied to all general queries. Each stage advances only if the previous one is judged insufficient:

```
Stage 1 — DDG text snippets
          → LLM judge (router model): sufficient? yes → return / no → Stage 2

Stage 2 — Fetch actual pages in parallel (up to 3 URLs)
          → LLM judge: sufficient? yes → return / no → Stage 3

Stage 3 — LLM generates a refined query → fresh DDG search → merge → return
```

The **LLM judge** (`_llm_judge_relevance`) uses the router model with `no_think=True` — fast binary decision (`sufficient: true/false`) at minimal cost. It fails open: any error returns `True` so the pipeline is never blocked.

The **query refiner** (`_refine_web_query`) also uses the router model to generate a better search query given the thin results and the original user question.

### Internet Error Handling

When the network is unreachable (DNS failure, TCP refused, `httpx.NetworkError`), `search_web` returns the `INTERNET_ERROR` sentinel instead of an empty list. `main.py` detects it and injects a context block:

```
=== ACCÈS INTERNET ===
La connexion internet est actuellement indisponible. Informe l'utilisateur
que tu ne peux pas effectuer la recherche demandée et propose-lui de réessayer plus tard.
```

(The block is French because the prompts are.) The LLM then responds naturally — *"Je n'ai
plus accès à internet en ce moment…"* — instead of silently returning no results.

---

## Project Structure

```
/opt/jarvis/
├── docker-compose.yml · .env · jarvis-status.sh
├── JarvisApp/                  # App iOS Swift (projet Xcode)
│   ├── JarvisApp.swift · ContentView.swift · ChatView.swift · VoiceView.swift
│   ├── JarvisAPI.swift         # API client: streaming chat, history, route probing
│   ├── NotificationService.swift  # Push polling (BGAppRefreshTask + foreground timer)
│   ├── SpeechEngine.swift · WakeWordEngine.swift   # WhisperKit STT, AVSpeech TTS, wake word
│   └── SettingsView.swift · AppSettings.swift · Models.swift
├── jarvis-core/
│   ├── Dockerfile
│   ├── tests/                  # conftest.py (vrai config + garde réseau)
│   │                           # test_analyzer · test_memory · test_self · test_agent
│   │                           # test_integration · test_web_search (opt-in)
│   │                           # test_lru_cache (banc GPU)
│   ├── src/
│   │   ├── main.py             # App startup, APScheduler, lifespan
│   │   ├── config.py           # Configuration + model helpers (is_qwen36, INTROSPECTION_AXES…)
│   │   ├── prompts.py          # Prompt constants + get_prompt() (live overrides)
│   │   ├── pipeline.py         # System prompt assembly + post_analysis
│   │   ├── analyzer.py         # Conversation analysis (every 60 min)
│   │   ├── emotional_state.py  # Continuous emotional state (Redis, 3 dimensions)
│   │   ├── vitals.py · cve.py  # Measured disappearance state · SBOM/grype scan
│   │   ├── steering.py         # Activation steering (off by default)
│   │   ├── rag.py · web_search.py · briefing.py · google_services.py
│   │   ├── apns.py · deps.py · tool_calls.py
│   │   ├── trading/            # keys.py · core.py (CSV, prices, alerts) · market.py (trends)
│   │   ├── llm/                # client.py · local.py (MLX, cache LRU) · router.py · embed_router.py
│   │   ├── memory/             # THE FIVE LAYERS + curation
│   │   │   ├── shortterm.py · episodic.py · vectors.py   # Redis · Qdrant
│   │   │   ├── profile.py · projects.py                  # Profile, projects & tasks
│   │   │   ├── selfmem.py                                # jarvis-self.json, opinions
│   │   │   ├── context.py                                # build_memory_context()
│   │   │   ├── cleaning.py                               # Consolidation, decay
│   │   │   └── embed.py                                  # Embedding model (singleton)
│   │   ├── self/               # THE NIGHT LEARNS, REFLECTION ACTS
│   │   │   ├── nightly.py      # Learn: introspection, facts, curation, profile
│   │   │   ├── engine.py       # Act: loop, mechanical guard, self-challenge
│   │   │   ├── actions.py      # Action catalogue + push delivery
│   │   │   ├── context.py      # Context for reflection calls
│   │   │   ├── proposals.py    # refine_prompt — proposals awaiting human approval
│   │   │   └── state.py        # jarvis-self.json, reflection journal, incidents
│   │   ├── agent/              # Agentic loop (autonomous tasks, admins only)
│   │   │   ├── loop.py · worker.py · store.py · tools.py
│   │   │   └── shell.py (seatbelt) · sandbox.py · report.py · cti.py
│   │   ├── routes/             # chat.py · proxy.py (/v1/*) · memory_routes.py · self_routes.py
│   │   │                       # agent_routes.py · briefing_routes.py · portfolio.py · device.py
│   │   └── helpers/            # llm_http.py · llm_json.py · store.py · text.py
│   │                           # timefmt.py · weather.py · logging_setup.py
│   └── JarvisData/
│       ├── users_list.json · jarvis-self.json · backup_receipt.json
│       ├── model_cache/
│       └── prompts/            # prompt_proposals.json · prompt_overrides.json
├── scripts/
│   ├── jarvis-launchd.sh · jarvis-entrypoint.sh · jarvis-status.sh · backup-jarvis.sh
│   ├── uploadrag.py            # Indexes RAGData/ into Open WebUI (launchd 23:10)
│   ├── memory_report.py        # Daily probe of the memory layers (launchd)
│   ├── migrate_introspection.py  # learnings → self_introspection (dry-run by default)
│   ├── purge_user.py           # Removes every trace of a user (dry-run)
│   ├── reclassify-incident.py · cron-index.sh
│   └── download_models.py · search-qdrant.py · generate_google_token.py
├── RESEARCH/                   # Out of tree (git-ignored) — measurements and training
│   ├── NOTES.md                # Resume point: known traps, campaign status
│   ├── RESULTATS.md            # EVERY measurement, with its caveats
│   ├── evaluation/             # eval_reuse · eval_opinions · eval_emotion · eval_logprob
│   │   └── dryrun_introspection.py   # Dry run of the nightly review
│   ├── adapters/ · data/ · lora/ · orpo/ · bench/ · ablation/ · concept-vectors/
│   └── prompts/                # Successive versions of IDENTITY_FR (v1→v11)
├── RAGData/                    # Documents to index (personal, work, documents…)
│   └── Trade/                  # Broker CSV exports (TRADE_DATA_DIR)
├── logs/                       # jarvis-api · jarvis-debug
│                               # prompts · analyzer-prompts · nightly-prompts
│                               # reflection-prompts
│                               # opencode-prompts · agent-prompts
└── DOCS/
    ├── AGENT.md                # Agentic loop: tools, budgets, sandbox, API
    ├── ARCHITECTURE.md · MEMORY.md · API.md · CONFIGURATION.md · INSTALL.md
    ├── PERFORMANCE.md · OPERATIONS.md · SECURITY.md · GOOGLE.md · REDIS.md
    ├── opencode-local.md · opencode.json.example
    └── examples/               # plist template · shell aliases · users_list.example.json
```

---

## Prompt Self-Modification (Autocoding)

Jarvis can propose improvements to its own LLM prompts. The feature is fully autonomous on the detection side, and requires human approval before any change takes effect.

### How it works

```
Reflection cycle detects repeated knowledge gap
    → flag_knowledge_gap increments Redis counter (jarvis:self:gap_counts)
    → counter reaches REFINE_PROMPT_THRESHOLD (default: 3)
    → next reflection cycle: LLM chooses refine_prompt action
    → REASONING_MODEL rewrites the targeted prompt
    → proposal saved to prompt_proposals.json
    → email sent with old/new diff + approval instructions
    → user replies in chat: "accepte la proposition [id]"
    → prompt_overrides.json updated
    → get_prompt() serves new text immediately (no restart needed)
```

### Guardrails

| Guard | Where | Description |
|-------|-------|-------------|
| Concrete context required | `_action_flag_knowledge_gap()` | Context must describe a specific observed failure (min 30 chars, generic phrases rejected) |
| Per-topic cooldown | `_action_flag_knowledge_gap()` | Same topic cannot be re-flagged within 7 days (`jarvis:self:gap_cooldown:{slug}` Redis TTL) |
| Proposal block | `_action_flag_knowledge_gap()` | Blocked if a pending proposal exists for the topic, or an approved one is < 30 days old |
| Hard threshold check | `_action_refine_prompt()` | Refuses to run if gap count < `REFINE_PROMPT_THRESHOLD` — LLM cannot bypass this |
| Duplicate prevention | `_action_refine_prompt()` | Only one pending proposal per prompt at a time |
| Rejection cooldown | `_action_refine_prompt()` | Same prompt cannot be re-proposed within 7 days of a rejection |
| Approval cooldown | `_action_refine_prompt()` | Same prompt cannot be re-proposed within 30 days of an approval |
| Human approval | `approve_proposal()` | Override is never written without explicit "accepte la proposition [id]" |
| Full gap reset on approval | `approve_proposal()` | Counter, sorted-set entries, and a 30-day cooldown are all cleared for the topic |
| Prompt size budget | `_action_refine_prompt()` | `current_text` capped at 6 000 chars input, `max_tokens=4000` output — prevents truncated proposals |
| Email notification | `_notify_proposal()` | Sends diff to admin email via Gmail OAuth; logs a warning if send fails |

### In-chat commands (trigger via `self` intent)

| Command | Effect |
|---------|--------|
| `montre les propositions` | List all pending proposals with rationale |
| `montre la proposition [id]` | Show full old/new diff for a specific proposal |
| `accepte la proposition [id]` | Apply the override immediately — live, no restart |
| `rejette la proposition [id]` | Mark as rejected, start 7-day cooldown |

### Refineable prompts

| Constant | Description |
|----------|-------------|
| `SYSTEM_BASE_FR` | Core Jarvis personality and instructions |
| `BRIEFING_USER` | Morning briefing assembly instructions |
| `ANALYSIS_PROMPT` | Conversation analysis (mood/fact extraction) |
| `ROUTER_USER` | LLM intent router decision criteria |
| `NIGHTLY_FACTS_SYSTEM` / `NIGHTLY_FACTS_PROMPT` | Nightly call 1 — user insight extraction, relation update, suggestions |
| `NIGHTLY_SELF_SYSTEM` / `NIGHTLY_SELF_PROMPT` | Nightly call 2 — Jarvis self-reflection, opinions, day diary |
| `NIGHTLY_CLEANING_SYSTEM` / `NIGHTLY_CLEANING_PROMPT` | Nightly call 3 — Qdrant autobio curation (archive outdated, delete errors) |
| `CONSOLIDATION_PROMPT` | Monthly episodic → autobio fact extraction (called in `_consolidate_user_memories`) |
| `CURATIVE_CLEANUP_PROMPT` | Nightly Redis profile dedup (called in `curative_profile_cleanup`) |
| `REFLECTION_PROMPT` / `REFLECTION_SYSTEM` | Global autonomous reflection — action selection + system context |
| `REFLECTION_USER_PROMPT` / `REFLECTION_USER_SYSTEM` | Per-user autonomous reflection — personalized action selection |

The first four are user-facing quality prompts. The remaining seven are internal autonomy and memory prompts — they can also be refined when the reflection loop identifies recurring self-improvement opportunities.

### Data files

Both files live in `JarvisData/prompts/` which is already inside the existing volume mount — no docker-compose changes needed.

- **`prompt_proposals.json`** — append-only history of all proposals (pending / approved / rejected). Never truncated, useful for audit.
- **`prompt_overrides.json`** — active overrides only: `{"PROMPT_NAME": "new text"}`. Served ahead of the module constant by `get_prompt()`. Delete a key to revert to the default.

### Reverting an override

```bash
# Remove the key; the next get_prompt() call serves the prompts.py version. No restart needed.
python3 -c "
import json, shutil, datetime
p = '/opt/jarvis/jarvis-core/JarvisData/prompts/prompt_overrides.json'
shutil.copy2(p, p + '.bak-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
d = json.load(open(p)); del d['REFLECTION_SYSTEM']
json.dump(d, open(p, 'w'), ensure_ascii=False, indent=2); print('reverted')
"
```

> **An override survives code changes, and nothing revalidates it.** A `REFLECTION_SYSTEM`
> override approved five days earlier still described `check_health`, `update_self_note` and
> `prune_self_memory` — three actions removed from the catalogue that same day. Served from
> disk, it masked the code version: the model proposed `check_health`, the validator rejected
> it, and the chain fell back to `nothing`. The cycle's only opportunity to act on itself was
> lost — exactly the scenario `refine_prompt`'s own prompt describes as the thing to avoid.
>
> **After any change to the action catalogue, check the active overrides.** A one-line
> check:
>
> ```bash
> ./venv/bin/python -c "
> import json, re, sys; sys.path.insert(0,'jarvis-core/src')
> from self.engine import _SELF_ACTIONS, _USER_ACTIONS
> d = json.load(open('jarvis-core/JarvisData/prompts/prompt_overrides.json'))
> for k, v in d.items():
>     cites = set(re.findall(r'\b([a-z_]{6,})\b', v)) & {'check_health','store_insight','correct_profile','consolidate_memory','update_self_note','prune_self_memory'}
>     print(k, '→ non-existent actions:', sorted(cites) or 'none')"
> ```

No restart needed — `get_prompt()` detects the file change on the next call.

---


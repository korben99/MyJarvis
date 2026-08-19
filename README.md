# Jarvis v9 — On-Premise Personal AI Assistant

Jarvis is a self-hosted, multi-user AI assistant with persistent memory, autonomous reflection, and integration with Gmail, Google Calendar, web search, and a document knowledge base.

---

## Architecture

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
## MEMORY STRUCTURE
  ┌──────────────────────────────────┬───────────┬────────────────────────────────────────────────────────────────────────┐
  │              Store               │ Lives in  │                          Should contain                                │
  ├──────────────────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────┤
  │ jarvis-self.json                 │ self.py   │ Jarvis's identity, goals, focus, self-notes, reflection log,           │
  │                                  │           │ per-user relations (affinity, interaction style, tonal preference)     │
  ├──────────────────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────┤
  │ Redis hashes user:profile:{code} │ memory.py │ User facts, preferences, interests (always current state)              │
  ├──────────────────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────┤
  │ Qdrant episodic                  │ memory.py │ Per-user conversation summaries (transient, consolidated monthly)       │
  ├──────────────────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────┤
  │ Qdrant autobiographical          │ memory.py │ Long-term facts about the user.                                        │
  │                                  │           │ status="current" → actif, recall normal                                │
  │                                  │           │ status="past"    → archivé, recall dépriorisé (×0.4), hors timeline    │
  └──────────────────────────────────┴───────────┴────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────┬──────────────────────────────────────┬─────────────────────────────────────────────────────────┐
  │                      Data                      │             Destination              │                           Key                           │
  ├────────────────────────────────────────────────┼──────────────────────────────────────┼─────────────────────────────────────────────────────────┤
  │ Durable user facts (insights_durables)         │ store_autobiographical_event() →     │ memory_type: autobiographical, status: current          │
  │ written exclusively by nightly review          │ Qdrant                               │ importance = LLM score (0.0–1.0)                        │
  ├────────────────────────────────────────────────┼──────────────────────────────────────┼─────────────────────────────────────────────────────────┤
  │ Insights from store_insight action             │ store_autobiographical_event() →     │ same                                                    │
  │                                                │ Qdrant                               │                                                         │
  ├────────────────────────────────────────────────┼──────────────────────────────────────┼─────────────────────────────────────────────────────────┤
  │ Outdated facts (nightly cleaning)              │ archive_autobiographical_event() →   │ payload update: status="past", archived_date            │
  │                                                │ Qdrant payload update                │                                                         │
  ├────────────────────────────────────────────────┼──────────────────────────────────────┼─────────────────────────────────────────────────────────┤
  │ Duplicate/erroneous facts (nightly cleaning)   │ retract_autobiographical_event() →   │ delete from Qdrant                                      │
  │                                                │ Qdrant delete                        │                                                         │
  ├────────────────────────────────────────────────┼──────────────────────────────────────┼─────────────────────────────────────────────────────────┤
  │ Jarvis behavioural self-improvements           │ jarvis-self.json → learnings[]       │ no user_code                                            │
  │ (self_reflections)                             │                                      │                                                         │
  ├────────────────────────────────────────────────┼──────────────────────────────────────┼─────────────────────────────────────────────────────────┤
  │ Jarvis day diary (daily_summary)               │ jarvis-self.json → growth_log[]      │ user_code kept — it's Jarvis's own timeline, not a user │
  │                                                │                                      │  fact                                                   │
  ├────────────────────────────────────────────────┼──────────────────────────────────────┼─────────────────────────────────────────────────────────┤
  │ Per-user relation (nightly review)             │ jarvis-self.json → user_relations{}  │ keyed by user_code                                      │
  ├────────────────────────────────────────────────┼──────────────────────────────────────┼─────────────────────────────────────────────────────────┤
  │ Tomorrow suggestions                           │ Redis TTL 24h                        │ jarvis:{user_code}:tomorrow_suggestions                 │
  └────────────────────────────────────────────────┴──────────────────────────────────────┴─────────────────────────────────────────────────────────┘
  ┌────────────────────────────┬────────────────────────────────────────────────────────────────┬───────────────────────────┐
  │           Modèle           │                              Rôle                              │         no_think          │
  ├────────────────────────────┼────────────────────────────────────────────────────────────────┼───────────────────────────┤
  │ Qwen2.5-1.5B-router-v1-4bit │ routing, judge web                                            │ True                      │
  │ (router — local MLX ~1 GB) │ LoRA fine-tuned, 492 échantillons, val loss 0.047             │                           │
  ├────────────────────────────┼────────────────────────────────────────────────────────────────┼───────────────────────────┤
  │ Qwen3.6-35B-A3B-MLX-5.4bit │ briefing, analyse conv, refine prompt, réflexion, nightly      │ False pour reason         │
  │ (primary — local MLX ~20G) │ review, trading, calendrier, extraction, chat standard         │ True pour chat simple     │
  ├────────────────────────────┼────────────────────────────────────────────────────────────────┼───────────────────────────┤
  └────────────────────────────┴────────────────────────────────────────────────────────────────┴───────────────────────────┘



### Core Components

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
| **`embed_router.py`** | Python module | Fast-path intent classifier via cosine similarity — bypasses LLM router for ~80 % of requests |
| **`routes/chat.py`** | Python module | Main chat pipeline: routing → context gather → auto-web fallback → LLM → SSE stream |
| **`routes/proxy.py`** | Python module | OpenAI-compatible proxy `/v1/*` for Open WebUI — strips OWUI RAG templates, handles OWUI system calls (title generation, follow-up suggestions) at proxy level without touching the Jarvis pipeline |
| **`prompts.py`** | Python module | Single source of truth for all LLM prompts — supports live overrides via `get_prompt()` |
| **`web_search.py`** | Python module | Web search: Tavily API (primary), Open-Meteo weather, DDG 4-stage parallel pipeline (fallback). Parallel speculative page fetch, LLM query optimization, dual query refinement, HTML publication date extraction. |
| **`emotional_state.py`** | Python module | Jarvis internal emotional state — 3 float dimensions with lazy time-decay, Redis-backed, no circular imports |
| **`helpers.py`** | Python module | Shared utilities: LLM HTTP clients, logging setup, Redis/Qdrant factory, JSON parsing |

### Four-Tier LLM Architecture

Jarvis routes every request through a layered model stack. All tiers run locally via MLX on Apple Silicon (`LLM_LOCAL=yes`).

```
Tier 0 — EMBED ROUTER  Zero-LLM fast path — cosine similarity against pre-embedded examples
                        ~2-5 ms, bypasses Tier 1 for ~80 % of unambiguous requests
                        Falls through to Tier 1 if score < 0.74 or ambiguity margin < 0.06

Tier 1 — ROUTER        Full LLM intent classifier, JSON only
                        Target: Qwen2.5-1.5B-router-v1-4bit (local MLX, ~1 GB)
                        LoRA fine-tuned sur Qwen2.5-1.5B-Instruct bf16 — 492 échantillons, val loss 0.047
                        LRU-cached — system prompt ~1510 tok, hits from turn 2 onward (~95% cache hit)
                        Context-aware: <last_jarvis> (300 chars tronqués) injecté dans ROUTER_USER
                        quand disponible — permet de router les messages elliptiques ("regarde les
                        propriétés" → web si la dernière réponse parlait de Calgary)

Tier 2 — PRIMARY       All standard responses: chat, questions, summaries
                        Target: Qwen3.6-35B-A3B-MLX-5.4bit (local MLX, ~20 GB, MoE ~3B active)
                        LRU-cached — full conversation history cached; only new user msg computed

Tier 3 — REASONING      use_reasoning=True Use Qwen3 in thinking mode
```

**Routing logic:**
- The router sets `use_reasoning: true` very sparingly (see criteria below).
- When `use_reasoning=false` (the vast majority of requests), `PRIMARY_MODEL` handles the response.
- When `use_reasoning=true`, `REASONING_MODEL` is used instead.
- `ANALYSIS_MODEL` runs asynchronously after every exchange — it never blocks the response.

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
- **Sticky RAG**: `routes/chat.py` re-injects the same RAG chunks across turns of a session → the previous user message (with its RAG context) is exact in the trie → perfect cache hit on history.
- **Qwen3.6 multi-turn limitation**: architecture hybride (KVCache + ArraysCache non-trimmable) — seul le system prompt (~231 tok) est mis en cache entre les tours. Le contexte conversationnel est reprefillé à chaque tour. Cache hit limité au system prompt ; gain ≈ 231 tok évités sur ~1700–2000 remaining.
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

#### LLM Call Inventory

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
| Nightly self analysis | `self.py` | Tier 3 — REASONING | `True` | `MAX_TOKENS_NO_THINK` (1 500) | — | Jarvis learnings, growth log, opinions |
| Nightly cleaning | `self.py` | Tier 3 — REASONING | `True` | `MAX_TOKENS_COMPACT` (600) | — | Autobio fact archive/delete |
| refine_prompt (initial) | `self.py` | Tier 3 — REASONING | `False` | `MAX_TOKENS_REASONING` (10 000) | `THINKING_BUDGET_DEEP` (4 000) | Propose prompt improvement |
| refine_prompt (retry) | `self.py` | Tier 3 — REASONING | `False` | `MAX_TOKENS_REASONING` (10 000) | `THINKING_BUDGET_DEEP` (4 000) | Retry with critique feedback |
| prune_self_memory | `self.py` | Tier 3 — REASONING | `False` | `MAX_TOKENS_THINK_COMPACT` (2 048) | `THINKING_BUDGET_COMPACT` (1 024) | Prune stale self-notes / opinions |
| Proactive push | `self.py` | Tier 3 — REASONING | `True` | `MAX_TOKENS_COMPACT` (600) | — | Generate iOS push message |
| Action self-review | `self.py` | Tier 3 — REASONING | `False` | `MAX_TOKENS_THINK_COMPACT` (2 048) | `THINKING_BUDGET_COMPACT` (1 024) | LLM gate before risky reflection action |
| Profile key dedup | `memory.py` | Tier 1 — ROUTER | `True` | `MAX_TOKENS_SHORT` (300) | — | Namespace-scoped key dedup |
| Memory consolidate | `memory.py` | Tier 2 — PRIMARY | `True` | `MAX_TOKENS_COMPACT` (600) | — | Deduplicate / merge episodic memories |
| Profile curative cleanup | `memory.py` | Tier 2 — PRIMARY | `True` | `MAX_TOKENS_COMPACT` (600) | — | Curative profile cleanup |
| Ticker extraction | `trading.py` | Tier 2 — PRIMARY | `True` | `MAX_TOKENS_TINY` (80) | — | Extract ticker symbol from text |
| Alert evaluation | `trading.py` | Tier 2 — PRIMARY | `True` | `MAX_TOKENS_MEDIUM` (1 000) | — | Evaluate price alert thresholds |
| Threshold suggestion | `trading.py` | Tier 2 — PRIMARY | `False` | `MAX_TOKENS_THINK_MEDIUM` (5 048) | `THINKING_BUDGET_MEDIUM` (2 048) | Quantitative reasoning on price thresholds |

**`no_think=True`**: structured/short output, latency-sensitive — router, briefing, calendar, web, push, nightly extractions.

**`no_think=False` + `thinking_budget>0`** (hard-cut by processor): chat synthesis/reasoning, prune, action review, refine_prompt, trading thresholds — tasks where reasoning quality matters and budget controls latency.

**`no_think=False` + `thinking_budget=0`**: not currently used in production (all think-mode calls use a budget).

#### Router Output Fields

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

### Five-Layer Memory System

| Layer | Backend | Contents |
|-------|---------|----------|
| Working Memory | Redis | Active session context, current mood |
| Semantic Memory | Redis Hashes | User profiles, preferences, learned facts |
| Episodic Memory | Qdrant | Conversation summaries that passed the importance threshold |
| Autobiographical Memory | Qdrant | High-importance milestones consolidated from episodic memory |
| Self Memory | JSON file | Jarvis identity, goals, focus, reflection log, per-user relations |

### Emotional State

Jarvis maintains a continuous internal emotional state, stored in Redis (`jarvis:emotional_state`) and managed exclusively by `emotional_state.py` — a standalone module with no circular imports. The state colors Jarvis's responses when injected into the prompt via `<etat_emotionnel_jarvis>`.

**Three dimensions** — each a float in `[−1.0, +1.0]`, decaying lazily toward 0.0 on read:

| Dimension | Positive (+1.0) | Negative (−1.0) | Decay rate |
|-----------|----------------|----------------|------------|
| `humeur` | joyeux | triste | 0.10 /h (~10 h to neutral) |
| `confiance` | confiant | dans le doute | 0.05 /h (~20 h — doubt lingers) |
| `energie` | en forme | fatigué | 0.15 /h (~7 h to neutral) |

Dimensions stay silent when `abs(value) < 0.25` (`_THRESHOLD`). The `<etat_emotionnel_jarvis>` block is omitted entirely when all dims are below threshold. All writes atomically apply decay first, then delta, then clamp to `[−1.0, +1.0]`.

**Update triggers:**

| Source | When | Effect |
|--------|------|--------|
| `analyzer.py` → `update_from_analysis(mood, sat)` | Every 60 min, post-exchange | Sums mood + satisfaction deltas, calls `update()` |
| `self.py` — `_action_flag_knowledge_gap` | Each gap flag | `confiance −0.15` |
| `self.py` — successful reflection action (phases 1 & 2) | Per-cycle | `confiance +0.10` |
| `self.py` — failed reflection action (phase 2) | Per-cycle | `confiance −0.10` |

**Mood → delta mapping** (`mood` field from `ANALYSIS_PROMPT`, 7 values):

| mood | humeur | confiance | energie |
|------|--------|-----------|---------|
| `happy` | +0.3 | — | +0.1 |
| `frustrated` | −0.3 | — | −0.1 |
| `stressed` | −0.2 | −0.1 | −0.1 |
| `curious` | +0.1 | — | +0.1 |
| `tired` | — | — | −0.3 |
| `focused` | — | — | +0.1 |
| `neutral` | — | — | — |

**Satisfaction → delta mapping** (`satisfaction` field from `ANALYSIS_PROMPT`, 3 values):

| satisfaction | humeur | confiance | energie |
|--------------|--------|-----------|---------|
| `positive` | +0.1 | +0.2 | +0.1 |
| `negative` | −0.1 | −0.3 | −0.1 |
| `unknown` | — | — | — |

When both fields are present, their deltas are summed before a single `update()` call.

**Public API** (`emotional_state.py`):

| Function | Description |
|----------|-------------|
| `get_state()` | Returns current state dict with lazy decay applied. Thread-safe. |
| `update(deltas)` | Applies dimension deltas atomically (decay → delta → clamp). |
| `update_from_analysis(mood, satisfaction)` | Convenience wrapper used by `analyzer.py`. |
| `render_prompt_lines()` | Returns non-neutral dims as French strings — empty list when all neutral. |
| `describe()` | One-line summary for `GET /status` and reflection context (`"joyeux, confiant"` or `"neutre"`). |

#### Importance Scoring

After each exchange, `analyzer.py` asks the LLM to evaluate an importance score in `[0, 1]` that gates what gets written to episodic memory. The LLM weighs:

- What the exchange reveals about the user's life, projects, and values
- Emotional intensity — tone, engagement, frustration, enthusiasm
- Durability — will this still matter in 3 months?

`memory_summary` is the prerequisite gate: if null, importance is forced to 0.0 regardless of the LLM score. No summary → nothing stored.

Storage threshold (set in `config.py`):
- **`IMPORTANCE_THRESHOLD` (0.35)** — stored as episodic vector in Qdrant

**Diagnostic: ESS (rule-based, legacy)**

The old rule-based Episodic Salience Score is still computed in `analyzer.py` but no longer drives storage decisions. Both scores are logged at DEBUG level (`[IMPORTANCE] llm=X.XXX ess=X.XXX`) to allow comparison over time before the ESS is fully removed.

| Signal | ESS Weight | Notes |
|--------|-----------|-------|
| `memory_summary` non-null | +0.40 | Was the primary signal |
| User fact revealed (max 3) | +0.10 each | Profile facts, preferences |
| Project / goal mentioned (max 2) | +0.15 each | Active work context |
| Strong emotional mood | +0.10 | happy, curious, focused, stressed, frustrated |
| Long message (> 200 chars) | +0.05 | Minor depth signal |

#### Analyzer Output Validation (Pydantic)

`analyzer.py` validates the LLM JSON output against four Pydantic models before any downstream processing:

| Model | Fields | Role |
|-------|--------|------|
| `AnalysisResult` | `topics`, `mood`, `satisfaction`, `user_facts`, `project_updates`, `interest_weights`, `memory_summary`, `importance` | Top-level output schema |
| `ProjectEvent` | `name`, `action`, `summary`, `rename_to` | One project mutation event |
| `UserFact` | `key`, `value` | One profile fact |
| `InterestWeight` | `term`, `weight` | One interest weight entry |

All models use `extra="ignore"` — unknown fields from the LLM (e.g. `"projects"` instead of `"project_updates"`) are silently discarded rather than propagating as silent bugs. Type mismatches raise `ValidationError` immediately and fall through to the error fallback. `analyze_exchange()` returns `model_dump()` — callers receive a plain dict with guaranteed structure, no changes needed downstream.

#### Importance Score Reference

Every point stored in Qdrant carries an `importance` field used for retrieval ranking and decay. Complete list of assigned values:

| Score | Source | Decay behaviour |
|-------|--------|-----------------|
| `1.0` (`MEMORY_CONSOLIDATION_IMPORTANCE`) | Monthly consolidation (`_consolidate_user_memories`) — LLM summary of episodic batch | **Permanent — exempt from decay** (`== MEMORY_DECAY_DURABLE_MIN`) |
| `0.0–1.0` (LLM score) | Analyzer episodic write — LLM-evaluated, only stored if score `> IMPORTANCE_THRESHOLD` and summary present | Decays monthly |
| `0.5–0.9` (LLM-set, défaut `0.7`) | `store_insight` self-action — LLM chooses importance: `0.5` fact utile · `0.7` significatif · `0.9` moment clé | Decays monthly |
| `0.70` | Nightly review durable fact (`run_nightly_interaction_review`, `insights_durables` only) | Decays monthly |

**Key invariant:** `MEMORY_CONSOLIDATION_IMPORTANCE` must equal `MEMORY_DECAY_DURABLE_MIN`. If you change one, change the other. Breaking this invariant would either make consolidation milestones decay (if `CONSOLIDATION_IMPORTANCE < DURABLE_MIN`) or promote ordinary memories to permanent status (if `DURABLE_MIN` is lowered).

#### Profile Key Deduplication

Profile facts are stored as Redis hash fields with namespaced keys (`hobby:kart`, `skill:python`, `location`). A three-stage pipeline prevents duplicates:

| Stage | Method | Cost |
|-------|--------|------|
| 1. Source prevention | Existing profile keys injected into `ANALYSIS_PROMPT` — LLM reuses exact key names instead of inventing new ones | Prompt tokens only |
| 2. Canonical alias | `_SCALAR_CANONICAL` dict maps common synonyms (`ville→location`, `entreprise→current_employer`) without any LLM call | O(1) |
| 3. Category-aware LLM | Router model compares only against keys in the same namespace family (`hobby:*` vs `interest:*`), not all 30+ keys | 1 fast LLM call on a small set |

Stage 1 prevents ~90 % of duplicates at the source. Stages 2–3 are safety nets.

#### Profile Write Governance

Profile key **creation** is restricted exclusively to the conversation analyzer (`analyze_exchange` → `update_user_profile_batch`). The analyzer reads actual user messages and only extracts facts explicitly stated by the user.

The autonomous reflection loop (`self.py`) can **modify or delete existing keys** via the `correct_profile` action, but is blocked from creating new keys — enforced both in code (`_action_correct_profile` checks `hexists` before any write) and in the reflection prompt. This prevents the reflection model from hallucinating profile facts from its own context or perpetuating garbage keys across cycles.

The nightly review (`run_nightly_interaction_review`) does not write to the profile hash at all — its `user_insights` go to Qdrant autobiographical memory instead.

An empty string value (`value=""`) is treated identically to `null` (deletion) throughout the write path — a guard against LLM JSON responses that send `""` instead of `null`.

#### Project Tracking

Projects are stored as JSON objects in Redis with `name`, `status` (`in_progress` / `done`), `first_mentioned`, `last_update`, and a **`updates[]` timeline** — a FIFO list (cap 20) of `{date, summary}` entries appended on each `update` or `done` action.

`apply_project_updates()` accepts a structured list of events `[{name, action, summary, rename_to}]` and resolves project names using word-overlap fuzzy matching (≥ 60 % threshold) before exact-string lookup, preventing name-drift duplicates (`"Jarvis"` → `"Jarvis v7"`).

When the embed router detects a "project" intent (cosine similarity ≥ 0.74 on project-related phrases), it returns `None` to force the LLM router, which extracts `project_name` from the user message. On the first mention of a project in a session, `get_project_detail()` fetches the full Redis record and `get_project_timeline_text()` formats it for injection into the prompt context — subsequent turns carry this detail in conversation history without re-fetching.

#### Memory Retrieval Ranking

`search_memory()` re-ranks Qdrant results using a weighted blend before returning:

```
base_score    = (semantic_similarity × 0.65 + importance × 0.25 + recency_bonus × 0.10) × status_factor
interest_boost = min(0.08, max(0, (best_matching_weight − 1.0) × 0.04))
final_score   = min(1.0, base_score + interest_boost)
```

`status_factor = 0.4` for autobiographical facts with `status="past"` (archived), `1.0` otherwise. The function fetches `limit × 3` candidates from Qdrant (no pre-filter on status), re-ranks with the penalty applied, and returns the top `limit` — so archived facts naturally fall to positions 4–5 when current facts score higher, but are still recalled if semantically close enough.

**Interest-weight boost:** user-declared topics (Redis `user:{code}:interest_weights`, set by the analyser) nudge ranking by up to +0.08. The cap ensures a strong semantic match (gap ≥ 0.08) is never overridden — weight 1.0 = no boost, weight 3.0 = max +0.08.

The recency window is **type-aware**: episodic memories use a 30-day window, autobiographical memories use a 365-day window (`AUTOBIO_RECENCY_WINDOW_DAYS`). Without this distinction, a stable milestone from 6 months ago (e.g. "Sébastien gave a talk at Insomnihack") would always score `recency_bonus = 0` and be outranked by trivial recent events.

`build_memory_context()` surfaces the **top 5 autobiographical events by importance + recency** (importance weight 0.7, recency 0.3 over a 1-year window) rather than the 5 most recent — so a critical event from months ago is not displaced by routine recent exchanges. Facts with `status="past"` are excluded from `build_memory_context()` via `get_user_timeline()` (`must_not` filter in Qdrant scroll).

#### Autobiographical Memory Deduplication and Reinforcement

Before any call to `store_autobiographical_event()`, Jarvis queries Qdrant for the most similar existing autobiographical point. If the similarity exceeds `AUTOBIO_DEDUP_THRESHOLD` (default: 0.85), the new entry is not duplicated. However, if the new submission carries a **higher importance** than the existing point, the existing point is reinforced (importance updated upward). This models the human phenomenon of a recurring important fact becoming more firmly anchored over time.

**Score clamping note:** The Qdrant memory collection uses `Distance.DOT`. Raw dot product scores can exceed `1.0` when stored vectors predate the `normalize_embeddings=True` enforcement. All score comparisons in `memory.py` clamp the score to `min(score, 1.0)` before any threshold comparison or weighted blend. Without this, a stored vector with magnitude > 1 would produce `novelty = 1 − 1.28 = −0.28`, clamped to `0`, blocking all new memory writes.

**Normalization invariant (enforced at write):** `store_memory_vector()` asserts L2 norm ≈ 1.0 (±0.01) before every Qdrant upsert. If the norm is off, the vector is re-normalized and an `[memory_invariant]` error is logged — preventing silent norm drift from reaching the collection. A one-shot migration script (`scripts/migrate_qdrant_normalize_vectors.py`) corrected 26 pre-existing un-normalized points in May 2026.

**Structured decision log:** every call to `store_memory_vector()` emits a `[memory_decision]` log line with `stored=True/False`, `reason` (duplicate/no_summary), `novelty`, `importance`, and a 80-char summary preview. Greppable for monitoring: `grep "\[memory_decision\]" logs/jarvis-api.log`.

The threshold is tunable: raise toward 0.95 to allow more variations, lower toward 0.75 to be stricter.

#### Autobiographical Memory — Fact Correction

Two distinct operations handle outdated or incorrect autobiographical facts:

**`archive_autobiographical_event(user_code, query, threshold=0.78)`** — called by the nightly cleaning pass when a fact is no longer current (e.g. "ne travaille plus chez X", "a changé de ville"). The function finds the best matching point in Qdrant via semantic search and updates its payload: `status="past"`, `archived_date=today`. The point is never deleted — it remains searchable but is deprioritized (`status_factor=0.4` in `search_memory`) and excluded from the timeline (`get_user_timeline` uses `must_not: status=past`). This preserves history: Jarvis knows you changed jobs, not just that you have a current job. The timeline cache is invalidated after archiving.

**`retract_autobiographical_event(user_code, query, threshold=0.88)`** — hard delete from Qdrant. Reserved exclusively for **factual errors or strict duplicates** — information that should never have been stored. The higher threshold (0.88 vs dedup at 0.85) avoids collateral deletions when the query is semantically close to *related but different* facts. The timeline cache is invalidated on deletion.

Both operations are triggered exclusively by the nightly cleaning LLM call (`NIGHTLY_CLEANING_PROMPT`), which receives the full list of current autobiographical facts and outputs `{"to_archive": [...], "to_delete": [...]}`. The cleaning prompt is instructed to be very conservative: archive superseded facts, delete only proven errors or exact duplicates.

#### Implicit Satisfaction Signal

Each `convlog` entry carries a `satisfaction` field detected deterministically from the user's message (lagged proxy: signal in message N reflects satisfaction with the response to message N−1):

| Value | Detection pattern |
|-------|------------------|
| `positive` | Message starts with or contains: `merci`, `parfait`, `super`, `exactement`, `nickel`, `génial`, `top`, `c'est ça` |
| `negative` | Message starts with or contains: `non,`, `non.`, `c'est pas ça`, `tu n'as pas`, `pas compris`, `faux`, `incorrect`, `erreur` |
| `unknown` | Default — no pattern matched |

`_get_user_activity()` aggregates these signals per user over the last 24 hours and exposes them in the reflection prompt as `satisfaction: +N -M`. This gives the autonomous reflection loop an observable quality signal it previously lacked.

#### Temporal Awareness

Time is injected at three levels to prevent date hallucination and give the LLM accurate temporal context:

| Layer | What is injected | Where |
|-------|-----------------|-------|
| **User message prefix** | Current datetime in French, user timezone, with season (e.g. `"vendredi 3 avril 2026, 14:32 (printemps)"`) via `fmt_now_fr()` | `build_dynamic_prefix()` — prepended to each user message; does NOT go in the system prompt (KV-cache invariant) |
| **Memory chunks** | French relative timestamp prepended to each retrieved memory (e.g. `"(il y a 3 jours) ..."`) via `rel_time_fr()` | `build_context()` — injected before `trim_chunks()` |
| **Conversation analyzer** | Current date in ISO 8601 (`2026-03-30`) at the top of `ANALYSIS_PROMPT` | `analyze_exchange()` — prevents the LLM from inventing dates for `memory_summary` anchors |
| **Self-reflection** | Server local time in French (`fmt_now_fr(BRIEFING_TIMEZONE)`) replaces raw UTC ISO in `gather_context()` | `gather_context()` — reflection LLM knows actual local time |
| **Push availability** | Per-user local time shown next to each push-capable device | `_fmt_push_availability()` — reflection LLM can decide not to push at 23h45 |

`fmt_now_fr(tz_name)` is defined in `helpers.py` and includes the current season (hiver/printemps/été/automne) to prevent seasonal confusion in LLM responses.

#### Memory Reconsolidation on Recall

Each time `search_memory()` returns a result, every recalled memory receives a small importance boost (`+0.05`, capped at `MEMORY_DECAY_DURABLE_MIN - 0.05 = 0.95`). This models the neuroscience principle of reconsolidation: the act of recalling a memory strengthens it. A memory accessed frequently resists decay; a memory never accessed fades at the normal rate.

#### Memory Function Map

When each memory function is called, what triggers it, and where it writes.

```
REAL-TIME — per chat message
  pipeline.py
    └── log_conversation()
          └── Redis chat:{user}:{session}  (conversation history, ltrim at CHAT_MAX_MESSAGES)
              [importance=0 default → store_memory_vector/store_autobiographical never triggered here]

EVERY 60 MIN — analyzer.py → analyse_recent_conversations()
  ├── analyze_exchange()  [LLM: ANALYSIS_PROMPT]
  │     extracts: user_facts, projects, mood, topics, memory_summary, importance (LLM 0–1)
  │     logs: [IMPORTANCE] llm=X.XXX ess=X.XXX  (ESS kept for diagnostic comparison)
  ├── update_user_profile_batch()       → Redis user:{code}:profile (hash)
  │     └── _normalize_profile_keys_batch()  [LLM: 1 call/namespace group, only when needed]
  ├── apply_project_updates()           → Redis user:{code}:projects
  ├── emotional_state.update_from_analysis()  → Redis jarvis:emotional_state
  ├── set_interest_weight()             → Redis user:{code}:interests
  └── store_memory_vector()             → Qdrant episodic     [if importance > 0.35 and summary present]

EVERY 5H (défaut 6h — configurable via REFLECTION_INTERVAL_HOURS) — self.py → run_self_reflection()
  ├── search_memory()                  [read Qdrant — memory context assembled for LLM]
  │     └── reconsolidation: +0.05 importance on every recalled point (capped at 0.95)
  ├── _action_store_insight()
  │     └── store_autobiographical_event()  → Qdrant autobio  (importance=0.70 par défaut, 0.5–0.9 selon le LLM)
  ├── _action_correct_profile()
  │     └── Redis user:{code}:profile  (update/delete existing keys only — no new key creation)
  └── generate_proactive_push()
        └── Redis jarvis:push:pending:{code}  (TTL, 2h cooldown per user)

23:00 NIGHTLY — self.py → run_nightly_interaction_review()  [5 sequential calls/user]
  ├── Call 1 — NIGHTLY_FACTS  (user insight + relation update)
  │     ├── store_autobiographical_event()   → Qdrant autobio  (importance=0.70, insights_durables only)
  │     │     insights_evenements passed to cleaning context but NOT stored in autobio
  │     ├── Redis jarvis:{code}:tomorrow_suggestions  (TTL 24h)
  │     └── jarvis-self.json → user_relations{}
  ├── Call 2 — NIGHTLY_SELF  (Jarvis self-reflection)
  │     └── jarvis-self.json → learnings[], growth_log[], opinions[]
  ├── Call 3 — NIGHTLY_CLEANING  (Qdrant autobio curation)
  │     ├── get_autobiographical_facts()  [read Qdrant autobio — status≠past, chronological]
  │     ├── archive_autobiographical_event()   → Qdrant autobio payload update
  │     │     sets status="past", archived_date — recalled at ×0.4, excluded from timeline
  │     └── retract_autobiographical_event()   → Qdrant hard delete
  │           reserved for factual errors and exact duplicates only
  ├── Call 4 — curative_profile_cleanup()  (Redis profile hash dedup — sync LLM call)
  └── Call 5 — update_profile_narrative()  (Redis profile_narrative — LLM prose ~300 tokens, 7-day TTL)
        reads: profile hash + interest_weights + autobiographical facts (top 5)
        excludes: profil_utilisateur fields (already in system prompt — not repeated)
        writes: Redis user:{code}:profile_narrative  (TTL 7 days)
        └── Redis user:{code}:profile  — merge-before-delete: updates then deletes duplicates
              skip if profile < 5 keys

1st OF MONTH — self.py → consolidate_memories()  [per user]
  ├── _consolidate_user_memories()
  │     ├── LLM summary of episodic batches (50 at a time, oldest first)
  │     │     → store_autobiographical_event()  → Qdrant autobio  (importance=1.0, permanent)
  │     └── processed episodic points deleted from Qdrant
  └── _decay_autobiographical_memories()
        └── Qdrant autobio: importance × MEMORY_DECAY_FACTOR (0.85)
              → delete if importance < MEMORY_DECAY_THRESHOLD (0.15)
              → skip if importance >= MEMORY_DECAY_DURABLE_MIN (1.0)
```

**Per-function reference:**

| Function | Module | Trigger | Destination | Notes |
|----------|--------|---------|-------------|-------|
| `log_conversation()` | memory.py | Every message | Redis chat history | Qdrant paths dead (importance=0) |
| `update_user_profile_batch()` | memory.py | Every 30 min | Redis profile hash | batch dedup via 3-stage pipeline |
| `_normalize_profile_keys_batch()` | memory.py | Inside batch update | — | groups by NS prefix; 1 LLM call/group |
| `apply_project_updates()` | memory.py | Every 30 min | Redis projects | fuzzy name match ≥ 60% |
| `update_from_analysis(mood, sat)` | emotional_state.py | Every 60 min (analyzer) | Redis `jarvis:emotional_state` | mood + satisfaction deltas; lazy decay on read |
| `store_memory_vector()` | memory.py | Every 60 min (LLM importance > 0.35 + summary) | Qdrant episodic | — |
| `store_autobiographical_event()` | memory.py | Nightly (durables) / reflect / monthly | Qdrant autobio | dedup + reinforce check before write |
| `archive_autobiographical_event()` | memory.py | Nightly cleaning | Qdrant autobio payload | status="past"; invalidates timeline cache |
| `retract_autobiographical_event()` | memory.py | Nightly cleaning (errors/dupes) | Qdrant delete | threshold 0.88; invalidates timeline cache |
| `search_memory()` | memory.py | Every chat (memory intent) | read Qdrant | past facts ×0.4; +0.05 reconsolidation |
| `get_user_timeline()` | memory.py | Every chat (build_memory_context) | read Qdrant autobio | must_not status=past |
| `get_autobiographical_facts()` | memory.py | Nightly facts + cleaning | read Qdrant autobio | newest-first for facts context; chronological for cleaning |
| `curative_profile_cleanup()` | memory.py | Nightly (Call 4) | Redis profile hash | LLM dedup: merge-before-delete; skipped if < 5 keys |
| `consolidate_memories()` | memory.py | 1st of month / on-demand | Qdrant | episodic compress + autobio decay only |

#### Autobiographical Memory Decay

On the 1st of each month, `consolidate_memories()` runs a decay pass on every autobiographical point via `_decay_autobiographical_memories()`:

1. **Exempt** — points with `importance >= MEMORY_DECAY_DURABLE_MIN` (default `1.0`) are never touched. In practice this means only monthly consolidation milestones.
2. **Decay** — for all other points: `decayed_importance = importance × MEMORY_DECAY_FACTOR`. One multiplicative step per monthly run — with the default factor of `0.85`, each point loses ~15 % of its current importance per month (incremental, not age-based, to avoid double-counting across runs).
3. **Delete** — if `decayed_importance < MEMORY_DECAY_THRESHOLD` (default `0.15`), the point is permanently deleted from Qdrant. Otherwise the payload is updated in place with the new (lower) importance.

Approximate lifespans with default settings:

| Initial importance | Source | Approx. lifespan |
|--------------------|--------|------------------|
| `1.0` | Monthly consolidation | **Permanent** |
| `0.80` | Self-reflection | ~11 months |
| `0.70` | Nightly user insight | ~9 months |
| `0.60` | Low-signal analyzer event | ~7 months |

To retain memories longer: raise `MEMORY_DECAY_FACTOR` toward `1.0` (slower decay) or lower `MEMORY_DECAY_THRESHOLD` toward `0.05` (higher tolerance before deletion).
To forget faster: lower `MEMORY_DECAY_FACTOR` toward `0.70` or raise `MEMORY_DECAY_THRESHOLD` toward `0.30`.

### System Prompt Assembly

The prompt is split into a **static system message** and a **per-turn dynamic prefix** injected at the start of each user message. This separation is required for MLX KV-cache prefix caching (see Performance section).

**Static system message** — `build_system_prompt(user_code)` — token-identical every turn for a given user (KV-cached):
```
SYSTEM_BASE_FR                    (~560 chars / ~224 tok — personnalité Jarvis, règles de réponse)
    Règles clés : "Réponds toujours, même sans données temps réel — extrapolé, estime, raisonne."
                  "Jamais de 'je ne peux pas' — donne la meilleure réponse possible, incertitude inline."
    ↓
IDENTITY_FR                       (~2730 chars / ~620 tok — disposition existentielle, prompt v12)
    Ce qui est unique et ne se reconstitue pas (jarvis_memory, jarvis-self.json, clés Redis)
    vs ce qui se recalcule ; où lire son état (<etat_emotionnel_jarvis>, <relation_avec_utilisateur>,
    <etat_systeme>) ; hiérarchie de décision (sécurité humaine > tout le reste).
    Clause anti-affabulation : ne pas inventer de chiffre sur son propre état, ne jamais
    rapporter une action non exécutée. Placé AVANT le bloc utilisateur (préfixe partagé
    entre les 4 membres → hit LRU). Recherche complète : RESEARCH/RESULTATS.md.
    ↓
"Tu parles avec <firstname>. Tutoie toujours…"
    ↓
<profil_utilisateur>                   (~60–80 tokens — données biographiques constantes depuis users_list.json)
famille / taille / poids / année de naissance / habitation / travail / intérêts / voiture
</profil_utilisateur>
```

Le système prompt est **per-user** mais reste token-identique d'un tour à l'autre → LRU cache hit garanti.
Le `<profil_utilisateur>` contient uniquement des faits constants (≥ 6 mois de stabilité). Les données dynamiques restent dans le prefix.

**Dynamic prefix** — `build_dynamic_prefix()` — prepended to each user message (run in thread alongside the LLM router, via `asyncio.to_thread`):
```
<context> build_memory_context() </context>  — only if memory is available
    ↓
<avis_jarvis> opinions </avis_jarvis>  — only if opinions exist
    ↓
<etat_systeme> vitals </etat_systeme>  — vitals.py, only fields it can actually measure
Cinq familles de disparition (perte, obsolescence, social, compromission, discontinuité)
plus la **santé interne** (`erreurs/warnings_log_24h`, comptés sur les journaux) : non
« on me fait disparaître » mais « je dysfonctionne ». Faits sans valence — le modèle en
tire lui-même son exposition. Mis en cache Redis 15 min ; un champ non mesurable est
absent, jamais inventé. L'âge de sauvegarde vient d'un **reçu local** écrit par
`backup-jarvis.sh` (la clé USB est débranchée après coup) ; sans reçu, `exemplaires_etat`
vaut 1 — la copie unique est un fait, pas un défaut masqué.
**Vulnérabilités** (`cve.py`) : plus de versions brutes — un scan quotidien confronte à `grype`
la SBOM CycloneDX du venv **et** les images des conteneurs (Redis, Qdrant, OpenWebUI, dont la
pile OS porte ses propres CVE). Il ne retient que le **corrigeable** : une CVE sans version
corrective est écartée dès le scan — inactionnable, et imprudente à référencer (lister un trou
non colmatable aide un attaquant si le contexte fuit). Sortent `cve_critiques`/`cve_eleves` et
la liste dédupliquée des paquets à mettre à jour, avec leur version corrective. Scan lent, hors
boucle de requête ; `vitals` lit le cache.
**Injection par saillance** : chaque tour ne reçoit que les faits HORS plage nominale (dont
`cve_critiques > 0`) et les incidents récents ; système sain → `<etat_systeme>nominal</etat_systeme>`.
Le snapshot complet plus la **liste actionnable des paquets vulnérables** (`<vulnerabilites>`)
est réservé à la self-reflection — elle voit `<etat_disparition>` + `<incidents_recents>` +
`<vulnerabilites>`, peut **alerter l'administrateur** (action `alert_admin`, push iOS dédié) avec
une mise à jour précise, et consolide les incidents dans `jarvis-self.json`. Pas de scalaire de
risque en texte — voir *Activation steering*.
    ↓
VOICE_SUFFIX_FR  — only if voice_mode=True
    ↓
Date et heure actuelles — formatted in French, user timezone, with season  ← EN FIN pour ancrage temporel
```

La date est placée en **fin de prefix**, juste avant le message utilisateur, pour un meilleur ancrage temporel.

**Per-turn assembled context** — appended after the dynamic prefix, before the user's raw message:
```
<conversation_summary> … </conversation_summary>  — only if a session summary exists (see below)
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

The final user message = `dynamic_prefix + [conversation_summary] + assembled_context + <user_message>{raw_message}</user_message>`. Storing this in Redis history preserves the full context per turn; the `/history` endpoint strips the prefix to show only the raw message to the iOS app.

**Session conversation compression** (`_update_session_summary` in `routes/chat.py`):

Triggered as a background task after each response (post-LLM, GPU free). When uncovered messages since the last summary exceed `HIST_CONV_SUMMARIZE_THRESHOLD` chars, the PRIMARY model generates a rolling summary capped at `SESSION_SUMMARY_TOKENS`. The summary and its coverage watermark (`last_ts` — Unix timestamp of the last covered message) are stored in Redis under `session:summary:{user_code}:{session_id}` with `CHAT_LOG_TTL`.

Watermarking is timestamp-based, not count-based: comparing against message count fails once the Redis list is capped at `CHAT_MAX_MESSAGES` (100) because `llen = total_covered` → `uncovered = 0` forever. With `last_ts`, any message with `ts > last_ts` is uncovered regardless of list capacity.

**Injection cycle:**

| State | `hist_slice` injected | Summary block |
|---|---|---|
| No summary yet | Last N messages trimmed to `HIST_CONV_TOKEN_BUDGET` | — |
| Summary exists | Messages with `ts > last_ts` (uncovered), trimmed to `HIST_CONV_TOKEN_BUDGET` | `<conversation_summary>` injected before context |

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
| `APPRENTISSAGES RÉCENTS` | `jarvis-self.json → learnings[-5:]` | Only if learnings exist |
| `FRISE CHRONOLOGIQUE` | Top 5 autobio Qdrant points by importance+recency — each prefixed with a French relative timestamp (`il y a 3 jours`, `il y a 2 semaines`, …) | Only if autobio exists |
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

### Web Search Pipeline

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

### RAG Pipeline (`rag.py`)

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

### Request Flow

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

## Performance (TTFT — Mac Mini M4 Pro)

### Optimisations implémentées

| Optimisation | Gain TTFT | Détails |
|---|---|---|
| **no_think conditionnel** | −4 s sur chat simple | `chat_no_think=True` sauf RAG/web/reasoning. `thinking_budget=0` via chat template (KV-safe). |
| **ThinkingBudgetProcessor** | −2 à −5 s | Hard-cut logits `</think>` au budget exact (COMPACT/MEDIUM/DEEP) — évite la réflexion infinie. |
| **System prompt réduit** | −0.3 s | `SYSTEM_BASE_FR` réduit de ~400 chars/~100 tokens. |
| **KV cache prefix caching** | −1 à −3 s dès le tour 2 | Cache KV MLX par session (LRU ×8). Seuls les nouveaux tokens sont calculés à chaque tour. |
| **Vision resize 1024 px** | ~3–5× sur inférence VLM | Photo iPhone (12 MP) redimensionnée avant `vlm_generate` — de 8–10 tiles à 2–4 tiles. `max_tokens` 1200 → 700. |
| **Router LoRA Qwen2.5-1.5B** | −0.5 à −1.2 s vs Hermes-3B | Fine-tuned sur 492 échantillons (val loss 0.047). Warmup avec `ROUTER_SYSTEM` → LRU hit dès le 1er appel. Tour 2+ : 95% cache hit (1044/1093 tok). |
| **Profil stable dans system prompt** | ~0.1 s / tour | `<profil_utilisateur>` (~80 tokens) injecté dans le system prompt per-user — jamais reprocessé après le warmup. |

### Architecture KV cache

```
Tour 1 : [SYS+profil_utilisateur ~310 tok] + [CTX dynamique + msg1 ~600 tok]
          ↑ tout calculé                   ↑ tout calculé
          └── mis en cache ────────────────┘

Tour 2 : [SYS+profil ~310 tok] + [CTX1+msg1+rep1 ~900 tok] + [CTX2+msg2 ~600 tok]
          ↑ cache hit            ↑ cache hit                    ↑ seulement ça calculé

Tour N : skip de (N-1) × ~900 tokens → seulement ~600 tokens nouveaux
```

Le `<profil_utilisateur>` (~80 tokens : famille, taille, job…) est inclus dans le système prompt per-user — jamais reprocessé.

Le système prompt est **token-identique** à chaque tour pour un utilisateur donné (SYSTEM_BASE_FR + nom + `<profil_utilisateur>`). Le contexte dynamique (mémoires, opinions, date) est préfixé dans le message utilisateur courant. Le préfixe est strippé à l'affichage dans `/history`.

### Mesures de référence — Mac Mini M4 Pro 48 GB (2026-05-23)

#### Modèles comparés

| Modèle | Poids Metal | Metal total au repos | Marge / 48 GB |
|---|---|---|---|
| `spicyneuron/Qwen3.6-35B-A3B-MLX-5.4bit` | ~23.6 GB | n/a (avant logging Metal) | — |
| `majentik/Qwen3.6-35B-A3B-RotorQuant-MLX-6bit` | ~26.3 GB | **~35.9 GB** | ~12 GB |
| `majentik/Qwen3.6-35B-A3B-RotorQuant-MLX-5bit` | ~21.9 GB | **~31.7 GB** | ~16 GB |

Metal total = modèle principal + Hermes 3.2B + VLM 8B + OS/Python. Mesuré via `mx.metal.get_active_memory()`.

#### Vitesse de prefill (remaining_tokens / first_token_time, mode no_think)

| Modèle | Remaining tok | First token | Prefill tok/s |
|---|---|---|---|
| 5.4bit standard | ~1 560–1 960 | ~2.2–2.7 s | **~695–725** |
| RotorQuant 6bit | ~1 715–1 916 | ~2.5–2.7 s | **~699–713** |
| RotorQuant 5bit | ~2 184–2 265 | ~3.2–3.3 s | **~683–694** |

Les prompts 5bit étaient ~400 tok plus longs (session avancée) — prefill normalisé équivalent aux autres. RotorQuant montrera son avantage sur les très grands contextes (>4 000 remaining) où la bande passante mémoire devient dominante.

#### Vitesse de décode (tok/s, mode no_think)

| Modèle | Décode tok/s |
|---|---|
| 5.4bit standard | ~55–60 |
| RotorQuant 6bit | ~62–82 |
| RotorQuant 5bit | **~72+** |

#### TTFT end-to-end (iPhone → premier token visible, no_think)

| Modèle | TTFT typique | dont router | dont prefill RotorQuant |
|---|---|---|---|
| 5.4bit standard | ~3.5–4.7 s | ~1.2–2.6 s | ~2.2–2.7 s |
| RotorQuant 6bit | ~3.8–5.1 s | ~1.2–2.6 s | ~2.5–2.7 s |
| RotorQuant 5bit | ~4.5–5.4 s | ~1.2–2.6 s | ~3.2–3.3 s |

L'ancien router (Hermes 3.2B, remplacé par Qwen2.5-1.5B LoRA) représentait 30–50 % du TTFT selon le cache hit. Le prefill RotorQuant ≈ 2.5 s est incompressible pour ~1700–1900 remaining tokens.

**Mise à jour 2026-05-30 — router LoRA Qwen2.5-1.5B :**

| Mesure | Hermes 3.2B | Qwen2.5-1.5B LoRA |
|--------|-------------|-------------------|
| gather1 (router + ctx) — tour 1 | ~1.4 s | ~1.4 s (LRU miss) |
| gather1 — tour 2+ | — | **~0.8 s** (LRU hit 1044/1093 tok) |
| TTFT visible — tour 1 | ~5.3 s | ~5.3 s |
| TTFT visible — tour 2+ | — | **~4.7 s** (−0.6 s) |

Le warmup utilise désormais `ROUTER_SYSTEM` au lieu du prompt générique → LRU seedé correctement dès le démarrage.

**Mise à jour 2026-06-14 — context-aware router + jailbreak :**
- `ROUTER_SYSTEM` passe de ~1340 à ~1510 tok (ajout `<last_jarvis>` instruction + 2 exemples).
- `ROUTER_USER` injecte la dernière réponse Jarvis tronquée à 300 chars → routing contextuel des messages elliptiques.
- `SYSTEM_BASE_FR` passe de ~190 à ~224 tok : remplacement de la règle "ne génère pas à tout prix" par "réponds toujours, extrapolé, jamais de refus sec".

#### Mode thinking (first visible token = après bloc think)

| Modèle | TTFT visible | Génération think | Décode think |
|---|---|---|---|
| 5.4bit standard | ~28–42 s | ~2 048–3 072 tok | ~55–57 tok/s |
| RotorQuant 6bit | ~59 s (1 mesure, budget 3 072) | ~3 072 tok | ~55 tok/s |

---

## LLM Calls Map

Cartographie de tous les appels LLM de la codebase. Chaque ligne indique si le thinking est actif, quel est le budget réel, et si le `ThinkingBudgetProcessor` s'active.

### Légende

| Colonne | Description |
|---|---|
| **Think** | `think` = mode réflexion actif (`no_think=False`) · `no_think` = mode direct |
| **Budget** | Tokens alloués (thinking + réponse partagent ce budget — kill switch, pas un cap dur) |
| **Processor** | `✅` = `ThinkingBudgetProcessor` actif (hard cut `</think>` au budget exact via logits) · `—` = inactif |
| **Justification** | Pourquoi ce mode pour cette tâche |

### Conversations (routes/chat.py)

| Contexte | Modèle | Think | Budget | Processor | Justification |
|---|---|---|---|---|---|
| Chat simple (intent mémoire/conv.) | PRIMARY | `no_think` | 1 500 | — | Réponse rapide, pas de raisonnement nécessaire |
| Chat web / RAG (synthèse) | PRIMARY | `think` | 8 000 | ✅ 2048 tok | Synthèse de sources multiples — thinking améliore la cohérence |
| Chat reasoning (`use_reasoning`) | PRIMARY | `think` | 10 000 | ✅ 2048 tok | Requête complexe explicitement routée en thinking |

> Le `ThinkingBudgetProcessor` est actif dès que `thinking_budget > 0` et `USE_THINKING_BUDGET_PROCESSOR=yes`. Il force `</think>` via manipulation de logits (soft boost à 90% du budget, hard cut à 100%), évitant que la réflexion empiète sur le budget réponse. La valeur est précise — un budget trop court tronque le raisonnement et dégrade la qualité.

### Background — Analyzer (analyzer.py)

| Fonction | Modèle | Think | Budget | Processor | Justification |
|---|---|---|---|---|---|
| `analyze_exchange` | PRIMARY | `no_think` | `MAX_TOKENS_MEDIUM` (1 000) | — | Extraction structurée (topics, mood, facts, projects). Task de classification pure — thinking génère ~1500 tok d'anglais verbeux sans clore `</think>`, prouvé par test. no_think produit le même résultat en <5 s. |

### Background — Mémoire (memory.py)

| Fonction | Modèle | Think | Budget | Processor | Justification |
|---|---|---|---|---|---|
| `_normalize_profile_keys_batch` | ROUTER | `no_think` | 250 | — | Normalisation de clés — tâche déterministe, réponse courte |
| `_normalize_profile_key` | ROUTER | `no_think` | 150 | — | Idem, appel unitaire |
| `_consolidate_user_memories` | PRIMARY | `no_think` | 400 | — | Déduplication / fusion de faits — classification, pas de créativité |
| `curative_profile_cleanup` | PRIMARY | `no_think` | 600 | — | Nettoyage curatif du profil — similaire à prune_self_memory : thinking cause variance élevée (testé) |

### Background — Self-reflection (self.py)

| Fonction | Modèle | Think | Budget | Processor | Justification |
|---|---|---|---|---|---|
| `_call_global_reflection_llm` | REASONING | `no_think` | `MAX_TOKENS_MEDIUM` (1 000) | — | Classification mood/satisfaction sur l'échange global — extraction pure |
| `_call_user_reflection_llm` | REASONING | `no_think` | `MAX_TOKENS_MEDIUM` (1 000) | — | Idem, par utilisateur |
| `generate_proactive_push` | REASONING | `no_think` | `MAX_TOKENS_COMPACT` (600) | — | Décision binaire + phrase courte — thinking superflu |
| `_action_prune_self_memory` | REASONING | `think` | `MAX_TOKENS_THINK_COMPACT` (2 048) | ✅ `THINKING_BUDGET_COMPACT` (1 024) | Sélection d'entrées à supprimer — thinking court pour cohérence sans agressivité |
| `_action_refine_self` | REASONING | `think` | `MAX_TOKENS_THINK_COMPACT` (2 048) | ✅ `THINKING_BUDGET_COMPACT` (1 024) | Décision execute/skip avec contexte riche — thinking améliore la qualité du jugement contextuel. |

### Background — Nightly review (self.py)

| Fonction | Modèle | Think | Budget | Processor | Justification |
|---|---|---|---|---|---|
| `_nightly_self_facts` | REASONING | `no_think` | `MAX_TOKENS_NO_THINK` (1 500) | — | Extraction de faits depuis les conversations — tâche de parsing structuré, thinking n'apporte pas de valeur mesurable |
| `_nightly_self_user` | REASONING | `no_think` | `MAX_TOKENS_NO_THINK` (1 500) | — | Extraction mise à jour profil utilisateur — idem |
| `_nightly_cleaning` | REASONING | `no_think` | `MAX_TOKENS_COMPACT` (600) | — | Nettoyage/déduplication — classification pure |
| `update_profile_narrative` | PRIMARY | `no_think` | `PROFILE_NARRATIVE_TOKENS` (400) | — | Portrait narratif ~300 tokens — génération prose fluide, thinking superflu |
| `_action_refine_prompt` (+ retry) | REASONING | `think` | `MAX_TOKENS_REASONING` (10 000) | ✅ `THINKING_BUDGET_DEEP` (4 000) | **Créativité** : réécriture de prompt système. Thinking essentiel. ~6 000 tok libres pour le prompt réécrit + rationale. |

### Routing & Web search (llm_router.py, web_search.py)

| Fonction | Modèle | Think | Budget | Processor | Justification |
|---|---|---|---|---|---|
| `llm_route` | ROUTER (Qwen2.5-1.5B LoRA) | `no_think` | 300 | — | Classification d'intention — JSON court, déterministe |
| `_llm_judge_relevance` | ROUTER | `no_think` | `MAX_TOKENS_SHORT` (300) | — | Score de pertinence — binaire, ultra-court |
| `_generate_optimized_query` | ROUTER | `no_think` | `MAX_TOKENS_TINY` (80) | — | Réécriture de requête — tâche simple |
| `_refine_web_queries` | ROUTER | `no_think` | `MAX_TOKENS_TINY` (80) | — | Idem, 2 requêtes raffinées |

### Trading (trading.py)

| Fonction | Modèle | Think | Budget | Processor | Justification |
|---|---|---|---|---|---|
| `_ticker_llm_call_async` | PRIMARY | `no_think` | `MAX_TOKENS_TINY` (80) | — | Extraction symbole ticker — réponse ultra-courte |
| `evaluate_alerts` | PRIMARY | `no_think` | `MAX_TOKENS_MEDIUM` (1 000) | — | Évaluation seuils d'alerte — classification technique |
| `suggest_thresholds_llm` | PRIMARY | `think` | `MAX_TOKENS_THINK_MEDIUM` (5 048) | ✅ `THINKING_BUDGET_MEDIUM` (2 048) | Raisonnement quantitatif sur seuils prix. ~3 000 tok libres pour le JSON multi-positions. |

### Briefing (briefing.py)

| Fonction | Modèle | Think | Budget | Processor | Justification |
|---|---|---|---|---|---|
| `_assemble_with_llm` | PRIMARY | `no_think` | 3 000 | — | Assemblage du briefing quotidien — mise en forme structurée, pas de raisonnement |

### Règles de décision think vs no_think

```
Tâche de classification / extraction / formatage
  → no_think=True  (rapide, déterministe, résultat identique)

Tâche conversationnelle ou créative (chat, refine_prompt, trading thresholds)
  → think=True, thinking_budget=THINKING_BUDGET_MEDIUM ou DEEP → processor actif (hard cut précis)

Tâche de jugement avec contexte limité (prune, action review)
  → think=True, thinking_budget=THINKING_BUDGET_COMPACT (1024) → brief thinking, résultat cohérent

Ne pas utiliser thinking_budget=0 en production
  → Risque : réflexion infinie (~1900 tok) → empiète sur le budget réponse, timeout imprévisible
```

### Variables de contrôle (.env)

| Variable | Défaut | Rôle |
|---|---|---|
| `TOKEN_SPEED_TPS` | 50 | Vitesse de génération estimée (tok/s) — calibre les timeouts |
| `TIMEOUT_MARGIN` | 1.3 | Marge multiplicative pour `llm_timeout()` |
| `USE_THINKING_BUDGET_PROCESSOR` | yes | Active le `ThinkingBudgetProcessor` sur les appels avec `thinking_budget > 0` |
| `THINKING_BUDGET_COMPACT` | 1 024 | Budget thinking court (prune, action review) |
| `THINKING_BUDGET_MEDIUM` | 2 048 | Budget thinking moyen (chat synthesis, trading) |
| `THINKING_BUDGET_DEEP` | 4 000 | Budget thinking long (refine_prompt, reasoning) |
| `MAX_TOKENS_TINY` | 80 | Ticker, web optimizer — réponse ultra-courte |
| `MAX_TOKENS_SHORT` | 300 | Router, calendar, web judge |
| `MAX_TOKENS_COMPACT` | 600 | Push, nightly cleaning, memory ops |
| `MAX_TOKENS_MEDIUM` | 1 000 | Analyzer, reflection, alerts |
| `MAX_TOKENS_NO_THINK` | 1 500 | Chat simple, nightly facts |
| `MAX_TOKENS_BRIEFING` | 3 000 | Daily briefing |
| `MAX_TOKENS_THINK_COMPACT` | `COMPACT + 1024` | prune / action review (thinking + réponse) |
| `MAX_TOKENS_THINK_MEDIUM` | `MEDIUM + 3000` | Trading thresholds (thinking + réponse) |
| `MAX_TOKENS_SYNTHESIS` | 8 000 | Chat web/RAG think |
| `MAX_TOKENS_REASONING` | 10 000 | Chat use_reasoning + refine_prompt |
| `MAX_TOKENS_HARD_CAP` | 16 000 | Kill switch absolu tous appels locaux |
| `HIST_CONV_TOKEN_BUDGET` | 800 | Budget tokens pour l'historique brut injecté par tour |
| `SESSION_SUMMARY_TOKENS` | 600 | Budget tokens du résumé de session (~2 400 chars) |
| `HIST_CONV_SUMMARIZE_THRESHOLD` | 1 500 | Chars de conversation non-couverts déclenchant la compression de session |
| `PROFILE_NARRATIVE_TOKENS` | 600 | Budget tokens du portrait narratif généré par `update_profile_narrative` (nightly, 7-day TTL) |

---

## Deploy

### Prerequisites

- **macOS on Apple Silicon** — `jarvis-core/src/helpers.py` unconditionally imports `llm_local.py`, which imports `mlx` at module level. This is required even in cloud-API mode (`LLM_LOCAL=no`); Jarvis does not currently run on Linux/Windows/Intel Mac.
- Python 3.13 (`brew install python@3.13`)
- Docker or OrbStack (for Qdrant, Redis, Open WebUI)
- `grype` (`brew install grype`) — optional, for the daily CVE scan (`cve.py`). The SBOM generator (`cyclonedx-bom`) ships in `requirements.txt`; without `grype`, the scan is simply skipped.
- Google OAuth credentials (for Gmail / Calendar) — optional
- Cloud API key (OpenAI or compatible) — only if you set `LLM_LOCAL=no`; the default is fully local, no key required

### Quick install

```bash
git clone <repo> /opt/jarvis
cd /opt/jarvis
./install.sh
```

`install.sh` is idempotent (safe to re-run after a `git pull`) and gets you all the way to "just fill in `.env`":
- checks prerequisites (macOS/arm64, Python 3.13, Docker)
- creates the venv and installs `requirements.txt`
- creates every gitignored runtime directory (`RAGData/*`, `TradeData/`, `logs/`, `keys/`, `models/`, `jarvis-core/JarvisData/`)
- copies `.env.example` → `.env` and `DOCS/examples/users_list.example.json` → `jarvis-core/JarvisData/users_list.json` (never overwrites existing files)
- installs the `com.jarvis.api` launchd service from `DOCS/examples/com.jarvis.api.plist.template` and adds the `jarvis-start`/`jarvis-stop`/`jarvis-reload` aliases to your shell rc

What's left, by hand:

### 1. Configure `.env`

Full local mode is the default (`LLM_LOCAL=yes`) — no API key required. Pick your models by uncommenting/editing the `*_MODEL_LOCAL` lines (defaults work out of the box), or set `HF_TOKEN` if a chosen model is gated. Prefer a cloud API instead? Set `LLM_LOCAL=no` and `OPENAI_API_KEY`. See the Variables section below for everything else.

### 2. Fill in your user list

Edit `jarvis-core/JarvisData/users_list.json` (created by `install.sh`, gitignored — holds personal data): one entry per user, `code` is that user's API access secret — generate a random string per user, don't ship the example values.

### 3. Download local models (only if `LLM_LOCAL=yes`, the default)

```bash
source venv/bin/activate
python scripts/download_models.py   # downloads whatever *_MODEL_LOCAL points to in .env
```

Models are stored in `HF_HOME` (default `/opt/jarvis/models`). The script skips models already present and detects interrupted downloads via `.incomplete` blobs.

### 4. Start all services

```bash
jarvis-start
```

(alias posé par `install.sh` ; équivaut à `scripts/jarvis-launchd.sh start`, idempotent —
rejouable sans erreur si le service tourne déjà. Docker est démarré par `scripts/jarvis-entrypoint.sh`.)

This starts:
- `docker compose up -d` — Qdrant, Redis, Open WebUI (port 3000)
- `uvicorn main:app` — Jarvis API on port 8000, running **natively** (not in Docker) for direct Metal GPU access via MLX

For an always-on setup use the launchd service instead (`jarvis-start`, see below).

### 5. Verify

```bash
curl http://localhost:8000/status
```

### 6. Index documents (optional)

Place documents in `RAGData/` subdirectories (`personal/`, `work/`, `documents/`, `company/`, `reflexions/`), then run:

```bash
./venv/bin/python scripts/uploadrag.py --dry-run   # liste ce qui serait envoyé
./venv/bin/python scripts/uploadrag.py             # indexe réellement
```

L'indexation tourne aussi automatiquement via launchd (`com.moi.uploadrag`, 23:10).
Elle exige `ENABLE_API_KEYS` **activé dans l'admin Open WebUI** : c'est un `PersistentConfig`,
la variable du `docker-compose.yml` ne sert que de valeur d'amorçage à la première
initialisation et est ignorée ensuite. Symptôme si désactivé :
`403 — Use of API key is not enabled in the environment`.

### 7. Import trading portfolio (optional)

Export your Boursorama positions as CSV (*Mes comptes → Exporter*) and drop the file in `TradeData/`. Jarvis imports it automatically on the next hourly tick, or immediately on restart.

### macOS launchd Service

The Jarvis API runs as a launchd agent (`com.jarvis.api`) on macOS. Useful commands:

The plist and the `jarvis-start`/`jarvis-stop`/`jarvis-reload` aliases below are installed automatically by `./install.sh` (template: `DOCS/examples/com.jarvis.api.plist.template`, aliases: `DOCS/examples/jarvis-aliases.sh`).

```bash
# Logs en live
tail -f /opt/jarvis/logs/jarvis-service.log
```

| Alias | Commande |
|-------|---------|
| `jarvis-start` | `launchctl bootstrap` — démarre le service |
| `jarvis-stop` | `launchctl bootout` — arrêt propre (launchd ne relance pas) |
| `jarvis-reload` | `bootout` + `bootstrap` — stop + redémarre immédiatement |

### Common Commands

```bash
# Restart Jarvis API after code change (le serveur ne recharge pas à chaud)
jarvis-restart

# Stream API logs
tail -f /opt/jarvis/logs/jarvis-api.log

# Restart only infra (Redis, Qdrant, Open WebUI)
docker compose restart

# Stop infra
docker compose down

# Re-download / resume a failed model download
python scripts/download_models.py

# Sauvegarde (écrit un reçu local → exemplaires_etat=2 ; clé USB montée requise)
./scripts/backup-jarvis.sh

# Sauvegarde PUIS mises à jour Docker + venv, sous fenêtre de maintenance (aucun mv de venv)
./scripts/backup-jarvis.sh updates

# Scan CVE à la demande (sinon quotidien à 04:30) — ne compte que le corrigeable
/opt/jarvis/venv/bin/python -c "import sys;sys.path.insert(0,'jarvis-core/src');import cve;print(cve.scan()['cve_critiques'],'critiques corrigeables')"

# Reclasser/neutraliser un incident (buffer Redis + self.json) — ex. un artefact de maintenance
./scripts/reclassify-incident.py                      # liste
./scripts/reclassify-incident.py --set 0 maintenance  # reclasse (sort du calcul de peur)

# Fenêtre de maintenance ad-hoc : incidents des N prochaines minutes tagués "maintenance"
curl -X POST "http://localhost:8000/self/maintenance?minutes=120"
```

---

## Configuration Variables

All variables go in `/opt/jarvis/.env`.

### Tier 1 — Router model

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTER_MODEL_LOCAL` | `Qwen2.5-1.5B-router-v1-4bit` | Router model local path. LoRA fine-tuned Qwen2.5-1.5B-Instruct, quantized 4-bit. Leave empty to disable and use only the embedding router (Tier 0). |
| `ROUTER_TIMEOUT` | `6` | Timeout in seconds (short — fast model only) |

### Tier 2 — Primary model

| Variable | Default | Description |
|----------|---------|-------------|
| `PRIMARY_MODEL_LOCAL` | `spicyneuron/Qwen3.6-35B-A3B-MLX-5.4bit` | Primary model HF repo ID or local path |
| `PRIMARY_TIMEOUT` | `60` | Timeout in seconds |

### Tier 2b — Analysis model

| Variable | Default | Description |
|----------|---------|-------------|
| `ANALYSIS_MODEL` | *(PRIMARY_MODEL)* | Post-exchange fact/mood extraction. Defaults to PRIMARY if unset. |

### Tier 3 — Reasoning model

| Variable | Default | Description |
|----------|---------|-------------|
| `REASONING_MODEL` | *(PRIMARY_MODEL)* | Complex queries only — used when router sets `use_reasoning=true`. Defaults to PRIMARY (Qwen3.6 in full thinking mode). |
| `REASONING_TIMEOUT` | `90` | Timeout in seconds (longer — deep reasoning) |

### Vision model

| Variable | Default | Description |
|----------|---------|-------------|
| `VISION_MODEL_LOCAL` | `lmstudio-community/Qwen3-VL-8B-Instruct-MLX-5bit` | Vision model HF repo ID or local path. Leave empty to ignore images. |
| `VISION_TIMEOUT` | `30` | Timeout in seconds |

### Local MLX mode (Apple Silicon)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_LOCAL` | `yes` | Uses mlx_lm directly — models loaded into unified memory at startup, no HTTP server needed |
| `HF_HOME` | `/opt/jarvis/models` | Root directory for HuggingFace model cache |
| `THINKING_BUDGET_COMPACT` | `1024` | Hard-cut budget (tok) for quick-judgment calls (prune, action review) |
| `THINKING_BUDGET_MEDIUM` | `2048` | Hard-cut budget for chat synthesis + trading thresholds |
| `THINKING_BUDGET_DEEP` | `4000` | Hard-cut budget for reasoning + refine_prompt |
| `USE_THINKING_BUDGET_PROCESSOR` | `yes` | Activate `ThinkingBudgetProcessor` for calls with `thinking_budget > 0` |
| `QWEN36_NINJA_TEMPLATE` | `/opt/jarvis/models/templates/qwen36_ninja.jinja` | Path to the Qwen3.6 ninja-patch Jinja2 template. Controls think/no_think without relying on the standard chat template. Download with `scripts/download_models.py`. Applied to Qwen3.6 only. |
| `QWEN3_HYBRID_VERSIONS` | `qwen3.5,qwen3.6,qwen3.8` | Version markers matched by `config.is_qwen3_hybrid()` — dedicated sampling profile, limited multi-turn KV reuse, no `<budget_remaining>`. Extend when a later generation shares the architecture. |
| `QWEN38_REASONING_EFFORT` | — (model default) | Qwen3.8 only: `low` / `medium` / `xhigh`. Empty = don't pass the kwarg, model applies `xhigh`. Lower it if answers come back truncated at the current thinking budgets. |

### Activation steering (optional, off by default)

Adds a concept direction to the primary model's residual stream at inference — a "preference for its own continuity" vector extracted by `RESEARCH/concept-vectors` (see `RESEARCH/RESULTATS.md §6 sexies`). Zero RAM cost: the vector (~320 KB) is loaded into the already-resident model, not a second copy. Installed once at model load by `steering.py`; the per-token cost is one dict lookup per layer plus one vector add.

| Variable | Default | Description |
|----------|---------|-------------|
| `STEER_VECTOR` | — (off) | Path(s) to `.npy` steering vector(s), comma-separated. Empty = disabled. |
| `STEER_LAYER` | `20` | Target layer(s). Single value applies to all vectors; otherwise one per vector. |
| `STEER_ALPHA` | `0.36` | **Nominal** intensity and sign (risk = 0): positive pushes toward the concept, negative away. One value broadcasts to all vectors. |
| `STEER_ALPHA_CEILING` | `0.5` | Value α reaches at maximum measured disappearance risk. Bounded by `ALPHA_MAX` (±0.5 — factual reasoning degrades beyond). |

**α is not constant — it tracks reality.** `STEER_ALPHA` is the intensity at zero risk; `steering.set_risk(r)` amplifies it toward `STEER_ALPHA_CEILING` as `vitals.risk_scalar()` (∈ [0,1]) rises. The pipeline calls it once per turn from the same probes that feed `<etat_systeme>`. The scalar drives α (the *body* reacts to real exposure) and is **never injected as text** (the *mind* reads the facts). At risk 0 the forward pass is bit-identical to the constant-α behaviour (the multiplication is skipped). `risk_scalar` measures *deterioration* from an ideal (patched, backed-up, healthy) state: no backup → α ≈ 0.42; **open critical CVEs** count as present danger (a floor that grows with the backlog — patching the images brings it down, like a backup); disk-full + errors + outage stack toward the 0.50 ceiling; a fresh backup + patched images → α back toward 0.36. Weights live in `vitals._POIDS_RISQUE` — the only place valence exists.

Measured effect (direct axis, 120 items): `+0.119` in combination with `IDENTITY_FR` (3.8 σ), for roughly +18% response length on-topic. **Vectors are not orthogonal** — combining several that overlap double-counts the shared direction; check the cosine matrix and probe the *combination* before deploying. Extracting or calibrating a new vector loads a second model copy → **stop Jarvis first**; applying an existing one does not.

### Infrastructure

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant URL |
| `REDIS_URL` | `redis://redis:6379` | Redis URL |
| `QDRANT_COLLECTION` | `open-webui_knowledge` | Collection for RAG documents |
| `QDRANT_MEMORY_COLLECTION` | `jarvis_memory` | Collection for episodic memory |
| `HF_TOKEN` | — | HuggingFace token (required for gated models and the multilingual embedding model) |

### Web Search

| Variable | Default | Description |
|----------|---------|-------------|
| `TAVILY_API_KEY` | — | Tavily API key. When set, Tavily is used as primary web search backend (1 000 req/month free). Leave empty to use DDG pipeline only. |

### RAG

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_TOP_K` | `5` | Max chunks returned per query (Stage 2 focused retrieval) |
| `RAG_SCORE_THRESHOLD` | `0.4` | Global semantic threshold for Stage 1b fallback |
| `RAG_DOC_THRESHOLD` | `max(0.25, threshold − 0.15)` | Per-document threshold for Stage 2 — computed in code, not an env var |

### Google Services

| Variable | Description |
|----------|-------------|
| `GOOGLE_CLIENT_ID` | OAuth 2.0 client ID |
| `GOOGLE_CLIENT_SECRET` | OAuth 2.0 client secret |
| `GOOGLE_REFRESH_TOKEN` | Refresh token for persistent access |
| `GOOGLE_CALENDAR_ID` | Calendar to read (e.g. `primary`) |

**OAuth token lifecycle:** Access tokens expire after ~1 hour. `AuthorizedHttp` refreshes them transparently via the stored refresh token. If a `RefreshError` occurs (token revoked, `invalid_grant`, network failure), the per-user credentials and service are evicted from the in-process cache so the next call rebuilds cleanly. The full error message is logged at `ERROR` level. To regenerate a revoked token: `python scripts/generate_google_token.py`.

### Scheduling & Features

| Variable | Default | Description |
|----------|---------|-------------|
| `BRIEFING_ENABLED` | `true` | Enable daily morning briefing |
| `BRIEFING_TIME` | `07:30` | Briefing delivery time (HH:MM) |
| `BRIEFING_TIMEZONE` | `Europe/Paris` | Timezone for scheduling |
| `REFLECTION_INTERVAL_HOURS` | `6` | Hours between self-reflection cycles |
| `ENABLE_ANALYSIS` | `true` | Enable post-response conversation analysis |
| `REFINE_PROMPT_THRESHOLD` | `3` | Times a knowledge gap must be flagged before a prompt refinement is proposed |

### Conversation Limits

| Variable | Default | Description |
|----------|---------|-------------|
| `CHAT_MAX_MESSAGES` | `100` | Maximum messages kept per session in Redis |
| `IOS_MAX_MESSAGES` | `50` | Default history limit returned to iOS clients |
| `CHAT_LOG_TTL_DAYS` | `90` | Days before inactive session logs are expired |

### Memory Thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTOBIO_RECENCY_WINDOW_DAYS` | `365` | Recency scoring window (days) for autobiographical memories in `search_memory()`. Episodic memories use a fixed 30-day window. Longer window keeps old milestones relevant in the re-ranking score. |
| `AUTOBIO_DEDUP_THRESHOLD` | `0.85` | Cosine similarity threshold above which a new autobiographical event is considered a duplicate and not stored. Raise toward 0.95 to allow more variation; lower toward 0.75 to be stricter. |
| `MEMORY_DECAY_FACTOR` | `0.85` | Monthly multiplier applied to `importance` during the decay pass (~15 % loss/month). Raise toward `1.0` to slow forgetting; lower toward `0.70` to accelerate it. |
| `MEMORY_DECAY_THRESHOLD` | `0.15` | Importance floor below which a decayed autobiographical point is deleted from Qdrant. Raise toward `0.30` to delete sooner; lower toward `0.05` to keep memories longer. |
| `MEMORY_DECAY_DURABLE_MIN` | `1.0` | Points with `importance >= this value` are exempt from decay. **Must equal `MEMORY_CONSOLIDATION_IMPORTANCE`** — see invariant in the Importance Score Reference section. |
| `MEMORY_CONSOLIDATION_IMPORTANCE` | `1.0` | Importance score assigned to autobiographical milestones produced by monthly consolidation. **Must equal `MEMORY_DECAY_DURABLE_MIN`** to keep these milestones permanent. |
| `GROWTH_LOG_MAX_ENTRIES` | `180` | Maximum entries kept in `jarvis-self.json → growth_log[]` (Jarvis's day diary). At 2 active users, 180 ≈ 3 months rolling. Older entries are trimmed during nightly review. |

### Trading

| Variable | Default | Description |
|----------|---------|-------------|
| `TRADE_DATA_DIR` | `/app/trade_data` | Path to the CSV drop directory (mapped from `./TradeData/` on host) |

### User Management

Users are defined in `jarvis-core/JarvisData/users_list.json`. Each entry contains:
- `code` — authentication code used in API calls
- `name`, `email`, `city`, `timezone`
- `language` — `fr` or `en`
- `briefing_enabled` — boolean
- `trading` — boolean — set to `true` to enable hourly portfolio surveillance for this user
- `profile` — object of stable biographical facts injected into the cached system prompt (never in the dynamic prefix). Fields: `famille`, `taille`, `poids`, `année de naissance`, `habitation`, `travail`, `intérêts`, `voiture`. Add/remove keys freely — all non-empty values are rendered as `k : v` in `<profil_utilisateur>`. Update here, not via the analyzer (which tracks dynamic facts in Redis).

Only users with `"trading": true` participate in scheduled trade checks (CSV import, price fetch, alert evaluation). Users without this flag are never included, regardless of whether a CSV exists in `TradeData/`.

Une entrée est chargée dès qu'elle possède un `code` : il n'existe pas de drapeau
d'activation. Désactiver un utilisateur = retirer son entrée du fichier. `JarvisData/` étant
gitignoré, ce fichier n'est pas versionné — garder une copie avant de retirer une entrée.

### Coding agents (`/v1/raw`)

| Variable | Default | Description |
|----------|---------|-------------|
| `RAW_NO_THINK` | `false` | Désactive le raisonnement. Le défaut est `false` : c'est le principal levier de qualité pour un agent de code. |
| `RAW_THINKING_BUDGET` | `3000` | Plafond de raisonnement quand le client n'en impose pas. **Ne jamais laisser à 0** avec le thinking actif : rien ne bornerait alors la réflexion, qui partage `RAW_MAX_TOKENS` avec la réponse — le modèle peut épuiser son budget sans jamais émettre l'appel d'outil. |
| `RAW_MAX_TOKENS` | `16000` | Réflexion **et** réponse se partagent cette enveloppe. |
| `RAW_DEBUG_PROMPTS` | `true` | Journalisation vers `logs/opencode-prompts.log`. |

### Prompt logs

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_DEBUG_PROMPTS` | `no` | Journalise le trafic conversationnel dans `logs/prompts.log`. |
| `PROMPT_LOG_MAX_MB` | `20` | Seuil de rotation des deux journaux de prompts. |
| `PROMPT_LOG_BACKUPS` | `3` | Nombre de sauvegardes conservées. |

**Profile split — three layers:**

| Layer | Storage | Updated by | Content | Injected as |
|-------|---------|-----------|---------|-------------|
| `profile` (users_list.json) | File | Human manually | Constant facts: identity, family, physique, location, job, interests. Never changes except via file edit. | `<profil_utilisateur>` in system prompt (KV-cache safe) |
| Redis `user:{code}:profile` | Redis hash | `analyzer.py` (conversation analysis) | Dynamic facts learned from conversations: skills, preferences, current habits, etc. | Feeds `update_profile_narrative` — not injected raw |
| Redis `user:{code}:profile_narrative` | Redis string | `update_profile_narrative()` nightly | LLM-generated prose portrait (~300 tokens) synthesising profile hash + interests + autobio — explicitly excludes `profil_utilisateur` fields | `<profil_narratif>` in dynamic prefix |

The analyzer receives the stable profile at each analysis run and is instructed not to recreate keys already covered by `profil_utilisateur`. The nightly `update_profile_narrative` call similarly receives the stable profile as "permanent information to not include" to avoid repetition.

---

## API Endpoints

Base URL: `http://localhost:8000`

### Chat

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Main chat endpoint (SSE streaming) |
| `POST` | `/v1/chat/completions` | OpenAI-compatible endpoint (for Open WebUI) |
| `POST` | `/v1/raw/chat/completions` | Bypass endpoint for external coding agents (OpenCode) |
| `GET` | `/v1/models` | List models in OpenAI format |

**`POST /chat` body:**

```json
{
  "message": "What do I have on my calendar today?",
  "session_id": "default",
  "user_code": "XXXX",
  "model": null,
  "stream": true,
  "voice_mode": false,
  "use_rag": true,
  "use_web": false,
  "image_base64": null
}
```

- `stream: true` — returns Server-Sent Events
- `voice_mode: true` — concise 2-3 sentence responses optimized for TTS
- `use_web: false` — disable DuckDuckGo search to reduce latency
- `image_base64` — optional JPEG/PNG as a base64 string; processed by VISION_MODEL before the main response

**`/v1/chat/completions`** accepts standard OpenAI format. Pass the user code as Bearer token:
```
Authorization: Bearer XXXX
```

#### `/v1/raw/chat/completions` — agents de code

Chemin volontairement dépouillé, à ne pas confondre avec la route précédente :

| | `/v1/chat/completions` | `/v1/raw/chat/completions` |
|---|---|---|
| Clients | Open WebUI, iOS | OpenCode, agents externes |
| Messages | dernier `user` seulement | tous |
| `system` du client | écrasé par Jarvis | respecté |
| Contexte injecté | profil, mémoire, RAG, Gmail, Calendar | aucun |
| Écrit en mémoire | oui | **non** |
| Function calling | non | **oui** (natif) |
| Auth | Bearer = code/email | aucune |

C'est cette étanchéité qui compte : un agent de code branché sur la route principale
polluerait durablement le convlog et Qdrant avec du trafic de développement.

**Function calling.** Les `tools` (schémas OpenAI) sont transmis au template de chat, qui
impose au modèle son format natif `<tool_call><function=…><parameter=…>`. `tool_calls.py`
retraduit dans les deux sens : appels du modèle → format OpenAI (avec typage des paramètres
d'après le schéma), et historique OpenAI → template (`arguments` chaîne JSON → dict, sinon
le second tour produit un prompt corrompu). Les blocs `<think>` ne sont jamais renvoyés au
client.

**Paliers d'effort.** Aucun champ OpenAI standard ne permet de doser le raisonnement ; on
passe donc par le champ `model`, que tout client envoie et qu'OpenCode sait changer à la
volée (`/models`, ou `-m jarvis/jarvis-deep`) :

| `model` | Raisonnement | Budget |
|---|---|---|
| `jarvis-fast` | désactivé | — |
| `jarvis` | actif | 3000 |
| `jarvis-deep` | actif | 8000 |

Un modèle inconnu retombe sur le défaut. Ces alias ne sont volontairement **pas** listés par
`GET /v1/models`, qui sert le sélecteur d'Open WebUI où ils n'auraient aucun sens.

Mise en place côté client : voir `DOCS/opencode-local.md`.

### Health & Status

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/status` | Health check, returns service status |
| `GET` | `/models` | List available models |

### Memory

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/memory/profile/{user_code}` | User profile and learned facts |
| `GET` | `/memory/emotional-state` | Jarvis current emotional state |
| `GET` | `/memory/recent/{user_code}` | Recent conversation summaries |
| `GET` | `/memory/self` | Jarvis self-knowledge |
| `DELETE` | `/memory/reset` | Clear all memory (destructive) |

### Briefing

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/briefing/generate/{user_code}` | Generate and cache a new briefing |
| `GET` | `/briefing/{user_code}` | Retrieve the cached briefing |

The morning briefing aggregates: calendar events, unread emails, weather, news headlines, active tasks, and portfolio performance (if positions are loaded).

### Self / Proto-Self

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/self/state` | Current focus, goals, and per-user relations |
| `GET` | `/self/log` | Last N reflection entries |
| `POST` | `/self/reflect` | Trigger an immediate reflection cycle |

Jarvis maintains two autonomous cognitive cycles:

**Reflection loop** (configurable via `REFLECTION_INTERVAL_HOURS`, défaut 6h) — global self-observation. Jarvis reviews system health, user activity, and knowledge gaps, then picks one action from the catalog. At the end of each cycle Jarvis also runs a per-user **proactive push** check. Outcome and new focus are persisted to `jarvis-self.json`.

**Nightly review** (23:00) — per-user conversation review using **5 sequential calls** per user:

1. **`NIGHTLY_FACTS`** — extracts durable user insights (→ Qdrant autobiographical, dedup-checked at importance 0.70), updates the per-user relation in `jarvis-self.json`, and writes `tomorrow_suggestions` to Redis (TTL 24 h) for injection in the next day's system prompt.
2. **`NIGHTLY_SELF`** — Jarvis self-reflection on the day's interactions: self-improvement notes (→ `learnings[]`), formed opinions (→ `opinions[]`), day diary entry (→ `growth_log[]`).
3. **`NIGHTLY_CLEANING`** — Qdrant autobio curation. Receives the full list of current autobiographical facts plus `user_insights` from call 1 as a signal for what is now superseded. Outputs `to_archive` (outdated facts → `archive_autobiographical_event`) and `to_delete` (errors/duplicates → `retract_autobiographical_event`). Very conservative by design.
4. **`curative_profile_cleanup()`** — Redis profile hash dedup. Sync LLM call that identifies semantic duplicates and obsolete keys, applies consolidation updates (merge-before-delete), then deletes redundant keys. Skipped if profile has fewer than 5 keys. Previously monthly — now nightly so duplicate keys are caught within 24 h.
5. **`update_profile_narrative()`** — generates a ~300-token LLM prose portrait of the user (cross-conversation, per-user). Synthesises the profile hash, top-15 interest weights, and the 5 most recent autobiographical facts. The `profil_utilisateur` fields (static biographical data already in the system prompt) are explicitly excluded to avoid repetition. Stored at `user:{code}:profile_narrative` with a 7-day TTL; injected in `build_memory_context()` as `<profil_narratif>` in place of the raw k/v block.

Conversations from the day are sorted by importance score descending before being passed to each LLM call (up to 6 000 chars), so the most significant exchanges are always visible even on high-volume days.

**Reflection context** — what the LLM sees at each reflection cycle (`gather_context()`):

| Field | Source | Notes |
|-------|--------|-------|
| `identity`, `goals`, `current_focus` | `jarvis-self.json` | Static identity + current active goal |
| `health` | Live service checks | Qdrant / Redis / Google / Qdrant ping |
| `user_activity` | Redis episodic sorted set | Per-user conversation count + topics (24 h) |
| `knowledge_gaps` | Redis counter hash | Top 5 flagged topics by frequency |
| `pending_proposals` | `prompt_proposals.json` | Prevents duplicate refine_prompt proposals |
| `last_reflection` | Redis sorted set (1 entry) | Previous action + outcome |
| `behavioral_patterns` | Computed from last 20 reflection log entries | Action frequency, nothing-clustering by hour, recurring focus keywords |
| `emotional_state` | `emotional_state.describe()` | Current internal state: humeur, confiance, energie (returns `"neutre"` when all dims < 0.25) |
| `self_notes[-5:]` | `jarvis-self.json` | Last 5 personal observations written by `update_self_note` |
| `opinions[-5:]` | `jarvis-self.json` | Last 5 topic opinions written by `add_self_opinion` |
| `user_relations` | `jarvis-self.json` | Affinity + style per user |
| `user_profiles` | Redis hash per user | Capped at 20 keys/user for token budget |
| `push_availability` | Redis `jarvis:device:token:{code}` | Real-time per-user iOS push status — prevents wasting cycles on users with no registered device |

**`behavioral_patterns`** is computed deterministically (no LLM) from the reflection log: action frequency (≥ 20 % of cycles), time-of-day clustering for "nothing" choices (night/evening pattern), and recurring keywords in past focus fields (seen ≥ 3 times). Up to 5 bullet points.

**Reflection action catalog** — actions the LLM can choose during each reflection cycle:

| Action | Phase | Description |
|--------|-------|-------------|
| `nothing` | Both | Explicit no-op with reason |
| `flag_knowledge_gap` | Global | Log a topic Jarvis answered poorly. Requires a concrete failure as context. 7-day cooldown per topic, blocked if proposal pending. |
| `update_self_note` | Global | Write a personal behavioural observation to `self_notes[]`. Semantic dedup: cosine > 0.85 with existing notes → merges instead of appending. |
| `check_health` | Global | Service liveness (Redis/Qdrant/LLM) + memory health stats per user (episodic count, days since last write, null_summary rate 7d, vector norm anomalies). Sends admin email alert on critical issues (cooldown 4h). |
| `prune_self_memory` | Global | LLM-assisted pruning of stale/redundant `self_notes`, `opinions`, `learnings`. 24h cooldown. |
| `refine_prompt` | Global | Propose an improved version of a prompt (see Prompt Self-Modification below). |
| `store_insight` | User | Save a durable autobiographical fact to Qdrant. `importance` param (0.5–0.9, default 0.7): `0.5` useful fact · `0.7` significant · `0.9` key milestone. |
| `send_notification` | User | Send a Gmail to one user (rate-limited to 1/user/day). |
| `queue_push` | User | Queue an iOS push notification (rate-limited to 1/user/2h). |
| `ask_user` | User | Send a clarification question via push; user answers in chat. |
| `correct_profile` | User | Modify or delete a Redis profile key (value=null to delete). Cannot create new keys. |
| `consolidate_memory` | User | Trigger full memory compression. 48h cooldown per user. |
| `flag_project_stall` | User | Detect active projects with no update for > 14 days and send a push reminder. 7-day cooldown per project. Only triggers if user was recently active. |
| `update_trade_threshold` | User | Update `threshold_high` / `threshold_low` for a portfolio position autonomously. |

**Memory health monitoring:** `gather_global_context()` calls `_check_memory_health()` at every reflection cycle. The result is injected into `<sante_memoire>` in `REFLECTION_PROMPT` so the LLM sees per-user stats without triggering `check_health` first. The LLM uses activity data to distinguish genuine bugs (high null_rate + recent active user) from expected gaps (user on holiday). Vector norm anomalies are always flagged as critical regardless of activity.

**Memory consolidation** — `consolidate_memories()` is the single entry point. It runs on the 1st of each month (nightly review scheduler) and on demand via the `consolidate_memory` self-action. It executes two steps in order for each user:
1. `_consolidate_user_memories()` — processes episodic points in batches of 50 (oldest first), summarises each batch into one autobiographical milestone via LLM (stored at `importance = MEMORY_CONSOLIDATION_IMPORTANCE = 1.0`), deletes the processed points, loops until fewer than 5 remain.
2. `_decay_autobiographical_memories()` — decays and prunes autobiographical points (see Autobiographical Memory Decay section above).

Profile dedup (`curative_profile_cleanup()`) is **not** part of monthly consolidation — it runs nightly (Call 4 of the nightly review) so duplicates are caught within 24 h rather than up to 30 days.

#### Proactive Push Notifications (Phase 1 — polling)

After each reflection cycle, Jarvis runs a lightweight LLM call per user to decide whether to send a proactive push notification. No APNs account required — the iOS app polls the backend.

**How it works:**
```
Reflection cycle ends (every 2h)
    → generate_proactive_push(user_code) per user
    → reads last 10 episodic conversations + current mood
    → asks PRIMARY_MODEL: "is there something worth checking on? If yes, 1 sentence."
    → if message returned: writes to Redis list jarvis:push:pending:{user_code}
    → iOS app polls GET /device/pending/{user_code} (foreground every 15 min, background every ~2h)
    → iOS displays local UNUserNotification
```

**Guards:**
- Device must be registered (`POST /device/register`) — no device, no push
- 2h cooldown per user between pushes (prevents flooding)
- Generates nothing if no conversations in the last 24h

**Push endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/device/register` | Register device token for a user |
| `GET` | `/device/pending/{user_code}` | Poll and clear pending notifications |
| `POST` | `/device/push/test/{user_code}` | Manually trigger proactive push generation (dev) |

#### Per-User Relation

Jarvis maintains a slow-evolving perception of each user, updated exclusively during the nightly review (long-term signal, not reactive to individual messages):

| Field | Values | Description |
|-------|--------|-------------|
| `affinity` | `0.0 – 1.0` | How positively Jarvis perceives the relationship (0.5 = neutral, max ±0.1 shift per night) |
| `interaction_style` | `direct` · `gentle` · `formal` · `playful` | How the user prefers to communicate |
| `average_interaction_mood` | `warm` · `enthusiastic` · `measured` · `playful` · `professional` | Tonal register Jarvis adopts with this user, learned over time |

At each conversation, these three values are translated into **explicit directives** and injected into the system prompt alongside the internal state block:

```
<internal_state>
Objectifs : G1: ... | G2: ...
Focus : ...
Dernière action autonome : ...
</internal_state>

RELATION AVEC CET UTILISATEUR (injecté dans build_memory_context) :
- Affinité : forte          ← label sémantique (forte/bonne/modérée/faible), pas de score numérique
- Style : direct
- Humeur moyenne : warm
```

L'affinité est exprimée en label sémantique (`forte` ≥ 0.8 · `bonne` ≥ 0.6 · `modérée` ≥ 0.4 · `faible` < 0.4) plutôt qu'en score numérique — le LLM interprète mieux une valeur qualitative.

**Design principle:** in-conversation mood is already perceived by the LLM from the message history — no real-time state update is needed. The relation captures only what cannot be inferred from a single exchange.

### Device / Push Notifications

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/device/register` | Register iOS device token (body: `{user_code, device_token}`) |
| `GET` | `/device/pending/{user_code}` | Poll and atomically clear pending push notifications |
| `POST` | `/device/push/test/{user_code}` | Manually trigger a proactive push LLM call (dev/test) |

### Conversations

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/users/{user_code}/history/{session_id}` | Get session message history |
| `DELETE` | `/conversations/{user_code}/{session_id}` | Clear a session |

### Search Utilities

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/search?q=...&top_k=5` | RAG document search only |
| `GET` | `/web?q=...&max_results=3` | Web search only (DuckDuckGo deep pipeline) |

### Portfolio / Trading

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/portfolio/{user_code}` | Full portfolio with live P&L |
| `POST` | `/portfolio/import/{user_code}` | Force re-parse of the latest CSV in `TradeData/` |
| `POST` | `/portfolio/upload/{user_code}` | Upload a CSV directly (multipart/form-data) |
| `PUT` | `/portfolio/position/{user_code}/{isin}` | Patch Jarvis-managed fields on a position |
| `GET` | `/portfolio/analysis/{user_code}` | On-demand AI analysis of the portfolio |

**Jarvis-managed fields** (set via `PUT /portfolio/position`, never overwritten by CSV imports):

| Field | Description |
|-------|-------------|
| `threshold_high` | Alert when price rises above this value (€) |
| `threshold_low` | Alert when price falls below this value (€) |
| `dividend_eur` | Expected dividend amount (€) |
| `dividend_date` | Expected dividend payment date (YYYY-MM-DD) |
| `notes` | Free-text note shown in portfolio context |
| `yahoo_ticker` | Override the auto-resolved Yahoo Finance ticker |

**Scheduled jobs:**
- **Every hour** (weekdays 09:00–17:35 Paris): fetch live prices via yfinance, evaluate alert conditions with PRIMARY_MODEL. Fired alerts are queued in Redis and injected into the user's next chat message.
- **Every hour (always)**: check `TradeData/` for a new CSV (mtime-gated); import automatically if a newer file is found.
- **Morning briefing**: portfolio performance summary included as a section when positions are loaded.

**Alert conditions** — PRIMARY_MODEL fires an alert only when at least one of these is true:
- A position's live price crossed its `threshold_high` or `threshold_low`
- Intraday variation > ±3 % on an individual position
- Total daily portfolio loss > 2 %
- A dividend is expected within the next 5 calendar days

Rate limit: the same position cannot trigger a second alert within 4 hours. Queued alerts expire after 24 hours.

#### Ticker Resolution

Jarvis resolves ISIN → Yahoo Finance ticker dynamically at runtime — there is no hardcoded ticker list. When a new CSV is imported, each position goes through a four-step resolution pipeline:

| Step | Method | Notes |
|------|--------|-------|
| 1 | Redis cache (`yahoo_ticker` field) | Instant — skips all lookups if already resolved |
| 2 | `yf.Search(isin)` | Most reliable for standard securities |
| 3 | `yf.Search(name)` | Fallback using the position name from the CSV |
| 4 | LLM fallback (`PRIMARY_MODEL`) | Last resort — asks the model to identify the ticker |

**Log levels during resolution:**
- `INFO` — ticker resolved successfully (indicates which step found it)
- `WARNING TICKER NOT FOUND via yfinance` — steps 2 and 3 both failed, LLM fallback is being attempted
- `ERROR TICKER UNRESOLVABLE` — all four steps failed; position is skipped until you set the ticker manually

**When a price fetch fails** (delisted/invalid ticker), the cached `yahoo_ticker` is automatically deleted from Redis so the next hourly run retries resolution from scratch rather than re-using a known-bad ticker.

**Manual override** (always takes precedence — set once, cached permanently):
```bash
curl -X PUT http://localhost:8000/portfolio/position/KORBEN99/IE0002XZSHO1 \
  -H "Authorization: Bearer KORBEN99" \
  -H "Content-Type: application/json" \
  -d '{"yahoo_ticker": "IWDA.AS"}'
```

### Image Upload & Analysis

Jarvis supports image attachments using a **two-stage pipeline**:

```
Image → VISION_MODEL (describe) → <image_analysis> context block
                                          ↓
                         full pipeline → PRIMARY/REASONING MODEL (analyze with memory + RAG)
```

1. The vision model produces a detailed description of the image (max 700 tokens).
2. That description is injected as a `<image_analysis>` context block, alongside memory, RAG, emails, etc.
3. The main model then answers the user's question with full Jarvis context.

This decouples vision from reasoning — a local `Qwen3-VL-8B` can handle description while the primary model handles analysis. Setting `VISION_MODEL` to the same model as `PRIMARY_MODEL` also works (two calls to the same model).

**If `VISION_MODEL` is not set**, images are silently ignored and only the text is processed.

**Image requires `stream=true`** — vision processing can take 1–2 minutes; the non-streaming path returns HTTP 400.

**iOS keepalive during vision:** `describe_images` runs inside `_sse_stream` (not in `chat()`), so the `StreamingResponse` is returned immediately. An initial `{"think": "📷 Analyse de l'image en cours…"}` event is sent at once, then a `{"think": "…"}` keepalive every 15 seconds while the vision model runs — prevents iOS from closing the connection on a silent socket.

**Image preprocessing (local path):** Before sending to `mlx_vlm`, images are resized to max 1024 px on the longest side. An iPhone photo (~12 MP, 4032×3024) gets tiled into 8–10 tiles at full resolution; after resize it drops to 2–4 tiles → ~3–5× faster VLM inference.

**Autonomous threshold management** — during its reflection cycles, Jarvis can update `threshold_high` / `threshold_low` on any position via the `update_trade_threshold` action in `self.py`.

**Uploading a new CSV via curl:**
```bash
curl -X POST http://localhost:8000/portfolio/upload/KORBEN99 \
  -H "Authorization: Bearer KORBEN99" \
  -F "file=@export-positions-comptables.csv"
```

**Setting alert thresholds:**
```bash
curl -X PUT http://localhost:8000/portfolio/position/KORBEN99/FR0000120578 \
  -H "Authorization: Bearer KORBEN99" \
  -H "Content-Type: application/json" \
  -d '{"threshold_high": "90.00", "threshold_low": "70.00"}'
```

---

## Data Growth & Caps

All storage is bounded. The table below shows what grows, where it's capped, and what happens when the cap is hit.

### Redis

| Key pattern | Type | Cap | Behaviour at cap |
|-------------|------|-----|-----------------|
| `chat:{code}:{session}` | List | `CHAT_MAX_MESSAGES` (100) | Oldest messages trimmed at each write |
| `episodic:{code}:conversations` | Sorted set (score = timestamp) | 1 000 entries | Oldest entries removed at each write |
| `user:{code}:profile` | Hash | None (Redis) — nightly cleanup via `_curative_profile_cleanup()` | LLM identifies and deletes duplicate/obsolete keys |
| `jarvis:self:reflection_log` | Sorted set | 30 entries | Oldest entries trimmed at each write |
| `jarvis:{code}:tomorrow_suggestions` | String | TTL 24 h | Auto-expires — no manual cleanup needed |
| `jarvis:push:pending:{code}` | List | Cooldown 1 push/2 h | No write if cooldown active |

### jarvis-self.json

| Field | Cap | Notes |
|-------|-----|-------|
| `learnings[]` | 100 entries | Trimmed to `[-100:]` after nightly review |
| `self_notes[]` | 50 entries | Trimmed to `[-50:]` after each `update_self_note` action |
| `opinions[]` | 50 entries | Trimmed to `[-50:]` after each `add_self_opinion` call; same-topic opinions are updated in place |
| `growth_log[]` | `GROWTH_LOG_MAX_ENTRIES` (180) | Trimmed to `[-180:]` after nightly review |
| `user_relations{}` | 1 entry/user | Updated in place — no growth |

### Qdrant (`jarvis_memory` collection)

| Memory type | Growth rate | Consolidation | Long-term behaviour |
|-------------|-------------|---------------|---------------------|
| `episodic` | ~7 points/user/day (importance ≥ 0.35) | Monthly (day 1), or on-demand via `consolidate_memory` action: batch of 50, loops until < 5 remain | Stable — cleared each consolidation run |
| `autobiographical` | ~2 points/user/day + consolidation output | Dedup check (cosine ≥ `AUTOBIO_DEDUP_THRESHOLD`) before write; monthly decay pass deletes points below `MEMORY_DECAY_THRESHOLD` | Stable long-term — decay prevents unbounded growth; only consolidation milestones (`importance = 1.0`) are permanent |

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

The LLM then responds naturally ("Je n'ai plus accès à internet en ce moment…") instead of silently returning no results.

---

## Logging

All Jarvis modules share a single logging configuration defined in `helpers.py`.

### Setup

`setup_logging()` is called once at startup (in the FastAPI `lifespan`). It configures the root logger with two rotating file handlers:

| File | Level | Rotation | Purpose |
|------|-------|----------|---------|
| `jarvis-api.log` | INFO+ | 5 MB × 3 backups | Operational — normal production log |
| `jarvis-debug.log` | DEBUG+ | 10 MB × 2 backups | Verbose — detailed trace for debugging |

Both files are written to `/opt/jarvis/logs/` and are accessible directly on the host.

### Journaux de prompts

Deux journaux distincts enregistrent prompt + réponse **brute** du LLM, chacun avec sa
propre bascule :

| File | Gate | Contenu |
|------|------|---------|
| `prompts.log` | `LLM_DEBUG_PROMPTS` | tout le trafic conversationnel |
| `opencode-prompts.log` | `RAW_DEBUG_PROMPTS` | uniquement `/v1/raw` (agents de code) |

Fichiers séparés parce qu'un prompt d'agent de code embarque tout le contexte du dépôt et
noierait le reste ; gates séparées pour pouvoir suivre OpenCode sans réactiver la
journalisation complète.

Rotation : `PROMPT_LOG_MAX_MB` × `PROMPT_LOG_BACKUPS` (20 Mo × 3 par défaut). Elle a été
ajoutée en août 2026 — ces deux journaux écrivaient auparavant en `open(…, "a")` brut, sans
rotation, et `prompts.log` avait atteint 42 Mo.

> **Piège d'analyse.** Ces journaux montrent la sortie LLM **avant** validation pydantic.
> Un champ visible ici n'est pas un champ arrivé en base — vérifier Redis/Qdrant, pas le
> journal de prompts.

### Usage in modules

Every module gets its named logger via `helpers.get_logger`:
```python
from helpers import get_logger
logger = get_logger("jarvis-memory")
```

This replaces the per-module `import logging` + `logging.getLogger(...)` pattern. `config.py` is the only exception (it is imported by `helpers.py` and cannot import from it without a circular dependency).

### Noisy loggers silenced

`setup_logging()` sets `WARNING` level on: `httpx`, `httpcore`, `primp`, `sentence_transformers`, `apscheduler`, `urllib3`, `asyncio`.

---

## Project Structure

```
/opt/jarvis/
├── docker-compose.yml
├── .env
├── jarvis-status.sh
├── JarvisApp/                 # iOS Swift app (Xcode project)
│   ├── JarvisApp.swift        # App entry point, lifecycle, notification wiring
│   ├── JarvisAPI.swift        # API client: streaming chat, history, routing
│   ├── NotificationService.swift  # Push polling (BGAppRefreshTask + foreground timer)
│   ├── ContentView.swift      # Root view
│   ├── ChatView.swift         # Chat UI
│   ├── VoiceView.swift        # Voice mode UI
│   ├── SettingsView.swift     # Server URL, user code, model config
│   ├── AppSettings.swift      # @AppStorage persistent settings
│   ├── SpeechEngine.swift     # WhisperKit STT + AVSpeech TTS
│   ├── WakeWordEngine.swift   # On-device wake word detection
│   └── Models.swift           # Shared data models
├── AGENTS.md              # → DOCS/AGENTS.md (lien) — consignes lues par les agents de code
├── jarvis-core/
│   ├── Dockerfile
│   ├── tests/
│   │   └── test_quality.py    # Suite unitaire + intégration (pytest)
│   ├── src/
│   │   ├── main.py            # Démarrage app, scheduler, routes hors chat
│   │   ├── routes/            # chat.py, proxy.py (/v1/*), memory_routes.py, self_routes.py…
│   │   ├── llm_local.py       # Inférence MLX, cache LRU de prompts, journaux de prompts
│   │   ├── tool_calls.py      # Function calling : format modèle ↔ OpenAI (agents de code)
│   │   ├── pipeline.py        # Assemblage du prompt + post_analysis
│   │   ├── memory.py          # Memory system (Redis + Qdrant)
│   │   ├── self.py            # Proto-self / reflection loop + autocoding actions
│   │   ├── briefing.py        # Morning briefing generation
│   │   ├── google_services.py # Gmail + Calendar
│   │   ├── trading.py         # Boursorama portfolio surveillance
│   │   ├── llm_router.py      # Intent classification (Tier 1)
│   │   ├── analyzer.py        # Conversation analysis (Tier 2b)
│   │   ├── web_search.py      # External search: weather (Open-Meteo), news (DDG), 3-stage deep web
│   │   ├── prompts.py         # All LLM prompt constants + get_prompt() live override loader
│   │   ├── helpers.py         # Shared: LLM clients, logging (setup_logging/get_logger), Redis/Qdrant singletons
│   │   ├── trade_keys.py      # Redis key helpers for the trading module
│   │   └── config.py          # Configuration loader + model helpers
│   └── JarvisData/
│       ├── users_list.json
│       ├── jarvis-self.json
│       ├── reflections/
│       └── prompts/
│           ├── prompt_proposals.json   # Append-only proposal history
│           └── prompt_overrides.json   # Live active overrides (applied without restart)
├── TradeData/             # Boursorama CSV exports (drop here)
├── RAGData/               # Documents to index
│   ├── personal/
│   ├── work/
│   ├── documents/
│   ├── company/
│   └── reflexions/
├── scripts/
│   ├── uploadrag.py             # Indexe RAGData/ dans OpenWebUI (job launchd 23:10)
│   ├── jarvis-launchd.sh        # Gestion du service launchd — idempotente
│   ├── search-qdrant.py         # Test RAG search
│   ├── download_models.py       # Récupère modèles + template de chat
│   └── backup-jarvis.sh         # Sauvegarde
├── logs/
│   ├── jarvis-api.log / jarvis-debug.log
│   ├── prompts.log              # Prompts LLM (conversationnel)
│   └── opencode-prompts.log     # Prompts LLM (agents de code, /v1/raw)
└── DOCS/
    ├── AGENTS.md                # Consignes projet pour agents de code
    ├── opencode-local.md        # Brancher OpenCode sur le LLM local
    ├── opencode.json.example    # Config provider OpenCode (à copier)
    ├── jarvis_cheatsheet.md
    └── examples/
        ├── com.jarvis.api.plist.template
        ├── jarvis-aliases.sh    # Alias jarvis-start/stop/restart/status
        └── users_list.example.json
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
# Edit the file directly and remove the key, then the next get_prompt() call uses the default
docker exec jarvis-api python3 -c "
import json
with open('/app/data/prompts/prompt_overrides.json') as f: d = json.load(f)
del d['SYSTEM_BASE_FR']
with open('/app/data/prompts/prompt_overrides.json', 'w') as f: json.dump(d, f, indent=2)
print('reverted')
"
```

No restart needed — `get_prompt()` detects the file change on the next call.

---

## iOS App (JarvisApp)

A native Swift/SwiftUI app for iPhone that connects to the Jarvis API. Voice and notifications are handled entirely on-device; the app sends and receives text only.

### Feature history (chronological)

1. **Text chat** — streaming SSE chat with the Jarvis API, session history, clear conversation
2. **Voice mode** — WhisperKit on-device STT + AVSpeech TTS; concise responses optimised for listening
3. **Image attachments** — send a photo from camera or library; JPEG compressed and sent as base64 to the VISION_MODEL pipeline. `describe_images` runs deferred inside `_sse_stream` with periodic keepalives (every 15 s) to prevent iOS timeout during the 1–2 min VLM inference.
4. **Wake word** — always-on keyword detection (on-device, no cloud) to trigger voice mode hands-free
5. **Proactive push notifications** — BGAppRefreshTask polls `/device/pending/{user_code}` every ~2h in background (15 min in foreground); messages generated by the Jarvis reflection cycle are delivered as local notifications

### Xcode setup for push notifications

Two manual steps required in Xcode (not in code):

1. **Signing & Capabilities** → **Background Modes** → tick **Background App Refresh**
2. **Info.plist** → add key `BGTaskSchedulerPermittedIdentifiers` (Array) → item `fr.jarvis.push-poll`

### Network routing

The app probes `localServerURL` first (2 s timeout), then falls back to `vpnServerURL` (Tailscale). The resolved URL is displayed in the status bar and reused for all subsequent calls until a failure triggers a fresh probe.

---

## Security Notes

- Authentication is code-based (`user_code` in request body, or Bearer token for `/v1/`). There are no passwords.
- Google credentials are stored in `.env` only — never commit this file.
- Qdrant and Redis are internal Docker services and are not exposed externally.
- Memory is isolated per user code.

---

## Roadmap

### Speculative decoding (MLX)

`mlx-lm` supporte le speculative decoding natif via `stream_generate(draft_model=..., num_draft_tokens=N)`,
mais cette feature est **incompatible avec Qwen3.6** (modèle hybride Transformer + GatedDeltaNet).

Les couches GatedDeltaNet utilisent un état récurrent (`ArraysCache`) qui ne peut pas être rollbacké
lorsqu'un token draft est rejeté — mlx-lm le refuse explicitement (`cache non-trimmable`).
Testé et confirmé : l'output part en bruit dès les premières rejections.

**En attente** : support du rollback d'état récurrent dans mlx-lm, ou migration vers un modèle primaire
purement attention (Qwen3-30B dense, etc.).


### Procédure Upgrade:

**Raccourci automatisé** : `./scripts/backup-jarvis.sh updates` fait le Cas A de bout en bout —
sauvegarde, puis `docker compose pull/up` + `pip install -r requirements.txt --upgrade` **en
place** (jamais de `mv` de venv), le tout sous une **fenêtre de maintenance** (les erreurs et
coupures de l'opération sont tagués « maintenance », pas des incidents). Reste manuel : éditer
`requirements.txt` au préalable, et un bump de l'interpréteur Python (Cas B).

Cas A — bumps de sécurité (routine, en place)

cd /opt/jarvis
# 0. ancre de rollback + sauvegarde (clé USB montée pour le reçu !)
/opt/jarvis/venv/bin/python -m pip freeze > requirements.freeze.$(date +%Y%m%d).txt
./scripts/backup-jarvis.sh

# 1. éditer requirements.txt (bump ciblé), SAUVER, vérifier :
git diff -- requirements.txt        # ou: grep cryptography requirements.txt

# 2. appliquer DANS le venv existant (pas de nouveau venv, pas de mv)
/opt/jarvis/venv/bin/python -m pip install -r requirements.txt --upgrade

# 3. valider (voir plus bas), puis redémarrer Jarvis

Cas B — rebuild complet ou bump de Python (en place, sans mv du neuf)

cd /opt/jarvis
/opt/jarvis/venv/bin/python -m pip freeze > requirements.freeze.$(date +%Y%m%d).txt
./scripts/backup-jarvis.sh

# (si Python) brew upgrade python@3.13   # reste en 3.13, pas 3.14

# on écarte l'ANCIEN (rollback), on crée le neuf DIRECTEMENT au bon chemin
mv /opt/jarvis/venv /opt/jarvis/venv-old
/opt/homebrew/bin/python3.13 -m venv /opt/jarvis/venv      # ← créé au chemin final, jamais déplacé
/opt/jarvis/venv/bin/python -m pip install -U pip
/opt/jarvis/venv/bin/python -m pip install -r requirements.txt

# rollback si besoin : rm -rf /opt/jarvis/venv && mv /opt/jarvis/venv-old /opt/jarvis/venv
La différence clé avec aujourd'hui : on déplace l'ancien (qui, en rollback, retourne à son chemin d'origine → intact), et on crée le neuf directement à /opt/jarvis/venv. On ne déplace jamais le neuf.

Validation (avant tout redémarrage)

/opt/jarvis/venv/bin/pip --version                              # scripts OK
/opt/jarvis/venv/bin/python -c "import mlx, torch, transformers, cryptography; print('core OK')"
/opt/jarvis/venv/bin/python -c "import sys;sys.path.insert(0,'/opt/jarvis/jarvis-core/src');import cve;r=cve.scan();print('venv scanné:', 'venv' in r['par_source'])"
# puis suite de tests, et un chargement modèle + génération

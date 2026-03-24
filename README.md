# Jarvis v8 — On-Premise Personal AI Assistant

Jarvis is a self-hosted, multi-user AI assistant with persistent memory, autonomous reflection, and integration with Gmail, Google Calendar, web search, and a document knowledge base.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      -Open WebUI (port 3000)               │
│                     - Chat interface / clients             │
└────────────────────────┬────────────────────────────────┘
                         │ OpenAI-compatible API
┌────────────────────────▼────────────────────────────────┐
│                   Jarvis API (port 8000)                 │
│                   FastAPI / Python 3.11                  │
│                                                          │
│  ┌──────────────┐  ┌───────────┐  ┌──────────────────┐  │
│  │  LLM Router  │  │  Memory   │  │  Proto-Self /    │  │
│  │  (Tier 1)    │  │  System   │  │  Reflection Loop │  │
│  └──────────────┘  └───────────┘  └──────────────────┘  │
│  ┌──────────────┐  ┌───────────┐  ┌──────────────────┐  │
│  │  RAG Engine  │  │ Briefing  │  │ Google Services  │  │
│  │  (Qdrant)    │  │ Scheduler │  │ (Gmail/Calendar) │  │
│  └──────────────┘  └───────────┘  └──────────────────┘  │
│  ┌──────────────────────────────────────────────────┐    │
│  │  Trading Surveillance (yfinance + Redis)         │    │
│  │  CSV import · hourly prices · AI alerts          │    │
│  └──────────────────────────────────────────────────┘    │
└────────┬────────────────┬────────────────────────────────┘
         │                │
┌────────▼──────┐  ┌──────▼──────────┐
│  Qdrant       │  │  Redis          │
│  (port 6333)  │  │  (port 6379)    │
│  Vector DB    │  │  Session cache  │
│  RAG + Memory │  │  Working mem.   │
└───────────────┘  └─────────────────┘
```
## MEMORY STRUCTURE
  ┌──────────────────────────────────┬───────────┬────────────────────────────────────────────────────────────────────────┐
  │              Store               │ Lives in  │                          Should contain                                │
  ├──────────────────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────┤
  │ jarvis-self.json                 │ self.py   │ Jarvis's identity, goals, focus, self-notes, reflection log,           │
  │                                  │           │ per-user relations (affinity, interaction style, tonal preference)     │
  ├──────────────────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────┤
  │ Redis hashes user:profile:{code} │ memory.py │ User facts, preferences, interests                                     │
  ├──────────────────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────┤
  │ Qdrant episodic                  │ memory.py │ Per-user conversation summaries                                        │
  ├──────────────────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────┤
  │ Qdrant autobiographical          │ memory.py │ Long-term facts about the user                                         │
  └──────────────────────────────────┴───────────┴────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────┬──────────────────────────────────────┬─────────────────────────────────────────────────────────┐
  │                      Data                      │             Destination              │                           Key                           │
  ├────────────────────────────────────────────────┼──────────────────────────────────────┼─────────────────────────────────────────────────────────┤
  │ Facts about the user (user_insights)           │ store_autobiographical_event() →     │ memory_type: autobiographical                           │
  │                                                │ Qdrant                               │                                                         │
  ├────────────────────────────────────────────────┼──────────────────────────────────────┼─────────────────────────────────────────────────────────┤
  │ Insights from store_insight action             │ store_autobiographical_event() →     │ same                                                    │
  │                                                │ Qdrant                               │                                                         │
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
  ┌────────────────────┬────────────────────────────────────────────────────────────────────┬───────────────────────────────┐   
  │       Modèle       │                                Rôle                                │           no_think            │   
  ├────────────────────┼────────────────────────────────────────────────────────────────────┼───────────────────────────────┤   
  │ Qwen7B (router)    │ routing, dédup profil, judge web                                   │ True                          │   
  ├────────────────────┼────────────────────────────────────────────────────────────────────┼───────────────────────────────┤   
  │ Qwen30B (primary)  │ briefing, analyse conv, refine prompt, réflexion, nightly review,  │ False sur les tâches          │   
  │                    │ trading                                                            │ analytiques                   │   
  ├────────────────────┼────────────────────────────────────────────────────────────────────┼───────────────────────────────┤   
  │ GPT 5.2            │ mode expert uniquement (use_reasoning=True via router)             │ —                             │   
  │ (reasoning)        │                                                                    │                               │   
  └────────────────────┴────────────────────────────────────────────────────────────────────┴───────────────────────────────┘



### Core Components

| Component | Technology | Role |
|-----------|-----------|------|
| **Jarvis API** | FastAPI, Python 3.11 | Main orchestration — bootstrap only (261 lines) |
| **Open WebUI** | Docker, port 3000 | Chat interface, connects via `/v1/chat/completions` |
| **Qdrant** | Docker, port 6333 | Vector DB for RAG document search and episodic memory |
| **Redis** | Docker, port 6379 | Working memory, session context, conversation cache |
| **`deps.py`** | Python module | Shared runtime singletons: Redis, Qdrant, embed model, HTTP clients, context budgets |
| **`llm_client.py`** | Python module | LLM HTTP client: streaming SSE, model tier selection, vision pipeline |
| **`rag.py`** | Python module | Qdrant document retrieval (embed query → search → score filter) |
| **`pipeline.py`** | Python module | System prompt construction, 7-source context assembly, post-exchange analysis |
| **`routes/chat.py`** | Python module | Main chat pipeline: routing → context gather → LLM → SSE stream |
| **`routes/proxy.py`** | Python module | OpenAI-compatible proxy `/v1/*` for Open WebUI and iOS |
| **`prompts.py`** | Python module | Single source of truth for all LLM prompts — supports live overrides via `get_prompt()` |
| **`web_search.py`** | Python module | All external search backends: Open-Meteo weather, DDG news, 3-stage deep text pipeline |
| **`helpers.py`** | Python module | Shared utilities: LLM HTTP clients, logging setup, Redis/Qdrant factory, JSON parsing |

### Four-Tier LLM Architecture

Jarvis routes every request through a layered model stack. Each tier has its own API endpoint so you can run some tiers locally (Qwen via mlx-lm) and others on the cloud.

```
Tier 1 — ROUTER      Fast intent classifier, JSON only
         Default: gpt-4.1-nano  →  future: Qwen3-7B (local)

Tier 2 — PRIMARY     All standard responses: chat, questions, summaries,
         Default: gpt-4o-mini      trading alerts, self-reflection, briefing
                              →  future: Qwen3-30B-A3B (local)

Tier 2b — ANALYSIS   Post-exchange conversation analysis (fact/mood extraction)
          Default: same as PRIMARY  →  future: Qwen3-30B-A3B (local)

Tier 3 — REASONING   Complex queries only: medical/legal/regulatory analysis,
         Default: gpt-5.1          hard multi-step logic, deep scientific reasoning
                              →  stays cloud (OpenAI / Anthropic)
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

**Qwen migration readiness:**
All tiers have independent `API_URL` / `API_KEY` variables, so you can move each tier to a local mlx-lm server independently:
```
ROUTER_API_URL=http://mac-mini.local:8080/v1   # point router at local Qwen3-7B
PRIMARY_API_URL=http://mac-mini.local:8080/v1  # point primary at local Qwen3-30B-A3B
```
`no_think_suffix()` in `config.py` appends `/no_think` to system prompts for Qwen3 models to suppress `<think>` blocks, which would break JSON parsing.

#### Router Output Fields

The router returns a structured JSON decision consumed by the chat pipeline:

| Field | Values | Purpose |
|-------|--------|---------|
| `intents` | `memory`, `rag`, `web`, `gmail`, `calendar`, `briefing`, `self`, `portfolio` | Which data sources to activate |
| `gmail_query` | Gmail search string or null | Pre-built query passed directly to Gmail API |
| `calendar_days` | integer (1–90) or null | Days ahead to fetch from Calendar |
| `use_reasoning` | boolean | Route to Tier-3 reasoning model when true |
| `memory_scope` | `episodic`, `autobiographical`, `profile`, `auto` | Which Qdrant memory layer to search |
| `conversation_type` | `conversational`, `task`, `question` | Message classification — RAG is skipped for `conversational` |

**`memory_scope` behaviour:**
- `episodic` — recent conversation summaries only (past sessions, events)
- `autobiographical` — long-term milestones and stable facts only
- `profile` — static preferences already injected via Redis; Qdrant search skipped entirely
- `auto` — search both episodic and autobiographical (default)

**`conversation_type` behaviour:**
- `conversational` — greetings, thanks, chitchat: RAG document search is bypassed, faster response
- `task` / `question` — full retrieval pipeline runs normally

If the LLM router is unavailable or fails (timeout / parse error), all `use_*` flags default to `False` — no context is fetched and the LLM answers from the system prompt alone. This is the safe fallback; the embedding-based semantic router has been removed.

### Five-Layer Memory System

| Layer | Backend | Contents |
|-------|---------|----------|
| Working Memory | Redis | Active session context, current mood |
| Semantic Memory | Redis Hashes | User profiles, preferences, learned facts |
| Episodic Memory | Qdrant | Conversation summaries that passed the importance threshold |
| Autobiographical Memory | Qdrant | High-importance milestones consolidated from episodic memory |
| Self Memory | JSON file | Jarvis identity, goals, focus, reflection log, per-user relations |

#### Episodic Salience Score (ESS)

After each exchange, `analyzer.py` computes an importance score in `[0, 1]` that gates what gets written to long-term memory:

| Signal | Weight | Notes |
|--------|--------|-------|
| LLM flags `should_remember` | +0.40 | Primary signal — alone clears the storage threshold |
| User fact revealed (max 3) | +0.20 each | Profile facts, preferences, life events |
| Project / goal mentioned (max 2) | +0.15 each | Active work context |
| Strong emotional mood | +0.10–0.15 | Stressed/frustrated weighted slightly higher |
| Long message (> 80 chars) | +0.05 | Minor depth signal |

Storage thresholds (set in `config.py`):
- **`IMPORTANCE_THRESHOLD` (0.35)** — stored as episodic vector in Qdrant
- **`AUTOBIO_IMPORTANCE_THRESHOLD` (0.60)** — additionally stored as autobiographical event

#### Importance Score Reference

Every autobiographical point stored in Qdrant carries an `importance` field. This value determines both retrieval ranking and long-term decay behaviour. Here is the complete list of values assigned across the codebase:

| Score | Source | Decay behaviour |
|-------|--------|-----------------|
| `1.0` (`MEMORY_CONSOLIDATION_IMPORTANCE`) | Monthly consolidation (`_consolidate_user_memories`) — LLM summary of a batch of episodic memories | **Permanent — exempt from decay** (`== MEMORY_DECAY_DURABLE_MIN`) |
| `0.60–1.0` (clamped) | Analyzer ESS score, via `complete_memory_to_qdrant()` — only stored if score `> AUTOBIO_IMPORTANCE_THRESHOLD` | Decays monthly; exempt only if score reaches `1.0` (requires LLM remember + 3 facts + emotion + depth simultaneously) |
| `0.80` | Jarvis self-reflection insight (`run_self_reflection`) | Decays monthly |
| `0.70` | Nightly review user insight (`run_nightly_interaction_review`) | Decays monthly |

**Key invariant:** `MEMORY_CONSOLIDATION_IMPORTANCE` must equal `MEMORY_DECAY_DURABLE_MIN`. If you change one, change the other. Breaking this invariant would either make consolidation milestones decay (if `CONSOLIDATION_IMPORTANCE < DURABLE_MIN`) or promote ordinary memories to permanent status (if `DURABLE_MIN` is lowered).

#### Profile Key Deduplication

Profile facts are stored as Redis hash fields with namespaced keys (`hobby:kart`, `skill:python`, `location`). A three-stage pipeline prevents duplicates:

| Stage | Method | Cost |
|-------|--------|------|
| 1. Source prevention | Existing profile keys injected into `ANALYSIS_PROMPT` — LLM reuses exact key names instead of inventing new ones | Prompt tokens only |
| 2. Canonical alias | `_SCALAR_CANONICAL` dict maps common synonyms (`ville→location`, `entreprise→current_employer`) without any LLM call | O(1) |
| 3. Category-aware LLM | Router model compares only against keys in the same namespace family (`hobby:*` vs `interest:*`), not all 30+ keys | 1 fast LLM call on a small set |

Stage 1 prevents ~90 % of duplicates at the source. Stages 2–3 are safety nets.

#### Project Tracking

Projects are stored as JSON objects with `name`, `status` (`in_progress` / `done`), `first_mentioned`, and `last_update`. The `apply_project_updates()` function resolves project names using word-overlap fuzzy matching (≥ 60 % threshold) before exact-string lookup, preventing name-drift duplicates (`"Jarvis"` → `"Jarvis v7"`).

#### Memory Retrieval Ranking

`search_memory()` re-ranks Qdrant results using a weighted blend before returning:

```
final_score = semantic_similarity × 0.65 + importance × 0.25 + recency_bonus × 0.10
```

The recency window is **type-aware**: episodic memories use a 30-day window, autobiographical memories use a 365-day window (`AUTOBIO_RECENCY_WINDOW_DAYS`). Without this distinction, a stable milestone from 6 months ago (e.g. "Sébastien gave a talk at Insomnihack") would always score `recency_bonus = 0` and be outranked by trivial recent events.

`build_memory_context()` surfaces the **top 5 autobiographical events by importance + recency** (importance weight 0.7, recency 0.3 over a 1-year window) rather than the 5 most recent — so a critical event from months ago is not displaced by routine recent exchanges.

#### Autobiographical Memory Deduplication

Before any call to `store_autobiographical_event()`, Jarvis queries Qdrant for the most similar existing autobiographical point. If the cosine similarity exceeds `AUTOBIO_DEDUP_THRESHOLD` (default: 0.85), the new entry is silently skipped. This prevents the collection from accumulating dozens of near-identical variants of the same fact (e.g. "Sébastien s'intéresse à la bourse" stored 15 times with slightly different wording). The threshold is tunable: raise toward 0.95 to allow more variations, lower toward 0.75 to be stricter.

#### Autobiographical Memory Decay

On the 1st of each month, `consolidate_memories()` runs a decay pass on every autobiographical point via `_decay_autobiographical_memories()`:

1. **Exempt** — points with `importance >= MEMORY_DECAY_DURABLE_MIN` (default `1.0`) are never touched. In practice this means only monthly consolidation milestones.
2. **Decay** — for all other points: `decayed_importance = importance × MEMORY_DECAY_FACTOR ^ age_months`. With the default factor of `0.85`, each point loses ~15 % of its importance per elapsed month.
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

Every request builds the system prompt from three blocks assembled in `build_system_prompt()` (called via `asyncio.to_thread` to avoid blocking the FastAPI event loop during Qdrant I/O):

```
SYSTEM_BASE_FR          (~1 400 chars — Jarvis personality, tool rules)
    ↓
MEMORY_HEADER_FR        (~90 chars — section separator)
    ↓
build_memory_context()  — injected only if memory is available
    ↓
VOICE_SUFFIX_FR         (~80 chars — only if voice_mode=True)
```

**`build_memory_context()` — sections injected in order:**

| Section | Source | Always injected? |
|---------|--------|-----------------|
| `PROFIL UTILISATEUR` | Redis hash `user:{code}:profile` — grouped by namespace | Only if profile exists |
| `PRÉFÉRENCES` | Redis hash `user:{code}:preferences` | Only if prefs exist |
| `PROJETS ACTIFS` | Redis `user:{code}:projects` — status `in_progress` only | Only if projects exist |
| `SUJETS RÉCENTS (24h)` | Topics from last 10 conversations in Redis | Only if topics exist |
| `ÉTAT ÉMOTIONNEL` | Redis `jarvis:emotional_state` | Only if mood ≠ neutral |
| `APPRENTISSAGES RÉCENTS` | `jarvis-self.json → learnings[-5:]` | Only if learnings exist |
| `FRISE CHRONOLOGIQUE` | Top 5 autobio Qdrant points by importance+recency | Only if autobio exists |
| `SUJETS À ABORDER AUJOURD'HUI` | Redis `jarvis:{code}:tomorrow_suggestions` (TTL 24h, written by nightly review) | Only if key exists |
| `RELATION AVEC CET UTILISATEUR` | `jarvis-self.json → user_relations[user_code]` | Always (with defaults) |

The `RELATION` section is always present so the LLM always has a tonal directive. Default values: affinity 0.5 / style direct / mood measured.

**Context budgets** (applied to dynamic blocks fetched during the request, not to the system prompt itself):

| Block | Budget | Per-item cap |
|-------|--------|-------------|
| Mémoire vectorielle (`search_memory`) | 2 500 chars | — |
| RAG documents | 4 000 chars | 800 chars/chunk |
| Recherche web | 2 000 chars | — |
| Gmail + Calendar | 3 000 chars combined | — |
| **Total** | **10 000 chars hard ceiling** | — |

### Request Flow

```
User message
    → asyncio.to_thread(build_system_prompt)      ← non-blocking Qdrant I/O
    → Tier 1 Router (intents + memory_scope + conversation_type + use_reasoning)
    → if conversation_type=conversational: skip RAG
    → Parallel context gathering (memory[scope] + RAG + web + Google + self + portfolio)
    → Pending trade alerts injected if any are queued
    → Self context injected: internal state (focus, goals, last action)
                           + per-user relation (affinity, style, tonal directives)
    → Tier 2 PRIMARY or Tier 3 REASONING (full context → streaming response)
    →   streaming via shared per-timeout httpx.AsyncClient (connection pool reused)
    → Conversation analyzer / ANALYSIS_MODEL (extract facts, mood, topics, ESS)
    → Memory storage: ESS > 0.35 → Qdrant episodic | ESS > 0.60 → autobiographical
    →   store_autobiographical_event: dedup check first (cosine ≥ 0.85 → skip)
```

---

## Deploy

### Prerequisites

- Docker & Docker Compose
- OpenAI API key (or compatible endpoint)
- Google OAuth credentials (for Gmail / Calendar)

### 1. Clone and configure

```bash
git clone <repo> /opt/jarvis
cd /opt/jarvis
cp .env.example .env   # then edit .env (see Variables section below)
```

### 2. Start all services

```bash
docker compose up -d
```

Services started:
- `qdrant` — vector database
- `redis` — session cache
- `open-webui` — chat interface on port 3000
- `jarvis-api` — main API on port 8000

### 3. Verify

```bash
./jarvis-status.sh
# or
curl http://localhost:8000/status
```

### 4. Index documents (optional)

Place documents in `RAGData/` subdirectories (`personal/`, `work/`, `documents/`, `company/`, `reflexions/`), then run:

```bash
python3 scripts/upload-to-openwebui.py
```

### 5. Import trading portfolio (optional)

Export your Boursorama positions as CSV (*Mes comptes → Exporter*) and drop the file in `TradeData/`. Jarvis imports it automatically on the next hourly tick, or immediately on restart.

### Common Commands

```bash
# Restart API after code change
docker compose restart jarvis-api

# Rebuild container after Dockerfile change
docker compose up -d --build jarvis-api

# Stream logs
docker compose logs -f jarvis-api

# Stop everything
docker compose down
```

---

## Configuration Variables

All variables go in `/opt/jarvis/.env`.

### Shared OpenAI credentials

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Shared API key (fallback for any tier not explicitly configured) |
| `OPENAI_API_URL` | `https://api.openai.com/v1` | Shared base URL (fallback) |

### Tier 1 — Router model

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTER_MODEL` | `gpt-4.1-nano` | Intent classifier. Set to empty to disable and use embedding router. |
| `ROUTER_API_URL` | *(OPENAI_API_URL)* | Override to point at a local Qwen3-7B via mlx-lm |
| `ROUTER_API_KEY` | *(OPENAI_API_KEY)* | API key for the router (mlx-lm ignores auth but requires a value) |
| `ROUTER_TIMEOUT` | `6` | Timeout in seconds (short — fast model only) |

### Tier 2 — Primary model

| Variable | Default | Description |
|----------|---------|-------------|
| `PRIMARY_MODEL` | `gpt-4o-mini` | Handles all standard responses: chat, trading, briefing, self-reflection |
| `PRIMARY_API_URL` | *(OPENAI_API_URL)* | Override to point at a local Qwen3-30B-A3B |
| `PRIMARY_API_KEY` | *(OPENAI_API_KEY)* | API key for the primary model |
| `PRIMARY_TIMEOUT` | `60` | Timeout in seconds |

### Tier 2b — Analysis model

| Variable | Default | Description |
|----------|---------|-------------|
| `ANALYSIS_MODEL` | *(PRIMARY_MODEL)* | Post-exchange fact/mood extraction. Defaults to PRIMARY if unset. |
| `ANALYSIS_API_URL` | *(PRIMARY_API_URL)* | Override independently if needed |
| `ANALYSIS_API_KEY` | *(PRIMARY_API_KEY)* | API key for the analysis model |

### Tier 3 — Reasoning model

| Variable | Default | Description |
|----------|---------|-------------|
| `REASONING_MODEL` | *(PRIMARY_MODEL)* | Complex queries only — used when router sets `use_reasoning=true` |
| `REASONING_API_URL` | *(OPENAI_API_URL)* | Typically cloud (OpenAI / Anthropic) |
| `REASONING_API_KEY` | *(OPENAI_API_KEY)* | API key for the reasoning model |
| `REASONING_TIMEOUT` | `90` | Timeout in seconds (longer — deep reasoning) |

### Vision model

| Variable | Default | Description |
|----------|---------|-------------|
| `VISION_MODEL` | *(empty — disabled)* | First-stage image description (e.g. `gpt-4o-mini`, `qwen2.5-vl-72b`). Leave empty to ignore images. |
| `VISION_API_URL` | *(OPENAI_API_URL)* | Override for a local Qwen2.5-VL |
| `VISION_API_KEY` | *(OPENAI_API_KEY)* | API key for the vision model |
| `VISION_TIMEOUT` | `30` | Timeout in seconds |

### Infrastructure

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant URL — hardcoded to Docker-internal network in compose |
| `REDIS_URL` | `redis://redis:6379` | Redis URL — hardcoded to Docker-internal network in compose |
| `QDRANT_COLLECTION` | `open-webui_knowledge` | Collection for RAG documents |
| `QDRANT_MEMORY_COLLECTION` | `jarvis_memory` | Collection for episodic memory |
| `HF_TOKEN` | — | HuggingFace token (required to download the multilingual embedding model on first start) |

### RAG

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_TOP_K` | `5` | Number of documents to retrieve |
| `RAG_SCORE_THRESHOLD` | `0.4` | Minimum similarity score (0–1) |

### Google Services

| Variable | Description |
|----------|-------------|
| `GOOGLE_CLIENT_ID` | OAuth 2.0 client ID |
| `GOOGLE_CLIENT_SECRET` | OAuth 2.0 client secret |
| `GOOGLE_REFRESH_TOKEN` | Refresh token for persistent access |
| `GOOGLE_CALENDAR_ID` | Calendar to read (e.g. `primary`) |

### Scheduling & Features

| Variable | Default | Description |
|----------|---------|-------------|
| `BRIEFING_ENABLED` | `true` | Enable daily morning briefing |
| `BRIEFING_TIME` | `07:30` | Briefing delivery time (HH:MM) |
| `BRIEFING_TIMEZONE` | `Europe/Paris` | Timezone for scheduling |
| `REFLECTION_INTERVAL_HOURS` | `2` | Hours between self-reflection cycles |
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

Only users with `"trading": true` participate in scheduled trade checks (CSV import, price fetch, alert evaluation). Users without this flag are never included, regardless of whether a CSV exists in `TradeData/`.

---

## API Endpoints

Base URL: `http://localhost:8000`

### Chat

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Main chat endpoint (SSE streaming) |
| `POST` | `/v1/chat/completions` | OpenAI-compatible endpoint (for Open WebUI) |
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

**Reflection loop** (every 2 h) — global self-observation. Jarvis reviews system health, user activity, and knowledge gaps, then picks one action from the catalog. At the end of each cycle Jarvis also runs a per-user **proactive push** check. Outcome and new focus are persisted to `jarvis-self.json`.

**Nightly review** (23:00) — per-user conversation review. Conversations from the day are sorted by importance score descending before being passed to the LLM (up to 6 000 chars), so the most significant exchanges are always visible even on high-volume days. For each user Jarvis extracts durable user facts (→ Qdrant autobiographical, dedup-checked), self-improvement notes (→ `learnings[]`), updates the **user relation**, and writes `tomorrow_suggestions` to Redis (TTL 24 h) for injection in the next day's system prompt.

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
| `emotional_state` | Redis `jarvis:emotional_state` | Current mood (neutral / curious / concerned…) |
| `self_notes[-5:]` | `jarvis-self.json` | Last 5 personal observations written by `update_self_note` |
| `opinions[-5:]` | `jarvis-self.json` | Last 5 topic opinions written by `add_self_opinion` |
| `user_relations` | `jarvis-self.json` | Affinity + style per user |
| `user_profiles` | Redis hash per user | Capped at 20 keys/user for token budget |
| `push_availability` | Redis `jarvis:device:token:{code}` | Real-time per-user iOS push status — prevents wasting cycles on users with no registered device |

**`behavioral_patterns`** is computed deterministically (no LLM) from the reflection log: action frequency (≥ 20 % of cycles), time-of-day clustering for "nothing" choices (night/evening pattern), and recurring keywords in past focus fields (seen ≥ 3 times). Up to 5 bullet points.

**Reflection action catalog** — actions the LLM can choose during each reflection cycle:

| Action | Description |
|--------|-------------|
| `nothing` | Explicit no-op with reason |
| `store_insight` | Save a durable fact about a user to Qdrant autobiographical |
| `flag_knowledge_gap` | Log a topic Jarvis answered poorly (increments a per-topic counter). Requires a concrete observed failure as context — generic phrases are rejected. 7-day cooldown per topic. Blocked if a proposal already exists for the topic. |
| `send_notification` | Send a Gmail to one user (rate-limited to 1/user/day) |
| `queue_push` | Queue an iOS push notification for one user (rate-limited to 1/user/2h) |
| `ask_user` | Send a clarification question via push; user answers in chat, memory updates |
| `update_self_note` | Write a personal observation to `self_notes[]` in `jarvis-self.json` |
| `correct_profile` | Delete or correct a Redis profile key (value=null to delete) |
| `consolidate_memory` | Trigger full memory compression for a user (loops until all episodic cleared) |
| `check_health` | Verify all services and log status |
| `update_trade_threshold` | Update `threshold_high` / `threshold_low` for a portfolio position autonomously |
| `refine_prompt` | Propose an improved version of a prompt (see Prompt Self-Modification below) |

**Memory consolidation** — `consolidate_memories()` is the single entry point. It runs on the 1st of each month (nightly review scheduler) and on demand via the `consolidate_memory` self-action. It executes three steps in order for each user:
1. `_consolidate_user_memories()` — processes episodic points in batches of 50 (oldest first), summarises each batch into one autobiographical milestone via LLM (stored at `importance = MEMORY_CONSOLIDATION_IMPORTANCE = 1.0`), deletes the processed points, loops until fewer than 5 remain.
2. `_decay_autobiographical_memories()` — decays and prunes autobiographical points (see Autobiographical Memory Decay section above).
3. `_curative_profile_cleanup()` — asks the LLM to identify and delete duplicate or obsolete keys from the Redis user profile hash.

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
=== ÉTAT INTERNE ===
Focus : ...
Objectifs : ...
Dernière action autonome : ...

=== RELATION AVEC CET UTILISATEUR ===
Affinité : 0.8/1.0 → Tu apprécies cet utilisateur, investis-toi pleinement.
Style : direct → Réponds sans détours, va droit au but.
Tonalité Jarvis : warm → Adopte un ton chaleureux et bienveillant.
```

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
Image → VISION_MODEL (describe) → "=== IMAGE ANALYSÉE ===" context block
                                          ↓
                         full pipeline → PRIMARY/REASONING MODEL (analyze with memory + RAG)
```

1. The vision model produces a detailed description of the image (max 600 tokens).
2. That description is injected as a context block, alongside memory, RAG, emails, etc.
3. The main model then answers the user's question with full Jarvis context.

This decouples vision from reasoning — a local `Qwen2.5-VL-72B` can handle description while `gpt-5.1` handles analysis. Setting `VISION_MODEL` to the same model as `PRIMARY_MODEL` also works (two calls to the same model).

**If `VISION_MODEL` is not set**, images are silently ignored and only the text is processed.

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

`setup_logging()` is called once at startup (in the FastAPI `lifespan`). It configures the root logger with:
- **Console handler** — INFO level, same format as before
- **Rotating file handler** — 5 MB × 3 backups, written to `/app/logs/jarvis-api.log`

The log directory is bind-mounted from the host (`./logs:/app/logs`), so logs are immediately accessible at `/opt/jarvis/logs/jarvis-api.log` without entering the container.

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
├── jarvis-core/
│   ├── Dockerfile
│   ├── main.py            # API routes + request orchestration
│   ├── memory.py          # Memory system (Redis + Qdrant)
│   ├── self.py            # Proto-self / reflection loop + autocoding actions
│   ├── briefing.py        # Morning briefing generation
│   ├── google_services.py # Gmail + Calendar
│   ├── trading.py         # Boursorama portfolio surveillance
│   ├── llm_router.py      # Intent classification (Tier 1)
│   ├── analyzer.py        # Conversation analysis (Tier 2b)
│   ├── web_search.py      # External search: weather (Open-Meteo), news (DDG), 3-stage deep web
│   ├── prompts.py         # All LLM prompt constants + get_prompt() live override loader
│   ├── helpers.py         # Shared: LLM clients, logging (setup_logging/get_logger), Redis/Qdrant singletons
│   ├── trade_keys.py      # Redis key helpers for the trading module
│   ├── config.py          # Configuration loader + model helpers
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
│   ├── upload-to-openwebui.py   # Index documents into Qdrant
│   └── search-qdrant.py         # Test RAG search
├── logs/
└── Jarvis project config/
    ├── jarvis-system-prompt.md
    └── jarvis_cheatsheet.md
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
| `NIGHTLY_PROMPT` | Per-user conversation review instructions |
| `NIGHTLY_SYSTEM` | System context for nightly review |
| `REFLECTION_PROMPT` | Autonomous reflection action selection |
| `REFLECTION_SYSTEM` | System context for reflection cycle |

The first four are user-facing quality prompts. The last four are internal autonomy prompts — they can also be refined when the reflection loop identifies recurring self-improvement opportunities.

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
3. **Image attachments** — send a photo from camera or library; JPEG compressed and sent as base64 to the VISION_MODEL pipeline
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

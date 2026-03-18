# Jarvis v7 — On-Premise Personal AI Assistant

Jarvis is a self-hosted, multi-user AI assistant with persistent memory, autonomous reflection, and integration with Gmail, Google Calendar, web search, and a document knowledge base.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Open WebUI (port 3000)               │
│                     Chat interface / clients             │
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



### Core Components

| Component | Technology | Role |
|-----------|-----------|------|
| **Jarvis API** | FastAPI, Python 3.11 | Main orchestration, routing, response generation |
| **Open WebUI** | Docker, port 3000 | Chat interface, connects via `/v1/chat/completions` |
| **Qdrant** | Docker, port 6333 | Vector DB for RAG document search and episodic memory |
| **Redis** | Docker, port 6379 | Working memory, session context, conversation cache |
| **`prompts.py`** | Python module | Single source of truth for all LLM prompts — supports live overrides via `get_prompt()` |

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

If the LLM router is unavailable or fails, the embedding-based semantic router takes over with `memory_scope=auto` and `conversation_type` defaulting to full retrieval — no degradation in correctness.

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

#### Memory Retrieval Ranking

`search_memory()` re-ranks Qdrant results using a weighted blend before returning:

```
final_score = semantic_similarity × 0.65 + importance × 0.25 + recency_bonus × 0.10
```

The context injected into each prompt (`build_memory_context`) surfaces the **top 5 autobiographical events by importance + recency** (importance weight 0.7, recency 0.3 over a 1-year window) rather than the 5 most recent — so a critical event from months ago is not displaced by routine recent exchanges.

### Request Flow

```
User message
    → Tier 1 Router (intents + memory_scope + conversation_type + use_reasoning)
    → if conversation_type=conversational: skip RAG
    → Parallel context gathering (memory[scope] + RAG + web + Google + self + portfolio)
    → Pending trade alerts injected if any are queued
    → Self context injected: internal state (focus, goals, last action)
                           + per-user relation (affinity, style, tonal directives)
    → Tier 2 PRIMARY or Tier 3 REASONING (full context → streaming response)
    → Conversation analyzer / ANALYSIS_MODEL (extract facts, mood, topics, ESS)
    → Memory storage: ESS > 0.35 → Qdrant episodic | ESS > 0.60 → autobiographical
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

**Nightly review** (23:00) — per-user conversation review. For each user who had conversations that day, Jarvis extracts durable user facts (→ Qdrant autobiographical), self-improvement notes (→ `learnings[]`), and updates the **user relation** for that user.

**Reflection action catalog** — actions the LLM can choose during each reflection cycle:

| Action | Description |
|--------|-------------|
| `nothing` | Explicit no-op with reason |
| `store_insight` | Save a learning about a user to `jarvis-self.json` |
| `flag_knowledge_gap` | Log a topic Jarvis answered poorly (increments a per-topic counter) |
| `send_notification` | Send a Gmail to one user (rate-limited to 1/user/day) |
| `queue_push` | Queue an iOS push notification for one user (rate-limited to 1/user/2h) |
| `update_self_note` | Write a personal observation to `self_notes` |
| `consolidate_memory` | Trigger memory compression for a user |
| `check_health` | Verify all services and log status |
| `update_trade_threshold` | Update `threshold_high` / `threshold_low` for a portfolio position autonomously |
| `refine_prompt` | Propose an improved version of a prompt (see Prompt Self-Modification below) |

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
| `GET` | `/web?q=...&max_results=3` | Web search only (DuckDuckGo) |

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
│   ├── prompts.py         # All LLM prompt constants + get_prompt() live override loader
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
| Hard threshold check | `_action_refine_prompt()` | Refuses to run if gap count < `REFINE_PROMPT_THRESHOLD` — LLM cannot bypass this |
| Duplicate prevention | `_action_refine_prompt()` | Only one pending proposal per prompt at a time |
| Rejection cooldown | `_action_refine_prompt()` | Same prompt cannot be re-proposed within 7 days of a rejection |
| Human approval | `approve_proposal()` | Override is never written without explicit "accepte la proposition [id]" |

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

All other prompts remain read-only (reflection, nightly review, etc.) — only the four user-facing quality prompts are exposed to the refinement pipeline.

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

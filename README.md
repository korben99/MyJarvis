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

### Core Components

| Component | Technology | Role |
|-----------|-----------|------|
| **Jarvis API** | FastAPI, Python 3.11 | Main orchestration, routing, response generation |
| **Open WebUI** | Docker, port 3000 | Chat interface, connects via `/v1/chat/completions` |
| **Qdrant** | Docker, port 6333 | Vector DB for RAG document search and episodic memory |
| **Redis** | Docker, port 6379 | Working memory, session context, conversation cache |

### Two-Tier LLM Architecture

- **Tier 1 — Router:** Fast intent classifier (`gpt-4.1-nano` by default, future: local Qwen2.5-7B). Determines which tools and context sources to activate, classifies the message type, and hints which memory layer to search.
- **Tier 2 — Reasoning:** Full response generation (`gpt-5.1` by default). Receives aggregated context (memory + RAG + web + Google).

#### Router Output Fields

The router returns a structured JSON decision consumed by the chat pipeline:

| Field | Values | Purpose |
|-------|--------|---------|
| `intents` | `memory`, `rag`, `web`, `gmail`, `calendar`, `briefing`, `self`, `portfolio` | Which data sources to activate |
| `gmail_query` | Gmail search string or null | Pre-built query passed directly to Gmail API |
| `calendar_days` | integer (1–90) or null | Days ahead to fetch from Calendar |
| `use_reasoning` | boolean | Route to Tier-2 reasoning model when true |
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
| Self Memory | JSON file | Jarvis identity, goals, reflection log |

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
    → Tier 2 LLM (full context → streaming response)
    → Conversation analyzer (extract facts, mood, topics, ESS importance score)
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

### LLM / OpenAI

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key (required) |
| `OPENAI_API_URL` | `https://api.openai.com/v1` | API base URL (override for local LLM) |
| `PRIMARY_MODEL` | `gpt-4o-mini` | Fallback model when tier models are unset |
| `ANALYSIS_MODEL` | `gpt-4o-mini` | Conversation analysis |
| `ROUTER_MODEL` | `gpt-4.1-nano` | Intent classification (Tier 1). Set to empty string to disable LLM router and fall back to embedding router. |
| `ROUTER_API_URL` | `https://api.openai.com/v1` | API base URL for the router model (override to point at a local Qwen2.5-7B) |
| `ROUTER_API_KEY` | *(OPENAI_API_KEY)* | API key for the router model (defaults to main key) |
| `ROUTER_TIMEOUT` | `6` | Router call timeout in seconds |
| `REASONING_MODEL` | *(PRIMARY_MODEL)* | Full reasoning model (Tier 2). Set to empty to fall back to PRIMARY_MODEL. |
| `REASONING_API_URL` | *(OPENAI_API_URL)* | API base URL for the reasoning model (override to point at a local Qwen2.5-32B) |
| `REASONING_API_KEY` | *(OPENAI_API_KEY)* | API key for the reasoning model |
| `REASONING_TIMEOUT` | `90` | Reasoning call timeout in seconds |
| `VISION_MODEL` | *(empty — disabled)* | Vision model for image description (e.g. `gpt-4o`, `qwen2.5-vl-72b`). Leave empty to ignore images. |
| `VISION_API_URL` | *(OPENAI_API_URL)* | API base URL for the vision model (override for a local Qwen2.5-VL) |
| `VISION_API_KEY` | *(OPENAI_API_KEY)* | API key for the vision model |
| `VISION_TIMEOUT` | `30` | Vision call timeout in seconds |
| `HF_TOKEN` | — | HuggingFace token (required to download the multilingual embedding model on first start) |
| `ENABLE_ANALYSIS` | `true` | Enable post-response conversation analysis (fact/memory extraction) |
| `SELF_MEMORY_PATH` | `/app/data/jarvis-self.json` | Path to Jarvis identity + reflection JSON file |

### Infrastructure

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://redis:6379` | Redis connection string |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant connection URL |
| `QDRANT_COLLECTION` | `open-webui_knowledge` | Collection for RAG documents |
| `QDRANT_MEMORY_COLLECTION` | `jarvis_memory` | Collection for episodic memory |

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

### Scheduling

| Variable | Default | Description |
|----------|---------|-------------|
| `BRIEFING_ENABLED` | `true` | Enable daily morning briefing |
| `BRIEFING_TIME` | `07:30` | Briefing delivery time (HH:MM) |
| `BRIEFING_TIMEZONE` | `Europe/Paris` | Timezone for scheduling |
| `REFLECTION_INTERVAL_HOURS` | `6` | Hours between self-reflection cycles |

### Trading

| Variable | Default | Description |
|----------|---------|-------------|
| `TRADE_DATA_DIR` | `/app/trade_data` | Path to the CSV drop directory (mapped from `./TradeData/` on host) |

### User Management

| Variable | Default | Description |
|----------|---------|-------------|
| `USERS_LIST` | `JarvisData/users_list.json` | Path to users file |

Users are defined in `jarvis-core/JarvisData/users_list.json`. Each entry contains:
- `code` — authentication code used in API calls
- `name`, `email`, `city`, `timezone`
- `language` — `fr` or `en`
- `briefing_enabled` — boolean

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
  "use_web": false
}
```

- `stream: true` — returns Server-Sent Events
- `voice_mode: true` — concise 2-3 sentence responses optimized for TTS
- `use_web: false` — disable DuckDuckGo search to reduce latency

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
| `GET` | `/self/state` | Current focus, mood, and active goals |
| `GET` | `/self/log` | Last N reflection entries |
| `POST` | `/self/reflect` | Trigger an immediate reflection cycle |

**Reflection action catalog** — actions the LLM can choose during each reflection cycle:

| Action | Description |
|--------|-------------|
| `nothing` | Explicit no-op with reason |
| `store_insight` | Save a learning about a user to `jarvis-self.json` |
| `flag_knowledge_gap` | Log a topic Jarvis answered poorly |
| `send_notification` | Send a Gmail to one user (rate-limited to 1/user/day) |
| `update_self_note` | Write a personal observation to `self_notes` |
| `consolidate_memory` | Trigger memory compression for a user |
| `check_health` | Verify all services and log status |
| `update_trade_threshold` | Update `threshold_high` / `threshold_low` for a portfolio position autonomously |

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
- **Every hour** (weekdays 09:00–17:35 Paris): fetch live prices via yfinance, evaluate alert conditions with REASONING_MODEL. Fired alerts are queued in Redis and injected into the user's next chat message.
- **Every hour (always)**: check `TradeData/` for a new CSV (mtime-gated); import automatically if a newer file is found.
- **Morning briefing**: portfolio performance summary included as a section when positions are loaded.

**Alert conditions** — REASONING_MODEL fires an alert only when at least one of these is true:
- A position's live price crossed its `threshold_high` or `threshold_low`
- Intraday variation > ±3 % on an individual position
- Total daily portfolio loss > 2 %
- A dividend is expected within the next 5 calendar days

Rate limit: the same position cannot trigger a second alert within 4 hours. Queued alerts expire after 24 hours.

### Image Upload & Analysis

Jarvis supports image attachments from Open WebUI using a **two-stage pipeline**:

```
Image → VISION_MODEL (describe) → "=== IMAGE ANALYSÉE ===" context block
                                          ↓
                         full pipeline → REASONING_MODEL (analyze with memory + RAG)
```

1. The vision model produces a detailed description of the image (max 600 tokens).
2. That description is injected as a context block, alongside memory, RAG, emails, etc.
3. The reasoning model then answers the user's question with full Jarvis context.

This decouples vision from reasoning — a local `Qwen2.5-VL-72B` can handle description while `gpt-5.1` handles analysis. Setting `VISION_MODEL` to the same model as `REASONING_MODEL` also works (two calls to the same model).

**If `VISION_MODEL` is not set**, images are silently ignored and only the text is processed.

**Autonomous threshold management** — during its reflection cycles, Jarvis can update `threshold_high` / `threshold_low` on any position via the `update_trade_threshold` action in `self.py`. This happens when the reflection LLM judges that a threshold needs revision (e.g. price has drifted far from the current threshold, or a significant market move warrants adjusting the stop-loss). Thresholds set this way are treated identically to those set via the API or CSV auto-population.

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
├── jarvis-core/
│   ├── Dockerfile
│   ├── main.py            # API routes + request orchestration
│   ├── memory.py          # Memory system (Redis + Qdrant)
│   ├── self.py            # Proto-self / reflection loop
│   ├── briefing.py        # Morning briefing generation
│   ├── google_services.py # Gmail + Calendar
│   ├── trading.py         # Boursorama portfolio surveillance
│   ├── llm_router.py      # Intent classification
│   ├── analyzer.py        # Conversation analysis
│   ├── config.py          # Configuration loader
│   └── JarvisData/
│       ├── users_list.json
│       └── reflections/
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

## Security Notes

- Authentication is code-based (`user_code` in request body, or Bearer token for `/v1/`). There are no passwords.
- Google credentials are stored in `.env` only — never commit this file.
- Qdrant and Redis are internal Docker services and are not exposed externally.
- Memory is isolated per user code.

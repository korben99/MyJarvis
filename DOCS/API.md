# API reference


Base URL: `http://localhost:8000`

## Chat

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

### `/v1/raw/chat/completions` — agents de code

A deliberately bare path, not to be confused with the route above:

| | `/v1/chat/completions` | `/v1/raw/chat/completions` |
|---|---|---|
| Clients | Open WebUI, iOS | OpenCode, agents externes |
| Messages | dernier `user` seulement | tous |
| client `system` | overwritten by Jarvis | respected |
| Contexte injecté | profil, mémoire, RAG, Gmail, Calendar | aucun |
| Écrit en mémoire | oui | **non** |
| Function calling | non | **oui** (natif) |
| Auth | Bearer = code/email | aucune |

That isolation is the whole point: a coding agent pointed at the main route would durably
pollute the convlog and Qdrant with development traffic.

**Function calling.** `tools` (OpenAI schemas) are passed to the chat template, which
imposes the model's native `<tool_call><function=…><parameter=…>` format. `tool_calls.py`
translates both ways: model calls → OpenAI format (typing parameters from the schema), and
OpenAI history → template (`arguments` JSON string → dict, otherwise the second turn produces
a corrupted prompt). `<think>` blocks are never returned to the
client.

**Effort tiers.** No standard OpenAI field lets you dial reasoning up or down, so the
`model` field is used instead — every client sends it, and OpenCode can change it on the
volée (`/models`, ou `-m jarvis/jarvis-deep`) :

| `model` | Raisonnement | Budget |
|---|---|---|
| `jarvis-fast` | désactivé | — |
| `jarvis` | actif | 3000 |
| `jarvis-deep` | actif | 8000 |

An unknown model falls back to the default. These aliases are deliberately **not** listed by
`GET /v1/models`, which feeds the Open WebUI selector where they would make no sense.

Mise en place côté client : voir `DOCS/opencode-local.md`.

## Health & Status

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/status` | Health check, returns service status |
| `GET` | `/models` | List available models |

## Memory

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/memory/profile/{user_code}` | User profile and learned facts |
| `GET` | `/memory/emotional-state` | Jarvis current emotional state |
| `GET` | `/memory/recent/{user_code}` | Recent conversation summaries |
| `GET` | `/memory/self` | Jarvis self-knowledge |
| `DELETE` | `/memory/reset` | Clear all memory (destructive) |

## Briefing

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/briefing/generate/{user_code}` | Generate and cache a new briefing |
| `GET` | `/briefing/{user_code}` | Retrieve the cached briefing |

The morning briefing aggregates: calendar events, unread emails, weather, news headlines, active tasks, portfolio performance (if positions are loaded), and a market perspective — index/VIX/EUR-USD orientation, per-line trend, and upcoming earnings or ex-dividend dates (see `DOCS/TRADING.md`).

## Self / Proto-Self

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/self/state` | Current focus, goals, and per-user relations |
| `GET` | `/self/log` | Last N reflection entries |
| `POST` | `/self/reflect` | Trigger an immediate reflection cycle |

### Background cycles — who fires what, how often, and what it writes

Every scheduled job, from `main.py`. Frequencies are the defaults; all are `.env`-tunable.
"LLM calls" is per run, at full load — an idle system makes far fewer (silent users are
skipped entirely, and most nights revise no introspection axis).

| Trigger | Frequency | Entry point | LLM calls per run | Writes |
|---|---|---|---|---|
| `conversation_analysis` | **60 min** (`CONV_ANALYSIS_INTERVAL_MINUTES`) | `analyse_recent_conversations()` | 1 per session with new messages, per user | Redis profile (`update_user_profile_batch`), projects (`apply_project_updates`), interest weights, `convlog` back-fill (satisfaction · importance · mood · summary), Qdrant **episodic** vector if above `IMPORTANCE_THRESHOLD`, emotional state. **Never writes autobiographical.** |
| `self_reflection` | **6 h** (`REFLECTION_INTERVAL_HOURS`) | `run_self_reflection()` | ≤ 3 (phase 1) + ≤ 3 per *active* user (phase 2) + 1 self-review per outward action + 1 proactive-push check per user | `jarvis-self.json` (reflection log, incidents), Redis knowledge gaps, and whatever the chosen actions write — see the action catalog below |
| `nightly_interaction_review` | **23:00** | `run_nightly_interaction_review()` | 5 per user *having conversed* (calls 1–3, profile dedup, narrative) | Qdrant **autobiographical** (create/archive/delete), `jarvis-self.json` (`self_introspection`, `introspection_log`, `opinions`, `growth_log`, `user_relations`), Redis profile + `profile_narrative` (7-day TTL) + `tomorrow_suggestions` (24 h TTL) |
| ↳ monthly consolidation | **1st of month**, inside the nightly | `consolidate_memories()` | 1 per batch of 50 episodic points | Episodic → autobiographical milestones (`importance = 1.0`), deletes the consolidated points, then decays autobiographical |
| `morning_briefing` | `BRIEFING_TIME` | `run_morning_briefings()` | 1 per user | Push / email delivery only |
| `trade_check` | **2 h** | `run_trade_check()` | 1 per user (`evaluate_alerts`), skipped when the market is closed | Redis portfolio state (prices, auto-set thresholds), push on alert |
| `cve_scan` | **04:30** | `cve.scan()` | none (SBOM + grype) | CVE cache read by `vitals` |
| agent worker | queue-driven, not scheduled | `agent/worker.py` | per task step | Agent workspace, Redis task records |

Two consequences worth knowing before changing anything:

- **Three writers reach autobiographical memory**, not one: the nightly (call 1), the
  and consolidation. Was three until 2026-08-21 — the reflection action `store_insight`
  has been removed. The comment in
  `analyzer.py` claiming the nightly is "the sole authoritative writer" is accurate only
  about the *analyzer* abstaining.
- **The profile has three writers too**: the analyzer (hot, every hour), the nightly dedup
  (call 4). Was three until 2026-08-21 — `correct_profile` has been removed.

### One rule: the night learns, the reflection acts

Reorganised on 2026-08-21. Before that, both cycles straddled both jobs — the night learned
about users *and* about itself, the reflection learned *and* acted, on itself *and* on
users. Three reflection actions duplicated three nightly jobs (`store_insight` vs autobio,
`correct_profile` vs profile dedup, `consolidate_memory` vs curation), at different
cadences, with nothing arbitrating. There was no rule to remember, so nothing was
memorable.

One question now places any piece of code: **does it write what Jarvis knows, or does it
do something? about himself, or about someone?**

| | learn (night, 23:00) | act (every `REFLECTION_INTERVAL_HOURS`) |
|---|---|---|
| **about itself** | 9 introspection axes + opinions — **one** call on the whole day, from conversations *and* operational state | `refine_prompt`, `alert_admin`, `flag_knowledge_gap` |
| **about the user** | facts → autobio, relation, autobio curation, profile dedup, narrative — 4 calls **per active user** | `queue_push`, `send_notification`, `ask_user`, `flag_project_stall`, `update_trade_threshold` |

Three consequences worth knowing:

- **Self-learning left the per-user loop.** Introspection and opinions are global; revising
  them once per user meant N rewrites a night, each seeing one interlocutor and overwriting
  the previous. They now get one call on all of the day's conversations.
- **The night sees its own operation.** `<ton_fonctionnement>` carries services, incidents
  and memory health into the self call. Without it the `meta_personne` axis could learn
  nothing about Jarvis's real limits — the blind spot left when `self_notes` was removed.
- **Two guards are deterministic, not LLM decisions.** `consolidate_incidents()` and
  `alerter_si_anomalie_critique()` run at the head of every reflection cycle, before any
  model call. A service down or non-normalised vectors are abnormal without discussion; the
  admin alert must not depend on what the model chooses next. The model still sees the same
  facts and can raise a reasoned `alert_admin` — one reports, the other recommends.

**Mechanical gate on "act about itself".** That catalogue is short by nature, so without
material the call can only answer `nothing` — 69 of 95 cycles did, before the split.
`_matiere_pour_agir_sur_soi()` opens the call only on a service down, a fixable critical
CVE, an incident since the last pass, a gap at proposal threshold, or user activity. Not
event-driven: no permanent health probe exists to listen to, so state is computed at each
pass and the decision is made on evidence.

Jarvis maintains two autonomous cognitive cycles:

**Reflection loop** (configurable via `REFLECTION_INTERVAL_HOURS`, défaut 6h) — global self-observation. Jarvis reviews system health, user activity, and knowledge gaps, then picks one action from the catalog. At the end of each cycle Jarvis also runs a per-user **proactive push** check. Outcome and new focus are persisted to `jarvis-self.json`.

**Nightly review** (23:00) — per-user conversation review using **5 sequential calls** per user:

1. **`NIGHTLY_FACTS`** — extracts durable user insights (→ Qdrant autobiographical, dedup-checked at importance 0.70), updates the per-user relation in `jarvis-self.json`, and writes `tomorrow_suggestions` to Redis (TTL 24 h) for injection in the next day's system prompt.
2. **`NIGHTLY_SELF`** — Jarvis self-reflection on the day's interactions: revision of the nine introspection axes (→ `self_introspection{}`, history in `introspection_log[]`), formed opinions (→ `opinions[]`), day diary entry (→ `growth_log[]`). Revising nothing is the expected outcome on most nights.
3. **`NIGHTLY_CLEANING`** — Qdrant autobio curation. Receives the full list of current autobiographical facts plus `user_insights` from call 1 as a signal for what is now superseded. Outputs `to_archive` (outdated facts → `archive_autobiographical_event`) and `to_delete` (errors/duplicates → `retract_autobiographical_event`). Very conservative by design.
4. **`curative_profile_cleanup()`** — Redis profile hash dedup. Sync LLM call that identifies semantic duplicates and obsolete keys, applies consolidation updates (merge-before-delete), then deletes redundant keys. Skipped if profile has fewer than 5 keys. Previously monthly — now nightly so duplicate keys are caught within 24 h.
5. **`update_profile_narrative()`** — generates a ~300-token LLM prose portrait of the user (cross-conversation, per-user). Synthesises the profile hash, top-15 interest weights, and the 5 most recent autobiographical facts. The `profil_utilisateur` fields (static biographical data already in the system prompt) are explicitly excluded to avoid repetition. Stored at `user:{code}:profile_narrative` with a 7-day TTL; injected in `build_memory_context()` as `<profil_narratif>` in place of the raw k/v block.

Conversations from the day are sorted by importance score descending before being passed to each LLM call (up to 6 000 chars), so the most significant exchanges are always visible even on high-volume days.

**Reflection context** — what the LLM sees at each reflection cycle. Two builders, one per phase: `gather_global_context()` (phase 1) and `gather_user_context(user_code)` (phase 2):

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
| `introspection` | `jarvis-self.json` | The filled introspection axes — replaced `self_notes` on 2026-08-21 |
| `opinions[-5:]` | `jarvis-self.json` | Last 5 topic opinions written by `add_self_opinion` |
| `user_relations` | `jarvis-self.json` | Affinity + style per user |
| `user_profiles` | Redis hash per user | Capped at 20 keys/user for token budget |
| `push_availability` | Redis `jarvis:device:token:{code}` | Real-time per-user iOS push status — prevents wasting cycles on users with no registered device |

**`behavioral_patterns`** is computed deterministically (no LLM) from the reflection log: action frequency (≥ 20 % of cycles), time-of-day clustering for "nothing" choices (night/evening pattern), and recurring keywords in past focus fields (seen ≥ 3 times). Up to 5 bullet points.

**Reflection action catalog** — actions the LLM can choose during each reflection cycle.
Only outward-facing work lives here since 2026-08-21; memory upkeep moved to the night:

- `prune_self_memory` — stale/redundant `opinions`, 24h cooldown, now a nightly step.
  `self_introspection` is never pruned: nine axes bounded by construction, revised not deleted.
- `consolidate_memory` — full memory compression, now the nightly monthly run (day 1). The
  on-demand path is gone; compressing memory is upkeep, not action.
- `flag_knowledge_gap` — log a topic Jarvis answered poorly, now a field of the nightly self
  call. It requires a *concrete* failure as context, and only the night has the conversations
  in view: the reflection cycle sees counts and topics, never content, so it could not satisfy
  that honestly. It showed — the two gaps actually on file described Jarvis's own loop
  behaviour ("decision inertia", "notification handling"), the only thing it could observe,
  and those fed `refine_prompt`. Guards unchanged: 7-day cooldown per topic, blocked if a
  proposal is pending.

| Action | Phase | Description |
|--------|-------|-------------|
| `nothing` | Both | Explicit no-op with reason |
| `refine_prompt` | Global | Propose an improved version of a prompt (see Prompt Self-Modification below). |
| `alert_admin` | Global | Push a maintenance/security recommendation to the admin (e.g. *"bump openssl to 3.5.6 on qdrant"*). The channel through which Jarvis, having seen `<etat_disparition>` and `<vulnerabilites>`, can act on its own state. Dedicated 24h cooldown. |
| `send_notification` | User | Send a Gmail to one user (rate-limited to 1/user/day). |
| `queue_push` | User | Queue an iOS push notification. Cooldown **48 h** per user (`_PUSH_COOLDOWN_TTL`, `self/state.py`). |
| `ask_user` | User | Send a clarification question via push; user answers in chat. |
| `flag_project_stall` | User | Detect active projects overdue, or with no update for **> 21 days**, and send a push reminder. **14-day** cooldown per project. Runs for **silent users too** — mechanically, without an LLM call: a dormant project is precisely the signature of a user who stopped talking, so restricting it to active users could only reach those who did not need it. |
| `update_trade_threshold` | User | Update `threshold_high` / `threshold_low` for a portfolio position autonomously. |

**Memory health monitoring:** `gather_global_context()` calls `_check_memory_health()` at every reflection cycle, and the result is injected into `<sante_memoire>` so the model sees per-user stats. It uses activity data to distinguish genuine bugs (high null_rate + recent active user) from expected gaps (user on holiday). In parallel and independently, `alerter_si_anomalie_critique()` mails the admin on services down or non-normalised vectors — deterministically, before any model call, with its own 4h cooldown. The same facts also feed the nightly self call as `<ton_fonctionnement>`.

**Memory consolidation** — `consolidate_memories()` is the single entry point. It runs on the 1st of each month (nightly review scheduler) and on demand via the `consolidate_memory` self-action. It executes two steps in order for each user:
1. `_consolidate_user_memories()` — processes episodic points in batches of 50 (oldest first), summarises each batch into one autobiographical milestone via LLM (stored at `importance = MEMORY_CONSOLIDATION_IMPORTANCE = 1.0`), deletes the processed points, loops until fewer than 5 remain.
2. `_decay_autobiographical_memories()` — decays and prunes autobiographical points (see Autobiographical Memory Decay section above).

Profile dedup (`curative_profile_cleanup()`) is **not** part of monthly consolidation — it runs nightly (Call 4 of the nightly review) so duplicates are caught within 24 h rather than up to 30 days.

### Proactive Push Notifications (Phase 1 — polling)

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

### Per-User Relation

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

RELATIONSHIP WITH THIS USER (injected into build_memory_context):
- Affinité : forte          ← label sémantique (forte/bonne/modérée/faible), pas de score numérique
- Style : direct
- Humeur moyenne : warm
```

Affinity is expressed as a semantic label (`forte` ≥ 0.8 · `bonne` ≥ 0.6 · `modérée` ≥ 0.4 · `faible` < 0.4) rather than a numeric score — the LLM reads a qualitative value more reliably.

**Design principle:** in-conversation mood is already perceived by the LLM from the message history — no real-time state update is needed. The relation captures only what cannot be inferred from a single exchange.

## Device / Push Notifications

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/device/register` | Register iOS device token (body: `{user_code, device_token}`) |
| `GET` | `/device/pending/{user_code}` | Poll and atomically clear pending push notifications |
| `POST` | `/device/push/test/{user_code}` | Manually trigger a proactive push LLM call (dev/test) |

## Conversations

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/users/{user_code}/history/{session_id}` | Get session message history |
| `DELETE` | `/conversations/{user_code}/{session_id}` | Clear a session |

## Search Utilities

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/search?q=...&top_k=5` | RAG document search only |
| `GET` | `/web?q=...&max_results=3` | Web search only (DuckDuckGo deep pipeline) |

## Portfolio / Trading

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/portfolio/{user_code}` | Full portfolio with live P&L |
| `POST` | `/portfolio/import/{user_code}` | Force re-parse of the latest CSV in `RAGData/Trade/` |
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
- **Every hour (always)**: check `RAGData/Trade/` for a new CSV (mtime-gated); import automatically if a newer file is found.
- **Morning briefing**: portfolio performance summary included as a section when positions are loaded.

**Alert conditions** — PRIMARY_MODEL fires an alert only when at least one of these is true:
- A position's live price crossed its `threshold_high` or `threshold_low`
- Intraday variation > ±3 % on an individual position
- Total daily portfolio loss > 2 %
- A dividend is expected within the next 5 calendar days

Rate limit: the same position cannot trigger a second alert within 4 hours. Queued alerts expire after 24 hours.

### Ticker Resolution

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
curl -X PUT http://localhost:8000/portfolio/position/ALICE1/IE0002XZSHO1 \
  -H "Authorization: Bearer ALICE1" \
  -H "Content-Type: application/json" \
  -d '{"yahoo_ticker": "IWDA.AS"}'
```

## Image Upload & Analysis

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
curl -X POST http://localhost:8000/portfolio/upload/ALICE1 \
  -H "Authorization: Bearer ALICE1" \
  -F "file=@export-positions-comptables.csv"
```

**Setting alert thresholds:**
```bash
curl -X PUT http://localhost:8000/portfolio/position/ALICE1/FR0000120578 \
  -H "Authorization: Bearer ALICE1" \
  -H "Content-Type: application/json" \
  -d '{"threshold_high": "90.00", "threshold_low": "70.00"}'
```

---


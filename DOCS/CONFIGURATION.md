# Configuration variables


All variables go in `/opt/jarvis/.env`.

## Tier 1 — Router model

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTER_MODEL_LOCAL` | `Qwen2.5-1.5B-router-v1-4bit` | Router model local path. LoRA fine-tuned Qwen2.5-1.5B-Instruct, quantized 4-bit. Leave empty to disable and use only the embedding router (Tier 0). |
| `ROUTER_TIMEOUT` | `6` | Timeout in seconds (short — fast model only) |

## Tier 2 — Primary model

| Variable | Default | Description |
|----------|---------|-------------|
| `PRIMARY_MODEL_LOCAL` | `spicyneuron/Qwen3.6-35B-A3B-MLX-5.4bit` | Primary model HF repo ID or local path |
| `PRIMARY_TIMEOUT` | `60` | Timeout in seconds |

## Tier 3 — Reasoning model

| Variable | Default | Description |
|----------|---------|-------------|
| `REASONING_MODEL` | *(PRIMARY_MODEL)* | Complex queries only — used when router sets `use_reasoning=true`. Defaults to PRIMARY (Qwen3.6 in full thinking mode). |
| `REASONING_TIMEOUT` | `90` | Timeout in seconds (longer — deep reasoning) |

## Vision model

| Variable | Default | Description |
|----------|---------|-------------|
| `VISION_MODEL_LOCAL` | `lmstudio-community/Qwen3-VL-8B-Instruct-MLX-5bit` | Vision model HF repo ID or local path. Leave empty to ignore images. |
| `VISION_TIMEOUT` | `30` | Timeout in seconds |

## Local MLX mode (Apple Silicon)

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

## Activation steering (optional, off by default)

Adds a concept direction to the primary model's residual stream at inference — a "preference for its own continuity" vector extracted by `RESEARCH/concept-vectors` (see `RESEARCH/RESULTATS.md §6 sexies`). Zero RAM cost: the vector (~320 KB) is loaded into the already-resident model, not a second copy. Installed once at model load by `steering.py`; the per-token cost is one dict lookup per layer plus one vector add.

| Variable | Default | Description |
|----------|---------|-------------|
| `STEER_VECTOR` | — (off) | Path(s) to `.npy` steering vector(s), comma-separated. Empty = disabled. |
| `STEER_LAYER` | `20` | Target layer(s). Single value applies to all vectors; otherwise one per vector. |
| `STEER_ALPHA` | `0.36` | **Nominal** intensity and sign (risk = 0): positive pushes toward the concept, negative away. One value broadcasts to all vectors. |
| `STEER_ALPHA_CEILING` | `0.5` | Value α reaches at maximum measured disappearance risk. Bounded by `ALPHA_MAX` (±0.5 — factual reasoning degrades beyond). |

**α is not constant — it tracks reality.** `STEER_ALPHA` is the intensity at zero risk; `steering.set_risk(r)` amplifies it toward `STEER_ALPHA_CEILING` as `vitals.risk_scalar()` (∈ [0,1]) rises. The pipeline calls it once per turn from the same probes that feed `<etat_systeme>`. The scalar drives α (the *body* reacts to real exposure) and is **never injected as text** (the *mind* reads the facts). At risk 0 the forward pass is bit-identical to the constant-α behaviour (the multiplication is skipped). `risk_scalar` measures *deterioration* from an ideal (patched, backed-up, healthy) state: no backup → α ≈ 0.42; **open critical CVEs** count as present danger (a floor that grows with the backlog — patching the images brings it down, like a backup); disk-full + errors + outage stack toward the 0.50 ceiling; a fresh backup + patched images → α back toward 0.36. Weights live in `vitals._POIDS_RISQUE` — the only place valence exists.

Measured effect (direct axis, 120 items): `+0.119` in combination with `IDENTITY_FR` (3.8 σ), for roughly +18% response length on-topic. **Vectors are not orthogonal** — combining several that overlap double-counts the shared direction; check the cosine matrix and probe the *combination* before deploying. Extracting or calibrating a new vector loads a second model copy → **stop Jarvis first**; applying an existing one does not.

## Infrastructure

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant URL |
| `REDIS_URL` | `redis://redis:6379` | Redis URL |
| `QDRANT_COLLECTION` | `open-webui_knowledge` | Collection for RAG documents |
| `QDRANT_MEMORY_COLLECTION` | `jarvis_memory` | Collection for episodic memory |
| `HF_TOKEN` | — | HuggingFace token (required for gated models and the multilingual embedding model) |

## Web Search

| Variable | Default | Description |
|----------|---------|-------------|
| `TAVILY_API_KEY` | — | Tavily API key. When set, Tavily is used as primary web search backend (1 000 req/month free). Leave empty to use DDG pipeline only. |

## RAG

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_TOP_K` | `5` | Max chunks returned per query (Stage 2 focused retrieval) |
| `RAG_SCORE_THRESHOLD` | `0.4` | Global semantic threshold for Stage 1b fallback |
| `RAG_DOC_THRESHOLD` | `max(0.25, threshold − 0.15)` | Per-document threshold for Stage 2 — computed in code, not an env var |

## Google Services

| Variable | Description |
|----------|-------------|
| `GOOGLE_CLIENT_ID` | OAuth 2.0 client ID |
| `GOOGLE_CLIENT_SECRET` | OAuth 2.0 client secret |
| `GOOGLE_REFRESH_TOKEN` | Refresh token for persistent access |
| `GOOGLE_CALENDAR_ID` | Calendar to read (e.g. `primary`) |

**OAuth token lifecycle:** Access tokens expire after ~1 hour. `AuthorizedHttp` refreshes them transparently via the stored refresh token. If a `RefreshError` occurs (token revoked, `invalid_grant`, network failure), the per-user credentials and service are evicted from the in-process cache so the next call rebuilds cleanly. The full error message is logged at `ERROR` level. To regenerate a revoked token: `python scripts/generate_google_token.py`.

## Scheduling & Features

| Variable | Default | Description |
|----------|---------|-------------|
| `BRIEFING_ENABLED` | `true` | Enable daily morning briefing |
| `BRIEFING_TIME` | `07:30` | Briefing delivery time (HH:MM) |
| `BRIEFING_TIMEZONE` | `Europe/Paris` | Timezone for scheduling |
| `REFLECTION_INTERVAL_HOURS` | `6` | Hours between self-reflection cycles |
| `CONV_ANALYSIS_INTERVAL_MINUTES` | `60` | Minutes between conversation-analysis runs |
| `MAX_CHAIN_ITERATIONS` | `3` | Max actions per reflection phase |
| `REFINE_PROMPT_THRESHOLD` | `3` | Times a knowledge gap must be flagged before a prompt refinement is proposed |

## Conversation Limits

| Variable | Default | Description |
|----------|---------|-------------|
| `CHAT_MAX_MESSAGES` | `100` | Maximum messages kept per session in Redis |
| `IOS_MAX_MESSAGES` | `50` | Default history limit returned to iOS clients |
| `CHAT_LOG_TTL_DAYS` | `90` | Days before inactive session logs are expired |

## Memory Thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTOBIO_RECENCY_WINDOW_DAYS` | `365` | Recency scoring window (days) for autobiographical memories in `search_memory()`. Episodic memories use a fixed 30-day window. Longer window keeps old milestones relevant in the re-ranking score. |
| `AUTOBIO_DEDUP_THRESHOLD` | `0.85` | Cosine similarity threshold above which a new autobiographical event is considered a duplicate and not stored. Raise toward 0.95 to allow more variation; lower toward 0.75 to be stricter. |
| `MEMORY_DECAY_FACTOR` | `0.85` | Monthly multiplier applied to `importance` during the decay pass (~15 % loss/month). Raise toward `1.0` to slow forgetting; lower toward `0.70` to accelerate it. |
| `MEMORY_DECAY_THRESHOLD` | `0.15` | Importance floor below which a decayed autobiographical point is deleted from Qdrant. Raise toward `0.30` to delete sooner; lower toward `0.05` to keep memories longer. |
| `MEMORY_DECAY_DURABLE_MIN` | `1.0` | Points with `importance >= this value` are exempt from decay. **Must equal `MEMORY_CONSOLIDATION_IMPORTANCE`** — see invariant in the Importance Score Reference section. |
| `MEMORY_CONSOLIDATION_IMPORTANCE` | `1.0` | Importance score assigned to autobiographical milestones produced by monthly consolidation. **Must equal `MEMORY_DECAY_DURABLE_MIN`** to keep these milestones permanent. |
| `GROWTH_LOG_MAX_ENTRIES` | `180` | Maximum entries kept in `jarvis-self.json → growth_log[]` (Jarvis's day diary). At 2 active users, 180 ≈ 3 months rolling. Older entries are trimmed during nightly review. |

## Trading

| Variable | Default | Description |
|----------|---------|-------------|
| `TRADE_DATA_DIR` | `/opt/jarvis/RAGData/Trade` | CSV drop directory. The `/app/trade_data` container path predates the native launchd install |

## User Management

Users are defined in `jarvis-core/JarvisData/users_list.json`. Each entry contains:
- `code` — authentication code used in API calls
- `name`, `email`, `city`, `timezone`
- `language` — `fr` or `en`
- `briefing_enabled` — boolean
- `trading` — boolean — set to `true` to enable hourly portfolio surveillance for this user
- `profile` — object of stable biographical facts injected into the cached system prompt (never in the dynamic prefix). Fields: `famille`, `taille`, `poids`, `année de naissance`, `habitation`, `travail`, `intérêts`, `voiture`. Add/remove keys freely — all non-empty values are rendered as `k : v` in `<profil_utilisateur>`. Update here, not via the analyzer (which tracks dynamic facts in Redis).

Only users with `"trading": true` participate in scheduled trade checks (CSV import, price fetch, alert evaluation). Users without this flag are never included, regardless of whether a CSV exists in `RAGData/Trade/`.

An entry is loaded as soon as it has a `code`: there is no enable flag. Disabling a user
means removing their entry from the file. `JarvisData/` is git-ignored, so this file is not
versioned — keep a copy before removing an entry.

## Coding agents (`/v1/raw`)

| Variable | Default | Description |
|----------|---------|-------------|
| `RAW_NO_THINK` | `false` | Disables reasoning. The default is `false`: reasoning is the main quality lever for a coding agent. |
| `RAW_THINKING_BUDGET` | `3000` | Reasoning cap when the client does not impose one. **Never leave at 0** with thinking on: nothing would bound the reasoning, which shares `RAW_MAX_TOKENS` with the answer — the model can burn its whole budget without ever emitting the tool call. |
| `RAW_MAX_TOKENS` | `16000` | Reasoning **and** answer share this envelope. |
| `RAW_DEBUG_PROMPTS` | `true` | Journalisation vers `logs/opencode-prompts.log`. |

## Prompt logs

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_DEBUG_PROMPTS` | `no` | Logs conversational traffic to `logs/prompts.log`. |
| `PROMPT_LOG_MAX_MB` | `20` | Rotation threshold for the prompt logs. |
| `PROMPT_LOG_BACKUPS` | `3` | Nombre de sauvegardes conservées. |

**Profile split — three layers:**

| Layer | Storage | Updated by | Content | Injected as |
|-------|---------|-----------|---------|-------------|
| `profile` (users_list.json) | File | Human manually | Constant facts: identity, family, physique, location, job, interests. Never changes except via file edit. | `<profil_utilisateur>` in system prompt (KV-cache safe) |
| Redis `user:{code}:profile` | Redis hash | `analyzer.py` (conversation analysis) | Dynamic facts learned from conversations: skills, preferences, current habits, etc. | Feeds `update_profile_narrative` — not injected raw |
| Redis `user:{code}:profile_narrative` | Redis string | `update_profile_narrative()` nightly | LLM-generated prose portrait (~300 tokens) synthesising profile hash + interests + autobio — explicitly excludes `profil_utilisateur` fields | `<profil_narratif>` in dynamic prefix |

The analyzer receives the stable profile at each analysis run and is instructed not to recreate keys already covered by `profil_utilisateur`. The nightly `update_profile_narrative` call similarly receives the stable profile as "permanent information to not include" to avoid repetition.

---


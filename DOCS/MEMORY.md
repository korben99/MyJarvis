# The memory system

> The five memory layers, introspection, emotional state, and growth caps.

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
  │                                  │           │ status="current" → active, normal recall                               │
  │                                  │           │ status="past"    → archived, deprioritised recall (×0.4), off timeline │
  └──────────────────────────────────┴───────────┴────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────┬──────────────────────────────────────┬─────────────────────────────────────────────────────────┐
  │                      Data                      │             Destination              │                           Key                           │
  ├────────────────────────────────────────────────┼──────────────────────────────────────┼─────────────────────────────────────────────────────────┤
  │ Durable user facts (insights_durables)         │ store_autobiographical_event() →     │ memory_type: autobiographical, status: current          │
  │ written exclusively by nightly review          │ Qdrant                               │ importance = LLM score (0.0–1.0)                        │
  ├────────────────────────────────────────────────┼──────────────────────────────────────┼─────────────────────────────────────────────────────────┤
  │ Durable insights (nightly call 1)              │ store_autobiographical_event() →     │ same                                                    │
  │                                                │ Qdrant                               │                                                         │
  ├────────────────────────────────────────────────┼──────────────────────────────────────┼─────────────────────────────────────────────────────────┤
  │ Outdated facts (nightly cleaning)              │ archive_autobiographical_event() →   │ payload update: status="past", archived_date            │
  │                                                │ Qdrant payload update                │                                                         │
  ├────────────────────────────────────────────────┼──────────────────────────────────────┼─────────────────────────────────────────────────────────┤
  │ Duplicate/erroneous facts (nightly cleaning)   │ retract_autobiographical_event() →   │ delete from Qdrant                                      │
  │                                                │ Qdrant delete                        │                                                         │
  ├────────────────────────────────────────────────┼──────────────────────────────────────┼─────────────────────────────────────────────────────────┤
  │ Jarvis behavioural self-knowledge              │ jarvis-self.json → self_introspection│ no user_code                                            │
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
  │           Model            │                              Role                              │         no_think          │
  ├────────────────────────────┼────────────────────────────────────────────────────────────────┼───────────────────────────┤
  │ Qwen2.5-1.5B-router-v1-4bit │ routing, judge web                                            │ True                      │
  │ (router — local MLX ~1 GB) │ LoRA fine-tuned, 492 samples, val loss 0.047                  │                           │
  ├────────────────────────────┼────────────────────────────────────────────────────────────────┼───────────────────────────┤
  │ Qwen3.6-35B-A3B-MLX-5.4bit │ briefing, conv analysis, refine prompt, reflection, nightly    │ False for reasoning       │
  │ (primary — local MLX ~20G) │ review, trading, calendar, extraction, standard chat           │ True for plain chat       │
  ├────────────────────────────┼────────────────────────────────────────────────────────────────┼───────────────────────────┤
  └────────────────────────────┴────────────────────────────────────────────────────────────────┴───────────────────────────┘



## Five-Layer Memory System

| Layer | Backend | Contents |
|-------|---------|----------|
| Working Memory | Redis | Active session context, current mood |
| Semantic Memory | Redis Hashes | User profiles, preferences, learned facts |
| Episodic Memory | Qdrant | Conversation summaries that passed the importance threshold |
| Autobiographical Memory | Qdrant | High-importance milestones consolidated from episodic memory |
| Self Memory | JSON file | Jarvis identity, introspection (9 axes), opinions, self-notes, growth log, per-user relations |

## Self-Introspection — nine fixed axes

What Jarvis knows about **his own conduct**, in `jarvis-self.json` under `self_introspection`.
Injected into every conversation as `<introspection_jarvis>`.

This replaced the former `learnings` list on 2026-08-20. That list accumulated insights
indexed *by subject* and grew without useful bound; measurement showed the model kept
rediscovering the same eight axes night after night (17 near-duplicate entries), and no
relevance-based recall mechanism survived testing — a learning is written in meta terms
(*"when the user lists blocked admin steps…"*), a message is concrete (*"still got the flat
to sell"*), and a follow-up (*"and for my mother?"*) carries no subject at all. Full
measurements in `RESEARCH/RESULTATS.md`.

Since a disposition applies everywhere rather than being recalled, **all non-empty axes are
injected on every turn**. Cost is bounded by construction (9 lines) instead of growing with
use.

The axes come from published frameworks rather than from the model's own output, which
would be circular — see `INTROSPECTION_AXES` in `config.py`:

| Axes | Framework |
|---|---|
| `controle`, `communion` | Interpersonal circumplex (Wiggins 1979) — the two orthogonal dimensions of interpersonal conduct. `controle` renders *agency*. |
| `meta_personne`, `meta_tache`, `meta_strategie` | Metacognitive knowledge (Flavell 1979) — person / task / strategy variables |
| `affect_antecedent`, `affect_reponse` | Process model of emotion regulation (Gross) — antecedent-focused vs response-focused |
| `autonomie_autre`, `competence_autre` | Self-determination theory (Deci & Ryan 1985) — the *interlocutor's* needs. Its third need, relatedness, is folded into `communion`. |

**How it is written.** The nightly review (Call 2) *revises* axes rather than appending.
One line per axis, the latest wins; a revision history is kept in `introspection_log`
(never injected). No quota either way — and doing nothing is the expected answer on most
nights. Axes outside the list are discarded, verbatim rewrites are filtered out, and an
empty axis stays empty.

Two measured rules govern what an axis line may contain:

- **The gesture that works, never the flaw.** The model reproduces whatever it is shown,
  defects included: an axis phrased *"I state biological mechanisms with unwarranted
  confidence"* cut referrals to a doctor from 8 to 3 across four health questions. Turned
  around into *"separating general science from the clinical case and referring to a doctor
  works better than an exposé of mechanisms"*, it cut the mechanism dumping by two thirds.
- **A statement, not a promise.** *"I must do better…"* cannot be verified or reused.

**Migration.** Existing installs must run `scripts/migrate_introspection.py` (dry-run by
default). Nothing crashes without it, but accumulated self-knowledge silently stops being
injected — hence a warning logged when `jarvis-self.json` is loaded on the old schema.

## Emotional State

Jarvis maintains a continuous internal emotional state, stored in Redis (`jarvis:emotional_state`) and managed exclusively by `emotional_state.py` — a standalone module with no circular imports. The state colors Jarvis's responses when injected into the prompt via `<etat_emotionnel_jarvis>`.

**Three dimensions** — each a float in `[−1.0, +1.0]`, decaying lazily toward 0.0 on read:

| Dimension | Positive (+1.0) | Negative (−1.0) | Decay rate |
|-----------|----------------|----------------|------------|
| `humeur` (mood) | `joyeux` (cheerful) | `triste` (sad) | 0.10 /h (~10 h to neutral) |
| `confiance` (confidence) | `confiant` (confident) | `dans le doute` (doubting) | 0.05 /h (~20 h — doubt lingers) |
| `energie` (energy) | `en forme` (energetic) | `fatigué` (tired) | 0.15 /h (~7 h to neutral) |

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

### Importance Scoring

After each exchange, `analyzer.py` asks the LLM to evaluate an importance score in `[0, 1]` that gates what gets written to episodic memory. The LLM weighs:

- What the exchange reveals about the user's life, projects, and values
- Emotional intensity — tone, engagement, frustration, enthusiasm
- Durability — will this still matter in 3 months?

`memory_summary` is the prerequisite gate: if null, importance is forced to 0.0 regardless of the LLM score. No summary → nothing stored.

**The summary is judged on the session, not on the hourly increment.** The analyzer runs
every `CONV_ANALYSIS_INTERVAL_MINUTES` on messages newer than the per-session watermark, so
a live conversation is cut into slices of one or two turns. Until 2026-08-21 the instruction
on `<historique_deja_analyse>` read *"N'en extrais RIEN : … aucun résumé"*, and the model
obeyed it literally: a slice was judged on its own and almost always came back null.

Measured on identical content, only the boundary moving:

| | result |
|---|---|
| 4 turns, all in `<echange>` | 0/3 null, importance 0.7 |
| 2 turns in `<echange>` + 2 in `<historique_deja_analyse>` | **3/3 null** |

The exception is now open to `memory_summary` alone — it may situate the increment in what
came before. Extraction (`user_facts`, `project_updates`, `topics`) still applies strictly to
the new exchange, which is what prevents re-writing what is already in memory. Duplicate
summaries are caught at write time, not after: `store_memory_vector()` rejects anything below
`NOVELTY_THRESHOLD` (0.25) — measured at 16% rejection over 86 logged decisions, with stored
novelty ranging 0.250–0.813. That threshold was set by hand and is worth re-measuring now
that the input distribution has changed; the `[memory_decision]` log lines carry everything
needed.

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

### Analyzer Output Validation (Pydantic)

`analyzer.py` validates the LLM JSON output against four Pydantic models before any downstream processing:

| Model | Fields | Role |
|-------|--------|------|
| `AnalysisResult` | `topics`, `mood`, `satisfaction`, `user_facts`, `project_updates`, `interest_weights`, `memory_summary`, `importance` | Top-level output schema |
| `ProjectEvent` | `name`, `action`, `summary`, `rename_to` | One project mutation event |
| `UserFact` | `key`, `value` | One profile fact |
| `InterestWeight` | `term`, `weight` | One interest weight entry |

All models use `extra="ignore"` — unknown fields from the LLM (e.g. `"projects"` instead of `"project_updates"`) are silently discarded rather than propagating as silent bugs. Type mismatches raise `ValidationError` immediately and fall through to the error fallback. `analyze_exchange()` returns `model_dump()` — callers receive a plain dict with guaranteed structure, no changes needed downstream.

### Importance Score Reference

Every point stored in Qdrant carries an `importance` field used for retrieval ranking and decay. Complete list of assigned values:

| Score | Source | Decay behaviour |
|-------|--------|-----------------|
| `1.0` (`MEMORY_CONSOLIDATION_IMPORTANCE`) | Monthly consolidation (`_consolidate_user_memories`) — LLM summary of episodic batch | **Permanent — exempt from decay** (`== MEMORY_DECAY_DURABLE_MIN`) |
| `0.0–1.0` (LLM score) | Analyzer episodic write — LLM-evaluated, only stored if score `> IMPORTANCE_THRESHOLD` and summary present | Decays monthly |
| `0.5–0.9` (LLM-set, défaut `0.7`) | Nightly `insights_durables` — LLM chooses importance: `0.5` fact utile · `0.7` significatif · `0.9` moment clé | Decays monthly |
| `0.70` | Nightly review durable fact (`run_nightly_interaction_review`, `insights_durables` only) | Decays monthly |

**Key invariant:** `MEMORY_CONSOLIDATION_IMPORTANCE` must equal `MEMORY_DECAY_DURABLE_MIN`. If you change one, change the other. Breaking this invariant would either make consolidation milestones decay (if `CONSOLIDATION_IMPORTANCE < DURABLE_MIN`) or promote ordinary memories to permanent status (if `DURABLE_MIN` is lowered).

### Profile Key Deduplication

Profile facts are stored as Redis hash fields with namespaced keys (`hobby:kart`, `skill:python`, `location`). A three-stage pipeline prevents duplicates:

| Stage | Method | Cost |
|-------|--------|------|
| 1. Source prevention | Existing profile keys injected into `ANALYSIS_PROMPT` — LLM reuses exact key names instead of inventing new ones | Prompt tokens only |
| 2. Canonical alias | `_SCALAR_CANONICAL` dict maps common synonyms (`ville→location`, `entreprise→current_employer`) without any LLM call | O(1) |
| 3. Category-aware LLM | Router model compares only against keys in the same namespace family (`hobby:*` vs `interest:*`), not all 30+ keys | 1 fast LLM call on a small set |

Stage 1 prevents ~90 % of duplicates at the source. Stages 2–3 are safety nets.

### Profile Write Governance

Profile key **creation** is restricted exclusively to the conversation analyzer (`analyze_exchange` → `update_user_profile_batch`). The analyzer reads actual user messages and only extracts facts explicitly stated by the user.

Until 2026-08-21 the reflection loop could modify profile keys via `correct_profile`. That action was removed: the nightly `curative_profile_cleanup()` applies `updates` as well as deletions, with the whole profile in view, and the analyzer writes new values hourly. Profile writing now has one owner per path instead of three.

The nightly review (`run_nightly_interaction_review`) does not write to the profile hash at all — its `user_insights` go to Qdrant autobiographical memory instead.

An empty string value (`value=""`) is treated identically to `null` (deletion) throughout the write path — a guard against LLM JSON responses that send `""` instead of `null`.

### Project Tracking

Projects are stored as JSON objects in Redis with `name`, `status` (`in_progress` / `done`), `first_mentioned`, `last_update`, and a **`updates[]` timeline** — a FIFO list (cap 20) of `{date, summary}` entries appended on each `update` or `done` action.

`apply_project_updates()` accepts a structured list of events `[{name, action, summary, rename_to}]` and resolves project names using word-overlap fuzzy matching (≥ 60 % threshold) before exact-string lookup, preventing name-drift duplicates (`"Jarvis"` → `"Jarvis v7"`).

When the embed router detects a "project" intent (cosine similarity ≥ 0.74 on project-related phrases), it returns `None` to force the LLM router, which extracts `project_name` from the user message. On the first mention of a project in a session, `get_project_detail()` fetches the full Redis record and `get_project_timeline_text()` formats it for injection into the prompt context — subsequent turns carry this detail in conversation history without re-fetching.

### Memory Retrieval Ranking

`search_memory()` re-ranks Qdrant results using a weighted blend before returning:

```
base_score    = (semantic_similarity × 0.65 + importance × 0.25 + recency_bonus × 0.10) × status_factor
interest_boost = min(0.08, max(0, (best_matching_weight − 1.0) × 0.04))
final_score   = min(1.0, base_score + interest_boost)
```

`status_factor = 0.4` for autobiographical facts with `status="past"` (archived), `1.0` otherwise. The function fetches `limit × 3` candidates from Qdrant (no pre-filter on status), re-ranks with the penalty applied, and returns the top `limit` — so archived facts naturally fall to positions 4–5 when current facts score higher, but are still recalled if semantically close enough.

**Interest-weight boost:** user-declared topics (Redis `user:{code}:interest_weights`, set by the analyser) nudge ranking by up to +0.08. The cap ensures a strong semantic match (gap ≥ 0.08) is never overridden — weight 1.0 = no boost, weight 3.0 = max +0.08.

The recency window is **type-aware**: episodic memories use a 30-day window, autobiographical memories use a 365-day window (`AUTOBIO_RECENCY_WINDOW_DAYS`). Without this distinction, a stable milestone from 6 months ago (e.g. "Alice gave a talk at a security conference") would always score `recency_bonus = 0` and be outranked by trivial recent events.

`build_memory_context()` surfaces the **top 5 autobiographical events by importance + recency** (importance weight 0.7, recency 0.3 over a 1-year window) rather than the 5 most recent — so a critical event from months ago is not displaced by routine recent exchanges. Facts with `status="past"` are excluded from `build_memory_context()` via `get_user_timeline()` (`must_not` filter in Qdrant scroll).

### Autobiographical Memory Deduplication and Reinforcement

Before any call to `store_autobiographical_event()`, Jarvis queries Qdrant for the most similar existing autobiographical point. If the similarity exceeds `AUTOBIO_DEDUP_THRESHOLD` (default: 0.85), the new entry is not duplicated. However, if the new submission carries a **higher importance** than the existing point, the existing point is reinforced (importance updated upward). This models the human phenomenon of a recurring important fact becoming more firmly anchored over time.

**Score clamping note:** The Qdrant memory collection uses `Distance.DOT`. Raw dot product scores can exceed `1.0` when stored vectors predate the `normalize_embeddings=True` enforcement. All score comparisons in `memory.py` clamp the score to `min(score, 1.0)` before any threshold comparison or weighted blend. Without this, a stored vector with magnitude > 1 would produce `novelty = 1 − 1.28 = −0.28`, clamped to `0`, blocking all new memory writes.

**Normalization invariant (enforced at write):** `store_memory_vector()` asserts L2 norm ≈ 1.0 (±0.01) before every Qdrant upsert. If the norm is off, the vector is re-normalized and an `[memory_invariant]` error is logged — preventing silent norm drift from reaching the collection. A one-shot migration script (`scripts/migrate_qdrant_normalize_vectors.py`) corrected 26 pre-existing un-normalized points in May 2026.

**Structured decision log:** every call to `store_memory_vector()` emits a `[memory_decision]` log line with `stored=True/False`, `reason` (duplicate/no_summary), `novelty`, `importance`, and a 80-char summary preview. Greppable for monitoring: `grep "\[memory_decision\]" logs/jarvis-api.log`.

The threshold is tunable: raise toward 0.95 to allow more variations, lower toward 0.75 to be stricter.

### Autobiographical Memory — Fact Correction

Two distinct operations handle outdated or incorrect autobiographical facts:

**`archive_autobiographical_event(user_code, query, threshold=0.78)`** — called by the nightly cleaning pass when a fact is no longer current (e.g. "ne travaille plus chez X", "a changé de ville"). The function finds the best matching point in Qdrant via semantic search and updates its payload: `status="past"`, `archived_date=today`. The point is never deleted — it remains searchable but is deprioritized (`status_factor=0.4` in `search_memory`) and excluded from the timeline (`get_user_timeline` uses `must_not: status=past`). This preserves history: Jarvis knows you changed jobs, not just that you have a current job. The timeline cache is invalidated after archiving.

**`retract_autobiographical_event(user_code, query, threshold=0.88)`** — hard delete from Qdrant. Reserved exclusively for **factual errors or strict duplicates** — information that should never have been stored. The higher threshold (0.88 vs dedup at 0.85) avoids collateral deletions when the query is semantically close to *related but different* facts. The timeline cache is invalidated on deletion.

Both operations are triggered exclusively by the nightly cleaning LLM call (`NIGHTLY_CLEANING_PROMPT`), which receives the full list of current autobiographical facts and outputs `{"to_archive": [...], "to_delete": [...]}`. The cleaning prompt is instructed to be very conservative: archive superseded facts, delete only proven errors or exact duplicates.

### Implicit Satisfaction Signal

Each `convlog` entry carries a `satisfaction` field detected deterministically from the user's message (lagged proxy: signal in message N reflects satisfaction with the response to message N−1):

| Value | Detection pattern |
|-------|------------------|
| `positive` | Message starts with or contains: `merci`, `parfait`, `super`, `exactement`, `nickel`, `génial`, `top`, `c'est ça` |
| `negative` | Message starts with or contains: `non,`, `non.`, `c'est pas ça`, `tu n'as pas`, `pas compris`, `faux`, `incorrect`, `erreur` |
| `unknown` | Default — no pattern matched |

`_get_user_activity()` aggregates these signals per user over the last 24 hours and exposes them in the reflection prompt as `satisfaction: +N -M`. This gives the autonomous reflection loop an observable quality signal it previously lacked.

### Temporal Awareness

Time is injected at three levels to prevent date hallucination and give the LLM accurate temporal context:

| Layer | What is injected | Where |
|-------|-----------------|-------|
| **User message prefix** | Current datetime in French, user timezone, with season (e.g. `"vendredi 3 avril 2026, 14:32 (printemps)"`) via `fmt_now_fr()` | `build_dynamic_prefix()` — prepended to each user message; does NOT go in the system prompt (KV-cache invariant) |
| **Memory chunks** | French relative timestamp prepended to each retrieved memory (e.g. `"(il y a 3 jours) ..."`) via `rel_time_fr()` | `build_context()` — injected before `trim_chunks()` |
| **Conversation analyzer** | Current date in ISO 8601 (`2026-03-30`) at the top of `ANALYSIS_PROMPT` | `analyze_exchange()` — prevents the LLM from inventing dates for `memory_summary` anchors |
| **Self-reflection** | Server local time in French (`fmt_now_fr(BRIEFING_TIMEZONE)`) replaces raw UTC ISO | `gather_global_context()` / `gather_user_context()` — reflection LLM knows actual local time |
| **Push availability** | Per-user local time shown next to each push-capable device | `self/context.py` phase 2 (`has_push`, `push_cooldown_str`) — reflection LLM can decide not to push at 23h45 |

`fmt_now_fr(tz_name)` is defined in `helpers.py` and includes the current season (hiver/printemps/été/automne) to prevent seasonal confusion in LLM responses.

### Memory Reconsolidation on Recall

Each time `search_memory()` returns a result, every recalled memory receives a small importance boost (`+0.05`, capped at `MEMORY_DECAY_DURABLE_MIN - 0.05 = 0.95`). This models the neuroscience principle of reconsolidation: the act of recalling a memory strengthens it. A memory accessed frequently resists decay; a memory never accessed fades at the normal rate.

### Memory Function Map

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
  │     └── store_autobiographical_event()  → Qdrant autobio  (importance=0.70 default, 0.5–0.9 per the LLM)
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
  │     └── jarvis-self.json → self_introspection{}, growth_log[], opinions[]
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

### Autobiographical Memory Decay

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

## Data Growth & Caps

All storage is bounded. The table below shows what grows, where it's capped, and what happens when the cap is hit.

### Redis

| Key pattern | Type | Cap | Behaviour at cap |
|-------------|------|-----|-----------------|
| `chat:{code}:{session}` | List | `CHAT_MAX_MESSAGES` (100) | Oldest messages trimmed at each write |
| `episodic:{code}:conversations` | Sorted set (score = timestamp) | 1 000 entries | Oldest entries removed at each write |
| `user:{code}:profile` | Hash | None (Redis) — nightly cleanup via `curative_profile_cleanup()` | LLM identifies and deletes duplicate/obsolete keys |
| `jarvis:self:reflection_log` | Sorted set | 30 entries | Oldest entries trimmed at each write |
| `jarvis:{code}:tomorrow_suggestions` | String | TTL 24 h | Auto-expires — no manual cleanup needed |
| `jarvis:push:pending:{code}` | List | Cooldown 1 push/2 h | No write if cooldown active |

### jarvis-self.json

| Field | Cap | Notes |
|-------|-----|-------|
| `self_introspection{}` | 9 axes | Bounded by construction — revised in place, never appended |
| `introspection_log[]` | `INTROSPECTION_LOG_MAX_ENTRIES` (200) | Revision history, never injected |
| `opinions[]` | 50 entries | Trimmed to `[-50:]` after each `add_self_opinion` call; same-topic opinions are updated in place |
| `growth_log[]` | `GROWTH_LOG_MAX_ENTRIES` (180) | Trimmed to `[-180:]` after nightly review |
| `user_relations{}` | 1 entry/user | Updated in place — no growth |

### Qdrant (`jarvis_memory` collection)

| Memory type | Growth rate | Consolidation | Long-term behaviour |
|-------------|-------------|---------------|---------------------|
| `episodic` | ~7 points/user/day (importance ≥ 0.35) | Monthly (day 1), or on-demand via `consolidate_memory` action: batch of 50, loops until < 5 remain | Stable — cleared each consolidation run |
| `autobiographical` | ~2 points/user/day + consolidation output | Dedup check (cosine ≥ `AUTOBIO_DEDUP_THRESHOLD`) before write; monthly decay pass deletes points below `MEMORY_DECAY_THRESHOLD` | Stable long-term — decay prevents unbounded growth; only consolidation milestones (`importance = 1.0`) are permanent |

---


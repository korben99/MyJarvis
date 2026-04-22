"""
PROJECT JARVIS v8
Jarvis Conversation Analyzer
=============================
After each exchange, extracts:
- Topics discussed
- User facts to remember
- Mood/sentiment
- Projects mentioned

Uses the LLM to analyze — costs ~$0.001 per analysis.

Episodic Salience Score (ESS)
Le score d'importance devient la somme de plusieurs signaux :

Personal relevance
informations sur l'utilisateur (facts, projets)

Emotional intensity
émotions positives ou négatives fortes

Novelty
nouveau sujet ou nouvelle information

Goal relevance
lié à un projet ou une action concrète

Memory summary signal
le LLM a identifié quelque chose à retenir

Message depth
message long / détaillé
"""

import asyncio
import json
import time
from datetime import date

from config import (
    AUTOBIO_IMPORTANCE_THRESHOLD,
    CHAT_LOG_TTL,
    IMPORTANCE_THRESHOLD,
    PRIMARY_API_KEY,
    PRIMARY_API_URL,
    PRIMARY_MODEL,
    USER_CODES,
)
from helpers import call_llm_async, extract_llm_json, get_logger, get_redis
from prompts import get_prompt

logger = get_logger("jarvis-analyzer")


async def analyze_exchange(
    user_msg: str,
    assistant_msg: str,
    existing_projects: list = None,
    existing_profile_keys: list = None,
) -> dict:
    """Analyze a conversation exchange using the LLM."""
    try:
        # Show in_progress projects clearly + recent done projects (last 90 days)
        # so the LLM can avoid re-creating finished or versioned projects.
        import time as _t

        _now_ts = _t.time()
        _90d = 90 * 86400

        def _proj_label(p):
            if p.get("status") == "done":
                return f"{p['name']} (terminé)"
            return p["name"]

        projects_context = (
            ", ".join(
                _proj_label(p)
                for p in existing_projects
                if isinstance(p, dict)
                and p.get("name")
                and (
                    p.get("status") != "done"
                    or (
                        p.get("last_update")
                        and (
                            _now_ts
                            - _t.mktime(
                                _t.strptime(p["last_update"][:19], "%Y-%m-%dT%H:%M:%S")
                            )
                        )
                        < _90d
                    )
                )
            )
            or "aucun"
        )
        profile_keys_str = (
            ", ".join(existing_profile_keys) if existing_profile_keys else "aucune"
        )
        prompt = get_prompt("ANALYSIS_PROMPT").format(
            current_date=date.today().isoformat(),
            user_message=user_msg[:1500],
            assistant_message=assistant_msg[:1500],
            existing_projects=projects_context,
            existing_profile_keys=profile_keys_str,
        )

        content = await call_llm_async(
            [{"role": "user", "content": prompt}],
            model=PRIMARY_MODEL,
            api_url=PRIMARY_API_URL,
            api_key=PRIMARY_API_KEY,
            temperature=0.1,
            max_tokens=8000,  # thinking can exceed 4000 tok on rich conversations
            json_response=True,
            no_think=False,  # thinking improves fact extraction accuracy
            timeout=90.0,
        )
        logger.debug(f"[ANALYZER RAW] {content[:300]}")
        try:
            result = extract_llm_json(content)
        except json.JSONDecodeError as exc:
            logger.error("Analyzer JSON parse error: %s", exc.doc[:200])
            raise

        # Episodic Salience Score (ESS) — signals combined into [0, 1]
        # IMPORTANCE_THRESHOLD = 0.35 → stored as episodic vector
        # AUTOBIO_IMPORTANCE_THRESHOLD = 0.60 → stored as autobiographical
        importance = 0.0

        # LLM's own judgment is the primary signal: 0.4 alone clears
        # IMPORTANCE_THRESHOLD so any exchange the LLM deems worth
        # remembering is captured, even with no other signals.
        # The field is "memory_summary" (renamed from "should_remember" in prompt v2).
        memory_summary_text = result.get("memory_summary")
        _has_summary = isinstance(memory_summary_text, str) and bool(
            memory_summary_text.strip()
        )

        if _has_summary:
            importance += 0.40

        # Personal facts revealed by the user
        importance += min(len(result.get("user_facts", [])), 3) * 0.20

        # Projects / goal context
        importance += min(len(result.get("projects", [])), 2) * 0.15

        # Emotional intensity (mild boost — avoid over-storing rants)
        # Both positive and negative emotions weighted equally: high intensity = more memorable,
        # regardless of valence. Previous asymmetry (0.15 negative vs 0.10 positive) created
        # a bias toward storing frustration over joy in long-term memory.
        mood = result.get("mood", "neutral")
        if mood in ["happy", "curious", "focused", "stressed", "frustrated"]:
            importance += 0.10

        # Message depth (minor signal — long messages often carry more info)
        if len(user_msg) > 200:
            importance += 0.05

        # Clamp score
        importance = min(importance, 1.0)
        result["importance"] = round(importance, 3)

        # should_remember: ESS cleared threshold AND LLM provided a concrete summary
        result["should_remember"] = (
            result["importance"] > IMPORTANCE_THRESHOLD and _has_summary
        )
        # Normalise memory_summary: None if missing/empty (LLM may omit field or send null)
        if not _has_summary:
            result["memory_summary"] = None

        return result

    except Exception as e:
        logger.error("Analysis error: %s", e)
        return {
            "topics": [],
            "mood": "neutral",
            "satisfaction": "unknown",
            "user_facts": [],
            "projects": [],
            "importance": 0.0,
            "memory_summary": None,
            "should_remember": False,
        }


# ── Scheduled batch analysis ──────────────────────────────────────────────


async def analyse_recent_conversations(user_code: str | None = None) -> None:
    """
    Scheduled analysis of unanalyzed conversation exchanges (called every 30 min).

    For each user, scans all active chat sessions, collects messages appended
    since the last watermark, and runs a single analyze_exchange call on the
    accumulated cross-session context (chronologically merged).

    Watermark — Redis key `analysis_wm:{user_code}:{session_id}`:
      Stores the float timestamp (ts) of the last analyzed message.
      Messages with ts > watermark are considered new.
      Immune to ltrim truncation: LLEN-based watermarks break once the list
      hits CHAT_MAX_MESSAGES because llen stops growing — timestamps don't.
      Snapshot is taken BEFORE the LLM call so messages that arrive during
      analysis are included in the next batch (never dropped).

    After analysis:
      - user_facts  → Redis profile via update_user_profile_batch
      - projects    → apply_project_updates
      - mood        → update_emotional_state
      - importance > IMPORTANCE_THRESHOLD   → store_memory_vector (Qdrant épisodique)
      - importance > AUTOBIO_IMPORTANCE_THRESHOLD → store_autobiographical_event
    """
    # Import here to avoid circular imports at module load time
    from memory import (
        apply_project_updates,
        get_user_profile,
        get_user_projects,
        set_interest_weight,
        store_autobiographical_event,
        store_memory_vector,
        update_emotional_state,
        update_user_profile_batch,
    )

    r = get_redis()
    users = [user_code] if user_code else list(USER_CODES.keys())

    _mood_to_state = {
        "happy":      {"mood": "happy",     "energy": 0.8},
        "stressed":   {"mood": "attentive",  "concern": 0.6},
        "frustrated": {"mood": "supportive", "concern": 0.7},
        "curious":    {"mood": "engaged",    "curiosity": 0.8},
        "tired":      {"mood": "gentle",     "energy": 0.4},
        "focused":    {"mood": "focused",    "energy": 0.7},
    }

    for uc in users:
        try:
            # ── Discover active sessions ──────────────────────────────────
            session_keys: list[str] = []
            cursor: int | str = "0"
            while True:
                cursor, keys = r.scan(cursor, match=f"chat:{uc}:*", count=100)
                session_keys.extend(k.decode() if isinstance(k, bytes) else k for k in keys)
                if str(cursor) == "0":
                    break

            if not session_keys:
                continue

            # ── Collect new messages (since watermark) from all sessions ──
            all_new: list[dict] = []
            wm_updates: dict[str, float] = {}  # wm_key → new watermark ts (snapshot)

            for chat_key in session_keys:
                session_id = chat_key.removeprefix(f"chat:{uc}:")
                wm_key = f"analysis_wm:{uc}:{session_id}"

                # Watermark is the float timestamp of the last analyzed message.
                # Using ts instead of LLEN makes the watermark immune to ltrim:
                # once the list is capped at CHAT_MAX_MESSAGES, llen stops
                # growing and llen-based watermarks silently freeze.
                wm_ts = float(r.get(wm_key) or 0)

                session_new: list[dict] = []
                for raw in r.lrange(chat_key, 0, -1):
                    try:
                        msg = json.loads(raw)
                        if msg.get("ts", 0) > wm_ts:
                            msg["_session_id"] = session_id
                            session_new.append(msg)
                    except (json.JSONDecodeError, ValueError):
                        pass

                if not session_new:
                    continue

                # Snapshot BEFORE the LLM call: record max ts seen in this batch
                wm_updates[wm_key] = max(m["ts"] for m in session_new)
                all_new.extend(session_new)

            if not all_new:
                logger.debug("[SCHEDULER] analyse_recent_conversations: no new messages for %s", uc)
                continue

            # ── Merge chronologically across sessions ─────────────────────
            all_new.sort(key=lambda m: m.get("ts", 0))

            user_parts = [m["content"] for m in all_new if m.get("role") == "user"]
            asst_parts  = [m["content"] for m in all_new if m.get("role") == "assistant"]

            if not user_parts:
                continue

            acc_user = "\n---\n".join(user_parts)[:3000]
            acc_asst  = "\n---\n".join(asst_parts)[:3000]

            # ── LLM analysis ──────────────────────────────────────────────
            existing_projects = get_user_projects(uc)
            existing_profile_keys = list(get_user_profile(uc).keys())
            analysis = await analyze_exchange(
                acc_user, acc_asst, existing_projects, existing_profile_keys
            )

            # ── Update watermarks (analysis succeeded) ────────────────────
            for wm_key, new_wm_ts in wm_updates.items():
                r.set(wm_key, new_wm_ts, ex=CHAT_LOG_TTL)

            # ── Apply results ─────────────────────────────────────────────
            importance     = analysis.get("importance", 0)
            mood           = analysis.get("mood", "neutral")
            memory_summary = analysis.get("memory_summary")
            satisfaction   = analysis.get("satisfaction", "unknown")

            if mood in _mood_to_state:
                update_emotional_state(_mood_to_state[mood])

            # ── Back-fill LLM satisfaction into convlog entries ───────────
            # Replaces the regex-based _detect_satisfaction() stored at log time.
            if satisfaction in ("positive", "negative"):
                _ts_list = [m.get("ts", 0) for m in all_new if m.get("ts")]
                if _ts_list:
                    _min_ts = min(_ts_list) - 1
                    _max_ts = max(_ts_list) + 1
                    _clog_key = f"convlog:{uc}"
                    _raw_entries = r.zrangebyscore(_clog_key, _min_ts, _max_ts, withscores=True)
                    if _raw_entries:
                        _pipe = r.pipeline()
                        for _raw, _score in _raw_entries:
                            try:
                                _e = json.loads(_raw)
                                if _e.get("satisfaction") != satisfaction:
                                    _e["satisfaction"] = satisfaction
                                    _pipe.zrem(_clog_key, _raw)
                                    _pipe.zadd(_clog_key, {json.dumps(_e, ensure_ascii=False): _score})
                            except (json.JSONDecodeError, ValueError):
                                pass
                        _pipe.execute()

            projects = analysis.get("projects", [])
            if projects:
                apply_project_updates(uc, projects)

            user_facts = [f for f in analysis.get("user_facts", []) if "key" in f and "value" in f]
            if user_facts:
                await asyncio.to_thread(update_user_profile_batch, uc, user_facts)

            for iw in analysis.get("interest_weights") or []:
                if "term" in iw and "weight" in iw:
                    set_interest_weight(uc, iw["term"], float(iw["weight"]))

            # ── Qdrant vectorisation (importance-gated) ───────────────────
            if importance > IMPORTANCE_THRESHOLD and memory_summary:
                entry = {
                    "session_id": "batch",
                    "timestamp": time.time(),
                    "user": acc_user[:500],
                    "assistant": acc_asst[:500],
                    "mood": mood,
                    "topics": analysis.get("topics", []),
                    "importance": importance,
                    "memory_summary": memory_summary,
                }
                await asyncio.to_thread(store_memory_vector, uc, entry)

            if importance > AUTOBIO_IMPORTANCE_THRESHOLD and memory_summary:
                await asyncio.to_thread(
                    store_autobiographical_event, uc, memory_summary, importance
                )

            logger.info(
                "[SCHEDULER] analyse_recent_conversations: user=%s sessions=%d "
                "new_msgs=%d mood=%s facts=%d importance=%.2f memory=%s",
                uc, len(wm_updates), len(all_new), mood,
                len(analysis.get("user_facts", [])), importance,
                "→Qdrant" if importance > IMPORTANCE_THRESHOLD else "skip",
            )

        except Exception as exc:
            logger.error(
                "[SCHEDULER] analyse_recent_conversations failed for %s: %s", uc, exc
            )

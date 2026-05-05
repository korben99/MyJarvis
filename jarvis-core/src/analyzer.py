"""
PROJECT JARVIS v9
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
    CHAT_LOG_TTL,
    IMPORTANCE_THRESHOLD,
    PRIMARY_MODEL,
    USER_CODES,
)
from helpers import extract_llm_json, get_logger, get_redis
from llm_local import call_llm_local_async_bg
from prompts import get_prompt

logger = get_logger("jarvis-analyzer")


async def analyze_exchange(
    conversation: str,
    existing_projects: list = None,
    existing_profile_keys: list = None,
) -> dict:
    """Analyze a conversation exchange using the LLM."""
    try:
        # Show in_progress projects clearly + recent done projects (last 90 days)
        # so the LLM can avoid re-creating finished or versioned projects.
        _now_ts = time.time()
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
                            - time.mktime(
                                time.strptime(
                                    p["last_update"][:19], "%Y-%m-%dT%H:%M:%S"
                                )
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
            conversation=conversation[:5000],
            existing_projects=projects_context,
            existing_profile_keys=profile_keys_str,
        )

        # Appel LLM : priorité basse (bg) pour ne pas bloquer le chat.
        # thinking_budget=600 : cap de raisonnement — assez pour extraire faits/projets
        # proprement, sans risquer que le thinking consume tout le budget comme avant
        # (sans budget, on obtenait {"key":"...","value":"..."} ou [] faute de tokens).
        # max_tokens = thinking_budget(600) + marge réponse(900) = 1500.
        content = await call_llm_local_async_bg(
            [{"role": "user", "content": prompt}],
            model=PRIMARY_MODEL,
            temperature=0.0,  # use model profile default (Qwen3.6: 1.0 thinking)
            max_tokens=1500,
            json_response=True,
            no_think=False,
            thinking_budget=600,
        )

        logger.debug(f"[ANALYZER RAW] {content[:300]}")
        try:
            result = extract_llm_json(content)
        except json.JSONDecodeError as exc:
            logger.error("Analyzer JSON parse error: %s", exc.doc[:200])
            raise

        memory_summary_text = result.get("memory_summary")
        _has_summary = isinstance(memory_summary_text, str) and bool(
            memory_summary_text.strip()
        )

        # ── Importance : jugement LLM direct (Option C) ───────────────────
        # Le LLM évalue 0.0–1.0 en fonction du ton, de l'émotion et de la durabilité.
        # memory_summary null → importance 0.0 (invariant : pas de stockage sans résumé).
        importance = (
            round(min(max(float(result.get("importance") or 0.0), 0.0), 1.0), 3)
            if _has_summary
            else 0.0
        )
        result["importance"] = importance

        # ── ESS rule-based — conservé pour comparaison diagnostique ───────
        # Ne pilote plus le stockage. Les deux scores sont loggés pour évaluer
        # la cohérence LLM vs règles sur plusieurs semaines avant de retirer l'ESS.
        _ess = 0.40 if _has_summary else 0.0
        _durable_facts = [
            f
            for f in result.get("user_facts", [])
            if isinstance(f.get("value"), str) and len(f["value"]) > 10
        ]
        _ess += min(len(_durable_facts), 3) * 0.10
        _ess += min(len(result.get("projects", [])), 2) * 0.15
        mood = result.get("mood", "neutral")
        if mood in ["happy", "curious", "focused", "stressed", "frustrated"]:
            _ess += 0.10
        if len(conversation) > 200:
            _ess += 0.05
        _ess = round(min(_ess, 1.0), 3)

        logger.debug(
            "[IMPORTANCE] llm=%.3f ess=%.3f summary=%s",
            importance,
            _ess,
            _has_summary,
        )

        # should_remember: LLM importance cleared threshold AND summary present
        result["should_remember"] = importance > IMPORTANCE_THRESHOLD and _has_summary
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

    Analyse par session indépendante (vs batch multi-sessions précédent) :
      - Précision : mood/satisfaction par session, pas moyennés sur tout le batch
      - Résilience : watermark mis à jour session par session (succès partiel OK)
      - Qdrant : un vecteur par session significative (vs un seul batch global)
      - Back-fill convlog : satisfaction + importance + mood (vs satisfaction seule)

    Non-bloquant GPU : utilise call_llm_local_async_bg → cède si chat en attente.

    Watermark — Redis key `analysis_wm:{user_code}:{session_id}` :
      Float timestamp du dernier message analysé. Mis à jour immédiatement après
      chaque analyse de session (immune à ltrim, résiliente aux échecs partiels).
    """
    from memory import (
        apply_project_updates,
        get_user_profile,
        get_user_projects,
        set_interest_weight,
        store_memory_vector,
        update_emotional_state,
        update_user_profile_batch,
    )

    r = get_redis()
    users = [user_code] if user_code else list(USER_CODES.keys())

    _mood_to_state = {
        "happy": {"mood": "happy", "energy": 0.8},
        "stressed": {"mood": "attentive", "concern": 0.6},
        "frustrated": {"mood": "supportive", "concern": 0.7},
        "curious": {"mood": "engaged", "curiosity": 0.8},
        "tired": {"mood": "gentle", "energy": 0.4},
        "focused": {"mood": "focused", "energy": 0.7},
        "neutral": {"mood": "neutral", "energy": 0.7, "concern": 0.0},
    }

    for uc in users:
        try:
            # ── Découverte des sessions actives ───────────────────────────
            session_keys: list[str] = []
            cursor: int | str = "0"
            while True:
                cursor, keys = r.scan(cursor, match=f"chat:{uc}:*", count=100)
                session_keys.extend(
                    k.decode() if isinstance(k, bytes) else k for k in keys
                )
                if str(cursor) == "0":
                    break

            if not session_keys:
                continue

            # ── Collecte des nouveaux messages par session ─────────────────
            existing_projects = get_user_projects(uc)
            existing_profile_keys = list(get_user_profile(uc).keys())

            sessions_data: list[dict] = []
            for chat_key in session_keys:
                session_id = chat_key.removeprefix(f"chat:{uc}:")
                wm_key = f"analysis_wm:{uc}:{session_id}"
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

                user_parts = [
                    m["content"] for m in session_new if m.get("role") == "user"
                ]
                if not user_parts:
                    continue

                sessions_data.append(
                    {
                        "session_id": session_id,
                        "wm_key": wm_key,
                        "new_wm_ts": max(m["ts"] for m in session_new),
                        "max_ts": max(m["ts"] for m in session_new),
                        "msgs": session_new,
                        "user_parts": user_parts,
                        "asst_parts": [
                            m["content"]
                            for m in session_new
                            if m.get("role") == "assistant"
                        ],
                    }
                )

            if not sessions_data:
                logger.debug("[SCHEDULER] no new messages for %s", uc)
                continue

            total_new_msgs = sum(len(sd["msgs"]) for sd in sessions_data)

            # ── Analyse par session — ordre chronologique ─────────────────
            # Résultats à fusionner après toutes les sessions
            merged_facts: dict[
                str, dict
            ] = {}  # key → fact (session la plus récente gagne)
            all_projects: list[str] = []
            merged_iw: dict[str, dict] = {}  # term → iw (poids max)
            most_recent_analysis: dict | None = None

            for sd in sorted(sessions_data, key=lambda x: x["max_ts"]):
                turns = []
                for m in sorted(sd["msgs"], key=lambda x: x.get("ts", 0)):
                    role = "Utilisateur" if m.get("role") == "user" else "Jarvis"
                    content = m.get("content", "").strip()[:800]
                    if content:
                        turns.append(f"{role} : {content}")
                conversation = "\n".join(turns)[:6000]

                try:
                    analysis = await analyze_exchange(
                        conversation, existing_projects, existing_profile_keys
                    )
                except Exception as exc:
                    logger.error(
                        "[SCHEDULER] analyze_exchange failed for %s/%s: %s",
                        uc,
                        sd["session_id"],
                        exc,
                    )
                    continue

                # Watermark mis à jour immédiatement — si la session suivante échoue,
                # celle-ci est déjà marquée comme analysée (pas de double-analyse).
                r.set(sd["wm_key"], sd["new_wm_ts"], ex=CHAT_LOG_TTL)
                most_recent_analysis = analysis  # la dernière réussie en ordre chrono

                # ── Fusion incrémentale ───────────────────────────────────
                for f in analysis.get("user_facts", []):
                    if "key" in f and "value" in f:
                        merged_facts[f["key"]] = f  # session plus récente écrase

                all_projects.extend(analysis.get("projects", []))

                for iw in analysis.get("interest_weights") or []:
                    if "term" in iw and "weight" in iw:
                        t = iw["term"]
                        if t not in merged_iw or float(iw["weight"]) > float(
                            merged_iw[t]["weight"]
                        ):
                            merged_iw[t] = iw

                # ── Back-fill convlog : satisfaction + importance + mood ────
                # (#30 fix : filtre strict session_id — pas de contamination croisée)
                # (#33 fix : importance + mood ajoutés au pipeline)
                _sat = analysis.get("satisfaction", "unknown")
                _imp = round(analysis.get("importance", 0.0), 3)
                _mood_s = analysis.get("mood", "neutral")
                _should_backfill = (
                    _sat in ("positive", "negative") or _imp > 0 or _mood_s != "neutral"
                )
                if _should_backfill:
                    _ts_list = [m.get("ts", 0) for m in sd["msgs"] if m.get("ts")]
                    if _ts_list:
                        _min_ts = min(_ts_list) - 1
                        _max_ts_bf = max(_ts_list) + 1
                        _clog_key = f"convlog:{uc}"
                        _raw_entries = r.zrangebyscore(
                            _clog_key, _min_ts, _max_ts_bf, withscores=True
                        )
                        if _raw_entries:
                            _pipe = r.pipeline()
                            for _raw, _score in _raw_entries:
                                try:
                                    _e = json.loads(_raw)
                                    # Strict : ne toucher que les entrées de CETTE session
                                    if _e.get("session_id") != sd["session_id"]:
                                        continue
                                    _changed = False
                                    if (
                                        _sat in ("positive", "negative")
                                        and _e.get("satisfaction") != _sat
                                    ):
                                        _e["satisfaction"] = _sat
                                        _changed = True
                                    if _imp > 0 and _e.get("importance", 0.0) == 0.0:
                                        _e["importance"] = _imp
                                        _changed = True
                                    if _mood_s != "neutral" and not _e.get("mood"):
                                        _e["mood"] = _mood_s
                                        _changed = True
                                    if _changed:
                                        _pipe.zrem(_clog_key, _raw)
                                        _pipe.zadd(
                                            _clog_key,
                                            {
                                                json.dumps(
                                                    _e, ensure_ascii=False
                                                ): _score
                                            },
                                        )
                                except (json.JSONDecodeError, ValueError):
                                    pass
                            _pipe.execute()

                # ── Qdrant par session (importance-gated) ─────────────────
                _imp_s = analysis.get("importance", 0.0)
                _mem_s = analysis.get("memory_summary")
                if _imp_s > IMPORTANCE_THRESHOLD and _mem_s:
                    entry = {
                        "session_id": sd["session_id"],
                        "timestamp": sd["max_ts"],
                        "conversation": conversation[:500],
                        "mood": _mood_s,
                        "topics": analysis.get("topics", []),
                        "importance": _imp_s,
                        "memory_summary": _mem_s,
                    }
                    await asyncio.to_thread(store_memory_vector, uc, entry)

                # Autobio writes are handled exclusively by the nightly review,
                # which has full-day context and is the sole authoritative writer.

            if most_recent_analysis is None:
                logger.warning("[SCHEDULER] all sessions failed for %s", uc)
                continue

            # ── Application des résultats fusionnés ───────────────────────
            mood = most_recent_analysis.get("mood", "neutral")
            if mood in _mood_to_state:
                update_emotional_state(_mood_to_state[mood])

            if merged_facts:
                await asyncio.to_thread(
                    update_user_profile_batch, uc, list(merged_facts.values())
                )

            if all_projects:
                apply_project_updates(uc, all_projects)

            for iw in merged_iw.values():
                set_interest_weight(uc, iw["term"], float(iw["weight"]))

            logger.info(
                "[SCHEDULER] analyse_recent_conversations: user=%s sessions_analysed=%d "
                "new_msgs=%d mood=%s facts=%d",
                uc,
                len(sessions_data),
                total_new_msgs,
                mood,
                len(merged_facts),
            )

        except Exception as exc:
            logger.error(
                "[SCHEDULER] analyse_recent_conversations failed for %s: %s", uc, exc
            )

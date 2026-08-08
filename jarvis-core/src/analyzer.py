"""
PROJECT JARVIS v10
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
from datetime import date, datetime, timezone

from config import (
    CHAT_LOG_TTL,
    DEFAULT_TEMP,
    IMPORTANCE_THRESHOLD,
    MAX_TOKENS_MEDIUM,
    PRIMARY_MODEL,
    USER_CODES,
    USERS,
)
from helpers import extract_llm_json, get_logger, get_redis
from llm_local import call_llm_local_async_bg
from prompts import get_prompt
from pydantic import BaseModel, Field, ValidationError

logger = get_logger("jarvis-analyzer")

# Historique déjà analysé joint à chaque analyse de session.
#
# Le budget global seul ne suffit pas : au plafond normal de 800 car./message, 2000 car.
# ne tiennent que deux tours — et le référent recherché est presque toujours plus haut.
# Constaté sur le cas du 07/08/2026 : « Je dois préparer un Webinar Fortinet » était le
# 1er des 4 tours antérieurs, donc le premier évincé, ce qui annulait tout l'intérêt.
#
# On coupe donc les tours d'historique bien plus court : pour lever un « ça », il faut
# savoir DE QUOI on parle, pas les détails. Le sujet tient dans les premiers caractères,
# et couper court limite aussi la tentation de ré-extraire des faits déjà traités.
# 300 × ~6 tours ≈ 500 tokens sur un prompt d'environ 2500, appel en priorité bg.
ANALYSED_HISTORY_BUDGET_CHARS = 2000
ANALYSED_HISTORY_CHARS_PER_TURN = 300


def _format_turns(messages: list[dict], chars_per_turn: int = 800) -> list[str]:
    """Rend des messages de chat en lignes « Rôle : contenu », du plus ancien au plus récent.

    Les slash-commands utilisateur (/briefing, /agenda…) sont écartées : aucun fait à
    en extraire.
    """
    lines: list[str] = []
    for message in sorted(messages, key=lambda m: m.get("ts", 0)):
        role = "Utilisateur" if message.get("role") == "user" else "Jarvis"
        content = message.get("content", "").strip()[:chars_per_turn]
        if content and not (message.get("role") == "user" and content.startswith("/")):
            lines.append(f"{role} : {content}")
    return lines


def _build_analysed_history(messages: list[dict], budget_chars: int) -> str:
    """Tours déjà analysés à joindre en lecture seule, dans un budget de caractères.

    On garde les plus RÉCENTS (ce sont eux qui portent le référent d'un « ça »), mais on
    les restitue dans l'ordre chronologique pour que le modèle lise une conversation et
    non une pile inversée. Un budget plutôt qu'un nombre de tours fixe : un N constant
    laisse ressortir le référent dès que les tours s'allongent.
    """
    lines = _format_turns(messages, ANALYSED_HISTORY_CHARS_PER_TURN)
    kept: list[str] = []
    used = 0
    for line in reversed(lines):
        if used + len(line) + 1 > budget_chars:
            break
        kept.append(line)
        used += len(line) + 1
    kept.reverse()
    return "\n".join(kept)


class UserFact(BaseModel):
    model_config = {"extra": "ignore"}
    key: str
    value: str


class InterestWeight(BaseModel):
    model_config = {"extra": "ignore"}
    term: str
    weight: float


class ProjectEvent(BaseModel):
    model_config = {"extra": "ignore"}
    name: str
    action: str  # create | update | done | rename
    summary: str = ""
    rename_to: str = ""
    # Échéance ISO "AAAA-MM-JJ".
    due: str | None = None


class AnalysisResult(BaseModel):
    model_config = {"extra": "ignore"}
    topics: list[str] = Field(default_factory=list)
    mood: str = "neutral"
    satisfaction: str = "unknown"
    user_facts: list[UserFact] = Field(default_factory=list)
    project_updates: list[ProjectEvent] = Field(default_factory=list)
    interest_weights: list[InterestWeight] = Field(default_factory=list)
    memory_summary: str | None = None
    importance: float | None = None


async def analyze_exchange(
    conversation: str,
    existing_projects: list | None = None,
    existing_profile_keys: list | None = None,
    stable_profile: dict | None = None,
    analysed_history: str = "",
) -> dict:
    """Analyze a conversation exchange using the LLM.

    `analysed_history` : tours antérieurs de la même session, déjà analysés lors d'une
    passe précédente. Joints en lecture seule pour que le modèle puisse résoudre les
    références ("ça", "c'est terminé") au lieu de les rattacher au hasard à un projet
    connu. Le prompt lui interdit explicitement d'en extraire quoi que ce soit.
    """
    existing_projects = existing_projects or []
    try:
        # Show in_progress projects clearly + recent done projects (last 90 days)
        # so the LLM can avoid re-creating finished or versioned projects.
        _now_ts = time.time()
        _90d = 90 * 86400

        def _proj_label(p: dict) -> str:
            name = p["name"]
            updates = p.get("updates") or []
            last_summary = (
                updates[-1]["summary"] if updates else p.get("description", "")
            )
            lu = p.get("last_update", "")
            age_label = ""
            if lu:
                try:
                    age_days = (
                        _now_ts
                        - datetime.strptime(lu[:19], "%Y-%m-%dT%H:%M:%S")
                        .replace(tzinfo=timezone.utc)
                        .timestamp()
                    ) / 86400
                    age_label = f" [màj il y a {int(age_days)}j]"
                except (ValueError, TypeError):
                    pass
            short_summary = (
                last_summary[:60] + "…" if len(last_summary) > 60 else last_summary
            )
            detail = f" — {short_summary}" if short_summary else ""
            if p.get("status") == "done":
                return f"{name} (terminé{age_label}{detail})"
            return f"{name}{age_label}{detail}"

        projects_context = (
            "\n".join(
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
                            - datetime.strptime(
                                p["last_update"][:19], "%Y-%m-%dT%H:%M:%S"
                            )
                            .replace(tzinfo=timezone.utc)
                            .timestamp()
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
        stable_str = (
            "\n".join(f"  {k}: {v}" for k, v in stable_profile.items() if v)
            if stable_profile
            else "aucun"
        )
        prompt = get_prompt("ANALYSIS_PROMPT").format(
            current_date=date.today().isoformat(),
            conversation=conversation[:5000],
            existing_projects=projects_context,
            existing_profile_keys=profile_keys_str,
            stable_profile=stable_str,
            analysed_history=analysed_history or "aucun",
        )

        # Appel LLM : priorité basse (bg) pour ne pas bloquer le chat.
        # no_think=True : tâche d'extraction structurée — le modèle n'a pas besoin de
        # raisonner, il doit classifier et reformater. En mode think, Qwen3.6 génère
        # systématiquement ~1500 tok de thinking anglais sans fermer </think>, épuisant
        # tout le budget sans jamais produire le JSON. no_think donne la réponse en <5s.
        content = await call_llm_local_async_bg(
            [{"role": "user", "content": prompt}],
            model=PRIMARY_MODEL,
            temperature=DEFAULT_TEMP,
            max_tokens=MAX_TOKENS_MEDIUM,
            json_response=True,
            no_think=True,
        )

        logger.debug("[ANALYZER RAW] %s", content[:300])
        try:
            raw = extract_llm_json(content)
            result = AnalysisResult.model_validate(raw).model_dump()
        except json.JSONDecodeError as exc:
            logger.error("Analyzer JSON parse error: %s", exc.doc[:200])
            raise
        except ValidationError as exc:
            logger.error("Analyzer schema validation error: %s", exc)
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
        _ess += min(len(result.get("project_updates", [])), 2) * 0.15
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

        # Normalise memory_summary: None if missing/empty (LLM may omit field or send null)
        if not _has_summary:
            result["memory_summary"] = None

        return result

    except Exception as e:
        logger.error("Analysis error: %s", e)
        result = AnalysisResult().model_dump()
        result["importance"] = 0.0
        return result


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
    import emotional_state
    from memory import (
        apply_project_updates,
        get_user_profile,
        get_user_projects,
        set_interest_weight,
        store_memory_vector,
        update_user_profile_batch,
    )

    r = get_redis()
    users = [user_code] if user_code else list(USER_CODES.keys())

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
            stable_profile: dict = USERS.get(uc, {}).get("profile", {})

            sessions_data: list[dict] = []
            for chat_key in session_keys:
                session_id = chat_key.removeprefix(f"chat:{uc}:")
                wm_key = f"analysis_wm:{uc}:{session_id}"
                wm_ts = float(r.get(wm_key) or 0)

                new_messages: list[dict] = []
                already_analysed_messages: list[dict] = []
                for raw in r.lrange(chat_key, 0, -1):
                    try:
                        message = json.loads(raw)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if message.get("ts", 0) > wm_ts:
                        message["_session_id"] = session_id
                        new_messages.append(message)
                    else:
                        already_analysed_messages.append(message)

                if not new_messages:
                    continue

                user_parts = [
                    m["content"] for m in new_messages if m.get("role") == "user"
                ]
                if not user_parts:
                    continue

                sessions_data.append(
                    {
                        "session_id": session_id,
                        "wm_key": wm_key,
                        "new_wm_ts": max(m["ts"] for m in new_messages),
                        "max_ts": max(m["ts"] for m in new_messages),
                        "msgs": new_messages,
                        # Tours antérieurs, joints au prompt en lecture seule pour
                        # résoudre les références. DÉLIBÉRÉMENT hors de "msgs" : ce
                        # dernier définit la fenêtre de back-fill du convlog
                        # (new_message_timestamps plus bas), qui ne doit couvrir que les
                        # nouveaux messages — l'élargir réécrirait des entrées déjà
                        # renseignées, et satisfaction s'y écrase sans garde.
                        "already_analysed_messages": already_analysed_messages,
                        "user_parts": user_parts,
                        "asst_parts": [
                            m["content"]
                            for m in new_messages
                            if m.get("role") == "assistant"
                        ],
                    }
                )

            if not sessions_data:
                logger.debug("[SCHEDULER] no new messages for %s", uc)
                continue

            total_new_msgs = sum(len(s["msgs"]) for s in sessions_data)

            # ── Analyse par session — ordre chronologique ─────────────────
            # Résultats à fusionner après toutes les sessions
            merged_facts: dict[
                str, dict
            ] = {}  # key → fact (session la plus récente gagne)
            all_projects: list[str] = []
            merged_iw: dict[str, dict] = {}  # term → iw (poids max)
            most_recent_analysis: dict | None = None

            for session in sorted(sessions_data, key=lambda s: s["max_ts"]):
                conversation = "\n".join(_format_turns(session["msgs"]))[:6000]
                analysed_history = _build_analysed_history(
                    session["already_analysed_messages"],
                    ANALYSED_HISTORY_BUDGET_CHARS,
                )

                try:
                    analysis = await analyze_exchange(
                        conversation,
                        existing_projects,
                        existing_profile_keys,
                        stable_profile,
                        analysed_history,
                    )
                except Exception as exc:
                    logger.error(
                        "[SCHEDULER] analyze_exchange failed for %s/%s: %s",
                        uc,
                        session["session_id"],
                        exc,
                    )
                    continue

                # Watermark mis à jour immédiatement — si la session suivante échoue,
                # celle-ci est déjà marquée comme analysée (pas de double-analyse).
                r.set(session["wm_key"], session["new_wm_ts"], ex=CHAT_LOG_TTL)
                most_recent_analysis = analysis  # la dernière réussie en ordre chrono

                # ── Fusion incrémentale ───────────────────────────────────
                for f in analysis.get("user_facts", []):
                    if "key" in f and "value" in f:
                        merged_facts[f["key"]] = f  # session plus récente écrase

                all_projects.extend(analysis.get("project_updates", []))

                for iw in analysis.get("interest_weights") or []:
                    if "term" in iw and "weight" in iw:
                        t = iw["term"]
                        if t not in merged_iw or float(iw["weight"]) > float(
                            merged_iw[t]["weight"]
                        ):
                            merged_iw[t] = iw

                # ── Back-fill convlog : satisfaction + importance + mood + résumé ──
                # Résumé de session : recopié sur chaque échange de la session.
                session_satisfaction = analysis.get("satisfaction", "unknown")
                session_importance = round(analysis.get("importance", 0.0), 3)
                session_mood = analysis.get("mood", "neutral")
                # Déjà normalisé par analyse_conversation : str non vide, ou None.
                session_summary = analysis.get("memory_summary")
                has_something_to_backfill = (
                    session_satisfaction in ("positive", "negative")
                    or session_importance > 0
                    or session_mood != "neutral"
                    or session_summary
                )
                if has_something_to_backfill:
                    # Fenêtre volontairement restreinte aux NOUVEAUX messages : les tours
                    # déjà analysés ont leur propre entrée convlog, déjà renseignée.
                    new_message_timestamps = [
                        m.get("ts", 0) for m in session["msgs"] if m.get("ts")
                    ]
                    if new_message_timestamps:
                        window_start = min(new_message_timestamps) - 1
                        window_end = max(new_message_timestamps) + 1
                        convlog_key = f"convlog:{uc}"
                        convlog_entries = r.zrangebyscore(
                            convlog_key, window_start, window_end, withscores=True
                        )
                        if convlog_entries:
                            pipe = r.pipeline()
                            for raw_entry, score in convlog_entries:
                                try:
                                    entry = json.loads(raw_entry)
                                    # Strict : ne toucher que les entrées de CETTE session
                                    if entry.get("session_id") != session["session_id"]:
                                        continue
                                    changed = False
                                    if (
                                        session_satisfaction in ("positive", "negative")
                                        and entry.get("satisfaction")
                                        != session_satisfaction
                                    ):
                                        entry["satisfaction"] = session_satisfaction
                                        changed = True
                                    if (
                                        session_importance > 0
                                        and entry.get("importance", 0.0) == 0.0
                                    ):
                                        entry["importance"] = session_importance
                                        changed = True
                                    if session_mood != "neutral" and entry.get(
                                        "mood"
                                    ) in (None, "", "neutral"):
                                        entry["mood"] = session_mood
                                        changed = True
                                    if session_summary and not entry.get(
                                        "memory_summary"
                                    ):
                                        entry["memory_summary"] = session_summary
                                        changed = True
                                    if changed:
                                        pipe.zrem(convlog_key, raw_entry)
                                        pipe.zadd(
                                            convlog_key,
                                            {
                                                json.dumps(
                                                    entry, ensure_ascii=False
                                                ): score
                                            },
                                        )
                                except (json.JSONDecodeError, ValueError):
                                    pass
                            pipe.execute()

                # ── Qdrant par session (importance-gated) ─────────────────
                # Importance non arrondie ici, contrairement au back-fill ci-dessus :
                # c'est elle qui est comparée au seuil et stockée dans le vecteur.
                raw_importance = analysis.get("importance", 0.0)
                has_durable_content = bool(
                    analysis.get("user_facts") or analysis.get("project_updates")
                )
                if (
                    raw_importance > IMPORTANCE_THRESHOLD or has_durable_content
                ) and session_summary:
                    memory_vector_entry = {
                        "session_id": session["session_id"],
                        "timestamp": session["max_ts"],
                        "conversation": conversation[:500],
                        "mood": session_mood,
                        "topics": analysis.get("topics", []),
                        "importance": raw_importance,
                        "memory_summary": session_summary,
                    }
                    await asyncio.to_thread(
                        store_memory_vector, uc, memory_vector_entry
                    )

                # Autobio writes are handled exclusively by the nightly review,
                # which has full-day context and is the sole authoritative writer.

            if most_recent_analysis is None:
                logger.warning("[SCHEDULER] all sessions failed for %s", uc)
                continue

            # ── Application des résultats fusionnés ───────────────────────
            mood = most_recent_analysis.get("mood", "neutral")
            satisfaction = most_recent_analysis.get("satisfaction", "unknown")
            emotional_state.update_from_analysis(mood, satisfaction)

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

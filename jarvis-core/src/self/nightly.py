"""Revue nocturne par utilisateur (APScheduler 23:00) : 5 appels séquentiels par
utilisateur — faits durables (autobio + relation + suggestions), auto-réflexion de
Jarvis (introspection/opinions), curation mémoire, dedup de profil, narratif de profil.
Déclenche la consolidation mensuelle le 1er du mois.

Module indépendant : ni la réflexion ni les actions ne l'appellent (planifié à part).
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone

from config import (
    DEFAULT_TEMP,
    GROWTH_LOG_MAX_ENTRIES,
    INTROSPECTION_AXES,
    INTROSPECTION_LOG_MAX_ENTRIES,
    MAX_TOKENS_COMPACT,
    MAX_TOKENS_NO_THINK,
    REASONING_API_KEY,
    REASONING_API_URL,
    REASONING_MODEL,
    USER_CODES,
    USERS,
    llm_timeout,
)
from helpers import call_llm_async_bg, extract_llm_json, get_logger, get_redis
from memory import (
    archive_autobiographical_event,
    curative_profile_cleanup,
    get_autobiographical_facts,
    get_self_memory,
    retract_autobiographical_event,
    save_self_memory,
    self_memory_lock,
    store_autobiographical_event,
    update_profile_narrative,
)
from prompts import get_prompt

from .state import _DEFAULT_RELATION, _upsert_opinion_inplace, get_user_relation

logger = get_logger("jarvis-self")

_VALID_STYLES = {"direct", "gentle", "formal", "playful"}
_VALID_MOODS = {"warm", "enthusiastic", "measured", "playful", "professional"}


# ══════════════════════════════════════════════════
#  NIGHTLY REVIEW  (replaces nightly-reflection.py)
# ══════════════════════════════════════════════════

# prompts accessed via get_prompt() for live-override support


def _build_conv_text(conversations: list[dict]) -> str:
    """Sort conversations by importance desc and build the conv_text string."""
    sorted_convs = sorted(
        conversations, key=lambda c: c.get("importance", 0), reverse=True
    )
    conv_text = ""
    for c in sorted_convs:
        imp = c.get("importance", 0.0)
        summary = (c.get("memory_summary") or "").strip()
        topics = c.get("topics") or []
        mood = c.get("mood", "?")
        header = f"[importance:{imp:.2f}] [mood:{mood}]"
        if topics:
            header += f" [topics: {', '.join(topics)}]"
        if summary:
            # Analyzer already distilled this exchange — use summary only
            conv_text += f"{header}\n{summary}\n\n"
        else:
            # No summary available — fall back to raw exchange
            conv_text += (
                f"{header}\n"
                f"User: {c.get('user', '')[:350]}\n"
                f"Jarvis: {c.get('assistant', '')[:350]}\n\n"
            )
    return conv_text[:6000] or "(no conversation content)"


async def _nightly_facts_user(
    user_code: str, user_name: str, conversations: list[dict], review_date: str
) -> dict | None:
    """Call 1 — extract durable user facts, relation update, tomorrow suggestions."""
    current_relation = get_user_relation(user_code)
    existing_autobio = await asyncio.to_thread(
        get_autobiographical_facts, user_code, 8, True  # newest first
    )
    existing_autobio_str = (
        "\n".join(f"- {f}" for f in existing_autobio) if existing_autobio else "aucun"
    )
    prompt = get_prompt("NIGHTLY_FACTS_PROMPT").format(
        user_name=user_name,
        user_code=user_code,
        review_date=review_date,
        count=len(conversations),
        conv_text=_build_conv_text(conversations),
        current_relation=json.dumps(current_relation, ensure_ascii=False),
        existing_autobio=existing_autobio_str,
    )
    try:
        content = await call_llm_async_bg(
            [
                {"role": "system", "content": get_prompt("NIGHTLY_FACTS_SYSTEM")},
                {"role": "user", "content": prompt},
            ],
            model=REASONING_MODEL,
            api_url=REASONING_API_URL,
            api_key=REASONING_API_KEY,
            temperature=DEFAULT_TEMP,
            max_tokens=MAX_TOKENS_NO_THINK,
            json_response=True,
            no_think=True,
            timeout=llm_timeout(MAX_TOKENS_NO_THINK),
        )
        return extract_llm_json(content)
    except Exception as exc:
        logger.error(
            "Nightly facts LLM call failed for %s: %s",
            user_code,
            type(exc).__name__,
            exc_info=True,
        )
        return None


def _normalise_introspection(brut, actuel: dict | None = None) -> dict[str, str]:
    """Les révisions d'axes retenues, filtrées sur les noms d'axes connus.

    Un axe inventé est jeté sans bruit plutôt que créé : la liste est fermée par
    construction (config.INTROSPECTION_AXES), c'est ce qui borne le coût de réinjection et
    empêche la liste de redevenir un tas. Un objet vide — le cas attendu la plupart des
    nuits — ne produit aucune écriture.

    Une réécriture À L'IDENTIQUE est écartée : mesuré le 20/08/2026, le modèle réémet
    parfois un axe au mot près. Sans effet sur le contenu, mais elle gonflerait
    introspection_log et ferait croire à une révision là où il n'y en a pas.
    """
    if not isinstance(brut, dict):
        return {}
    actuel = actuel or {}
    revisions = {}
    for axe, texte in brut.items():
        cle = str(axe).strip().lower()
        if cle not in INTROSPECTION_AXES:
            logger.info("Nightly self: axe inconnu ignoré (%s)", cle[:40])
            continue
        if isinstance(texte, str) and texte.strip():
            propre = texte.strip()
            if propre == (actuel.get(cle) or "").strip():
                continue
            revisions[cle] = propre
    return revisions


def _render_introspection(data: dict) -> str:
    """L'état courant des axes, tel qu'il est montré à la revue nocturne.

    Les axes vides sont montrés VIDES et non masqués : c'est ce qui permet à la nuit de
    les remplir, et de voir qu'ils le sont restés.
    """
    slots = data.get("self_introspection") or {}
    return "\n".join(
        f"{axe} ({question})\n    {slots.get(axe) or '— vide —'}"
        for axe, question in INTROSPECTION_AXES.items()
    )


async def _nightly_self_user(
    user_code: str, user_name: str, conversations: list[dict], review_date: str
) -> dict | None:
    """Call 2 — Jarvis self-reflection and opinion formation."""
    data = get_self_memory()
    recent_opinions = [
        f"{o['topic']}: {o['opinion']}" for o in data.get("opinions", [])[-10:]
    ]
    prompt = get_prompt("NIGHTLY_SELF_PROMPT").format(
        user_name=user_name,
        user_code=user_code,
        review_date=review_date,
        count=len(conversations),
        conv_text=_build_conv_text(conversations),
        self_introspection=_render_introspection(data),
        recent_opinions=json.dumps(recent_opinions, ensure_ascii=False)
        if recent_opinions
        else "aucune",
    )
    try:
        content = await call_llm_async_bg(
            [
                {"role": "system", "content": get_prompt("NIGHTLY_SELF_SYSTEM")},
                {"role": "user", "content": prompt},
            ],
            model=REASONING_MODEL,
            api_url=REASONING_API_URL,
            api_key=REASONING_API_KEY,
            temperature=DEFAULT_TEMP,
            max_tokens=MAX_TOKENS_NO_THINK,
            json_response=True,
            no_think=True,
            timeout=llm_timeout(MAX_TOKENS_NO_THINK),
        )
        return extract_llm_json(content)
    except Exception as exc:
        logger.error(
            "Nightly self LLM call failed for %s: %s",
            user_code,
            type(exc).__name__,
            exc_info=True,
        )
        return None


async def _nightly_cleaning_user(
    user_code: str, user_name: str, user_insights: list[str], review_date: str
) -> dict | None:
    """Call 3 — memory curator: archive outdated facts, delete strict duplicates."""
    autobio_facts = await asyncio.to_thread(get_autobiographical_facts, user_code, 40)
    if not autobio_facts:
        logger.info("Nightly cleaning skipped for %s — no autobio facts yet", user_code)
        return None

    facts_numbered = "\n".join(
        f"{i + 1}. {text}" for i, text in enumerate(autobio_facts)
    )
    new_insights_str = (
        json.dumps(user_insights, ensure_ascii=False) if user_insights else "aucun"
    )
    prompt = get_prompt("NIGHTLY_CLEANING_PROMPT").format(
        user_name=user_name,
        review_date=review_date,
        facts_count=len(autobio_facts),
        autobio_facts=facts_numbered,
        new_user_insights=new_insights_str,
    )
    try:
        content = await call_llm_async_bg(
            [
                {"role": "system", "content": get_prompt("NIGHTLY_CLEANING_SYSTEM")},
                {"role": "user", "content": prompt},
            ],
            model=REASONING_MODEL,
            api_url=REASONING_API_URL,
            api_key=REASONING_API_KEY,
            temperature=DEFAULT_TEMP,
            max_tokens=MAX_TOKENS_COMPACT,
            json_response=True,
            no_think=True,
            timeout=llm_timeout(MAX_TOKENS_COMPACT),
        )
        return extract_llm_json(content)
    except Exception as exc:
        logger.error(
            "Nightly cleaning LLM call failed for %s: %s",
            user_code,
            type(exc).__name__,
            exc_info=True,
        )
        return None


async def run_nightly_interaction_review() -> None:
    """
    Nightly per-user conversation review. Called by APScheduler at 23:00.

    For each user with conversations yesterday (5 sequential LLM calls):
      Call 1 — NIGHTLY_FACTS  : user insights → Qdrant autobio + relation update + suggestions
      Call 2 — NIGHTLY_SELF   : Jarvis self-reflection → introspection, opinions, growth_log
      Call 3 — NIGHTLY_CLEANING: Qdrant autobio curation (archive outdated, delete errors)
      Call 4 — profile dedup  : curative_profile_cleanup() → Redis profile hash (sync, no LLM if < 5 keys)
      Call 5 — profile narrative: update_profile_narrative() → Redis user:{code}:profile_narrative (7-day TTL)

    Each user's write to jarvis-self.json is done under self_memory_lock
    immediately after the LLM call — no data is held across await points.
    Idempotent: Redis lock per user per date (TTL 25h).
    Triggers monthly consolidation (episodic compress + autobio decay) on day 1.
    """
    logger.info("=== Nightly interaction review starting ===")
    r = get_redis()
    now = datetime.now(timezone.utc)
    yesterday = now - timedelta(days=1)
    review_date = yesterday.strftime("%Y-%m-%d")
    start_ts = yesterday.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    end_ts = yesterday.replace(
        hour=23, minute=59, second=59, microsecond=999999
    ).timestamp()

    for user_code, user_name in USER_CODES.items():
        lock_key = f"jarvis:{user_code}:nightly_review:{review_date}"
        if not r.set(lock_key, "1", nx=True, ex=90000):  # 25h TTL
            logger.info(
                "Nightly review already done for %s on %s — skipping",
                user_code,
                review_date,
            )
            continue

        entries_raw = r.zrangebyscore(f"convlog:{user_code}", start_ts, end_ts)
        if not entries_raw:
            logger.info(
                "No conversations for %s on %s — skipping", user_code, review_date
            )
            continue

        conversations = []
        for raw in entries_raw:
            try:
                conversations.append(json.loads(raw))
            except Exception:
                pass

        logger.info(
            "Nightly review for %s: %d conversations", user_code, len(conversations)
        )

        # ── Call 1: extract user facts ────────────────────────────────────
        facts = await _nightly_facts_user(
            user_code, user_name, conversations, review_date
        )
        user_insights: list[str] = []

        if facts:
            durables = [i for i in facts.get("insights_durables", []) if i]
            evenements = [i for i in facts.get("insights_evenements", []) if i]
            user_insights = durables + evenements  # full context for cleaning

            # Only durable states go to autobio — nightly is the sole autobio writer
            for item in durables:
                if isinstance(item, dict):
                    insight = (item.get("text") or "").strip()
                    importance = round(max(0.5, min(0.9, float(item.get("importance", 0.7)))), 2)
                else:
                    insight = str(item).strip()
                    importance = 0.7
                if insight:
                    store_autobiographical_event(user_code, insight, importance=importance)

            # Store tomorrow's suggestions in Redis (24h)
            suggestions = facts.get("tomorrow_suggestions", [])
            if suggestions:
                r.setex(
                    f"jarvis:{user_code}:tomorrow_suggestions",
                    86400,
                    json.dumps(suggestions),
                )

        # ── Call 2: Jarvis self-reflection ────────────────────────────────
        self_result = await _nightly_self_user(
            user_code, user_name, conversations, review_date
        )

        # ── Persist facts + self-reflection → jarvis-self.json ───────────
        summary = facts.get("daily_summary", "") if facts else ""
        rel_update = facts.get("user_relation_update", {}) if facts else {}
        revisions = _normalise_introspection(
            (self_result or {}).get("self_introspection"),
            get_self_memory().get("self_introspection"),
        )
        new_opinions = [
            o
            for o in (self_result or {}).get("jarvis_opinions", [])
            if isinstance(o, dict) and o.get("topic") and o.get("opinion")
        ]

        if revisions or summary or rel_update or new_opinions:
            with self_memory_lock:
                data = get_self_memory()
                # Un axe est RÉÉCRIT, jamais empilé : une seule ligne par axe, la dernière.
                # C'est ce qui borne le coût de réinjection et évite que la connaissance de
                # soi redevienne la liste sans fin qu'elle était avant le 20/08/2026.
                axes = data.setdefault("self_introspection", {})
                for axe, texte in revisions.items():
                    axes[axe] = texte
                    data.setdefault("introspection_log", []).append(
                        {"axe": axe, "text": texte, "date": review_date,
                         "user_code": user_code}
                    )
                    logger.info("Nightly introspection: %s → %s", axe, texte[:70])
                for op in new_opinions:
                    _upsert_opinion_inplace(
                        data, op["topic"], op["opinion"].strip(), review_date
                    )
                    logger.info(
                        "Nightly opinion: %s → %s", op["topic"], op["opinion"][:60]
                    )
                if summary:
                    data.setdefault("growth_log", []).append(
                        {
                            "date": review_date,
                            "user_code": user_code,
                            "user_name": user_name,
                            "summary": summary,
                            "mood": (facts or {}).get("mood_summary", ""),
                            "conversations": len(conversations),
                        }
                    )
                if rel_update:
                    current = {
                        **_DEFAULT_RELATION,
                        **data.get("user_relations", {}).get(user_code, {}),
                    }
                    new_affinity = rel_update.get("affinity", current["affinity"])
                    try:
                        new_affinity = round(max(0.0, min(1.0, float(new_affinity))), 2)
                    except (TypeError, ValueError):
                        new_affinity = current["affinity"]
                    new_style = rel_update.get(
                        "interaction_style", current["interaction_style"]
                    )
                    if new_style not in _VALID_STYLES:
                        new_style = current["interaction_style"]
                    new_mood = rel_update.get(
                        "average_interaction_mood", current["average_interaction_mood"]
                    )
                    if new_mood not in _VALID_MOODS:
                        new_mood = current["average_interaction_mood"]
                    data.setdefault("user_relations", {})[user_code] = {
                        "affinity": new_affinity,
                        "interaction_style": new_style,
                        "average_interaction_mood": new_mood,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                    logger.info(
                        "User relation updated for %s: affinity=%.2f style=%s mood=%s",
                        user_code,
                        new_affinity,
                        new_style,
                        new_mood,
                    )
                # Les axes eux-mêmes sont bornés par construction (9). Seule leur trace
                # historique est tronquée — elle sert à relire comment un axe a évolué,
                # pas à alimenter les conversations.
                data["introspection_log"] = data.get("introspection_log", [])[
                    -INTROSPECTION_LOG_MAX_ENTRIES:
                ]
                data["growth_log"] = data.get("growth_log", [])[
                    -GROWTH_LOG_MAX_ENTRIES:
                ]
                data["last_nightly"] = review_date
                save_self_memory(data)

        # ── Call 3: memory cleaning (Qdrant autobio) ─────────────────────
        cleaning = await _nightly_cleaning_user(
            user_code, user_name, user_insights, review_date
        )
        if cleaning:
            # Hard cap: trust the prompt constraint but enforce it in code too.
            # A runaway LLM should not wipe out more than 3 memories in one night.
            to_archive = cleaning.get("to_archive", [])[:3]
            to_delete = cleaning.get("to_delete", [])[:2]
            for text in to_archive:
                if isinstance(text, str) and text.strip():
                    await asyncio.to_thread(
                        archive_autobiographical_event, user_code, text
                    )
            for text in to_delete:
                if isinstance(text, str) and text.strip():
                    await asyncio.to_thread(
                        retract_autobiographical_event, user_code, text
                    )
            rationale = cleaning.get("rationale", "")
            logger.info(
                "Nightly cleaning for %s — archive:%d delete:%d — %s",
                user_code,
                len(to_archive),
                len(to_delete),
                rationale[:80],
            )

        # ── Call 4: profile dedup (Redis profile hash) ────────────────────
        stable_profile = USERS.get(user_code, {}).get("profile", {})
        await asyncio.to_thread(curative_profile_cleanup, user_code, stable_profile)
        await asyncio.to_thread(update_profile_narrative, user_code, stable_profile)

        logger.info("Nightly review done for %s — %s", user_code, summary[:80])

        # Monthly memory consolidation on day 1
        if now.day == 1:
            try:
                from memory import consolidate_memories

                await asyncio.to_thread(consolidate_memories, user_code)
                logger.info("Monthly memory consolidation done for %s", user_code)
            except Exception as exc:
                logger.warning(
                    "Monthly consolidation failed for %s: %s",
                    user_code,
                    type(exc).__name__,
                )

    logger.info("=== Nightly interaction review complete ===")

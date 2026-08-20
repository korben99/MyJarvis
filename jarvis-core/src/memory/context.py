"""Constructeur de contexte mémoire injecté dans le system prompt : frise
autobiographique (cache 5 min) + profil filtré par pertinence + préférences +
projets actifs + état émotionnel + apprentissages/notes + relation utilisateur.
"""

import json
import time

from config import LEARNINGS_MAX_INJECTED, QDRANT_MEMORY_COLLECTION
from helpers import (
    get_logger,
    get_qdrant,
    get_redis,
    keyword_overlap_score,
    rel_time_fr,
)

import emotional_state as _es

from .selfmem import get_self_memory

logger = get_logger("jarvis-memory")


# ══════════════════════════════════════════════════
#  CONTEXT BUILDER — Assemble memory for prompts
# ══════════════════════════════════════════════════


_TIMELINE_CACHE_TTL = 300  # 5 minutes — invalidated on new autobio store


def get_user_timeline(user_code: str, limit: int = 20):
    """
    Retrieve the user's autobiographical timeline.

    Scrolls up to 200 points to ensure the top-N by importance+recency are
    correctly identified — scroll() returns in arbitrary Qdrant order.

    Result is cached in Redis for 5 minutes to avoid a 200-point Qdrant scroll
    on every chat request. Invalidated whenever a new autobiographical memory
    is stored (store_autobiographical_event calls _invalidate_timeline_cache).
    """
    cache_key = f"cache:timeline:{user_code}"
    r = get_redis()
    try:
        cached = r.get(cache_key)
        if cached:
            return json.loads(cached)[:limit]
    except Exception:
        pass

    try:
        qdrant = get_qdrant()

        results = qdrant.scroll(
            collection_name=QDRANT_MEMORY_COLLECTION,
            scroll_filter={
                "must": [
                    {"key": "user_code", "match": {"value": user_code}},
                    {"key": "memory_type", "match": {"value": "autobiographical"}},
                ],
                # Archived (past) facts are excluded from the chat timeline —
                # they remain searchable via search_memory but are not injected
                # into every conversation as if they were still current.
                "must_not": [{"key": "status", "match": {"value": "past"}}],
            },
            limit=200,  # over-fetch then rank — scroll order is arbitrary
        )[0]

        now_ts = time.time()
        timeline = []

        for r_ in results:
            ts = r_.payload.get("timestamp", now_ts)
            imp = r_.payload.get("importance", 0)
            # Recency bonus normalised over a 1-year window
            recency = max(0.0, 1.0 - (now_ts - ts) / (86400 * 365))
            rank = imp * 0.7 + recency * 0.3
            timeline.append(
                {
                    "text": r_.payload["text"],
                    "timestamp": ts,
                    "importance": imp,
                    "_rank": rank,
                }
            )

        # Most important + recent events first
        timeline.sort(key=lambda x: x["_rank"], reverse=True)
        for e in timeline:
            e.pop("_rank", None)

        # Cache top-50 (more than enough for any prompt injection)
        try:
            r.setex(
                cache_key,
                _TIMELINE_CACHE_TTL,
                json.dumps(timeline[:50], ensure_ascii=False),
            )
        except Exception:
            pass

        return timeline[:limit]

    except Exception as e:
        logger.error("Timeline retrieval failed: %s", e)
        return []


# Profile keys always injected regardless of message relevance.
# These provide essential grounding context (location, employer, age, language).
_PROFILE_ALWAYS_INJECT = {"location", "current_employer", "age", "langue", "name"}

# Only filter if profile exceeds this count — small profiles are injected whole.
_PROFILE_FILTER_THRESHOLD = 8

# Max "other" (non-always) keys to keep after relevance scoring.
_PROFILE_MAX_SCORED = 8



def build_memory_context(
    session_id: str,
    user_code: str,
    self_mem: dict | None = None,
    include_suggestions: bool = True,
    user_message: str = "",
) -> str:
    """Build a memory context string to inject into the system prompt.

    Pass an already-loaded *self_mem* dict to avoid a redundant JSON read when
    the caller (build_system_prompt) has already called get_self_memory().

    include_suggestions — set False for pure utility intents (weather/calendar/gmail)
                          to skip the SUJETS À ABORDER section (~50 tokens saved).
    user_message        — when provided, profile keys are filtered to the most
                          relevant ones (keyword overlap scoring). Always-inject
                          keys (location, employer…) are kept regardless of score.

    All Redis reads are batched into a single pipeline round-trip.
    """
    if self_mem is None:
        self_mem = get_self_memory()
    parts = []

    # ── Single Redis pipeline round-trip for all scalar/hash reads ──────────
    r = get_redis()
    pipe = r.pipeline(transaction=False)
    pipe.hgetall(f"user:{user_code}:profile")  # 0
    pipe.hgetall(f"user:{user_code}:preferences")  # 1
    pipe.get(f"user:{user_code}:projects")  # 2
    pipe.get(f"jarvis:{user_code}:tomorrow_suggestions")  # 3
    pipe.get(f"cache:timeline:{user_code}")  # 4 — avoids 2nd Redis RTT
    pipe.get(f"user:{user_code}:profile_narrative")  # 5
    _pipe_results = pipe.execute()

    profile = _pipe_results[0] or {}
    prefs = _pipe_results[1] or {}
    _proj_raw = _pipe_results[2]
    _sugg_raw = _pipe_results[3]
    _timeline_cached = _pipe_results[4]
    _profile_narrative = _pipe_results[5]

    # Context-aware profile filtering: keep always-inject keys + top-N by keyword overlap.
    # Only applied when user_message is provided and profile is large enough to be worth filtering.
    if user_message and len(profile) > _PROFILE_FILTER_THRESHOLD:
        _total = len(profile)
        always = {
            k: v
            for k, v in profile.items()
            if k.split(":")[0] in _PROFILE_ALWAYS_INJECT or k in _PROFILE_ALWAYS_INJECT
        }
        rest = {k: v for k, v in profile.items() if k not in always}
        scored = sorted(
            rest.items(),
            key=lambda kv: keyword_overlap_score(
                kv[0].replace(":", " ") + " " + kv[1], user_message
            ),
            reverse=True,
        )
        profile = {**always, **dict(scored[:_PROFILE_MAX_SCORED])}
        logger.debug(
            "profile context-aware: %d/%d keys injected for %s (msg=%r…)",
            len(profile),
            _total,
            user_code,
            user_message[:40],
        )

    # Narrative profile (generated nightly) — replaces k/v hash rendering.
    # Falls back to raw k/v if narrative not yet generated.
    if _profile_narrative:
        parts.append(
            "<profil_narratif>\n" + _profile_narrative + "\n</profil_narratif>"
        )
    elif profile:
        grouped: dict[str, list[str]] = {}
        scalars: list[tuple[str, str]] = []
        for k, v in profile.items():
            if ":" in k:
                category, subkey = k.split(":", 1)
                grouped.setdefault(category, []).append(f"{subkey}={v}")
            else:
                scalars.append((k, v))
        plines = [f"- {k}: {v}" for k, v in scalars]
        plines += [f"- {cat}: {', '.join(vals)}" for cat, vals in grouped.items()]
        parts.append(
            "<profil_narratif>\n" + "\n".join(plines) + "\n</profil_narratif>"
        )

    # User preferences
    if prefs:
        plines = [f"- {k}: {v}" for k, v in prefs.items()]
        parts.append("<preferences>\n" + "\n".join(plines) + "\n</preferences>")

    # Active projects only — done projects are not useful context for chat
    try:
        projects = json.loads(_proj_raw) if _proj_raw else []
    except Exception:
        projects = []
    active_projects = [
        p for p in projects if isinstance(p, dict) and p.get("status") != "done"
    ]
    if active_projects:
        # Tasks and projects share one list: a due date is what distinguishes them, so the
        # model reads an "échéance" or it doesn't — nothing to classify on its own.
        plines = [
            f"- {p.get('name', 'sans nom')}"
            + (f" (échéance : {p['due_at'][:10]})" if p.get("due_at") else "")
            for p in active_projects
        ]
        parts.append(
            "<projets_et_taches>\n"
            "[Exhaustif — absent = clôturé. Une échéance = à faire pour cette date.]\n"
            + "\n".join(plines)
            + "\n</projets_et_taches>"
        )

    # Injection etat emotionnel
    _emotion_lines = _es.render_prompt_lines()
    if _emotion_lines:
        parts.append(
            "<etat_emotionnel_jarvis>\n"
            + "\n".join(f"- {l}" for l in _emotion_lines)
            + "\n</etat_emotionnel_jarvis>"
        )

    # Self identity — apprentissages de Jarvis (guide interne, pas des faits sur l'utilisateur)
    #
    # Réinjection par RÉCENCE, en attendant que le ciblage soit tenable. Mesuré le
    # 20/08/2026 (RESULTATS.md), trois mécanismes de tri ont échoué : le recouvrement
    # lexical sur le topic (0 correspondance sur 8 questions), l'embedding sur un topic
    # libre (vecteurs attracteurs — « securite abattage arbre frelon » attirait aussi bien
    # l'effacement qu'une commande docker), et l'embedding sur le constat lui-même (le
    # constat est en méta, le message en concret : deux registres incomparables).
    #
    # Ce qui rend la récence acceptable : dans eval_reuse.py, la condition « hors-sujet »
    # a marqué EXACTEMENT comme l'absence de bloc (6 et 6). Un apprentissage bien formé
    # mais hors sujet ne nuit pas — il coûte des tokens. Le gain du ciblage serait donc en
    # tokens, pas en qualité, ce qui ne justifie pas d'expédier un tri qui se trompe.
    if self_mem.get("learnings"):
        retenus = self_mem["learnings"][-LEARNINGS_MAX_INJECTED:]
        parts.append(
            "<apprentissages_jarvis>\n"
            + "\n".join(f"- {ln['text']}" for ln in retenus)
            + "\n</apprentissages_jarvis>"
        )

    if self_mem.get("self_notes"):
        recent_notes = self_mem["self_notes"][-5:]
        _notes_lines = [f"- {n['text']}" for n in recent_notes if n.get("text")]
        if _notes_lines:
            parts.append(
                "<notes_jarvis>\n" + "\n".join(_notes_lines) + "\n</notes_jarvis>"
            )

    # User Timeline — served from pipeline cache hit [5]; fallback to Qdrant on miss
    if _timeline_cached:
        try:
            timeline = json.loads(_timeline_cached)[:7]
        except Exception:
            timeline = get_user_timeline(user_code, limit=7)
    else:
        timeline = get_user_timeline(user_code, limit=7)
    if timeline:
        plines = [
            f"({rel_time_fr(event['timestamp'])}) {event['text']}" for event in timeline
        ]
        parts.append(
            "<frise_chronologique>\n" + "\n".join(plines) + "\n</frise_chronologique>"
        )

    # Tomorrow suggestions — written by nightly review, consumed today.
    # Skipped for pure utility intents (weather/calendar/gmail) — irrelevant noise.
    if include_suggestions:
        try:
            suggestions = json.loads(_sugg_raw) if _sugg_raw else []
        except Exception:
            suggestions = []
        if suggestions:
            plines = [f"- {s}" for s in suggestions]
            parts.append(
                "<sujets_a_aborder>\n" + "\n".join(plines) + "\n</sujets_a_aborder>"
            )

    # User relation — always injected so every conversation has a tonal directive.
    # self_mem is already loaded at the top of this function (no extra I/O).
    _default_rel = {
        "affinity": 0.5,
        "interaction_style": "direct",
        "average_interaction_mood": "measured",
    }
    _STYLE_FR = {
        "direct": "direct",
        "gentle": "doux",
        "formal": "formel",
        "playful": "joueur",
    }
    _REL_MOOD_FR = {
        "warm": "chaleureux",
        "enthusiastic": "enthousiaste",
        "measured": "posé",
        "playful": "joueur",
        "professional": "professionnel",
    }
    rel = {**_default_rel, **self_mem.get("user_relations", {}).get(user_code, {})}
    _aff = rel["affinity"]
    _aff_label = (
        "forte"
        if _aff >= 0.8
        else "bonne"
        if _aff >= 0.6
        else "modérée"
        if _aff >= 0.4
        else "faible"
    )
    parts.append(
        f"<relation_avec_utilisateur>\n"
        f"- Affinité : {_aff_label}\n"
        f"- Style de communication préféré : {_STYLE_FR.get(rel['interaction_style'], rel['interaction_style'])}\n"
        f"- Humeur moyenne des échanges : {_REL_MOOD_FR.get(rel['average_interaction_mood'], rel['average_interaction_mood'])}\n"
        f"</relation_avec_utilisateur>"
    )

    return "\n\n".join(parts) if parts else ""

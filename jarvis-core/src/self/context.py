"""Collecte du contexte de réflexion + construction des prompts + appels LLM.

Phase 1 (global) : santé services/mémoire, activité, lacunes, vitals, incidents, CVE.
Phase 2 (par utilisateur) : profil, activité, relation, disponibilité push.
Formatage des sections et appels aux deux prompts de réflexion (global / user).
"""

import json
import time
from collections import Counter
from datetime import datetime, timezone

import httpx
import numpy as np
from config import (
    BRIEFING_TIMEZONE,
    DEFAULT_TEMP,
    MAX_TOKENS_THINK_MEDIUM,
    PRIMARY_API_KEY,
    PRIMARY_API_URL,
    PRIMARY_MODEL,
    REASONING_API_KEY,
    REASONING_API_URL,
    REASONING_MODEL,
    THINKING_BUDGET_MEDIUM,
    USER_CODES,
    USER_TIMEZONES,
    llm_timeout,
)
from helpers import (
    call_llm_async_bg,
    extract_llm_json,
    fmt_now_fr,
    get_logger,
    get_qdrant,
    get_redis,
)
import emotional_state
from memory import get_self_memory
from prompts import get_prompt

from .proposals import list_pending_proposals
from .state import (
    _DEVICE_TOKEN_PREFIX,
    _KNOWLEDGE_GAPS_KEY,
    _PUSH_COOLDOWN_PREFIX,
    _REFINE_COOLDOWN_PREFIX,
    _extract_behavioral_patterns,
    get_last_reflection,
    slug_de_sujet,
)

logger = get_logger("jarvis-self")


# ══════════════════════════════════════════════════
#  CONTEXT GATHERING
# ══════════════════════════════════════════════════


def _check_service_health() -> dict:
    """Quick liveness check on Redis, Qdrant, OpenAI."""
    health = {}
    # Redis
    try:
        get_redis().ping()
        health["redis"] = "ok"
    except Exception:
        health["redis"] = "unreachable"

    # Qdrant
    try:
        get_qdrant().get_collections()
        health["qdrant"] = "ok"
    except Exception:
        health["qdrant"] = "unreachable"

    # Primary LLM — local: check model files exist; remote: ping /models endpoint.
    try:
        from config import LLM_LOCAL
        if LLM_LOCAL:
            import os as _os
            model_dir = _os.path.join("/opt/jarvis/models/hub", PRIMARY_MODEL.replace("/", "--", 1).replace("/", "--"))
            # HuggingFace cache layout: models--org--name
            hf_dir = _os.path.join(
                "/opt/jarvis/models/hub",
                "models--" + PRIMARY_MODEL.replace("/", "--"),
            )
            health["llm"] = "ok" if (_os.path.isdir(hf_dir) or _os.path.isdir(model_dir)) else "model_missing"
        else:
            r = httpx.get(
                f"{PRIMARY_API_URL}/models",
                headers={"Authorization": f"Bearer {PRIMARY_API_KEY}"},
                timeout=5,
            )
            health["llm"] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
    except Exception:
        health["llm"] = "unreachable"

    return health


def _check_memory_health() -> dict:
    """
    Inspect episodic memory health for all users.

    Returns per-user stats:
      - episodic_count   : total episodic points in Qdrant
      - last_episodic    : ISO date of most recent episodic point (or None)
      - days_since       : days since last episodic storage (or None)
      - null_summary_7d  : conversations with no memory_summary in last 7 days
      - total_7d         : total conversations logged in last 7 days
      - null_rate_7d     : null_summary_7d / total_7d (0.0–1.0)
      - norm_anomalies   : number of non-unit vectors in sample of 30 most recent
    """
    from config import QDRANT_MEMORY_COLLECTION

    qdrant = get_qdrant()
    r = get_redis()
    now = time.time()
    cutoff_7d = now - 7 * 86400
    result: dict[str, dict] = {}

    for user_code in USER_CODES:
        stats: dict = {}

        # ── Qdrant episodic count + last timestamp ────────────────────────
        #
        # Le compte vient de `count(exact=True)` et non plus de la longueur d'un scroll
        # plafonné à 500 : au-delà de 500 points, le chiffre rendu au modèle sous le nom de
        # « total des points épisodiques » était simplement faux, et plafonné.
        #
        # La sonde de normes, elle, ne rapatrie plus que les 30 points les plus récents au
        # lieu de 500 avec leurs vecteurs — soit 500 × 1024 flottants par utilisateur et par
        # passage. `order_by` est possible depuis la correction de l'index `timestamp`
        # (déclaré `integer` pour des valeurs flottantes, donc vide : voir main.py).
        _filtre = {
            "must": [
                {"key": "user_code", "match": {"value": user_code}},
                {"key": "memory_type", "match": {"value": "episodic"}},
            ]
        }
        try:
            stats["episodic_count"] = qdrant.count(
                collection_name=QDRANT_MEMORY_COLLECTION,
                count_filter=_filtre,
                exact=True,
            ).count

            points, _ = qdrant.scroll(
                collection_name=QDRANT_MEMORY_COLLECTION,
                scroll_filter=_filtre,
                order_by={"key": "timestamp", "direction": "desc"},
                limit=30,
                with_payload=True,
                with_vectors=True,
            )

            if points:
                last_ts = max(p.payload.get("timestamp", 0) for p in points)
                stats["last_episodic"] = datetime.fromtimestamp(
                    last_ts, tz=timezone.utc
                ).date().isoformat()
                stats["days_since"] = round((now - last_ts) / 86400, 1)
            else:
                stats["last_episodic"] = None
                stats["days_since"] = None

            # ── Sample norm check (30 most recent) ───────────────────────
            anomalies = 0
            for pt in points:
                if pt.vector:
                    norm = float(np.linalg.norm(pt.vector))
                    if abs(norm - 1.0) > 0.01:
                        anomalies += 1
            stats["norm_anomalies"] = anomalies

        except Exception as exc:
            logger.warning("memory_health Qdrant check failed for %s: %s", user_code, exc)
            stats["episodic_count"] = -1
            stats["last_episodic"] = None
            stats["days_since"] = None
            stats["norm_anomalies"] = -1

        # ── Redis convlog: null_summary rate over last 7 days ────────────
        try:
            raw_entries = r.zrangebyscore(f"convlog:{user_code}", cutoff_7d, "+inf")
            total = len(raw_entries)
            null_count = 0
            for raw in raw_entries:
                try:
                    e = json.loads(raw)
                    if not e.get("memory_summary"):
                        null_count += 1
                except Exception:
                    pass
            stats["null_summary_7d"] = null_count
            stats["total_7d"] = total
            stats["null_rate_7d"] = round(null_count / total, 2) if total else 0.0
        except Exception as exc:
            logger.warning("memory_health Redis check failed for %s: %s", user_code, exc)
            stats["null_summary_7d"] = -1
            stats["total_7d"] = -1
            stats["null_rate_7d"] = -1

        result[user_code] = stats

    return result


def _fmt_memory_health(health: dict) -> str:
    lines = []
    for user_code, s in health.items():
        days = f"{s['days_since']}j" if s.get("days_since") is not None else "jamais"
        norm_warn = f" ⚠ {s['norm_anomalies']} vecteurs non-normalisés" if s.get("norm_anomalies", 0) > 0 else ""
        null_pct = f"{int(s.get('null_rate_7d', 0) * 100)}%"
        lines.append(
            f"  {user_code}: épisodique={s.get('episodic_count','?')} pts"
            f", dernier={s.get('last_episodic') or 'jamais'} ({days})"
            f", null_summary_7j={s.get('null_summary_7d','?')}/{s.get('total_7d','?')} ({null_pct})"
            f"{norm_warn}"
        )
    return "\n".join(lines) if lines else "  (aucun utilisateur)"


def _get_user_activity(hours: int = 24) -> dict:
    """
    Count recent conversations per user by scanning their episodic Redis log.
    Returns {user_code: {name, conversations, topics}}.
    """
    r = get_redis()
    cutoff = time.time() - hours * 3600
    activity = {}

    for code, name in USER_CODES.items():
        entries_raw = r.zrangebyscore(f"convlog:{code}", cutoff, "+inf")
        topics: set[str] = set()
        sat: Counter = Counter()
        for raw in entries_raw:
            try:
                e = json.loads(raw)
                topics.update(e.get("topics", []))
                s = e.get("satisfaction", "unknown")
                if s in ("positive", "negative"):
                    sat[s] += 1
            except Exception:
                pass
        activity[code] = {
            "name": name,
            "conversations": len(entries_raw),
            "topics": sorted(topics)[:8],
            "satisfaction": dict(sat),
        }

    return activity


# Au-delà, une lacune n'est plus une observation, c'est un vestige. Mesuré le 21/08/2026 :
# les trois lacunes en base dataient des 08/04, 22/04 et 04/06 — injectées dans chaque
# cycle depuis, sans jamais expirer, et décrivant pour deux d'entre elles le comportement
# de boucle de Jarvis lui-même, c'est-à-dire la pathologie que le déplacement de
# flag_knowledge_gap vers la revue nocturne devait justement supprimer.
_GAP_MAX_AGE_DAYS = 60


def _purger_lacunes_perimees(r) -> int:
    """Retire du zset les lacunes trop vieilles.

    Le balayage des compteurs a disparu avec les compteurs eux-mêmes (21/08/2026) : le
    seuil de récurrence qu'ils alimentaient n'était lu nulle part, et le slug tronqué à
    40 caractères sans rapprochement sémantique les rendait de toute façon incapables de
    mesurer une récurrence. refine_prompt est désormais borné en débit.
    """
    limite = time.time() - _GAP_MAX_AGE_DAYS * 86400
    perimees = r.zrangebyscore(_KNOWLEDGE_GAPS_KEY, "-inf", limite)
    if perimees:
        r.zremrangebyscore(_KNOWLEDGE_GAPS_KEY, "-inf", limite)
        logger.info(
            "Lacunes : %d entrée(s) purgée(s) (au-delà de %d jours)",
            len(perimees), _GAP_MAX_AGE_DAYS,
        )
    return len(perimees)



def _lacunes(n: int = 5) -> tuple[list[str], int]:
    """Les lacunes récentes, et combien d'entre elles sont ACTIONNABLES.

    Rend `(libellés, nombre d'actionnables)`. Une lacune est actionnable si son sujet n'a
    ni proposition en attente ni sommeil en cours — c'est-à-dire s'il reste quelque chose
    à faire dessus. Ce compte remplace `gap_max_count` dans le garde de matière de
    l'engine : l'ancien lisait un compteur d'occurrences qui n'était jamais décrémenté et
    laissait donc le garde ouvert à vie.

    Le libellé porte désormais la DATE plutôt qu'un décompte. Le décompte a disparu avec
    les compteurs (21/08/2026) ; la date, elle, dit au modèle si l'observation est fraîche,
    ce qui est la seule chose qu'il puisse en faire.
    """
    r = get_redis()
    _purger_lacunes_perimees(r)

    # Sujets déjà couverts par une proposition en attente. Le marquage était annoncé par
    # la docstring depuis l'origine et n'avait jamais été écrit : le modèle voyait une
    # lacune ouverte là où une proposition dormait déjà, et relançait dessus.
    en_attente = {
        slug_de_sujet(p.get("topic", "")) for p in list_pending_proposals()
    } - {""}

    raw = r.zrevrange(_KNOWLEDGE_GAPS_KEY, 0, n * 3 - 1)  # fetch extra to survive dedup
    seen_slugs: set[str] = set()
    results: list[str] = []
    actionnables = 0
    for item in raw:
        date = ""
        try:
            d = json.loads(item)
            topic = d.get("topic", item)
            date = (d.get("date") or "")[:10]
        except Exception:
            topic = item
        slug = slug_de_sujet(topic)
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)

        label = f"{topic}" + (f" (signalée le {date})" if date else "")
        if slug in en_attente:
            label += " — PROPOSITION DÉJÀ EN ATTENTE, ne pas reproposer"
        elif r.ttl(f"{_REFINE_COOLDOWN_PREFIX}:{slug}") > 0:
            label += " — sujet déjà tranché récemment, ne pas reproposer"
        else:
            actionnables += 1

        results.append(label)
        if len(results) >= n:
            break
    return results, actionnables


def _fmt_pending_proposals() -> str:
    proposals = list_pending_proposals()
    if not proposals:
        return "aucune"
    return "; ".join(
        f"{p['id']} — {p['prompt_name']} (sujet: {p['topic']})" for p in proposals
    )


def gather_global_context() -> dict:
    """Assemble global context for Phase 1 (Jarvis self-state, no user profiles)."""
    self_data = get_self_memory()
    health = _check_service_health()
    activity = _get_user_activity(24)
    gaps, gaps_actionnables = _lacunes(5)
    last_ref = get_last_reflection()

    # État de disparition + incidents : la réflexion doit VOIR ce qu'elle vient de
    # consolider, sinon elle stocke sans jamais commenter. Snapshot complet ici (pas la
    # version saillante réservée au bloc de tour). Isolé : indisponible ≠ bloquant.
    try:
        from vitals import get_vitals, recent_incidents
        vitals_snapshot = get_vitals()
        incidents = recent_incidents(30)
    except Exception as exc:
        logger.debug("gather_global_context: vitals indisponible (%s)", exc)
        vitals_snapshot, incidents = {}, []

    # Liste actionnable des paquets vulnérables (quoi mettre à jour, vers quelle version) :
    # la réflexion est la boucle de maintenance, elle peut en tirer une note ou une alerte.
    try:
        from cve import render_advice
        cve_conseil = render_advice(critical_only=True, limit=20)
    except Exception as exc:
        logger.debug("gather_global_context: cve indisponible (%s)", exc)
        cve_conseil = ""

    return {
        "timestamp": fmt_now_fr(BRIEFING_TIMEZONE),
        "identity": self_data.get("identity", {}),
        "goals": self_data.get("goals", []),
        "current_focus": self_data.get("current_focus", ""),
        "health": health,
        "memory_health": _check_memory_health(),
        "vitals": vitals_snapshot,
        "incidents": incidents,
        "cve_conseil": cve_conseil,
        "user_activity": activity,
        "knowledge_gaps": gaps,
        # Nombre de lacunes sur lesquelles il reste quelque chose à faire, pour le garde
        # mécanique de `run_self_reflection`. Rendu séparément parce que `knowledge_gaps`
        # ne porte que des libellés d'affichage : les analyser à la regex reviendrait à
        # décider sur une chaîne de présentation.
        "gaps_actionnables": gaps_actionnables,
        "pending_proposals": _fmt_pending_proposals(),
        "last_reflection": last_ref,
        "reflection_count": self_data.get("reflection_count", 0),
        "user_relations": self_data.get("user_relations", {}),
        "behavioral_patterns": _extract_behavioral_patterns(20),
        "emotional_state": emotional_state.get_state(),
        # Les neuf axes remplis remplacent l'ancienne liste `self_notes`, retirée le
        # 21/08/2026 : la connaissance de soi n'a plus qu'un foyer, et c'est celui-ci.
        "introspection": self_data.get("self_introspection", {}),
        "opinions": self_data.get("opinions", [])[-5:],
    }


def gather_user_context(user_code: str) -> dict:
    """Assemble per-user context for Phase 2 (single user's profile and activity)."""
    from memory import get_user_profile

    user_name = USER_CODES.get(user_code, user_code)
    full_activity = _get_user_activity(24)
    user_activity = full_activity.get(user_code, {})
    self_data = get_self_memory()
    user_relation = self_data.get("user_relations", {}).get(user_code, {})
    profile = {k: v for k, v in get_user_profile(user_code).items() if v}

    r = get_redis()
    tz_name = USER_TIMEZONES.get(user_code, "Europe/Paris")
    local_time = fmt_now_fr(tz_name)
    has_push = bool(r.exists(f"{_DEVICE_TOKEN_PREFIX}:{user_code}"))

    cooldown_key = f"{_PUSH_COOLDOWN_PREFIX}:{user_code}"
    cooldown_ttl = r.ttl(
        cooldown_key
    )  # -2 = key absent, -1 = no TTL, >0 = seconds remaining
    if cooldown_ttl > 0:
        h, m = divmod(cooldown_ttl // 60, 60)
        push_cooldown_str = (
            f"actif encore {h}h{m:02d}" if h else f"actif encore {m} min"
        )
    else:
        push_cooldown_str = "expiré (push disponible)"

    return {
        "user_code": user_code,
        "user_name": user_name,
        "profile": profile,
        "has_push": has_push,
        "push_cooldown_str": push_cooldown_str,
        "local_time": local_time,
        "user_activity": user_activity,
        "user_relation": user_relation,
    }


# ══════════════════════════════════════════════════
#  LLM REFLECTION CALL
# ══════════════════════════════════════════════════

# ── Helpers ───────────────────────────────────────────────────────────────


def _fmt_goals(goals: list[dict]) -> str:
    return "\n".join(
        f"  G{i + 1}. {g.get('label', '?')}: {g.get('description', '')}"
        for i, g in enumerate(goals)
    )


def _fmt_activity(activity: dict) -> str:
    lines = []
    for code, info in activity.items():
        topics = ", ".join(info["topics"]) or "aucun"
        sat = info.get("satisfaction", {})
        sat_parts = []
        if sat.get("positive"):
            sat_parts.append(f"+{sat['positive']}")
        if sat.get("negative"):
            sat_parts.append(f"-{sat['negative']}")
        sat_str = f" | satisfaction: {' '.join(sat_parts)}" if sat_parts else ""
        lines.append(
            f"  {info['name']} ({code}): {info['conversations']} conversations | sujets: {topics}{sat_str}"
        )
    return "\n".join(lines) or "  No activity."


def _fmt_introspection(axes: dict) -> str:
    """Les axes REMPLIS seulement — les vides n'apprennent rien à la réflexion."""
    lignes = [f"  {axe} : {texte}" for axe, texte in (axes or {}).items() if texte]
    return "\n".join(lignes) or "  aucun axe encore renseigné"


def _fmt_opinions(opinions: list[dict]) -> str:
    if not opinions:
        return "  aucune opinion"
    return "\n".join(
        f"  {o.get('topic', '?')} : {o.get('opinion', '')}" for o in opinions
    )


def _fmt_vitals(v: dict) -> str:
    """Snapshot complet, à plat et sans valence : la réflexion lit tous les faits, pas
    seulement les saillants du bloc de tour."""
    if not v:
        return "  indisponible"
    return "\n".join(f"  - {k} : {val}" for k, val in v.items())


def _fmt_incidents(items: list[dict]) -> str:
    if not items:
        return "  aucun incident récent"
    return "\n".join(
        f"  - [{it.get('iso', '')[:10]}] {it.get('kind', '?')} "
        f"({it.get('severity', '')}) : {it.get('detail', '')}"
        for it in items[-10:]
    )


def _fmt_previous_steps(steps: list[dict] | None) -> str:
    if not steps:
        return "  aucune (première itération)"
    return "\n".join(
        f"  {s['iteration']}. {s['action']} → {s['outcome']}" for s in steps
    )


async def _call_global_reflection_llm(
    context: dict, previous_steps: list[dict] | None = None
) -> dict | None:
    """Phase 1 — global self-reflection (Jarvis state, no user profiles)."""
    bp = context.get("behavioral_patterns", [])
    behavioral_patterns = (
        "\n".join(f"  • {p}" for p in bp) if bp else "  aucun pattern identifié"
    )

    prompt = get_prompt("REFLECTION_PROMPT").format(
        timestamp=context["timestamp"],
        identity=json.dumps(context["identity"], ensure_ascii=False),
        goals=_fmt_goals(context["goals"]),
        health=json.dumps(context["health"]),
        memory_health=_fmt_memory_health(context.get("memory_health", {})),
        vitals=_fmt_vitals(context.get("vitals", {})),
        incidents=_fmt_incidents(context.get("incidents", [])),
        vulnerabilites=context.get("cve_conseil") or "  aucune connue",
        activity=_fmt_activity(context["user_activity"]),
        gaps=", ".join(context["knowledge_gaps"]) or "aucune",
        pending_proposals=context["pending_proposals"],
        last_reflection=json.dumps(
            {k: v for k, v in context["last_reflection"].items() if k != "steps"},
            ensure_ascii=False,
        )
        if context["last_reflection"]
        else "aucune",
        behavioral_patterns=behavioral_patterns,
        emotional_state=json.dumps(
            context.get("emotional_state", {}), ensure_ascii=False
        ),
        introspection=_fmt_introspection(context.get("introspection", {})),
        opinions=_fmt_opinions(context.get("opinions", [])),
        user_relations=json.dumps(context["user_relations"], ensure_ascii=False),
        previous_steps=_fmt_previous_steps(previous_steps),
    )

    try:
        content = await call_llm_async_bg(
            [
                {"role": "system", "content": get_prompt("REFLECTION_SYSTEM")},
                {"role": "user", "content": prompt},
            ],
            model=REASONING_MODEL,
            api_url=REASONING_API_URL,
            api_key=REASONING_API_KEY,
            temperature=DEFAULT_TEMP,
            max_tokens=MAX_TOKENS_THINK_MEDIUM,
            thinking_budget=THINKING_BUDGET_MEDIUM,
            json_response=True,
            no_think=False,
            timeout=llm_timeout(MAX_TOKENS_THINK_MEDIUM),
        )
        return extract_llm_json(content)
    except ValueError as exc:
        logger.error(
            "Global reflection LLM failed: %s — truncated or malformed JSON",
            type(exc).__name__,
            exc_info=True,
        )
        return None
    except Exception as exc:
        logger.error(
            "Global reflection LLM failed: %s", type(exc).__name__, exc_info=True
        )
        return None


def _fmt_single_user_profile(profile: dict) -> str:
    """Format a single user's profile dict for the per-user reflection prompt."""
    if not profile:
        return "  (aucun profil)"
    lines = []
    for k, v in list(profile.items())[:20]:
        lines.append(f"  {k} = {str(v)[:80]}")
    return "\n".join(lines)


async def _call_user_reflection_llm(
    global_ctx: dict,
    user_ctx: dict,
    previous_steps: list[dict] | None = None,
) -> dict | None:
    """Phase 2 — per-user reflection (single user's profile, activity, relation)."""
    user_code = user_ctx["user_code"]
    user_activity_entry = user_ctx["user_activity"]
    # Format as a single-user activity line using existing helper
    activity_str = (
        _fmt_activity({user_code: user_activity_entry} if user_activity_entry else {})
        or "  Aucune activité récente."
    )

    push_status = "disponible ✓" if user_ctx["has_push"] else "indisponible"

    prompt = get_prompt("REFLECTION_USER_PROMPT").format(
        timestamp=global_ctx["timestamp"],
        user_name=user_ctx["user_name"],
        user_code=user_code,
        local_time=user_ctx["local_time"],
        push_status=push_status,
        user_activity=activity_str,
        user_relation=json.dumps(user_ctx["user_relation"], ensure_ascii=False),
        user_profile=_fmt_single_user_profile(user_ctx["profile"]),
        previous_steps=_fmt_previous_steps(previous_steps),
    )

    messages = [
        {"role": "system", "content": get_prompt("REFLECTION_USER_SYSTEM")},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(2):
        try:
            # Le vestige du workaround DWQ (Qwen3-30B-A3B-4bit-DWQ-0508, qui sortait du
            # bloc think sans émettre </think>) a été retiré le 22/08/2026 : il annonçait
            # sa propre désactivation depuis la migration vers Qwen3.6 tout en laissant
            # `no_think=True` en place.
            content = await call_llm_async_bg(
                messages,
                model=REASONING_MODEL,
                api_url=REASONING_API_URL,
                api_key=REASONING_API_KEY,
                temperature=DEFAULT_TEMP,
                max_tokens=MAX_TOKENS_THINK_MEDIUM,
                thinking_budget=THINKING_BUDGET_MEDIUM,
                json_response=True,
                no_think=False,
                timeout=llm_timeout(MAX_TOKENS_THINK_MEDIUM),
            )
            return extract_llm_json(content)
        except ValueError as exc:
            if attempt == 0:
                logger.warning(
                    "User reflection LLM malformed JSON (%s), retrying — %s",
                    user_code,
                    exc,
                )
                continue
            logger.error(
                "User reflection LLM failed (%s) after retry: %s", user_code, exc
            )
            return None
        except Exception as exc:
            logger.error(
                "User reflection LLM failed (%s): %s",
                user_code,
                type(exc).__name__,
                exc_info=True,
            )
            return None
    return None

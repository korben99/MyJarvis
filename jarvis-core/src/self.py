"""
Jarvis Proto-Self
=================
Gives Jarvis a reason to exist and an autonomous reflection loop.

Architecture:
  - Identity & goals  : loaded from jarvis-self.json (read-only goals)
  - Reflection loop   : scheduled every REFLECTION_INTERVAL_HOURS
  - Action catalog    : fixed set of bounded, safe actions
  - Reflection log    : Redis sorted set  jarvis:self:reflection_log  (capped 30)
  - Notification guard: Redis key         jarvis:self:notif:{user_code}:{date}
  - Push system       : Redis list        jarvis:push:pending:{user_code}  (polled by iOS)

Reflection cycle (two-phase):
  Phase 1 — global (Jarvis self-state):
    1. gather_global_context()      — health, activity, gaps, self-notes
    2. _call_global_reflection_llm() — returns {focus, action, reason, params}
    3. execute_action()             — dispatches to global action catalog
  Phase 2 — per-user (one chain per user):
    1. gather_user_context(code)    — single user's profile, activity, push status
    2. _call_user_reflection_llm()  — returns {focus, action, reason, params}
    3. execute_action()             — dispatches to user action catalog
  Final: log_reflection() + generate_proactive_push() per user

Action catalog (v1 — grows in future versions):
  nothing                  — explicit no-op, with reason
  store_insight            — save a learning to a user's memory
  flag_knowledge_gap       — log a topic Jarvis answered poorly
  send_notification        — send a relevant Gmail to one user (1/user/day max)
  queue_push               — queue an iOS push notification for one user
  update_self_note         — write an observation to self_notes in jarvis-self.json
  consolidate_memory       — trigger memory compression for a user
  check_health             — verify all services, log status
  update_trade_threshold   — update threshold_high / threshold_low for one portfolio position
"""

import asyncio
import json
import os
import re
import time
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone

import httpx
import pytz
from config import (
    BRIEFING_TIMEZONE,
    GROWTH_LOG_MAX_ENTRIES,
    MAX_CHAIN_ITERATIONS,
    PRIMARY_API_KEY,
    PRIMARY_API_URL,
    PRIMARY_MODEL,
    PRIMARY_TIMEOUT,
    PROMPT_DATA_DIR,
    REASONING_API_KEY,
    REASONING_API_URL,
    REASONING_MODEL,
    REASONING_TIMEOUT,
    THINKING_BUDGET_TOKENS,

    USER_ADMINS,
    USER_CODES,
    USER_EMAILS,
    USER_TIMEZONES,
    USERS,
)
from google_services import is_google_available, send_gmail_message
from helpers import (
    call_llm,
    call_llm_async,
    extract_llm_json,
    fmt_now_fr,
    get_logger,
    get_qdrant,
    get_redis,
)
from memory import (
    append_conversation_message,
    atomic_json_write,
    get_emotional_state,
    get_self_memory,
    save_self_memory,
    self_memory_lock,
    store_autobiographical_event,
)
from trade_keys import idx_key, pos_key

logger = get_logger("jarvis-self")

# ── Namespace guards for correct_profile ─────────────────────────────────
# Each entry describes one protected domain: which key prefixes belong to it
# and which terms must appear in the value for the write to be allowed.
# Add a new domain here without touching _action_correct_profile logic.
# extra_check: optional callable(value_lower) -> bool for non-keyword rules.
_NS_GUARDS: list[dict] = [
    {
        "name": "financial",
        "key_prefixes": frozenset({
            "placement", "capital", "per", "pea", "livret_a",
            "investissement", "epargne",
        }),
        "required_terms": frozenset({
            "€", "$", "%", "fonds", "fond", "etf", "action", "obligation",
            "livret", "pea", "per", "scpi", "crypto", "bourse", "placement",
            "investissement", "epargne", "portefeuille", "rendement", "taux",
            "assurance", "virement", "depot", "retrait", "titre",
        }),
        "extra_check": lambda v: any(c.isdigit() for c in v),
        "error_hint": "financial context (amount, fund name, asset type, %, €, etc.)",
    },
    {
        "name": "travel",
        "key_prefixes": frozenset({"travel_plans", "travel_preference", "voyages_prevus"}),
        "required_terms": frozenset({
            "voyage", "travel", "trip", "vacances", "destination", "hotel", "vol",
            "billet", "trajet", "sejour", "partir", "avion", "train", "city", "ville",
        }),
        "extra_check": None,
        "error_hint": "travel context",
    },
]

# ── Redis keys ────────────────────────────────────────────────────────────
_REFLECTION_LOG_KEY = "jarvis:self:reflection_log"
_REFLECTION_LOG_MAX = 30
_KNOWLEDGE_GAPS_KEY = "jarvis:self:knowledge_gaps"
_GAP_COUNTS_KEY = "jarvis:self:gap_counts"  # hash: slug → count
_NOTIF_KEY_PREFIX = "jarvis:self:notif"
_NOTIF_TTL = 86400  # 24h — one notification per user per day
_PUSH_PENDING_PREFIX = "jarvis:push:pending"  # list of pending push messages per user
_DEVICE_TOKEN_PREFIX = (
    "jarvis:device:token"  # device token per user (set by /device/register)
)
_PUSH_COOLDOWN_PREFIX = (
    "jarvis:push:cooldown"  # prevent push flooding (1 push per 2h per user)
)
_PUSH_COOLDOWN_TTL = 72000  # 20h


# ── Redis / Qdrant singletons ─────────────────────────────────────────────


def get_goals() -> list[dict]:
    return get_self_memory().get("goals", [])


def get_current_focus() -> str:
    return get_self_memory().get("current_focus", "")


_DEFAULT_RELATION = {
    "affinity": 0.5,
    "interaction_style": "direct",
    "average_interaction_mood": "measured",
}

_VALID_STYLES = {"direct", "gentle", "formal", "playful"}
_VALID_MOODS = {"warm", "enthusiastic", "measured", "playful", "professional"}


def get_user_relation(user_code: str) -> dict:
    """Return the current relation dict for a user (with defaults if missing)."""
    relations = get_self_memory().get("user_relations", {})
    return {**_DEFAULT_RELATION, **relations.get(user_code, {})}


def _update_self_fields(**fields) -> None:
    """Update specific top-level fields in jarvis-self.json under the shared lock."""
    with self_memory_lock:
        data = get_self_memory()
        data.update(fields)
        save_self_memory(data)


def _upsert_opinion_inplace(data: dict, topic: str, opinion: str, date: str) -> None:
    """Upsert an opinion into an already-loaded self-memory dict (no lock — caller holds it)."""
    topic = topic.strip().lower()
    opinions = data.setdefault("opinions", [])
    existing = next((o for o in opinions if o["topic"] == topic), None)
    if existing:
        existing["opinion"] = opinion
        existing["updated"] = date
    else:
        opinions.append({"topic": topic, "opinion": opinion, "created": date})
    data["opinions"] = opinions[-50:]


def add_self_opinion(topic: str, opinion: str) -> None:
    """Add or update a Jarvis opinion. Thread-safe — acquires self_memory_lock."""
    date = datetime.now(timezone.utc).isoformat()
    with self_memory_lock:
        data = get_self_memory()
        _upsert_opinion_inplace(data, topic, opinion, date)
        save_self_memory(data)
    logger.info("Opinion upserted: %s", topic)


# ══════════════════════════════════════════════════
#  REFLECTION LOG  (Redis sorted set)
# ══════════════════════════════════════════════════


def log_reflection(entry: dict) -> None:
    """Append a reflection result to the Redis log (capped at _REFLECTION_LOG_MAX)."""
    r = get_redis()
    score = time.time()
    r.zadd(_REFLECTION_LOG_KEY, {json.dumps(entry, ensure_ascii=False): score})
    # Trim to most recent entries
    excess = r.zcard(_REFLECTION_LOG_KEY) - _REFLECTION_LOG_MAX
    if excess > 0:
        r.zremrangebyrank(_REFLECTION_LOG_KEY, 0, excess - 1)


def get_reflection_log(n: int = 10) -> list[dict]:
    """Return the last n reflection entries, most recent first."""
    r = get_redis()
    raw = r.zrevrange(_REFLECTION_LOG_KEY, 0, n - 1)
    results = []
    for item in raw:
        try:
            results.append(json.loads(item))
        except json.JSONDecodeError:
            pass
    return results


def get_last_reflection() -> dict | None:
    entries = get_reflection_log(1)
    return entries[0] if entries else None


def _extract_behavioral_patterns(n: int = 20) -> list[str]:
    """Derive up to 5 recurring behavioral patterns from the last n reflection entries.

    Fully deterministic — no LLM call. Three signals are analysed:
      1. Action frequency: which actions dominate (≥ 20 % of cycles).
      2. Time-of-day clustering for "nothing" choices.
      3. Recurring keywords in focus fields (word seen ≥ 3 times).
    """
    from collections import Counter

    logs = get_reflection_log(n)
    if not logs:
        return []

    total = len(logs)
    patterns: list[str] = []

    # 1. Action frequency
    action_counts: Counter = Counter(e.get("action", "unknown") for e in logs)
    for action, count in action_counts.most_common(3):
        pct = round(count / total * 100)
        if pct >= 20:
            patterns.append(
                f"action « {action} » choisie dans {pct}% des cycles ({count}/{total})"
            )

    # 2. "nothing" time-of-day clustering
    nothing_hours = []
    for e in logs:
        if e.get("action") == "nothing" and e.get("timestamp"):
            try:
                nothing_hours.append(datetime.fromisoformat(e["timestamp"]).hour)
            except Exception:
                pass
    if len(nothing_hours) >= 3:
        avg_h = sum(nothing_hours) / len(nothing_hours)
        if avg_h >= 20 or avg_h <= 6:
            patterns.append(
                f"tend à ne rien faire la nuit/soirée (heure moyenne: {avg_h:.0f}h)"
            )

    # 3. Recurring keywords in focus fields
    word_counts: Counter = Counter()
    for e in logs:
        for word in e.get("focus", "").lower().split():
            if len(word) > 5:
                word_counts[word] += 1
    top_words = [w for w, c in word_counts.most_common(5) if c >= 3]
    if top_words:
        patterns.append(f"focus récurrents : {', '.join(top_words[:3])}")

    return patterns[:5]


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

    # Primary LLM (lightweight models list call)
    try:
        r = httpx.get(
            f"{PRIMARY_API_URL}/models",
            headers={"Authorization": f"Bearer {PRIMARY_API_KEY}"},
            timeout=5,
        )
        health["llm"] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
    except Exception:
        health["llm"] = "unreachable"

    return health


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


def _get_knowledge_gaps(n: int = 5) -> list[str]:
    """
    Return the most recently flagged knowledge gaps, annotated with their
    cumulative occurrence count so the reflection LLM can decide when to
    trigger a prompt refinement.
    Gaps whose prompt already has a pending proposal are marked so the LLM
    does not waste a cycle trying to re-propose.
    """
    r = get_redis()
    raw = r.zrevrange(_KNOWLEDGE_GAPS_KEY, 0, n - 1)
    counts = r.hgetall(_GAP_COUNTS_KEY)
    results = []
    for item in raw:
        try:
            d = json.loads(item)
            topic = d.get("topic", item)
        except Exception:
            topic = item
        slug = re.sub(r"\s+", "_", topic.lower())[:40]
        # Use actual Redis count (default 0 like _action_refine_prompt, not 1)
        # so the LLM sees the real progress toward REFINE_PROMPT_THRESHOLD.
        count = int(counts.get(slug, 0))
        label = f"{topic} (flaggé ×{count})"
        results.append(label)
    return results


def _fmt_pending_proposals() -> str:
    proposals = list_pending_proposals()
    if not proposals:
        return "none"
    return "; ".join(
        f"{p['id']} — {p['prompt_name']} (topic: {p['topic']})" for p in proposals
    )


def gather_global_context() -> dict:
    """Assemble global context for Phase 1 (Jarvis self-state, no user profiles)."""
    self_data = get_self_memory()
    health = _check_service_health()
    activity = _get_user_activity(24)
    gaps = _get_knowledge_gaps(5)
    last_ref = get_last_reflection()

    return {
        "timestamp": fmt_now_fr(BRIEFING_TIMEZONE),
        "identity": self_data.get("identity", {}),
        "goals": self_data.get("goals", []),
        "current_focus": self_data.get("current_focus", ""),
        "health": health,
        "user_activity": activity,
        "knowledge_gaps": gaps,
        "pending_proposals": _fmt_pending_proposals(),
        "last_reflection": last_ref,
        "reflection_count": self_data.get("reflection_count", 0),
        "user_relations": self_data.get("user_relations", {}),
        "behavioral_patterns": _extract_behavioral_patterns(20),
        "emotional_state": get_emotional_state(),
        "self_notes": self_data.get("self_notes", [])[-5:],
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
    cooldown_ttl = r.ttl(cooldown_key)  # -2 = key absent, -1 = no TTL, >0 = seconds remaining
    if cooldown_ttl > 0:
        h, m = divmod(cooldown_ttl // 60, 60)
        push_cooldown_str = f"actif encore {h}h{m:02d}" if h else f"actif encore {m} min"
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

from prompts import PROMPT_TOKEN_BUDGETS, get_prompt

# ── Helpers ───────────────────────────────────────────────────────────────


def _fmt_goals(goals: list[dict]) -> str:
    return "\n".join(
        f"  G{i + 1}. {g.get('label', '?')}: {g.get('description', '')}"
        for i, g in enumerate(goals)
    )


def _fmt_activity(activity: dict) -> str:
    lines = []
    for code, info in activity.items():
        topics = ", ".join(info["topics"]) or "none"
        sat = info.get("satisfaction", {})
        sat_parts = []
        if sat.get("positive"):
            sat_parts.append(f"+{sat['positive']}")
        if sat.get("negative"):
            sat_parts.append(f"-{sat['negative']}")
        sat_str = f" | satisfaction: {' '.join(sat_parts)}" if sat_parts else ""
        lines.append(
            f"  {info['name']} ({code}): {info['conversations']} conversations | topics: {topics}{sat_str}"
        )
    return "\n".join(lines) or "  No activity."


def _fmt_self_notes(notes: list[dict]) -> str:
    if not notes:
        return "  aucune note"
    return "\n".join(f"  [{n.get('date', '')[:10]}] {n.get('note', '')}" for n in notes)


def _fmt_opinions(opinions: list[dict]) -> str:
    if not opinions:
        return "  aucune opinion"
    return "\n".join(
        f"  {o.get('topic', '?')} : {o.get('opinion', '')}" for o in opinions
    )


def _fmt_user_profiles() -> str:
    """Compact profile dump for all users — passed to the reflection LLM.

    Each user block is clearly delimited so the LLM cannot confuse which
    key belongs to which user_code.  Empty-valued keys are filtered out —
    they are invalid state and should not appear in the reasoning context.
    """
    from memory import get_user_profile

    blocks = []
    for code, name in USER_CODES.items():
        # Filter out empty/None values — stale keys that were never properly cleaned.
        profile = {k: v for k, v in get_user_profile(code).items() if v}
        if not profile:
            continue
        lines = [f"<profil user=\"{name}\" code=\"{code}\">"]
        for k, v in list(profile.items())[:20]:  # cap at 20 keys for token budget
            lines.append(f"  {k} = {str(v)[:80]}")
        lines.append("</profil>")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) or "  No profiles."


def _fmt_push_availability() -> str:
    """Check Redis device tokens and return a push availability summary per user."""
    r = get_redis()
    with_push, without_push = [], []
    for code, name in USER_CODES.items():
        tz_name = USER_TIMEZONES.get(code, "Europe/Paris")
        local_time = fmt_now_fr(tz_name)
        label = f"{name} ({code}) — heure locale : {local_time}"
        if r.exists(f"jarvis:device:token:{code}"):
            with_push.append(label)
        else:
            without_push.append(label)
    lines = []
    if with_push:
        lines.append(
            f"  Push iOS disponible :\n" + "\n".join(f"    • {l}" for l in with_push)
        )
    if without_push:
        lines.append(
            f"  Push iOS indisponible (email ou attente) :\n"
            + "\n".join(f"    • {l}" for l in without_push)
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────

# Actions allowed per phase — LLM cannot hallucinate cross-phase actions
_GLOBAL_ACTIONS = frozenset({
    "nothing", "flag_knowledge_gap", "update_self_note",
    "check_health", "prune_self_memory", "refine_prompt",
})
_USER_ACTIONS = frozenset({
    "nothing", "store_insight", "send_notification", "queue_push",
    "correct_profile", "ask_user", "consolidate_memory", "update_trade_threshold",
})


def _fmt_previous_steps(steps: list[dict] | None) -> str:
    if not steps:
        return "  aucune (première itération)"
    return "\n".join(
        f"  [{s['iteration']}] {s['action']} → {s['outcome']}"
        for s in steps
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
        activity=_fmt_activity(context["user_activity"]),
        gaps=", ".join(context["knowledge_gaps"]) or "none flagged",
        pending_proposals=context["pending_proposals"],
        last_reflection=json.dumps(
            {k: v for k, v in context["last_reflection"].items() if k != "steps"},
            ensure_ascii=False,
        )
        if context["last_reflection"]
        else "none yet",
        behavioral_patterns=behavioral_patterns,
        emotional_state=json.dumps(
            context.get("emotional_state", {}), ensure_ascii=False
        ),
        self_notes=_fmt_self_notes(context.get("self_notes", [])),
        opinions=_fmt_opinions(context.get("opinions", [])),
        user_relations=json.dumps(context["user_relations"], ensure_ascii=False),
        previous_steps=_fmt_previous_steps(previous_steps),
    )

    try:
        content = await call_llm_async(
            [
                {"role": "system", "content": get_prompt("REFLECTION_SYSTEM")},
                {"role": "user", "content": prompt},
            ],
            model=REASONING_MODEL,
            api_url=REASONING_API_URL,
            api_key=REASONING_API_KEY,
            temperature=0.2,
            max_tokens=500,  # action JSON ~150 tok; early-stop kicks in before ceiling
            json_response=True,
            no_think=True,  # structured JSON action selection — thinking adds latency/failure risk
            timeout=120.0,
        )
        return extract_llm_json(content)
    except ValueError as exc:
        logger.error(
            "Global reflection LLM failed: %s — truncated or malformed JSON",
            type(exc).__name__, exc_info=True,
        )
        return None
    except Exception as exc:
        logger.error("Global reflection LLM failed: %s", type(exc).__name__, exc_info=True)
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
    activity_str = _fmt_activity(
        {user_code: user_activity_entry} if user_activity_entry else {}
    ) or "  Aucune activité récente."

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
            content = await call_llm_async(
                messages,
                model=REASONING_MODEL,
                api_url=REASONING_API_URL,
                api_key=REASONING_API_KEY,
                temperature=0.2,
                max_tokens=THINKING_BUDGET_TOKENS + 1500,  # think block + action JSON
                json_response=True,
                no_think=False,  # thinking improves profile decision quality
                timeout=90.0,
            )
            return extract_llm_json(content)
        except ValueError as exc:
            if attempt == 0:
                logger.warning(
                    "User reflection LLM malformed JSON (%s), retrying — %s", user_code, exc
                )
                continue
            logger.error(
                "User reflection LLM failed (%s) after retry: %s", user_code, exc
            )
            return None
        except Exception as exc:
            logger.error(
                "User reflection LLM failed (%s): %s", user_code, type(exc).__name__, exc_info=True,
            )
            return None
    return None


# ══════════════════════════════════════════════════
#  ACTION CATALOG
# ══════════════════════════════════════════════════


def _action_nothing(params: dict) -> str:
    reason = params.get("reason", "no reason given")
    logger.info("Self action: nothing (%s)", reason)
    return f"no-op: {reason}"


def _action_store_insight(params: dict) -> str:
    user_code = params.get("user_code", "")
    insight = params.get("insight", "").strip()
    if not user_code or not insight or user_code not in USER_CODES:
        return "store_insight: invalid params"

    store_autobiographical_event(user_code, insight, importance=0.8)
    logger.info("Self action: stored insight for %s", user_code)
    return f"stored insight for {user_code}"


_GAP_GENERIC_PHRASES = {
    "lacune de connaissance identifiée dans les capacités d'assistance",
    "lacune identifiée dans les capacités",
    "knowledge gap identified",
}
_GAP_COOLDOWN_TTL = 7 * 86400  # 7 days per topic


def _action_flag_knowledge_gap(params: dict) -> str:
    topic = params.get("topic", "").strip()
    context = params.get("context", "").strip()
    if not topic:
        return "flag_knowledge_gap: missing topic"

    # Guard 1 — context must be substantive (not generic filler)
    if len(context) < 30 or context.lower().rstrip(".") in _GAP_GENERIC_PHRASES:
        return (
            f"flag_knowledge_gap: context too generic for '{topic}' — "
            "describe a specific observed failure, not a general statement"
        )

    slug = re.sub(r"\s+", "_", topic.lower())[:40]
    r = get_redis()
    cooldown_key = f"jarvis:self:gap_cooldown:{slug}"

    # Guard 2 — per-topic cooldown (7 days)
    if r.exists(cooldown_key):
        ttl = r.ttl(cooldown_key)
        return f"flag_knowledge_gap: '{topic}' already flagged recently — cooldown active ({ttl // 3600}h remaining)"

    # Guard 3 — block if a proposal already exists for this topic (pending or approved < 30 days)
    now_ts = datetime.now(timezone.utc).timestamp()
    cutoff = now_ts - 30 * 86400
    for p in _load_proposals():
        p_slug = re.sub(r"\s+", "_", p.get("topic", "").lower())[:40]
        if p_slug != slug:
            continue
        if p.get("status") == "pending":
            return f"flag_knowledge_gap: proposal already pending for '{topic}' — no need to re-flag"
        if p.get("status") == "approved":
            approved_ts = datetime.fromisoformat(
                p.get("approved_at", "2000-01-01T00:00:00+00:00")
            ).timestamp()
            if approved_ts > cutoff:
                return f"flag_knowledge_gap: proposal for '{topic}' approved recently — cooldown active (30 days)"

    r.setex(cooldown_key, _GAP_COOLDOWN_TTL, "1")

    entry = json.dumps(
        {
            "topic": topic,
            "context": context,
            "date": datetime.now(timezone.utc).isoformat(),
        }
    )
    pipe = r.pipeline()
    pipe.zadd(_KNOWLEDGE_GAPS_KEY, {entry: time.time()})
    pipe.zremrangebyrank(_KNOWLEDGE_GAPS_KEY, 0, -51)  # keep last 50
    pipe.hincrby(_GAP_COUNTS_KEY, slug, 1)
    results = pipe.execute()
    count = int(results[2] or 0)

    logger.info("Self action: knowledge gap flagged — %s (count=%d)", topic, count)
    return f"flagged knowledge gap: {topic} (count={count})"


def _action_send_notification(params: dict) -> str:
    user_code = params.get("user_code", "")
    subject = params.get("subject", "").strip()
    message = params.get("message", "").strip()

    if not user_code or not subject or not message or user_code not in USER_CODES:
        return "send_notification: invalid params"

    to = USER_EMAILS.get(user_code, "")
    if not to:
        return f"send_notification: no email configured for {user_code}"

    if not is_google_available(user_code):
        return "send_notification: Google not configured"

    # One notification per user per day guard (uses user's local timezone)
    r = get_redis()
    user_tz_str = USERS.get(user_code, {}).get("timezone", "Europe/Paris")
    user_tz = pytz.timezone(user_tz_str)
    today = datetime.now(user_tz).strftime("%Y-%m-%d")
    notif_key = f"{_NOTIF_KEY_PREFIX}:{user_code}:{today}"
    if r.exists(notif_key):
        logger.info(
            "Self action: notification suppressed for %s (already sent today)",
            user_code,
        )
        return f"send_notification: suppressed (already sent to {user_code} today)"

    user_name = USER_CODES[user_code]
    html = f"<p>Bonjour {user_name},</p><p>{message}</p><p><em>— Jarvis</em></p>"
    success = send_gmail_message(
        to=to,
        subject=f"Jarvis — {subject}",
        html_body=html,
        text_body=message,
        user_code=user_code,
    )

    if success:
        r.setex(notif_key, _NOTIF_TTL, "1")
        logger.info("Self action: notification sent to %s (%s)", user_code, to)
        return f"notification sent to {user_code}"
    # Guard: mark as attempted today even on failure to avoid retry loops in the chain
    r.setex(notif_key, _NOTIF_TTL, "failed")
    return "send_notification: delivery failed"


def _action_queue_push(params: dict) -> str:
    """Queue an iOS push notification for a user. Polled by the app via GET /device/pending/{user_code}."""
    user_code = params.get("user_code", "")
    message = params.get("message", "").strip()

    if not user_code or user_code not in USER_CODES:
        return "queue_push: invalid user_code"
    if not message:
        return "queue_push: empty message"

    r = get_redis()

    # Device must be registered
    if not r.exists(f"{_DEVICE_TOKEN_PREFIX}:{user_code}"):
        return f"queue_push: no device registered for {user_code}"

    cooldown_key = f"{_PUSH_COOLDOWN_PREFIX}:{user_code}"

    pending_key = f"{_PUSH_PENDING_PREFIX}:{user_code}"
    r.rpush(
        pending_key,
        json.dumps(
            {
                "message": message,
                "queued_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
    )
    r.expire(pending_key, 86400)  # auto-expire if not polled within 24h
    r.setex(cooldown_key, _PUSH_COOLDOWN_TTL, "1")

    # Also inject into the persistent iOS conversation so the message is visible
    # when the user opens the app — even if the notification was missed.
    append_conversation_message(user_code, "iphone-main", "assistant", message)

    logger.info("Self action: push queued for %s — %s", user_code, message[:80])
    return f"push queued for {user_code}: {message[:80]}"


def _action_correct_profile(params: dict) -> str:
    """
    Directly write or delete a key in a user's Redis profile.
    Use for clear duplicates (hobby:montres + interest:montres) or stale/wrong values.
    value=null deletes the key; any string value overwrites it.
    """
    user_code = params.get("user_code", "")
    key = params.get("key", "").strip()
    value = params.get("value")  # None or "" means delete
    if value == "":
        value = None

    if not user_code or user_code not in USER_CODES:
        return "correct_profile: invalid user_code"
    if not key:
        return "correct_profile: missing key"
    # Deletions (value=null) are reserved for nightly review and prune_self_memory.
    # The reflection phase must never silently erase profile facts it cannot verify.
    if value is None:
        logger.warning(
            "Self correct_profile: BLOCKED deletion of '%s' for %s — "
            "deletions are not allowed in the reflection phase (use nightly review)",
            key, user_code,
        )
        return (
            f"correct_profile: deletion of '{key}' blocked — "
            "deletions are reserved for the nightly review phase"
        )

    from helpers import get_redis
    from memory import update_user_profile

    # Guard: reflection can only modify or delete existing keys.
    # New keys are created exclusively by the conversation analyzer (which reads
    # actual user statements). This prevents the reflection from hallucinating
    # profile facts from its own context.
    if value is not None:
        exists = get_redis().hexists(f"user:{user_code}:profile", key)
        if not exists:
            logger.warning(
                "Self correct_profile: REJECTED creation of new key '%s' for %s — "
                "new keys can only be created by the conversation analyzer",
                key, user_code,
            )
            return f"correct_profile: key '{key}' does not exist — use the analyzer to create new profile facts from conversation"

    if value is None:
        old_val = get_redis().hget(f"user:{user_code}:profile", key)
        if old_val is None:
            # Key does not exist for this user — likely a user-confusion error in the LLM.
            logger.warning(
                "Self correct_profile: key '%s' does not exist for %s — "
                "possible user confusion in reflection LLM (check _fmt_user_profiles context)",
                key, user_code,
            )
            return (
                f"correct_profile: key '{key}' does not exist for {user_code}. "
                f"Verify you have the correct user_code — this key may belong to another user."
            )
        logger.warning(
            "Self correct_profile: DELETE '%s' for %s (current value: %s)",
            key, user_code, old_val if old_val else "(empty)"
        )

    # ── Namespace protection: block cross-domain value contamination ────────
    # Prevents reflection from writing hobby/travel/unrelated values into
    # domain-specific keys (e.g. placement:amp20 = "pilote de kart").
    # Guard rules are declared in _NS_GUARDS at module level — add new
    # domains there without touching this function.
    if value is not None:
        _key_prefix_ns = key.split(":")[0]
        _value_lower = value.lower()

        for _guard in _NS_GUARDS:
            _in_ns = _key_prefix_ns in _guard["key_prefixes"] or key in _guard["key_prefixes"]
            if not _in_ns:
                continue
            _has_match = any(t in _value_lower for t in _guard["required_terms"])
            if not _has_match and _guard["extra_check"]:
                _has_match = _guard["extra_check"](_value_lower)
            if not _has_match:
                logger.warning(
                    "Self correct_profile: BLOCKED '%s' = '%s' for %s — "
                    "%s namespace requires %s (namespace protection)",
                    key, value, user_code, _guard["name"], _guard["error_hint"],
                )
                return (
                    f"correct_profile: BLOCKED '{key}' = '{value}' — "
                    f"{_guard['name']} namespace requires {_guard['error_hint']}"
                )

    update_user_profile(user_code, key, value if value is not None else None)
    op = f"deleted '{key}'" if value is None else f"set '{key}' = '{value}'"
    logger.info("Self action: correct_profile [%s] %s", user_code, op)
    return f"correct_profile [{user_code}]: {op}"


def _action_ask_user(params: dict) -> str:
    """
    Queue a short clarification question as an iOS push notification.
    The user answers naturally in the next chat message; the analyzer captures the reply.
    Uses the same push cooldown as queue_push (max 1 per 2h per user).
    """
    user_code = params.get("user_code", "")
    question = params.get("question", "").strip()

    if not user_code or user_code not in USER_CODES:
        return "ask_user: invalid user_code"
    if not question:
        return "ask_user: empty question"

    return _action_queue_push({"user_code": user_code, "message": question})


def _action_update_self_note(params: dict) -> str:
    note = params.get("note", "").strip()
    if not note:
        return "update_self_note: empty note"

    with self_memory_lock:
        data = get_self_memory()
        data.setdefault("self_notes", []).append(
            {
                "note": note,
                "date": datetime.now(timezone.utc).isoformat(),
            }
        )
        data["self_notes"] = data["self_notes"][-50:]
        save_self_memory(data)
    logger.info("Self action: self note written")
    return f"self note written: {note[:60]}"


def _action_consolidate_memory(params: dict) -> str:
    user_code = params.get("user_code", "")
    if not user_code or user_code not in USER_CODES:
        return "consolidate_memory: invalid user_code"
    try:
        from memory import consolidate_memories

        consolidate_memories(user_code)
        logger.info("Self action: memory consolidation triggered for %s", user_code)
        return f"memory consolidation triggered for {user_code}"
    except Exception as exc:
        return f"consolidate_memory: failed ({type(exc).__name__})"


# ══════════════════════════════════════════════════
#  NIGHTLY REVIEW  (replaces nightly-reflection.py)
# ══════════════════════════════════════════════════

# prompts accessed via get_prompt() for live-override support


async def _nightly_review_user(
    user_code: str, user_name: str, conversations: list[dict], review_date: str
) -> dict | None:
    """Call LLM for per-user nightly review. Returns parsed dict or None on failure."""
    # Sort by importance desc so the LLM sees the most significant exchanges first,
    # even if the 6000-char budget is hit before the end of the day's conversations.
    sorted_convs = sorted(
        conversations, key=lambda c: c.get("importance", 0), reverse=True
    )
    conv_text = ""
    for c in sorted_convs:
        conv_text += f"User: {c.get('user', '')[:400]}\nJarvis: {c.get('assistant', '')[:400]}\nMood: {c.get('mood', '?')}\n\n"

    data = get_self_memory()
    recent_self_reflections = [l["text"] for l in data.get("learnings", [])[-12:]]
    recent_opinions = [
        f"{o['topic']}: {o['opinion']}" for o in data.get("opinions", [])[-10:]
    ]
    current_relation = get_user_relation(user_code)

    prompt = get_prompt("NIGHTLY_PROMPT").format(
        user_name=user_name,
        user_code=user_code,
        review_date=review_date,
        count=len(conversations),
        conv_text=conv_text[:6000] or "(no conversation content)",
        recent_self_reflections=json.dumps(recent_self_reflections, ensure_ascii=False),
        recent_opinions=json.dumps(recent_opinions, ensure_ascii=False)
        if recent_opinions
        else "aucune",
        current_relation=json.dumps(current_relation, ensure_ascii=False),
    )

    try:
        content = await call_llm_async(
            [
                {"role": "system", "content": get_prompt("NIGHTLY_SYSTEM")},
                {"role": "user", "content": prompt},
            ],
            model=REASONING_MODEL,
            api_url=REASONING_API_URL,
            api_key=REASONING_API_KEY,
            temperature=0.3,
            max_tokens=THINKING_BUDGET_TOKENS + 3000,  # think block + JSON output (facts, profile updates)
            json_response=True,
            no_think=False,  # reasoning task — thinking improves nightly analysis quality
            timeout=90.0,
        )
        return extract_llm_json(content)
    except Exception as exc:
        logger.error(
            "Nightly review LLM call failed for %s: %s",
            user_code,
            type(exc).__name__,
            exc_info=True,
        )
        return None


async def run_nightly_interaction_review() -> None:
    """
    Nightly per-user conversation review. Called by APScheduler at 23:00.

    For each user with conversations yesterday:
      - Calls LLM to extract user facts (→ autobiographical Qdrant) and
        Jarvis self-improvement notes (→ jarvis-self.json learnings)
      - Stores tomorrow_suggestions in Redis (TTL 24h)
      - Triggers monthly memory consolidation on day 1

    Each user's write to jarvis-self.json is done under self_memory_lock
    immediately after the LLM call — no data is held across await points.
    Idempotent: Redis lock per user per date (TTL 25h).
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
        review = await _nightly_review_user(
            user_code, user_name, conversations, review_date
        )
        if review is None:
            continue

        # Store tomorrow's suggestions in Redis (24h)
        suggestions = review.get("tomorrow_suggestions", [])
        if suggestions:
            r.setex(
                f"jarvis:{user_code}:tomorrow_suggestions",
                86400,
                json.dumps(suggestions),
            )

        # ── Facts about the user → autobiographical Qdrant (memory.py) ──────
        for insight in review.get("user_insights", []):
            if insight:
                store_autobiographical_event(user_code, insight, importance=0.7)

        # ── Jarvis self-improvement notes + diary + user relation → jarvis-self.json
        # Lock here: no await is held while the lock is active.
        summary = review.get("daily_summary", "")
        self_refls = [s for s in review.get("self_reflections", []) if s]
        new_opinions = [
            o
            for o in review.get("jarvis_opinions", [])
            if isinstance(o, dict) and o.get("topic") and o.get("opinion")
        ]
        rel_update = review.get("user_relation_update", {})
        if self_refls or summary or rel_update or new_opinions:
            with self_memory_lock:
                data = get_self_memory()
                for refl in self_refls:
                    data.setdefault("learnings", []).append(
                        {
                            "text": refl,
                            "date": review_date,
                            "source": "nightly_review",
                        }
                    )
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
                            "mood": review.get("mood_summary", ""),
                            "conversations": len(conversations),
                        }
                    )
                if rel_update:
                    current = {
                        **_DEFAULT_RELATION,
                        **data.get("user_relations", {}).get(user_code, {}),
                    }
                    # Validate and clamp each field
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
                data["learnings"] = data.get("learnings", [])[-100:]
                data["growth_log"] = data.get("growth_log", [])[
                    -GROWTH_LOG_MAX_ENTRIES:
                ]
                data["last_nightly"] = review_date
                save_self_memory(data)

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


def _action_check_health(params: dict) -> str:
    health = _check_service_health()
    issues = [svc for svc, status in health.items() if status != "ok"]
    if issues:
        logger.warning("Self health check: issues detected — %s", issues)
        return f"health check: issues — {issues}"
    logger.info("Self health check: all services OK")
    return "health check: all services OK"


def _action_update_trade_threshold(params: dict) -> str:
    user_code = params.get("user_code", "")
    isin = params.get("isin", "").strip().upper()
    th = params.get("threshold_high")
    tl = params.get("threshold_low")

    if not user_code or user_code not in USER_CODES:
        return "update_trade_threshold: invalid user_code"
    if not isin:
        return "update_trade_threshold: missing isin"
    if th is None and tl is None:
        return "update_trade_threshold: at least one of threshold_high / threshold_low is required"

    r = get_redis()
    if not r.sismember(idx_key(user_code), isin):
        return f"update_trade_threshold: ISIN {isin} not in portfolio for {user_code}"

    key = pos_key(user_code, isin)
    mapping = {}
    parts = []

    if th is not None:
        try:
            th = round(float(th), 2)
        except (TypeError, ValueError):
            return "update_trade_threshold: threshold_high must be a number"
        mapping["threshold_high"] = str(th)
        parts.append(f"high={th}€")

    if tl is not None:
        try:
            tl = round(float(tl), 2)
        except (TypeError, ValueError):
            return "update_trade_threshold: threshold_low must be a number"
        mapping["threshold_low"] = str(tl)
        parts.append(f"low={tl}€")

    r.hset(key, mapping=mapping)
    pos_name = r.hget(key, "name") or isin

    result = f"threshold updated for {pos_name} ({isin}): {', '.join(parts)}"
    logger.info("Self action: %s", result)
    return result


# ══════════════════════════════════════════════════
#  AUTOCODING — PROMPT PROPOSALS
# ══════════════════════════════════════════════════


def _proposals_path() -> str:
    return os.path.join(PROMPT_DATA_DIR, "prompt_proposals.json")


def _overrides_path() -> str:
    return os.path.join(PROMPT_DATA_DIR, "prompt_overrides.json")


def _load_proposals() -> list[dict]:
    try:
        with open(_proposals_path(), encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_proposals(proposals: list) -> None:
    atomic_json_write(_proposals_path(), proposals)


def _load_overrides() -> dict:
    try:
        with open(_overrides_path(), encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_overrides(overrides: dict) -> None:
    atomic_json_write(_overrides_path(), overrides)


def list_pending_proposals() -> list[dict]:
    """Return all proposals with status='pending'."""
    return [p for p in _load_proposals() if p.get("status") == "pending"]


def approve_proposal(proposal_id: str) -> str:
    """Apply the proposed text to prompt_overrides.json and mark as approved."""
    proposals = _load_proposals()
    found = next((p for p in proposals if p["id"] == proposal_id), None)
    if not found:
        return f"Proposition `{proposal_id}` introuvable."
    if found["status"] != "pending":
        return f"Proposition `{proposal_id}` est déjà **{found['status']}**."

    # Write override
    overrides = _load_overrides()
    overrides[found["prompt_name"]] = found["proposed_text"]
    _save_overrides(overrides)

    # Mark approved
    found["status"] = "approved"
    found["approved_at"] = datetime.now(timezone.utc).isoformat()
    _save_proposals(proposals)

    # Full knowledge-gap reset for this topic:
    # 1. counter hash   — so refine_prompt threshold is not immediately re-crossed
    # 2. sorted set     — remove all entries for this topic so it no longer appears in LACUNES
    # 3. cooldown key   — prevent re-flagging for 30 days after approval
    topic_slug = re.sub(r"\s+", "_", found.get("topic", "").lower())[:40]
    if topic_slug:
        r = get_redis()
        # hdel is unconditional — never swallowed by a broad except.
        # A failed hdel would leave count intact and allow immediate re-crossing of threshold.
        r.hdel(_GAP_COUNTS_KEY, topic_slug)
        r.setex(f"jarvis:self:gap_cooldown:{topic_slug}", 30 * 86400, "1")
        # Remove sorted-set entries one by one; skip malformed JSON silently so
        # a single corrupt entry doesn't abort the whole cleanup.
        try:
            all_entries = r.zrange(_KNOWLEDGE_GAPS_KEY, 0, -1)
        except Exception as exc:
            logger.warning("gap cleanup: could not read knowledge_gaps set: %s", exc)
            all_entries = []
        for e in all_entries:
            try:
                e_slug = re.sub(r"\s+", "_", json.loads(e).get("topic", "").lower())[:40]
                if e_slug == topic_slug:
                    r.zrem(_KNOWLEDGE_GAPS_KEY, e)
            except Exception:
                # Malformed entry — zrem by raw value as fallback
                try:
                    r.zrem(_KNOWLEDGE_GAPS_KEY, e)
                except Exception:
                    pass

    # Invalidate prompts in-memory cache so the new text is returned immediately
    # (clears both the mtime sentinel and the cached dict — belt & suspenders)
    try:
        import prompts as _pm

        _pm._override_mtime = -1.0
        _pm._override_cache = {}
    except Exception:
        pass

    logger.info("Proposal %s approved: %s updated", proposal_id, found["prompt_name"])
    return (
        f"✓ Proposition `{proposal_id}` approuvée.\n"
        f"Le prompt **{found['prompt_name']}** est maintenant actif — aucun redémarrage nécessaire."
    )


def reject_proposal(proposal_id: str) -> str:
    """Mark a proposal as rejected."""
    proposals = _load_proposals()
    found = next((p for p in proposals if p["id"] == proposal_id), None)
    if not found:
        return f"Proposition `{proposal_id}` introuvable."
    if found["status"] != "pending":
        return f"Proposition `{proposal_id}` est déjà **{found['status']}**."

    found["status"] = "rejected"
    found["rejected_at"] = datetime.now(timezone.utc).isoformat()
    _save_proposals(proposals)

    logger.info("Proposal %s rejected", proposal_id)
    return f"✗ Proposition `{proposal_id}` rejetée."


def _notify_proposal(user_code: str, proposal: dict) -> None:
    """Send an email notification with the proposal diff."""
    import difflib
    import html as _html

    to = USER_EMAILS.get(user_code, "")
    if not to or not is_google_available():
        return

    pid = proposal["id"]
    name = proposal["prompt_name"]
    rationale = proposal["rationale"]
    current_text = proposal["current_text"]
    proposed_text = proposal["proposed_text"]

    # ── Unified diff (plain text) ──────────────────────────────────────────
    diff_lines = list(
        difflib.unified_diff(
            current_text.splitlines(),
            proposed_text.splitlines(),
            fromfile="actuel",
            tofile="proposé",
            lineterm="",
            n=5,
        )
    )
    diff_plain = "\n".join(diff_lines) if diff_lines else "(aucune différence détectée)"

    # ── Unified diff (HTML colorisé) ───────────────────────────────────────
    def _colorize_diff_html(lines: list[str]) -> str:
        parts = []
        for line in lines:
            escaped = _html.escape(line)
            if line.startswith("+++") or line.startswith("---"):
                parts.append(f"<span style='color:#555;font-weight:bold'>{escaped}</span>")
            elif line.startswith("+"):
                parts.append(f"<span style='background:#d4edda;color:#155724'>{escaped}</span>")
            elif line.startswith("-"):
                parts.append(f"<span style='background:#f8d7da;color:#721c24'>{escaped}</span>")
            elif line.startswith("@@"):
                parts.append(f"<span style='color:#0d6efd;font-weight:bold'>{escaped}</span>")
            else:
                parts.append(escaped)
        return "\n".join(parts)

    diff_html = (
        _colorize_diff_html(diff_lines)
        if diff_lines
        else "<em>(aucune différence détectée)</em>"
    )

    text = (
        f"Jarvis a identifié une opportunité d'amélioration du prompt « {name} ».\n\n"
        f"Raison : {rationale}\n\n"
        f"── DIFFÉRENCES ──\n{diff_plain}\n\n"
        f"── TEXTE ACTUEL (complet) ──\n{current_text}\n\n"
        f"── TEXTE PROPOSÉ (complet) ──\n{proposed_text}\n\n"
        f"Pour approuver : dis à Jarvis « accepte la proposition {pid} »\n"
        f"Pour rejeter  : dis à Jarvis « rejette la proposition {pid} »"
    )
    html = (
        f"<p>Jarvis a identifié une opportunité d'amélioration du prompt <strong>{name}</strong>.</p>"
        f"<p><strong>Raison :</strong> {_html.escape(rationale)}</p>"
        f"<h3>Différences</h3>"
        f"<pre style='background:#f8f9fa;padding:10px;font-size:12px;white-space:pre-wrap;border:1px solid #dee2e6;border-radius:4px'>{diff_html}</pre>"
        f"<h3>Texte actuel</h3>"
        f"<pre style='background:#f5f5f5;padding:10px;font-size:12px;white-space:pre-wrap'>{_html.escape(current_text)}</pre>"
        f"<h3>Texte proposé</h3>"
        f"<pre style='background:#e8f5e9;padding:10px;font-size:12px;white-space:pre-wrap'>{_html.escape(proposed_text)}</pre>"
        f"<p>Pour approuver : dis à Jarvis <strong>« accepte la proposition {pid} »</strong><br>"
        f"Pour rejeter : dis à Jarvis <strong>« rejette la proposition {pid} »</strong></p>"
        f"<p><em>— Jarvis</em></p>"
    )
    success = send_gmail_message(
        to=to,
        subject=f"Jarvis — Proposition de prompt #{pid} ({name})",
        html_body=html,
        text_body=text,
        user_code=user_code,
    )
    if not success:
        logger.warning(
            "_notify_proposal: email not sent for %s (Gmail unavailable?)", user_code
        )


def _action_refine_prompt(params: dict) -> str:
    """
    Call the reasoning model to propose an improved version of a prompt.
    Stores the proposal in prompt_proposals.json and notifies by email.
    Runs synchronously (called via asyncio.to_thread from run_reflection).
    """
    prompt_name = params.get("prompt_name", "").strip()
    topic = params.get("topic", "").strip()
    context_str = params.get("context", "").strip()
    user_code = params.get("user_code", "").strip()

    if not prompt_name or not topic:
        return "refine_prompt: missing prompt_name or topic"
    if user_code and user_code not in USER_CODES:
        return f"refine_prompt: unknown user_code {user_code!r}"

    current_text = get_prompt(prompt_name)
    if not current_text:
        return f"refine_prompt: unknown prompt {prompt_name!r}"

    # Guard: no duplicate pending proposal for the same prompt (data integrity, not a cooldown)
    existing = [p for p in list_pending_proposals() if p["prompt_name"] == prompt_name]
    if existing:
        return f"refine_prompt: proposal already pending for {prompt_name} (id={existing[0]['id']})"

    max_budget = PROMPT_TOKEN_BUDGETS.get(prompt_name, 600)
    current_token_count = len(current_text) // 4  # approximation : 1 token ≈ 4 chars

    refine_prompt_text = get_prompt("REFINE_PROMPT_USER").format(
        prompt_name=prompt_name,
        topic=topic,
        context=context_str or "aucun contexte supplémentaire",
        current_text=current_text[:6000],
        current_token_count=current_token_count,
        max_token_budget=max_budget,
    )

    try:
        content = call_llm(
            [
                {"role": "system", "content": get_prompt("REFINE_PROMPT_SYSTEM")},
                {"role": "user", "content": refine_prompt_text},
            ],
            model=REASONING_MODEL,
            api_url=REASONING_API_URL,
            api_key=REASONING_API_KEY,
            temperature=0.4,
            max_tokens=THINKING_BUDGET_TOKENS + 4000,  # think block + proposed prompt text + rationale
            json_response=True,
            no_think=False,
            timeout=REASONING_TIMEOUT,
        )
        result = extract_llm_json(content)
    except Exception as exc:
        logger.error("refine_prompt: LLM call failed: %s", exc)
        return f"refine_prompt: LLM call failed ({type(exc).__name__})"

    proposed_text = result.get("proposed_text", "").strip()
    rationale = result.get("rationale", "").strip()

    if not proposed_text:
        return "refine_prompt: LLM returned empty proposed_text"

    # Guard: format-string safety — detect unescaped JSON braces in proposed_text.
    # JSON literals like {"key":"..."} must be escaped as {{"key":"..."}} in format templates.
    # An unescaped {word} that isn't a known placeholder would crash str.format() with KeyError.
    _original_placeholders = set(re.findall(r'\{(\w+)\}', current_text))
    _proposed_new = set(re.findall(r'\{(\w+)\}', proposed_text)) - _original_placeholders
    if _proposed_new:
        logger.warning(
            "refine_prompt: proposed text for %s contains unescaped braces: %s — rejecting",
            prompt_name, _proposed_new,
        )
        return (
            f"refine_prompt: proposed text contains unescaped brace placeholders {_proposed_new} "
            f"that would break str.format(). JSON object literals must use {{{{ }}}} escaping. "
            f"Proposal discarded."
        )

    # Guard: reject if proposed text exceeds the token budget — retry once with explicit feedback
    proposed_token_count = len(proposed_text) // 4
    if proposed_token_count > max_budget:
        logger.warning(
            "refine_prompt: proposed text for %s is ~%d tokens (budget=%d) — retrying with feedback",
            prompt_name,
            proposed_token_count,
            max_budget,
        )
        retry_messages = [
            {"role": "system", "content": get_prompt("REFINE_PROMPT_SYSTEM")},
            {"role": "user", "content": refine_prompt_text},
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": (
                    f"Ton proposed_text fait ~{proposed_token_count} tokens mais le budget maximum "
                    f"est {max_budget} tokens. Tu dois le raccourcir. "
                    f"Retourne uniquement le JSON avec le proposed_text raccourci."
                ),
            },
        ]
        try:
            content = call_llm(
                retry_messages,
                model=REASONING_MODEL,
                api_url=REASONING_API_URL,
                api_key=REASONING_API_KEY,
                temperature=0.3,
                max_tokens=THINKING_BUDGET_TOKENS + 4000,  # retry — same budget as initial call
                json_response=True,
                no_think=False,
                timeout=REASONING_TIMEOUT,
            )
            result = extract_llm_json(content)
            proposed_text = result.get("proposed_text", "").strip()
            rationale = result.get("rationale", rationale).strip()
        except Exception as exc:
            logger.error("refine_prompt: retry failed: %s", exc)
            return f"refine_prompt: proposed text too long and retry failed ({type(exc).__name__})"

        proposed_token_count = len(proposed_text) // 4
        if not proposed_text or proposed_token_count > max_budget:
            logger.warning(
                "refine_prompt: retry still too long (%d tokens) — proposal rejected",
                proposed_token_count,
            )
            return (
                f"refine_prompt: proposed text still too long after retry "
                f"(~{proposed_token_count} tokens, budget={max_budget}) — proposal rejected"
            )
        logger.info("refine_prompt: retry succeeded (%d tokens)", proposed_token_count)

    proposal = {
        "id": uuid.uuid4().hex[:8],
        "prompt_name": prompt_name,
        "topic": topic,
        "current_text": current_text,
        "proposed_text": proposed_text,
        "rationale": rationale,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    proposals = _load_proposals()
    proposals.append(proposal)
    _save_proposals(proposals)

    # Notify admins only — prompt changes are a system-level action
    for _code in USER_ADMINS:
        _notify_proposal(_code, proposal)

    logger.info(
        "refine_prompt: proposal %s created for %s (topic: %s)",
        proposal["id"],
        prompt_name,
        topic,
    )
    return f"proposal {proposal['id']} created for {prompt_name}"


def handle_proposal_command(message: str, user_code: str) -> str | None:
    """
    Detect and execute proposal management commands from a chat message.
    Returns a formatted response string, or None if the message is not a proposal command.
    Called by main.py before the full LLM pipeline when use_self=True.
    """
    msg = message.strip().lower()

    # ── List pending proposals ──
    if any(
        kw in msg
        for kw in (
            "montre les propositions",
            "liste les propositions",
            "propositions en attente",
            "show proposals",
            "list proposals",
            "quelles propositions",
        )
    ):
        proposals = list_pending_proposals()
        if not proposals:
            return "Aucune proposition de prompt en attente."
        lines = [f"**{len(proposals)} proposition(s) en attente :**\n"]
        for p in proposals:
            lines.append(
                f"- `{p['id']}` — **{p['prompt_name']}** : {p['rationale'][:100]}"
            )
        lines.append(
            "\nDis « accepte la proposition [id] » ou « rejette la proposition [id] »."
        )
        return "\n".join(lines)

    # ── Approve ──
    m = re.search(
        r"(accepte?|approu?ve?)\s+(la\s+proposition\s+)?([a-f0-9]{6,8})\b", msg
    )
    if m:
        if user_code not in USER_ADMINS:
            return "⛔ Seul un administrateur peut approuver une proposition de prompt."
        return approve_proposal(m.group(3))

    # ── Reject ──
    m = re.search(
        r"(rejette?|refu?se?|reject)\s+(la\s+proposition\s+)?([a-f0-9]{6,8})\b", msg
    )
    if m:
        if user_code not in USER_ADMINS:
            return "⛔ Seul un administrateur peut rejeter une proposition de prompt."
        return reject_proposal(m.group(3))

    # ── Show specific proposal ──
    m = re.search(
        r"(montre?|show|détail)\s+(la\s+proposition\s+)?([a-f0-9]{6,8})\b", msg
    )
    if m:
        pid = m.group(3)
        proposals = _load_proposals()
        found = next((p for p in proposals if p["id"] == pid), None)
        if not found:
            return f"Proposition `{pid}` introuvable."
        import difflib as _difflib
        cur = found["current_text"]
        prop = found["proposed_text"]
        diff_lines = list(
            _difflib.unified_diff(
                cur.splitlines(), prop.splitlines(),
                fromfile="actuel", tofile="proposé", lineterm="", n=3,
            )
        )
        diff_block = "\n".join(diff_lines) if diff_lines else "(aucune différence)"
        return (
            f"**Proposition `{pid}` — {found['prompt_name']}** ({found['status']})\n\n"
            f"**Raison :** {found['rationale']}\n\n"
            f"**Diff :**\n```diff\n{diff_block}\n```\n\n"
            f"**Texte actuel :**\n```\n{cur}\n```\n\n"
            f"**Texte proposé :**\n```\n{prop}\n```"
        )

    return None


_PRUNE_COOLDOWN_KEY = "jarvis:self:last_prune"
_PRUNE_COOLDOWN_TTL = 86400  # 24h — one prune pass per day max


def _action_prune_self_memory(params: dict) -> str:
    """
    Call the Primary LLM to identify obsolete/redundant entries in self_notes,
    opinions, and learnings, then delete them from jarvis-self.json.
    Runs synchronously (called via asyncio.to_thread from run_self_reflection).
    """
    r = get_redis()
    if r.exists(_PRUNE_COOLDOWN_KEY):
        return "prune_self_memory: cooldown active (24h)"

    with self_memory_lock:
        data = get_self_memory()

    self_notes = data.get("self_notes", [])
    opinions = data.get("opinions", [])
    learnings = data.get("learnings", [])

    if max(len(self_notes), len(opinions), len(learnings)) < 2:
        return "prune_self_memory: nothing to prune (all lists have < 2 entries)"

    def _fmt(items: list) -> str:
        if not items:
            return "  (vide)"
        lines = []
        for i, item in enumerate(items):
            text = item.get("text", str(item)) if isinstance(item, dict) else str(item)
            date = (
                f" ({item['date']})"
                if isinstance(item, dict) and "date" in item
                else ""
            )
            lines.append(f"  [{i}] {text}{date}")
        return "\n".join(lines)

    user_prompt = get_prompt("PRUNE_SELF_MEMORY_USER").format(
        self_notes=_fmt(self_notes),
        opinions=_fmt(opinions),
        learnings=_fmt(learnings),
    )

    try:
        content = call_llm(
            [
                {"role": "system", "content": get_prompt("PRUNE_SELF_MEMORY_SYSTEM")},
                {"role": "user", "content": user_prompt},
            ],
            model=REASONING_MODEL,
            api_url=REASONING_API_URL,
            api_key=REASONING_API_KEY,
            temperature=0.2,
            max_tokens=800,
            json_response=True,
            no_think=True,
            timeout=30.0,
        )
    except Exception as exc:
        logger.error(
            "prune_self_memory LLM call failed: %s", type(exc).__name__, exc_info=True
        )
        return f"prune_self_memory: LLM call failed ({type(exc).__name__})"

    result = extract_llm_json(content)
    if not result or "to_delete" not in result:
        return "prune_self_memory: invalid LLM response"

    to_delete = result["to_delete"]
    total_deleted = 0

    with self_memory_lock:
        data = get_self_memory()
        for field in ("self_notes", "opinions", "learnings"):
            raw_indices = to_delete.get(field, [])
            if not raw_indices:
                continue
            lst = data.get(field, [])
            cap = max(0, len(lst) // 2)  # never delete more than 50 %
            indices = sorted(
                set(int(i) for i in raw_indices if 0 <= int(i) < len(lst))
            )[:cap]
            for i in reversed(indices):
                lst.pop(i)
            data[field] = lst
            if indices:
                total_deleted += len(indices)
                logger.info(
                    "prune_self_memory: deleted %d from %s: %s",
                    len(indices),
                    field,
                    indices,
                )
        save_self_memory(data)

    r.setex(_PRUNE_COOLDOWN_KEY, _PRUNE_COOLDOWN_TTL, "1")
    return f"prune_self_memory: deleted {total_deleted} entries total"


_ACTION_CATALOG = {
    "nothing": _action_nothing,
    "store_insight": _action_store_insight,
    "flag_knowledge_gap": _action_flag_knowledge_gap,
    "send_notification": _action_send_notification,
    "queue_push": _action_queue_push,
    "correct_profile": _action_correct_profile,
    "ask_user": _action_ask_user,
    "update_self_note": _action_update_self_note,
    "consolidate_memory": _action_consolidate_memory,
    "check_health": _action_check_health,
    "update_trade_threshold": _action_update_trade_threshold,
    "refine_prompt": _action_refine_prompt,
    "prune_self_memory": _action_prune_self_memory,
    # nightly_review is scheduled automatically — not in LLM action catalog
}


def _execute_action(action: str, params: dict) -> str:
    fn = _ACTION_CATALOG.get(action)
    if fn is None:
        logger.warning(
            "Self: unknown action requested — %r (defaulting to nothing)", action
        )
        return f"unknown action: {action}"
    return fn(params or {})


# ══════════════════════════════════════════════════
#  PROACTIVE PUSH GENERATION
# ══════════════════════════════════════════════════


def _get_active_projects(user_code: str) -> list[dict]:
    """Return in_progress / active projects for a user from Redis."""
    try:
        from memory import get_user_projects

        return [
            p
            for p in get_user_projects(user_code)
            if p.get("status") in ("in_progress", "active")
        ]
    except Exception:
        return []


def _last_conversation_ts(user_code: str) -> float:
    """Return Unix timestamp of the most recent episodic conversation, or 0."""
    r = get_redis()
    entries = r.zrevrangebyscore(
        f"convlog:{user_code}",
        "+inf",
        "-inf",
        start=0,
        num=1,
        withscores=True,
    )
    return entries[0][1] if entries else 0.0


async def generate_proactive_push(user_code: str) -> str:
    """
    Per-user LLM call: read recent conversations + active projects + mood,
    decide if there is something worth checking on proactively.

    Two trigger paths:
      A) Recent conversation (last 24h) — reactive follow-up on what was discussed
      B) Active project + silence > 48h — proactive check-in on ongoing work
         even when the user hasn't talked to Jarvis recently

    Guards:
      - Device must be registered (jarvis:device:token:{user_code})
      - Cooldown 2h between pushes per user (jarvis:push:cooldown:{user_code})
      - At least one of: recent conversation OR active project with silence > 48h
    """
    r = get_redis()

    # Guard: device registered?
    if not r.exists(f"{_DEVICE_TOKEN_PREFIX}:{user_code}"):
        return "no device registered"

    # Guard: cooldown active?
    if r.exists(f"{_PUSH_COOLDOWN_PREFIX}:{user_code}"):
        return "cooldown active"

    now = time.time()

    # ── Path A: recent conversations (last 24h) ──────────────────────────
    cutoff = now - 24 * 3600
    entries_raw = r.zrangebyscore(f"convlog:{user_code}", cutoff, "+inf")

    conv_lines: list[str] = []
    for raw in entries_raw[-10:]:
        try:
            e = json.loads(raw)
            user_msg = e.get("user", "")[:150]
            asst_msg = e.get("assistant", "")[:150]
            topics = ", ".join(e.get("topics", []))
            if user_msg:
                conv_lines.append(f"User: {user_msg}")
            if asst_msg:
                conv_lines.append(f"Jarvis: {asst_msg}")
            if topics:
                conv_lines.append(f"Topics: {topics}")
            conv_lines.append("")
        except Exception:
            pass

    # ── Path B: active projects + silence > 48h ──────────────────────────
    active_projects = _get_active_projects(user_code)
    last_ts = _last_conversation_ts(user_code)
    silence_hours = (now - last_ts) / 3600 if last_ts else 999

    project_lines: list[str] = []
    if active_projects and silence_hours > 48:
        for p in active_projects[:5]:
            project_lines.append(f"- {p['name']}: {p.get('description', '')[:120]}")

    # Neither path has anything to work with → skip
    if not conv_lines and not project_lines:
        return "no recent conversations and no active projects"

    # Get current mood from Redis emotional state
    mood = "measured"
    try:
        mood_raw = r.get("jarvis:emotional_state")
        if mood_raw:
            mood = json.loads(mood_raw).get("mood", "measured")
    except Exception:
        pass

    user_name = USER_CODES.get(user_code, user_code)
    conv_text = (
        "\n".join(conv_lines)[:2000] if conv_lines else "(aucune conversation récente)"
    )

    projects_section = ""
    if project_lines:
        projects_section = (
            f"\nProjets actifs de {user_name} (silence depuis {silence_hours:.0f}h) :\n"
            + "\n".join(project_lines)
            + "\n"
        )

    prompt = (
        f"Voici les échanges récents avec {user_name} :\n\n{conv_text}\n"
        f"{projects_section}\n"
        f"Humeur actuelle de Jarvis : {mood}\n\n"
        f"En tant que Jarvis, y a-t-il quelque chose qui mérite de reprendre contact de façon proactive ? "
        f"Par exemple : prendre des nouvelles d'un projet en cours, d'une situation mentionnée, "
        f"s'enquérir de la santé, relancer un sujet important. "
        f"Si oui, écris un message court (1 phrase max, en français, naturel et chaleureux). "
        f"Si non, réponds null.\n\n"
        f'Réponds UNIQUEMENT en JSON : {{"message": "..."}} ou {{"message": null}}'
    )

    try:
        content = await call_llm_async(
            [{"role": "user", "content": prompt}],
            model=REASONING_MODEL,
            api_url=REASONING_API_URL,
            api_key=REASONING_API_KEY,
            temperature=0.7,
            max_tokens=600,
            json_response=True,
            no_think=True,
            timeout=30.0,
        )
        message = extract_llm_json(content).get("message")
    except Exception as exc:
        logger.warning(
            "generate_proactive_push: LLM call failed for %s: %s",
            user_code,
            type(exc).__name__,
            exc_info=True,
        )
        return "LLM call failed"

    if not message:
        return "no proactive message generated"

    outcome = _action_queue_push({"user_code": user_code, "message": message})
    logger.info("generate_proactive_push for %s: %s", user_code, outcome)
    return outcome


# ══════════════════════════════════════════════════
#  ACTION SELF-REVIEW
# ══════════════════════════════════════════════════

# Actions that require a self-challenge LLM call before execution.
_REVIEW_REQUIRED_ACTIONS: frozenset[str] = frozenset(
    {"refine_prompt", "queue_push", "ask_user", "send_notification"}
)


def _build_review_context(
    action: str,
    global_ctx: dict,
    user_ctx: dict | None,
    params: dict | None = None,
) -> tuple[str, str]:
    """Return (context_str, criteria_str) tailored to the action being reviewed."""
    params = params or {}

    if action == "refine_prompt":
        topic = params.get("topic", "")
        prompt_name = params.get("prompt_name", "")

        # Raw gap count for this specific topic
        r = get_redis()
        slug = re.sub(r"\s+", "_", topic.lower())[:40]
        count = int(r.hget(_GAP_COUNTS_KEY, slug) or 0)

        # Recent proposal history for this prompt (last 3)
        all_proposals = _load_proposals()
        recent = [
            f"{p.get('status','?')} le {p.get('created_at','?')[:10]}"
            for p in all_proposals
            if p.get("prompt_name") == prompt_name
        ][-3:]
        proposals_history = "; ".join(recent) or "aucune"

        gaps = ", ".join(global_ctx.get("knowledge_gaps", [])) or "aucune"
        proposals_pending = global_ctx.get("pending_proposals", "aucune")

        context = (
            f"Topic proposé : '{topic}' — flaggé {count} fois dans les gaps\n"
            f"Lacunes connues : {gaps}\n"
            f"Historique des proposals pour '{prompt_name}' : {proposals_history}\n"
            f"Proposals en attente : {proposals_pending}"
        )
        criteria = (
            "refine_prompt est justifié si tu as des preuves concrètes que ce topic revient "
            "régulièrement dans les conversations (gap count significatif) ET qu'aucune proposal "
            "n'est déjà en attente ou n'a été soumise récemment pour ce prompt. "
            "Si les données ci-dessus ne montrent pas de problème récurrent réel, dis false."
        )

    elif action in ("queue_push", "ask_user", "send_notification") and user_ctx:
        has_push = user_ctx.get("has_push", False)
        last_push = user_ctx.get("push_cooldown_str", "inconnu")
        activity = str(user_ctx.get("user_activity", {}))[:300]
        context = (
            f"Push iOS disponible : {has_push}\n"
            f"Dernier push envoyé : {last_push}\n"
            f"Activité récente : {activity}"
        )
        criteria = (
            "Un push est justifié si : push disponible ET délai raisonnable depuis le dernier "
            "(au moins quelques heures) ET le message apporte une valeur concrète et urgente "
            "qui n'a pas déjà été envoyée. Si le dernier push est récent, dis false. "
            "Sois conservateur : mieux vaut ne pas envoyer que spammer."
        )

    else:
        context = "Contexte général — évalue selon le bon sens."
        criteria = "L'action doit apporter une valeur claire et concrète maintenant."

    return context, criteria


async def _llm_review_before_action(
    action: str,
    params: dict,
    global_ctx: dict,
    user_ctx: dict | None,
    previous_steps: list[dict],
) -> tuple[bool, str]:
    """
    Self-challenge LLM call before executing a consequential action.
    Uses the router model (fast, binary decision).
    Returns (should_execute, reason).
    Fail-open: if the review call fails, the action is allowed.
    """
    context_str, criteria_str = _build_review_context(action, global_ctx, user_ctx, params)

    steps_summary = (
        "; ".join(f"{s['action']}→{s['outcome'][:60]}" for s in previous_steps)
        or "aucune"
    )

    prompt = get_prompt("ACTION_REVIEW_USER").format(
        action=action,
        params=json.dumps(params, ensure_ascii=False, default=str),
        context=context_str,
        previous_steps=steps_summary,
        criteria=criteria_str,
    )

    try:
        content = await call_llm_async(
            [
                {"role": "system", "content": get_prompt("ACTION_REVIEW_SYSTEM")},
                {"role": "user", "content": prompt},
            ],
            model=REASONING_MODEL,
            api_url=REASONING_API_URL,
            api_key=REASONING_API_KEY,
            temperature=0.1,
            max_tokens=THINKING_BUDGET_TOKENS + 300,  # think block + JSON court (execute + reason)
            json_response=True,
            no_think=False,  # jugement contextuel — thinking améliore la qualité
            timeout=30.0,
        )
        result = extract_llm_json(content)
        execute = bool(result.get("execute", True))
        reason = result.get("reason", "")
        return execute, reason
    except Exception as exc:
        logger.warning("Action self-review failed (%s) — allowing action by default", exc)
        return True, "review failed — defaulting to execute"


# ══════════════════════════════════════════════════
#  MAIN REFLECTION ENTRY POINT
# ══════════════════════════════════════════════════


def _run_chain_step(
    result: dict,
    steps: list[dict],
    allowed_actions: frozenset,
    phase_label: str,
) -> tuple[str, str, str, dict, bool]:
    """
    Extract and validate one chain step from an LLM result.

    Returns (focus, action, reason, params, should_stop).
    should_stop=True means the caller must break the chain loop.
    """
    focus = result.get("focus", "").strip()
    action = result.get("action", "nothing").strip()
    reason = result.get("reason", "").strip()
    params = result.get("params", {})

    # Guard 1: action must be in the allowed catalog for this phase
    if action not in allowed_actions:
        logger.warning(
            "%s: invalid action %r (not in allowed set) — defaulting to nothing",
            phase_label, action,
        )
        action = "nothing"
        params = {"reason": f"invalid action for this phase: {result.get('action')}"}

    # Guard 2: detect exact duplicate to prevent infinite loops
    _sig = json.dumps({"action": action, "params": params}, sort_keys=True)
    if any(
        json.dumps({"action": s["action"], "params": s["params"]}, sort_keys=True) == _sig
        for s in steps
    ):
        logger.info("%s: duplicate action=%s — stopping chain", phase_label, action)
        return focus, action, reason, params, True

    return focus, action, reason, params, False


async def run_self_reflection() -> dict:
    """
    Two-phase self-reflection cycle. Called by APScheduler every REFLECTION_INTERVAL_HOURS.

    Phase 1 (global): Jarvis self-state — health, knowledge gaps, self-notes, prompts.
                      Up to MAX_CHAIN_ITERATIONS steps.
    Phase 2 (per-user): One LLM chain per user — profile, push, insights.
                        Up to MAX_CHAIN_ITERATIONS steps per user.

    Returns a log entry with all steps under the "steps" key.
    """
    logger.info(
        "=== Jarvis self-reflection starting (max %d steps/phase) ===",
        MAX_CHAIN_ITERATIONS,
    )

    global_ctx = gather_global_context()
    global_steps: list[dict] = []
    focus = ""

    # ── Phase 1: global self-state ─────────────────────────────────────────
    logger.info("--- Phase 1: global self-state ---")
    for i in range(MAX_CHAIN_ITERATIONS):
        result = await _call_global_reflection_llm(global_ctx, previous_steps=global_steps)

        if result is None:
            logger.warning("Global reflection LLM failed at step %d — stopping", i + 1)
            break

        focus, action, reason, params, stop = _run_chain_step(
            result, global_steps, _GLOBAL_ACTIONS, f"P1-step{i+1}"
        )
        params.setdefault("reason", reason)  # forward top-level reason into _action_nothing

        if action in _REVIEW_REQUIRED_ACTIONS:
            approved, rev_reason = await _llm_review_before_action(
                action, params, global_ctx, None, global_steps
            )
            if not approved:
                logger.info("P1 self-review rejected %s: %s", action, rev_reason)
                action = "nothing"
                params = {"reason": f"self-review: {rev_reason}"}
                stop = True

        outcome = await asyncio.to_thread(_execute_action, action, params)

        step = {
            "phase": "global",
            "iteration": i + 1,
            "focus": focus,
            "action": action,
            "reason": reason,
            "params": params,
            "outcome": outcome,
        }
        global_steps.append(step)
        logger.info("P1 step %d/%d: action=%s outcome=%s", i + 1, MAX_CHAIN_ITERATIONS, action, outcome)

        if stop or action == "nothing":
            break

    # ── Phase 2: per-user chains ───────────────────────────────────────────
    logger.info("--- Phase 2: per-user reflection (%d users) ---", len(USER_CODES))
    all_user_steps: list[dict] = []

    for user_code in USER_CODES:
        user_ctx = gather_user_context(user_code)
        user_steps: list[dict] = []
        _failed_actions: set[str] = set()  # actions that hit a system constraint this cycle
        logger.info("--- User: %s (%s) ---", user_code, user_ctx["user_name"])

        for i in range(MAX_CHAIN_ITERATIONS):
            result = await _call_user_reflection_llm(global_ctx, user_ctx, previous_steps=user_steps)

            if result is None:
                logger.warning(
                    "User reflection LLM failed at step %d for %s — stopping",
                    i + 1, user_code,
                )
                break

            ufocus, action, reason, params, stop = _run_chain_step(
                result, user_steps, _USER_ACTIONS, f"P2-{user_code}-step{i+1}"
            )
            if not focus:
                focus = ufocus

            params.setdefault("reason", reason)  # forward top-level reason into _action_nothing

            # Inject user_code into params for all user-scoped actions so the
            # LLM doesn't need to carry it reliably across iterations.
            _user_scoped = {
                "correct_profile", "store_insight", "queue_push",
                "send_notification", "ask_user", "consolidate_memory",
                "update_trade_threshold",
            }
            if action in _user_scoped and not params.get("user_code"):
                params["user_code"] = user_code

            # Don't retry an action that already hit a system-level constraint this cycle
            if action in _failed_actions:
                logger.info(
                    "P2 %s step %d/%d: action=%s previously failed — skipping to nothing",
                    user_code, i + 1, MAX_CHAIN_ITERATIONS, action,
                )
                action = "nothing"
                params = {"reason": f"previous {action} hit a system constraint — not retrying"}

            if action in _REVIEW_REQUIRED_ACTIONS:
                approved, rev_reason = await _llm_review_before_action(
                    action, params, global_ctx, user_ctx, user_steps
                )
                if not approved:
                    logger.info(
                        "P2 %s self-review rejected %s: %s", user_code, action, rev_reason
                    )
                    action = "nothing"
                    params = {"reason": f"self-review: {rev_reason}"}
                    stop = True

            outcome = await asyncio.to_thread(_execute_action, action, params)

            # Detect system-constraint failures: outcome format is "action: error"
            # (no "[user_code]" bracket), distinct from success "action [user_code]: ..."
            _looks_like_error = (
                outcome.startswith(f"{action}:")
                and not outcome.startswith(f"{action} [")
            )
            if _looks_like_error and action != "nothing":
                _failed_actions.add(action)

            step = {
                "phase": f"user:{user_code}",
                "iteration": i + 1,
                "focus": ufocus,
                "action": action,
                "reason": reason,
                "params": params,
                "outcome": outcome,
            }
            user_steps.append(step)
            all_user_steps.append(step)
            logger.info(
                "P2 %s step %d/%d: action=%s outcome=%s",
                user_code, i + 1, MAX_CHAIN_ITERATIONS, action, outcome,
            )

            if stop or action == "nothing":
                break

    # ── Persist focus + reflection metadata ────────────────────────────────
    all_steps = global_steps + all_user_steps
    now_iso = datetime.now(timezone.utc).isoformat()
    with self_memory_lock:
        data = get_self_memory()
        data["current_focus"] = focus
        data["last_reflection"] = now_iso
        data["reflection_count"] = data.get("reflection_count", 0) + 1
        save_self_memory(data)

    last = (
        all_steps[-1]
        if all_steps
        else {"action": "nothing", "reason": "no steps executed", "outcome": ""}
    )
    log_entry = {
        "timestamp": now_iso,
        "focus": focus,
        "action": last["action"],  # for _extract_behavioral_patterns
        "reason": last["reason"],
        "outcome": last["outcome"],
        "steps": all_steps,
        "health": global_ctx["health"],
    }
    log_reflection(log_entry)

    logger.info(
        "=== Reflection complete: %d global + %d user step(s), final=%s ===",
        len(global_steps),
        len(all_user_steps),
        last["action"],
    )

    # Proactive push: per-user LLM call — fully guarded (device check + cooldown)
    for code in USER_CODES:
        try:
            await generate_proactive_push(code)
        except Exception as exc:
            logger.warning(
                "generate_proactive_push error for %s: %s", code, type(exc).__name__
            )

    return log_entry

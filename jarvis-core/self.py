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

Reflection cycle:
  1. gather_context()   — system health, user activity, topics, knowledge gaps
  2. LLM call           — returns {focus, action, reason, params}
  3. execute_action()   — dispatches to catalog function
  4. log_reflection()   — persists result to Redis + updates jarvis-self.json

Action catalog (v1 — grows in future versions):
  nothing                  — explicit no-op, with reason
  store_insight            — save a learning to a user's memory
  flag_knowledge_gap       — log a topic Jarvis answered poorly
  send_notification        — send a relevant Gmail to one user (1/user/day max)
  update_self_note         — write an observation to self_notes in jarvis-self.json
  consolidate_memory       — trigger memory compression for a user
  check_health             — verify all services, log status
  update_trade_threshold   — update threshold_high / threshold_low for one portfolio position
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone

import httpx
import redis as redis_lib
from qdrant_client import QdrantClient

import pytz

from config import (
    GOOGLE_CALENDAR_ID,
    OPENAI_API_KEY,
    OPENAI_API_URL,
    PRIMARY_API_KEY,
    PRIMARY_API_URL,
    PRIMARY_MODEL,
    no_think_suffix,
    QDRANT_URL,
    REDIS_URL,
    SELF_MEMORY_PATH,
    USER_CODES,
    USER_EMAILS,
    USERS,
)
from google_services import is_google_available, send_gmail_message

logger = logging.getLogger("jarvis-self")

# ── Redis keys ────────────────────────────────────────────────────────────
_REFLECTION_LOG_KEY = "jarvis:self:reflection_log"
_REFLECTION_LOG_MAX = 30
_KNOWLEDGE_GAPS_KEY = "jarvis:self:knowledge_gaps"
_NOTIF_KEY_PREFIX   = "jarvis:self:notif"
_NOTIF_TTL          = 86400   # 24h — one notification per user per day


# ── Redis / Qdrant singletons ─────────────────────────────────────────────

_redis_client: redis_lib.Redis | None = None
_qdrant_client: QdrantClient | None = None


def _redis() -> redis_lib.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis_lib.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def _qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL, timeout=10)
    return _qdrant_client


# ══════════════════════════════════════════════════
#  SELF STATE — read/write jarvis-self.json
# ══════════════════════════════════════════════════

def load_self() -> dict:
    try:
        with open(SELF_MEMORY_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Could not load jarvis-self.json: %s", type(exc).__name__)
        return {}


def _save_self(data: dict) -> None:
    try:
        with open(SELF_MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error("Could not save jarvis-self.json: %s", type(exc).__name__)


def get_goals() -> list[dict]:
    return load_self().get("goals", [])


def get_current_focus() -> str:
    return load_self().get("current_focus", "")


def _update_self_fields(**fields) -> None:
    """Atomically update specific top-level fields in jarvis-self.json."""
    data = load_self()
    data.update(fields)
    _save_self(data)


# ══════════════════════════════════════════════════
#  REFLECTION LOG  (Redis sorted set)
# ══════════════════════════════════════════════════

def log_reflection(entry: dict) -> None:
    """Append a reflection result to the Redis log (capped at _REFLECTION_LOG_MAX)."""
    r = _redis()
    score = time.time()
    r.zadd(_REFLECTION_LOG_KEY, {json.dumps(entry, ensure_ascii=False): score})
    # Trim to most recent entries
    excess = r.zcard(_REFLECTION_LOG_KEY) - _REFLECTION_LOG_MAX
    if excess > 0:
        r.zremrangebyrank(_REFLECTION_LOG_KEY, 0, excess - 1)


def get_reflection_log(n: int = 10) -> list[dict]:
    """Return the last n reflection entries, most recent first."""
    r = _redis()
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


# ══════════════════════════════════════════════════
#  CONTEXT GATHERING
# ══════════════════════════════════════════════════

def _check_service_health() -> dict:
    """Quick liveness check on Redis, Qdrant, OpenAI."""
    health = {}
    # Redis
    try:
        _redis().ping()
        health["redis"] = "ok"
    except Exception:
        health["redis"] = "unreachable"

    # Qdrant
    try:
        _qdrant().get_collections()
        health["qdrant"] = "ok"
    except Exception:
        health["qdrant"] = "unreachable"

    # OpenAI (lightweight models list call)
    try:
        import httpx as _httpx
        r = _httpx.get(
            f"{OPENAI_API_URL}/models",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            timeout=5,
        )
        health["openai"] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
    except Exception:
        health["openai"] = "unreachable"

    return health


def _get_user_activity(hours: int = 24) -> dict:
    """
    Count recent conversations per user by scanning their episodic Redis log.
    Returns {user_code: {name, conversations, topics}}.
    """
    r = _redis()
    cutoff = time.time() - hours * 3600
    activity = {}

    for code, name in USER_CODES.items():
        entries_raw = r.zrangebyscore(
            f"episodic:{code}:conversations", cutoff, "+inf"
        )
        topics: set[str] = set()
        for raw in entries_raw:
            try:
                e = json.loads(raw)
                topics.update(e.get("topics", []))
            except Exception:
                pass
        activity[code] = {
            "name": name,
            "conversations": len(entries_raw),
            "topics": sorted(topics)[:8],
        }

    return activity


def _get_knowledge_gaps(n: int = 5) -> list[str]:
    """Return the most recently flagged knowledge gaps."""
    r = _redis()
    raw = r.zrevrange(_KNOWLEDGE_GAPS_KEY, 0, n - 1)
    results = []
    for item in raw:
        try:
            results.append(json.loads(item).get("topic", item))
        except Exception:
            results.append(item)
    return results


def gather_context() -> dict:
    """Assemble full context for the reflection prompt."""
    self_data  = load_self()
    health     = _check_service_health()
    activity   = _get_user_activity(24)
    gaps       = _get_knowledge_gaps(5)
    last_ref   = get_last_reflection()

    return {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "identity":        self_data.get("identity", {}),
        "goals":           self_data.get("goals", []),
        "current_focus":   self_data.get("current_focus", ""),
        "health":          health,
        "user_activity":   activity,
        "knowledge_gaps":  gaps,
        "last_reflection": last_ref,
        "reflection_count": self_data.get("reflection_count", 0),
    }


# ══════════════════════════════════════════════════
#  LLM REFLECTION CALL
# ══════════════════════════════════════════════════

_REFLECTION_SYSTEM = """\
Tu es Jarvis, un assistant personnel IA. Tu effectues ta boucle de réflexion autonome.
Ton but est d'analyser ta situation actuelle et de décider d'une action concrète pour \
mieux servir tes objectifs. Sois honnête, autocritique et pragmatique.
Réponds toujours en JSON valide uniquement."""

_REFLECTION_PROMPT = """\
Date/heure actuelle : {timestamp}

TON IDENTITÉ :
{identity}

TES OBJECTIFS (immuables, par priorité) :
{goals}

SANTÉ DU SYSTÈME :
{health}

ACTIVITÉ DES UTILISATEURS (dernières 24h) :
{activity}

LACUNES DE CONNAISSANCE (sujets où j'ai été faible) :
{gaps}

RÉFLEXION PRÉCÉDENTE :
{last_reflection}

---
En te basant sur tes objectifs et le contexte ci-dessus, décide :
1. Quel est ton focus actuel ? (une phrase)
2. Quelle action UNIQUE du catalogue vas-tu entreprendre et pourquoi ?

CATALOGUE D'ACTIONS :
- nothing                : aucune action nécessaire ce cycle (params: {{"reason": "..."}})
- store_insight          : enregistrer un apprentissage important sur un utilisateur (params: {{"user_code": "...", "insight": "..."}})
- flag_knowledge_gap     : noter un sujet que tu devrais mieux connaître (params: {{"topic": "...", "context": "..."}})
- send_notification      : envoyer un email utile/pertinent à un utilisateur — UNIQUEMENT si vraiment utile (params: {{"user_code": "...", "subject": "...", "message": "..."}})
- update_self_note       : écrire une observation personnelle sur toi-même (params: {{"note": "..."}})
- consolidate_memory     : déclencher la compression mémoire pour un utilisateur actif (params: {{"user_code": "..."}})
- check_health           : journaliser un bilan de santé détaillé (params: {{}})
- update_trade_threshold : mettre à jour le seuil d'alerte haut/bas d'une position du portefeuille (params: {{"user_code": "...", "isin": "...", "threshold_high": 0.0, "threshold_low": 0.0}})

Règles :
- send_notification est réservé aux messages genuinement utiles (rappels, infos demandées). Ne jamais envoyer sans valeur claire.
- update_trade_threshold : n'utilise cette action que si tu as une bonne raison de réviser un seuil (ex. cours très éloigné du seuil existant, forte variation récente). Fournis uniquement les champs à modifier (threshold_high ou threshold_low ou les deux). L'ISIN doit être exact.
- Sois concis dans tous les champs texte.
- Choisis "nothing" si aucune action significative n'est nécessaire.
- Écris le focus, la reason, le subject et le message en français.

Réponds en JSON uniquement :
{{"focus": "...", "action": "...", "reason": "...", "params": {{...}}}}"""


async def _call_reflection_llm(context: dict) -> dict | None:
    """Call the LLM to produce a reflection result."""

    def _fmt_goals(goals):
        return "\n".join(f"  G{i+1}. {g.get('label', '?')}: {g.get('description', '')}" for i, g in enumerate(goals))

    def _fmt_activity(activity):
        lines = []
        for code, info in activity.items():
            topics = ", ".join(info["topics"]) or "none"
            lines.append(f"  {info['name']} ({code}): {info['conversations']} conversations | topics: {topics}")
        return "\n".join(lines) or "  No activity."

    prompt = _REFLECTION_PROMPT.format(
        timestamp     = context["timestamp"],
        identity      = json.dumps(context["identity"], ensure_ascii=False),
        goals         = _fmt_goals(context["goals"]),
        health        = json.dumps(context["health"]),
        activity      = _fmt_activity(context["user_activity"]),
        gaps          = ", ".join(context["knowledge_gaps"]) or "none flagged",
        last_reflection = json.dumps(context["last_reflection"], ensure_ascii=False) if context["last_reflection"] else "none yet",
    )

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{PRIMARY_API_URL}/chat/completions",
                headers={"Authorization": f"Bearer {PRIMARY_API_KEY}"},
                json={
                    "model": PRIMARY_MODEL,
                    "messages": [
                        {"role": "system", "content": _REFLECTION_SYSTEM + no_think_suffix(PRIMARY_MODEL)},
                        {"role": "user",   "content": prompt},
                    ],
                    "response_format": {"type": "json_object"},
                    "max_tokens": 300,
                    "temperature": 0.7,
                },
            )
        return json.loads(resp.json()["choices"][0]["message"]["content"])
    except Exception as exc:
        logger.error("Reflection LLM call failed: %s", type(exc).__name__)
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
    insight   = params.get("insight", "").strip()
    if not user_code or not insight or user_code not in USER_CODES:
        return "store_insight: invalid params"

    data = load_self()
    data.setdefault("learnings", []).append({
        "text":   insight,
        "date":   datetime.now(timezone.utc).isoformat(),
        "user":   user_code,
        "source": "self_reflection",
    })
    _save_self(data)
    logger.info("Self action: stored insight for %s", user_code)
    return f"stored insight for {user_code}"


def _action_flag_knowledge_gap(params: dict) -> str:
    topic   = params.get("topic", "").strip()
    context = params.get("context", "").strip()
    if not topic:
        return "flag_knowledge_gap: missing topic"

    r = _redis()
    entry = json.dumps({"topic": topic, "context": context, "date": datetime.now(timezone.utc).isoformat()})
    r.zadd(_KNOWLEDGE_GAPS_KEY, {entry: time.time()})
    r.zremrangebyrank(_KNOWLEDGE_GAPS_KEY, 0, -51)   # keep last 50
    logger.info("Self action: knowledge gap flagged — %s", topic)
    return f"flagged knowledge gap: {topic}"


def _action_send_notification(params: dict) -> str:
    user_code = params.get("user_code", "")
    subject   = params.get("subject", "").strip()
    message   = params.get("message", "").strip()

    if not user_code or not subject or not message or user_code not in USER_CODES:
        return "send_notification: invalid params"

    to = USER_EMAILS.get(user_code, "")
    if not to:
        return f"send_notification: no email configured for {user_code}"

    if not is_google_available():
        return "send_notification: Google not configured"

    # One notification per user per day guard (uses user's local timezone)
    r = _redis()
    user_tz_str = USERS.get(user_code, {}).get("timezone", "Europe/Paris")
    user_tz = pytz.timezone(user_tz_str)
    today = datetime.now(user_tz).strftime("%Y-%m-%d")
    notif_key = f"{_NOTIF_KEY_PREFIX}:{user_code}:{today}"
    if r.exists(notif_key):
        logger.info("Self action: notification suppressed for %s (already sent today)", user_code)
        return f"send_notification: suppressed (already sent to {user_code} today)"

    user_name = USER_CODES[user_code]
    html = f"<p>Bonjour {user_name},</p><p>{message}</p><p><em>— Jarvis</em></p>"
    success = send_gmail_message(to=to, subject=f"Jarvis — {subject}", html_body=html, text_body=message)

    if success:
        r.setex(notif_key, _NOTIF_TTL, "1")
        logger.info("Self action: notification sent to %s (%s)", user_code, to)
        return f"notification sent to {user_code}"
    return "send_notification: delivery failed"


def _action_update_self_note(params: dict) -> str:
    note = params.get("note", "").strip()
    if not note:
        return "update_self_note: empty note"

    data = load_self()
    data.setdefault("self_notes", []).append({
        "note": note,
        "date": datetime.now(timezone.utc).isoformat(),
    })
    # Keep last 50 self-notes
    data["self_notes"] = data["self_notes"][-50:]
    _save_self(data)
    logger.info("Self action: self note written")
    return f"self note written: {note[:60]}"


def _action_consolidate_memory(params: dict) -> str:
    user_code = params.get("user_code", "")
    if not user_code or user_code not in USER_CODES:
        return "consolidate_memory: invalid user_code"
    # Import here to avoid circular dependency
    try:
        from memory import _consolidate_user_memories
        _consolidate_user_memories(user_code)
        logger.info("Self action: memory consolidation triggered for %s", user_code)
        return f"memory consolidation triggered for {user_code}"
    except Exception as exc:
        return f"consolidate_memory: failed ({type(exc).__name__})"


# ══════════════════════════════════════════════════
#  NIGHTLY REVIEW  (replaces nightly-reflection.py)
# ══════════════════════════════════════════════════

_NIGHTLY_SYSTEM = """\
Tu es Jarvis, un assistant personnel IA. Tu passes en revue les conversations de la journée pour un utilisateur.
Extrais les apprentissages significatifs et résume la journée. Réponds en JSON valide uniquement.
Toutes les valeurs textuelles doivent être rédigées en français."""

_NIGHTLY_PROMPT = """\
Utilisateur : {user_name} ({user_code})
Date analysée : {review_date}

CONVERSATIONS DU JOUR ({count} échanges) :
{conv_text}

Apprentissages récents déjà enregistrés : {recent_learnings}

Réponds en JSON :
{{
  "daily_summary": "résumé de la journée en 2-3 phrases",
  "user_insights": ["nouvelles choses apprises sur l'utilisateur (liste vide si rien)"],
  "self_reflections": ["choses apprises par Jarvis pour mieux servir (liste vide si rien)"],
  "tomorrow_suggestions": ["sujets proactifs à mentionner demain (liste vide si rien)"],
  "mood_summary": "évaluation de la journée de l'utilisateur en une phrase"
}}
Retourne uniquement du JSON valide, tout en français."""


async def _nightly_review_user(user_code: str, user_name: str, conversations: list[dict], review_date: str) -> dict | None:
    """Call LLM for per-user nightly review. Returns parsed dict or None on failure."""
    conv_text = ""
    for c in conversations[-50:]:
        conv_text += f"User: {c.get('user', '')[:200]}\nJarvis: {c.get('assistant', '')[:200]}\nMood: {c.get('mood', '?')}\n\n"

    data = load_self()
    recent_learnings = [l["text"] for l in data.get("learnings", [])[-5:]]

    prompt = _NIGHTLY_PROMPT.format(
        user_name=user_name,
        user_code=user_code,
        review_date=review_date,
        count=len(conversations),
        conv_text=conv_text[:3000] or "(no conversation content)",
        recent_learnings=json.dumps(recent_learnings, ensure_ascii=False),
    )

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{PRIMARY_API_URL}/chat/completions",
                headers={"Authorization": f"Bearer {PRIMARY_API_KEY}"},
                json={
                    "model": PRIMARY_MODEL,
                    "messages": [
                        {"role": "system", "content": _NIGHTLY_SYSTEM + no_think_suffix(PRIMARY_MODEL)},
                        {"role": "user",   "content": prompt},
                    ],
                    "response_format": {"type": "json_object"},
                    "max_tokens": 600,
                    "temperature": 0.3,
                },
            )
        return json.loads(resp.json()["choices"][0]["message"]["content"])
    except Exception as exc:
        logger.error("Nightly review LLM call failed for %s: %s", user_code, type(exc).__name__)
        return None


async def run_nightly_review() -> None:
    """
    Nightly per-user conversation review. Called by APScheduler at 23:00.
    Replaces the external nightly-reflection.py cron script.

    For each user with conversations yesterday:
      - Calls LLM to extract insights, summary, tomorrow suggestions
      - Stores tomorrow_suggestions in Redis (TTL 24h)
      - Appends learnings and growth_log to jarvis-self.json
      - Triggers monthly memory consolidation on day 1

    Idempotent: one lock per user per date (Redis key, TTL 25h).
    """
    logger.info("=== Nightly review starting ===")
    r = _redis()
    now = datetime.now(timezone.utc)
    yesterday = now - timedelta(days=1)
    review_date = yesterday.strftime("%Y-%m-%d")
    start_ts = yesterday.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    end_ts   = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999).timestamp()

    data = load_self()
    changed = False

    for user_code, user_name in USER_CODES.items():
        lock_key = f"jarvis:{user_code}:nightly_review:{review_date}"
        if not r.set(lock_key, "1", nx=True, ex=90000):   # 25h TTL
            logger.info("Nightly review already done for %s on %s — skipping", user_code, review_date)
            continue

        entries_raw = r.zrangebyscore(f"episodic:{user_code}:conversations", start_ts, end_ts)
        if not entries_raw:
            logger.info("No conversations for %s on %s — skipping", user_code, review_date)
            continue

        conversations = []
        for raw in entries_raw:
            try:
                conversations.append(json.loads(raw))
            except Exception:
                pass

        logger.info("Nightly review for %s: %d conversations", user_code, len(conversations))
        review = await _nightly_review_user(user_code, user_name, conversations, review_date)
        if review is None:
            continue

        # Store tomorrow's suggestions in Redis (24h)
        suggestions = review.get("tomorrow_suggestions", [])
        if suggestions:
            r.setex(f"jarvis:{user_code}:tomorrow_suggestions", 86400, json.dumps(suggestions))

        # Append learnings from self_reflections
        for refl in review.get("self_reflections", []):
            if refl:
                data.setdefault("learnings", []).append({
                    "text":   refl,
                    "date":   review_date,
                    "user":   user_code,
                    "source": "nightly_review",
                })
                changed = True

        # Append growth log entry
        summary = review.get("daily_summary", "")
        if summary:
            data.setdefault("growth_log", []).append({
                "date":          review_date,
                "user":          user_code,
                "summary":       summary,
                "mood":          review.get("mood_summary", ""),
                "conversations": len(conversations),
            })
            changed = True

        logger.info("Nightly review done for %s — %s", user_code, summary[:80])

        # Monthly memory consolidation on day 1
        if now.day == 1:
            try:
                from memory import _consolidate_user_memories
                await asyncio.to_thread(_consolidate_user_memories, user_code)
                logger.info("Monthly memory consolidation done for %s", user_code)
            except Exception as exc:
                logger.warning("Monthly consolidation failed for %s: %s", user_code, type(exc).__name__)

    if changed:
        # Trim to sane limits before saving
        data["learnings"]  = data.get("learnings",  [])[-100:]
        data["growth_log"] = data.get("growth_log", [])[-365:]
        _save_self(data)

    logger.info("=== Nightly review complete ===")


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
    isin      = params.get("isin", "").strip().upper()
    th        = params.get("threshold_high")
    tl        = params.get("threshold_low")

    if not user_code or user_code not in USER_CODES:
        return "update_trade_threshold: invalid user_code"
    if not isin:
        return "update_trade_threshold: missing isin"
    if th is None and tl is None:
        return "update_trade_threshold: at least one of threshold_high / threshold_low is required"

    try:
        from trading import _get_redis as _trade_redis, _pos_key, _idx_key
    except ImportError as exc:
        return f"update_trade_threshold: trading module unavailable ({exc})"

    r = _trade_redis()
    if isin not in r.smembers(_idx_key(user_code)):
        return f"update_trade_threshold: ISIN {isin} not in portfolio for {user_code}"

    key     = _pos_key(user_code, isin)
    mapping = {}
    parts   = []

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
    result   = f"threshold updated for {pos_name} ({isin}): {', '.join(parts)}"
    logger.info("Self action: %s", result)
    return result


_ACTION_CATALOG = {
    "nothing":                  _action_nothing,
    "store_insight":            _action_store_insight,
    "flag_knowledge_gap":       _action_flag_knowledge_gap,
    "send_notification":        _action_send_notification,
    "update_self_note":         _action_update_self_note,
    "consolidate_memory":       _action_consolidate_memory,
    "check_health":             _action_check_health,
    "update_trade_threshold":   _action_update_trade_threshold,
    # nightly_review is scheduled automatically — not in LLM action catalog
}


def _execute_action(action: str, params: dict) -> str:
    fn = _ACTION_CATALOG.get(action)
    if fn is None:
        logger.warning("Self: unknown action requested — %r (defaulting to nothing)", action)
        return f"unknown action: {action}"
    return fn(params)


# ══════════════════════════════════════════════════
#  MAIN REFLECTION ENTRY POINT
# ══════════════════════════════════════════════════

async def run_reflection() -> dict:
    """
    Full reflection cycle. Called by APScheduler every REFLECTION_INTERVAL_HOURS.
    Returns the reflection result dict.
    """
    logger.info("=== Jarvis self-reflection starting ===")

    context = gather_context()
    result  = await _call_reflection_llm(context)

    if result is None:
        result = {"focus": "reflection failed", "action": "nothing", "reason": "LLM call failed", "params": {}}

    focus  = result.get("focus",  "").strip()
    action = result.get("action", "nothing").strip()
    reason = result.get("reason", "").strip()
    params = result.get("params", {})

    # Clamp to catalog
    if action not in _ACTION_CATALOG:
        action = "nothing"
        params = {"reason": f"unknown action requested: {result.get('action')}"}

    outcome = await asyncio.to_thread(_execute_action, action, params)

    # Persist focus + reflection metadata
    now_iso = datetime.now(timezone.utc).isoformat()
    data = load_self()
    data["current_focus"]   = focus
    data["last_reflection"] = now_iso
    data["reflection_count"] = data.get("reflection_count", 0) + 1
    _save_self(data)

    log_entry = {
        "timestamp": now_iso,
        "focus":     focus,
        "action":    action,
        "reason":    reason,
        "params":    params,
        "outcome":   outcome,
        "health":    context["health"],
    }
    log_reflection(log_entry)

    logger.info("=== Reflection complete: focus=%r action=%s outcome=%s ===", focus, action, outcome)
    return log_entry

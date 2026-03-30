"""
Jarvis — Common Helpers
=======================
Shared utilities used across the project.

Time / Timezone
---------------
All user-visible datetimes must use the user's own timezone, read from
users_list.json.  Internal timestamps (Redis keys, logs, DB records)
stay in UTC.

Connections
-----------
Single connection singletons (Redis + Qdrant) shared by all modules.
Redis JSON helpers wrap the common get/set + json.loads/dumps pattern.

LLM
---
Robust JSON extraction from LLM responses (handles markdown code fences
and prose-prefixed output from older mlx-lm versions).

Weather
-------
WMO weather code → French description mapping (Open-Meteo standard).

Public API
----------
Timezone
  get_user_tz(user_code)            -> pytz.BaseTzInfo
  now_user(user_code)               -> datetime (tz-aware, user local)
  today_user(user_code)             -> date (user local)
  fmt_event_time(iso, user_code)    -> str  "DD/MM HH:MM" or "YYYY-MM-DD" for all-day

Connections
  get_redis()                       -> redis.Redis
  get_qdrant()                      -> QdrantClient
  redis_get_json(key, default)      -> any
  redis_set_json(key, data, ttl)    -> None

LLM calls
  call_llm(messages, *, model, api_url, api_key, ...)        -> str  (sync)
  call_llm_async(messages, *, model, api_url, api_key, ...)  -> str  (async)
  Both share a persistent, connection-pooled httpx client.
  no_think_suffix is applied automatically for Qwen models.
  API keys are never logged.

LLM parsing
  extract_llm_json(raw)             -> dict  (raises json.JSONDecodeError on failure)

Weather
  WEATHER_CODES                     -> dict[int, str]
"""

import json
import logging
import os
import re
from datetime import date, datetime
from logging.handlers import RotatingFileHandler
from threading import Lock

import httpx
import pytz
import redis
from qdrant_client import QdrantClient

from config import QDRANT_URL, REDIS_URL, USERS, no_think_suffix, tokens_param

# ══════════════════════════════════════════════════
#  LOGGING SETUP
# ══════════════════════════════════════════════════

_LOG_FORMAT = "%(asctime)s  %(name)-24s  %(levelname)s  %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
_logging_configured = False


def setup_logging(log_file: str = "/app/logs/jarvis-api.log") -> None:
    """
    Configure the root Jarvis logger once: console + rotating file handlers.
    - jarvis-api.log  : INFO+  (5 MB × 3, operational)
    - jarvis-debug.log: DEBUG+ (10 MB × 2, verbose — for review)
    Safe to call multiple times (no-op after first call).
    """
    global _logging_configured
    if _logging_configured:
        return
    _logging_configured = True

    fmt = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console — INFO only
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    # INFO+ rotating file: 5 MB × 3 backups
    fh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # DEBUG+ rotating file: 10 MB × 2 backups
    debug_file = os.path.join(log_dir, "jarvis-debug.log")
    dfh = RotatingFileHandler(debug_file, maxBytes=10 * 1024 * 1024, backupCount=2, encoding="utf-8")
    dfh.setLevel(logging.DEBUG)
    dfh.setFormatter(fmt)
    root.addHandler(dfh)

    # Quiet noisy third-party loggers
    for noisy in ("httpx", "httpcore", "primp", "sentence_transformers",
                  "apscheduler", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named Jarvis logger. Thin wrapper around logging.getLogger."""
    return logging.getLogger(name)


logger = get_logger("jarvis-helpers")

_UTC = pytz.UTC


# ══════════════════════════════════════════════════
#  TIMEZONE HELPERS
# ══════════════════════════════════════════════════

def get_user_tz(user_code: str) -> pytz.BaseTzInfo:
    """Return the pytz timezone for a user. Defaults to UTC on unknown code or bad name."""
    tz_name = USERS.get(user_code, {}).get("timezone", "UTC")
    try:
        return pytz.timezone(tz_name)
    except pytz.UnknownTimeZoneError:
        logger.warning("Unknown timezone %r for user %s — falling back to UTC", tz_name, user_code)
        return _UTC


def now_user(user_code: str) -> datetime:
    """Current datetime in the user's timezone."""
    return datetime.now(get_user_tz(user_code))


def today_user(user_code: str) -> date:
    """Current date in the user's timezone."""
    return now_user(user_code).date()


def build_iso_dt(date_str: str, time_str: str, tz_name: str) -> str:
    """
    Build an ISO 8601 datetime string with timezone offset.
    date_str: "YYYY-MM-DD", time_str: "HH:MM", tz_name: e.g. "Europe/Paris"
    Returns e.g. "2026-03-25T14:00:00+01:00"
    """
    tz = pytz.timezone(tz_name)
    naive = datetime(
        int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10]),
        int(time_str[:2]), int(time_str[3:5]),
    )
    return tz.localize(naive).isoformat()


def fmt_event_time(iso: str, user_code: str, fmt: str = "%d/%m %H:%M") -> str:
    """
    Convert an ISO 8601 datetime string (with or without UTC offset) to the
    user's local timezone and format it.

    All-day events (date-only strings like "2026-03-21") are returned as-is.
    Returns the raw string on any parse error.
    """
    if not iso or len(iso) <= 10:
        return iso  # all-day event — no time component

    try:
        # Python < 3.11 does not accept "Z" as UTC in fromisoformat — normalize first.
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_UTC)
        return dt.astimezone(get_user_tz(user_code)).strftime(fmt)
    except (ValueError, OverflowError):
        return iso


_JOURS_FR = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
_MOIS_FR  = ("", "janvier", "février", "mars", "avril", "mai", "juin",
             "juillet", "août", "septembre", "octobre", "novembre", "décembre")


def fmt_now_fr(tz_name: str) -> str:
    """Return current datetime formatted in French for the given IANA timezone.

    Example: 'lundi 30 mars 2026, 14:32'
    """
    now = datetime.now(pytz.timezone(tz_name))
    jour = _JOURS_FR[now.weekday()]
    return f"{jour} {now.day} {_MOIS_FR[now.month]} {now.year}, {now.strftime('%H:%M')}"


def rel_time_fr(ts: float) -> str:
    """Return a French relative time string for a Unix timestamp.

    Examples: 'il y a 3 jours', 'il y a 2 semaines', 'il y a 1 mois'
    """
    import time as _time
    delta = _time.time() - ts
    if delta < 3600:
        m = max(1, int(delta / 60))
        return f"il y a {m} min"
    if delta < 86400:
        h = int(delta / 3600)
        return f"il y a {h}h"
    if delta < 7 * 86400:
        d = int(delta / 86400)
        return f"il y a {d} jour{'s' if d > 1 else ''}"
    if delta < 30 * 86400:
        w = int(delta / (7 * 86400))
        return f"il y a {w} semaine{'s' if w > 1 else ''}"
    if delta < 365 * 86400:
        mo = int(delta / (30 * 86400))
        return f"il y a {mo} mois"
    y = int(delta / (365 * 86400))
    return f"il y a {y} an{'s' if y > 1 else ''}"


# ══════════════════════════════════════════════════
#  CONNECTIONS — Redis + Qdrant
# ══════════════════════════════════════════════════

_redis_client: redis.Redis | None = None
_redis_lock = Lock()
_qdrant_client: QdrantClient | None = None
_qdrant_lock = Lock()


def get_redis() -> redis.Redis:
    """Return the shared Redis connection (singleton, thread-safe)."""
    global _redis_client
    if _redis_client is None:
        with _redis_lock:
            if _redis_client is None:
                _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def get_qdrant() -> QdrantClient:
    """Return the shared Qdrant connection (singleton, thread-safe)."""
    global _qdrant_client
    if _qdrant_client is None:
        with _qdrant_lock:
            if _qdrant_client is None:
                _qdrant_client = QdrantClient(url=QDRANT_URL, timeout=5.0)
    return _qdrant_client


def redis_get_json(key: str, default=None):
    """
    Get a Redis key and deserialise it from JSON.
    Returns default if the key is missing or on any error.
    """
    try:
        data = get_redis().get(key)
        return json.loads(data) if data else default
    except Exception as exc:
        logger.warning("redis_get_json failed for key %s: %s", key, type(exc).__name__)
        return default


def redis_set_json(key: str, data, ttl: int | None = None) -> None:
    """
    Serialise data to JSON and store it in Redis.
    Pass ttl (seconds) to use SETEX instead of SET.
    """
    try:
        payload = json.dumps(data, ensure_ascii=False)
        r = get_redis()
        if ttl:
            r.setex(key, ttl, payload)
        else:
            r.set(key, payload)
    except Exception as exc:
        logger.warning("redis_set_json failed for key %s: %s", key, type(exc).__name__)


# ══════════════════════════════════════════════════
#  LLM JSON EXTRACTION
# ══════════════════════════════════════════════════

def extract_llm_json(raw: str) -> dict:
    """
    Parse JSON from a model response robustly.

    Strategy:
    1. Direct parse (works when response_format is supported).
    2. Strip markdown code fences if present.
    3. Extract the first {...} block (mlx-lm fallback for older versions
       that ignore response_format, or models that prefix with prose).

    Raises json.JSONDecodeError if all strategies fail.
    """
    raw = raw.strip()

    # 1. Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences
    if "```" in raw:
        inner = raw.split("```")[1]
        first_newline = inner.find("\n")
        if first_newline != -1 and not inner[:first_newline].strip().startswith("{"):
            inner = inner[first_newline:].strip()
        try:
            return json.loads(inner.strip())
        except json.JSONDecodeError:
            pass

    # 3. Extract first {...} block (handles prose-prefixed responses)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group())

    raise json.JSONDecodeError("No JSON found in LLM response", raw, 0)


# ══════════════════════════════════════════════════
#  LLM HTTP CLIENTS  (shared, connection-pooled)
# ══════════════════════════════════════════════════

# Single persistent client per mode — avoids TCP handshake overhead on every call.
# Sync: used by memory.py background tasks (not in async event loop).
# Async: used by all FastAPI route handlers and background coroutines.
_LLM_LIMITS = httpx.Limits(max_connections=20, max_keepalive_connections=10)
_llm_sync_client: httpx.Client | None = None
_llm_sync_lock = Lock()
_llm_async_client: httpx.AsyncClient | None = None


def _get_llm_sync_client() -> httpx.Client:
    global _llm_sync_client
    if _llm_sync_client is None:
        with _llm_sync_lock:
            if _llm_sync_client is None:
                _llm_sync_client = httpx.Client(limits=_LLM_LIMITS)
    return _llm_sync_client


def _get_llm_async_client() -> httpx.AsyncClient:
    """No lock needed: asyncio is single-threaded, no concurrent init risk."""
    global _llm_async_client
    if _llm_async_client is None:
        _llm_async_client = httpx.AsyncClient(limits=_LLM_LIMITS)
    return _llm_async_client


def _llm_headers(api_key: str) -> dict:
    """Build auth headers. Key value is never stored in logs or tracebacks."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _llm_body(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    json_response: bool,
    no_think: bool = False,
) -> dict:
    """
    Build the JSON body for a /chat/completions request.

    - no_think=True appends /no_think to the last message (Qwen only, no-op elsewhere).
      Use for fast JSON-output tasks (router, analyzer, query builders) where
      chain-of-thought would break parsing or waste tokens.
      Leave False (default) for reasoning-heavy tasks (self-reflection, user questions).
    - Uses tokens_param() to pick max_tokens vs max_completion_tokens.
    - Sets response_format when json_response=True.
    """
    if no_think:
        suffix = no_think_suffix(model)
        if suffix:
            last = messages[-1]
            messages = [*messages[:-1], {**last, "content": last["content"] + suffix}]

    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        tokens_param(model): max_tokens,
    }
    if json_response:
        body["response_format"] = {"type": "json_object"}
    return body


def call_llm(
    messages: list[dict],
    *,
    model: str,
    api_url: str,
    api_key: str,
    temperature: float = 0.1,
    max_tokens: int = 500,
    json_response: bool = True,
    no_think: bool = False,
    timeout: float = 30.0,
) -> str:
    """
    Synchronous OpenAI-compatible LLM call.

    Returns the model's raw text content (choices[0].message.content).
    Raises httpx.HTTPStatusError on non-2xx responses.
    Raises httpx.TimeoutException on timeout.
    API key is never logged.
    """
    resp = _get_llm_sync_client().post(
        f"{api_url}/chat/completions",
        headers=_llm_headers(api_key),
        json=_llm_body(messages, model, temperature, max_tokens, json_response, no_think),
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def call_llm_async(
    messages: list[dict],
    *,
    model: str,
    api_url: str,
    api_key: str,
    temperature: float = 0.1,
    max_tokens: int = 500,
    json_response: bool = True,
    no_think: bool = False,
    timeout: float = 30.0,
) -> str:
    """
    Async OpenAI-compatible LLM call.

    Returns the model's raw text content (choices[0].message.content).
    Raises httpx.HTTPStatusError on non-2xx responses.
    Raises httpx.TimeoutException on timeout.
    API key is never logged.
    """
    resp = await _get_llm_async_client().post(
        f"{api_url}/chat/completions",
        headers=_llm_headers(api_key),
        json=_llm_body(messages, model, temperature, max_tokens, json_response, no_think),
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ══════════════════════════════════════════════════
#  WEATHER CODES  (WMO standard — Open-Meteo)
# ══════════════════════════════════════════════════

WEATHER_CODES: dict[int, str] = {
    0: "Ciel dégagé",
    1: "Principalement dégagé",
    2: "Partiellement nuageux",
    3: "Couvert",
    45: "Brouillard",
    48: "Brouillard givrant",
    51: "Bruine légère",
    53: "Bruine modérée",
    55: "Bruine dense",
    61: "Pluie faible",
    63: "Pluie modérée",
    65: "Pluie forte",
    71: "Neige faible",
    73: "Neige modérée",
    75: "Neige forte",
    80: "Averses faibles",
    81: "Averses modérées",
    82: "Averses violentes",
    95: "Orage",
    96: "Orage avec grêle",
    99: "Orage violent avec grêle",
}

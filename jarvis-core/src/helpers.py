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
  filter_think_chunk(chunk, in_think) -> (visible_text, new_in_think)  (streaming)

Weather
  WEATHER_CODES                     -> dict[int, str]
"""

import json
import logging
import os
import re
import sys
import time
import unicodedata
from datetime import date, datetime
from logging.handlers import RotatingFileHandler
from threading import Lock

import httpx
import pytz
import redis
from config import (
    CHAT_LOG_TTL,
    LLM_LOCAL,
    PRIMARY_MODEL,
    QDRANT_URL,
    REASONING_MODEL,
    REDIS_URL,
    ROUTER_MODEL,
    USERS,
    tokens_param,
)
from llm_local import call_llm_local, call_llm_local_async
from qdrant_client import QdrantClient

_LOCAL_MODELS = {ROUTER_MODEL, PRIMARY_MODEL, REASONING_MODEL} if LLM_LOCAL else set()

# ══════════════════════════════════════════════════
#  LOGGING SETUP
# ══════════════════════════════════════════════════

_LOG_FORMAT = "%(asctime)s  %(name)-24s  %(levelname)s  %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
_logging_configured = False


def setup_logging(log_file: str = "/opt/jarvis/logs/jarvis-api.log") -> None:
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

    # Console — INFO only (stdout so launchd's StandardOutPath captures it)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    # INFO+ rotating file: 5 MB × 3 backups
    fh = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # DEBUG+ rotating file: 10 MB × 2 backups
    debug_file = os.path.join(log_dir, "jarvis-debug.log")
    dfh = RotatingFileHandler(
        debug_file, maxBytes=10 * 1024 * 1024, backupCount=2, encoding="utf-8"
    )
    dfh.setLevel(logging.DEBUG)
    dfh.setFormatter(fmt)
    root.addHandler(dfh)

    # Quiet noisy third-party loggers
    for noisy in (
        "httpx",
        "httpcore",
        "primp",
        "sentence_transformers",
        "apscheduler",
        "urllib3",
        "asyncio",
        "rustls",
        "hyper_util",
        "h2",
        "reqwest",
        "hyper",
        "ddgs",
        "ddgs.ddgs",
    ):
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
        logger.warning(
            "Unknown timezone %r for user %s — falling back to UTC", tz_name, user_code
        )
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
        int(date_str[:4]),
        int(date_str[5:7]),
        int(date_str[8:10]),
        int(time_str[:2]),
        int(time_str[3:5]),
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
_MOIS_FR = (
    "",
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
)


_SEASONS_FR = {
    12: "hiver",
    1: "hiver",
    2: "hiver",
    3: "printemps",
    4: "printemps",
    5: "printemps",
    6: "été",
    7: "été",
    8: "été",
    9: "automne",
    10: "automne",
    11: "automne",
}


def fmt_now_fr(tz_name: str) -> str:
    """Return current datetime + season formatted in French for the given IANA timezone.

    Example: 'lundi 30 mars 2026, 14:32 (printemps)'
    """
    now = datetime.now(pytz.timezone(tz_name))
    jour = _JOURS_FR[now.weekday()]
    saison = _SEASONS_FR[now.month]
    return f"{jour} {now.day} {_MOIS_FR[now.month]} {now.year}, {now.strftime('%H:%M')} ({saison})"


def fmt_date_fr(d: date) -> str:
    """Return a short French date label: 'Dimanche 10 mai'."""
    return f"{_JOURS_FR[d.weekday()].capitalize()} {d.day} {_MOIS_FR[d.month]}"


def rel_time_fr(ts: float) -> str:
    """Return a French relative time string for a Unix timestamp.

    Examples: 'il y a 3 jours', 'il y a 2 semaines', 'il y a 1 mois'
    """
    delta = time.time() - ts
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


def normalize_key(s: str) -> str:
    """
    Strong normalization for profile keys:
    - lowercase
    - remove accents
    - normalize separators (space, dash → underscore)
    - trim
    - collapse multiple underscores
    """
    if not s:
        return ""
    # Unicode normalize + remove accents
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    # Lowercase + trim
    s = s.lower().strip()
    # Normalize separators
    s = s.replace("-", "_").replace(" ", "_")
    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s)
    return s


_FR_STOPWORDS = {
    "le", "la", "les", "de", "du", "des", "un", "une", "en", "et", "ou", "à",
    "au", "aux", "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "que",
    "qui", "est", "ce", "se", "ne", "pas", "plus", "sur", "par", "pour", "avec",
    "dans", "mais", "si", "car", "donc", "son", "sa", "ses", "mon", "ma", "mes",
    "ton", "ta", "tes", "leur", "leurs", "on", "me", "te", "lui", "eux",
    # demonstratives / relatives
    "cela", "ceci", "cette", "cet", "ces", "dont", "aussi",
    # common adverbs that carry no topic content
    "très", "déjà",
}


def keyword_overlap_score(a: str, b: str) -> int:
    """Count shared content words between two French strings (stopwords excluded).

    Uppercase 2-char acronyms (IA, ML, AI…) are included even though the
    length threshold for lowercase words is 3+, because they carry real meaning
    in topic matching (an "IA" opinion should surface on IA-related queries).
    """
    def tokens(s: str) -> set[str]:
        result: set[str] = set()
        for w in re.sub(r"[^\w]", " ", s).split():
            wl = w.lower()
            if wl in _FR_STOPWORDS:
                continue
            # Uppercase acronyms (IA, ML, …): include at length ≥ 2
            if w.isupper() and len(wl) >= 2:
                result.add(wl)
            elif len(wl) > 2:
                result.add(wl)
        return result
    return len(tokens(a) & tokens(b))


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
                _redis_client = redis.from_url(
                    REDIS_URL,
                    decode_responses=True,
                    socket_keepalive=True,
                    health_check_interval=30,
                    retry_on_timeout=True,
                )
    return _redis_client


_SESSION_SUMMARY_PREFIX = "session:summary:"


def get_session_summary_data(user_code: str, session_id: str) -> dict | None:
    """Return {text, msg_count} for the session conversation summary, or None."""
    raw = get_redis().get(f"{_SESSION_SUMMARY_PREFIX}{user_code}:{session_id}")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def set_session_summary_data(user_code: str, session_id: str, text: str, msg_count: int) -> None:
    """Persist conversation summary with its coverage watermark."""
    data = json.dumps({"text": text, "msg_count": msg_count}, ensure_ascii=False)
    get_redis().setex(
        f"{_SESSION_SUMMARY_PREFIX}{user_code}:{session_id}",
        CHAT_LOG_TTL,
        data,
    )


_STICKY_RAG_PREFIX = "jarvis:sticky_rag:"


def get_sticky_rag(user_code: str, session_id: str) -> list | None:
    """Return the last RAG chunks stored for this session, or None."""
    raw = get_redis().get(f"{_STICKY_RAG_PREFIX}{user_code}:{session_id}")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def set_sticky_rag(user_code: str, session_id: str, chunks: list) -> None:
    """Persist RAG chunks for automatic re-injection on subsequent memory turns."""
    try:
        get_redis().setex(
            f"{_STICKY_RAG_PREFIX}{user_code}:{session_id}",
            CHAT_LOG_TTL,
            json.dumps(chunks, ensure_ascii=False),
        )
    except Exception as exc:
        logger.warning("set_sticky_rag failed: %s", exc)


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


def _repair_json(text: str) -> str:
    """Best-effort repair of common LLM JSON generation mistakes.

    Currently handles:
    - Missing opening quote on object keys:  action": "x"  →  "action": "x"
      Pattern: a bare word followed by `":` that is NOT already preceded by `"`.
    """
    return re.sub(r'(?<!")\b([a-zA-Z_]\w*)(":\s*)', r'"\1\2', text)


def filter_think_chunk(chunk: str, in_think: bool) -> tuple[str, str, bool]:
    """Split a single SSE chunk into visible text and think-block content.

    Correctly handles chunks that carry text *before* an opening tag or *after*
    a closing tag — cases a simple flag-only approach silently drops.

    Returns:
        (visible_text, think_fragment, new_in_think_state)
    """
    visible: list[str] = []
    thinking: list[str] = []
    while chunk:
        if not in_think:
            pos = chunk.find("<think>")
            if pos == -1:
                visible.append(chunk)
                break
            if pos > 0:
                visible.append(chunk[:pos])
            chunk = chunk[pos + 7 :]  # advance past <think>
            in_think = True
        else:
            pos = chunk.find("</think>")
            if pos == -1:
                thinking.append(chunk)  # whole remainder is think content
                break
            thinking.append(chunk[:pos])
            chunk = chunk[pos + 8 :]  # advance past </think>
            in_think = False
    return "".join(visible), "".join(thinking), in_think


def extract_llm_json(text: str) -> dict:
    """
    Extraction robuste de JSON depuis une réponse LLM.
    Gère :
    - reasoning avant/après
    - texte parasite
    - multiples blocs JSON
    """

    if not text:
        raise ValueError("Empty LLM response")

    # ── 1. Nettoyage agressif ─────────────────────────────

    # remove <think> blocks (Qwen3)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # remove known reasoning markers
    if "Final Answer:" in text:
        text = text.split("Final Answer:")[-1]

    if "Thinking Process:" in text:
        text = text.split("Thinking Process:")[-1]

    # Strip backtick quote-wrappers that some models emit instead of double-quotes.
    # e.g. {`"key"`: `"value"`}  →  {"key": "value"}  — backticks are never valid JSON.
    if "`" in text:
        text = text.replace("`", "")

    text = text.strip()

    # ── 2. Extraction JSON par parsing équilibré ──────────

    start = text.find("{")
    if start == -1:
        # Detect wrong-type JSON (array, scalar) vs total garbage
        try:
            parsed = json.loads(text)
            raise ValueError(
                f"LLM returned {type(parsed).__name__} instead of JSON object: {text[:200]}"
            )
        except json.JSONDecodeError:
            pass
        raise ValueError(f"No JSON found in LLM response: {text[:200]}")

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        char = text[i]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1

            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    # object_pairs_hook keeps the FIRST value when a model emits
                    # duplicate keys (e.g. Hermes router repeating its JSON 3× inside
                    # one {…}).  reversed() + dict-comp: later duplicates overwrite
                    # earlier ones, so after reversal the first occurrence wins.
                    return json.loads(
                        candidate,
                        object_pairs_hook=lambda pairs: {
                            k: v for k, v in reversed(list(pairs))
                        },
                    )
                except json.JSONDecodeError:
                    break  # fallback

    # ── 3. Retry with malformed-key fixes ────────────────────
    # Pattern A: fully unquoted key — action: "nothing" → "action": "nothing"
    _unquoted_key_re = re.compile(r"([{,\n]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:(?!\s*/))")
    # Pattern B: missing opening quote — ,params": → ,"params":
    # Covers Qwen3.6 bug where model emits ,key": instead of ,"key":
    _half_quoted_key_re = re.compile(r'([{,\n]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(":\s*)')

    matches = re.findall(r"\{.*\}", text, re.DOTALL)
    for candidate in reversed(matches):  # try biggest first
        try:
            fixed = _half_quoted_key_re.sub(
                lambda m: m.group(1) + '"' + m.group(2) + m.group(3),
                candidate,
            )
            fixed = _unquoted_key_re.sub(
                lambda m: m.group(1) + '"' + m.group(2) + '"' + m.group(3),
                fixed,
            )
            return json.loads(fixed)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Invalid JSON in LLM response: {text[:200]}")


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


_LOCAL_DEFAULT_MAX_TOKENS = (
    10000  # global ceiling — thinking + output are counted together in mlx-lm
)


def _llm_body(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int | None,
    json_response: bool,
) -> dict:
    """
    Build the JSON body for a /chat/completions request.

    - Uses tokens_param() to pick max_tokens vs max_completion_tokens.
    - Sets response_format when json_response=True.
    - max_tokens=None → field omitted → API uses model default (no truncation risk).
    - Thinking control (no_think) is handled at the MLX prompt level for local
      models (_build_prompt via enable_thinking), not at the HTTP body level.
    """

    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        body[tokens_param(model)] = max_tokens
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
    max_tokens: int | None = None,
    json_response: bool = True,
    no_think: bool = False,
    timeout: float = 30.0,
    thinking_budget: int = 0,
) -> str:
    """
    Synchronous LLM call — HTTP (cloud/mlx-lm server) ou MLX direct (LLM_LOCAL=yes).

    Returns the model's raw text content.
    API key is never logged.
    max_tokens=None → no explicit limit (model stops at EOS / closing JSON brace).
    thinking_budget > 0 : cap de tokens de thinking (local uniquement, ignoré pour HTTP).
    """
    if LLM_LOCAL and model in _LOCAL_MODELS:
        return call_llm_local(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens or _LOCAL_DEFAULT_MAX_TOKENS,
            no_think=no_think,
            json_response=json_response,
            thinking_budget=thinking_budget,
        )
    resp = _get_llm_sync_client().post(
        f"{api_url}/chat/completions",
        headers=_llm_headers(api_key),
        json=_llm_body(messages, model, temperature, max_tokens, json_response),
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
    max_tokens: int | None = None,
    json_response: bool = True,
    no_think: bool = False,
    timeout: float = 30.0,
    thinking_budget: int = 0,
) -> str:
    """
    Async LLM call — HTTP (cloud/mlx-lm server) ou MLX direct (LLM_LOCAL=yes).

    Returns the model's raw text content.
    API key is never logged.
    max_tokens=None → no explicit limit (model stops at EOS / closing JSON brace).
    thinking_budget > 0 : cap de tokens de thinking (local uniquement, ignoré pour HTTP).
    """
    if LLM_LOCAL and model in _LOCAL_MODELS:
        return await call_llm_local_async(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens or _LOCAL_DEFAULT_MAX_TOKENS,
            no_think=no_think,
            json_response=json_response,
            thinking_budget=thinking_budget,
        )
    resp = await _get_llm_async_client().post(
        f"{api_url}/chat/completions",
        headers=_llm_headers(api_key),
        json=_llm_body(messages, model, temperature, max_tokens, json_response),
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

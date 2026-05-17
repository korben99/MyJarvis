"""
Jarvis Google Services
======================
Gmail (read + send) and Google Calendar (read) integration.

Security:
- Credentials loaded exclusively from environment variables, never hardcoded
- Credential values are never logged (only error types are logged)
- Thread-safe token refresh via threading.Lock
- All API results cached in Redis (rate limiting by design)
- HttpError status codes logged, not response bodies

Use cases:
- fetch_gmail_messages(query)    : search ALL folders and labels
- fetch_calendar_events(days)    : weekly (7) or monthly (30) agenda view
- send_gmail_message(to, subj, html) : send an email (briefing delivery)
"""

import base64
import email.mime.multipart
import email.mime.text
import hashlib
import json
import re
import threading
from datetime import date as date_type
from datetime import datetime, timedelta, timezone
from email.utils import parseaddr

import google_auth_httplib2
import httplib2
import pytz
from config import (
    BRIEFING_TIMEZONE,
    GOOGLE_CALENDAR_ID,
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_USER_TOKENS,
    MAX_TOKENS_SHORT,
    PRIMARY_API_KEY,
    PRIMARY_API_URL,
    PRIMARY_MODEL,
    ROUTER_API_KEY,
    ROUTER_API_URL,
    ROUTER_MODEL,
)
from google.auth.exceptions import GoogleAuthError, RefreshError, TransportError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from helpers import call_llm_async, extract_llm_json, get_logger, get_redis

logger = get_logger("jarvis-google")

_GOOGLE_API_TIMEOUT = 15  # seconds for all Google API calls

# ── OAuth scopes ──
_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar",
]

# ── Cache TTLs (seconds) ──
_GMAIL_CACHE_TTL = 300  # 5 min
_CALENDAR_CACHE_TTL = 300  # 5 min

# ── Result limits ──
_GMAIL_MAX_RESULTS = 10
_CALENDAR_MAX_RESULTS = 50
_EMAIL_BODY_MAX = 400  # chars per email
_SUBJECT_MAX = 120

# ── Per-user credential / service caches ──
_credentials_cache: dict[str, Credentials] = {}
_gmail_service_cache: dict[str, object] = {}
_calendar_service_cache: dict[str, object] = {}
_creds_lock = (
    threading.RLock()
)  # reentrant: _get_calendar_service → _get_credentials both hold this lock


# ══════════════════════════════════════════════════
#  AVAILABILITY CHECK
# ══════════════════════════════════════════════════


def is_google_available(user_code: str | None = None) -> bool:
    """
    True when Google credentials are present and ready.
    - user_code provided → check that specific user has a token.
    - user_code=None     → True if at least one user has a token (status endpoint).
    """
    if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET):
        return False
    if user_code is not None:
        return user_code in GOOGLE_USER_TOKENS
    return bool(GOOGLE_USER_TOKENS)


# ══════════════════════════════════════════════════
#  CREDENTIALS  (thread-safe per-user refresh)
# ══════════════════════════════════════════════════


def _get_credentials(user_code: str) -> Credentials:
    """
    Return a valid Credentials object for user_code, refreshing when needed.
    Lock prevents concurrent refresh races. Credential values are never logged.
    """
    refresh_token = GOOGLE_USER_TOKENS.get(user_code, "")
    if not refresh_token:
        raise RuntimeError(f"No Google token for user {user_code}")

    with _creds_lock:
        creds = _credentials_cache.get(user_code)
        if creds is None:
            creds = Credentials(
                token=None,
                refresh_token=refresh_token,
                client_id=GOOGLE_CLIENT_ID,
                client_secret=GOOGLE_CLIENT_SECRET,
                token_uri="https://oauth2.googleapis.com/token",
                scopes=_SCOPES,
            )
            _credentials_cache[user_code] = creds

        if not creds.valid:
            try:
                creds.refresh(Request())
                logger.info("Google access token refreshed for %s", user_code)
            except (GoogleAuthError, TransportError) as exc:
                logger.error(
                    "Google credential refresh failed for %s: %s",
                    user_code,
                    type(exc).__name__,
                )
                raise RuntimeError("Google authentication failed") from exc

        return creds


# ══════════════════════════════════════════════════
#  SERVICE CACHE (per user)
# ══════════════════════════════════════════════════


def _make_authorized_http(creds) -> google_auth_httplib2.AuthorizedHttp:
    """Build an AuthorizedHttp with a per-call timeout."""
    return google_auth_httplib2.AuthorizedHttp(
        creds, http=httplib2.Http(timeout=_GOOGLE_API_TIMEOUT)
    )


def _get_gmail_service(user_code: str):
    if user_code not in _gmail_service_cache:
        with _creds_lock:
            if user_code not in _gmail_service_cache:
                creds = _get_credentials(user_code)
                _gmail_service_cache[user_code] = build(
                    "gmail",
                    "v1",
                    http=_make_authorized_http(creds),
                    cache_discovery=False,
                )
    return _gmail_service_cache[user_code]


def _get_calendar_service(user_code: str):
    if user_code not in _calendar_service_cache:
        with _creds_lock:
            if user_code not in _calendar_service_cache:
                creds = _get_credentials(user_code)
                _calendar_service_cache[user_code] = build(
                    "calendar",
                    "v3",
                    http=_make_authorized_http(creds),
                    cache_discovery=False,
                )
    return _calendar_service_cache[user_code]


# ══════════════════════════════════════════════════
#  REDIS CACHE
# ══════════════════════════════════════════════════


def _cache_get(key: str):
    try:
        data = get_redis().get(key)
        return json.loads(data) if data else None
    except Exception:
        return None


def _cache_set(key: str, data, ttl: int):
    try:
        get_redis().setex(key, ttl, json.dumps(data, ensure_ascii=False))
    except Exception:
        pass


def _cache_key(prefix: str, *parts: str) -> str:
    """Build a namespaced, hashed Redis key. Parts are never stored raw."""
    raw = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"google:{prefix}:{digest}"


# ══════════════════════════════════════════════════
#  GMAIL
# ══════════════════════════════════════════════════


def _extract_header(headers: list, name: str) -> str:
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def _decode_email_body(payload: dict) -> str:
    """
    Recursively extract text/plain from a Gmail message payload.
    Falls back to text/html only if no plain part exists.
    """
    mime_type = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data", "")

    if mime_type == "text/plain" and body_data:
        try:
            return (
                base64.urlsafe_b64decode(body_data + "==")
                .decode("utf-8", errors="replace")
                .strip()
            )
        except Exception:
            return ""

    parts = payload.get("parts", [])
    plain, html_fallback = "", ""

    for part in parts:
        part_mime = part.get("mimeType", "")
        part_data = part.get("body", {}).get("data", "")

        if part_mime == "text/plain" and part_data:
            try:
                plain += (
                    base64.urlsafe_b64decode(part_data + "==")
                    .decode("utf-8", errors="replace")
                    .strip()
                )
            except Exception:
                pass
        elif part_mime == "text/html" and part_data and not html_fallback:
            try:
                html_fallback = (
                    base64.urlsafe_b64decode(part_data + "==")
                    .decode("utf-8", errors="replace")
                    .strip()
                )
            except Exception:
                pass
        elif part_mime.startswith("multipart/"):
            plain += _decode_email_body(part)

    return plain or html_fallback


def fetch_gmail_messages(
    query: str, max_results: int = _GMAIL_MAX_RESULTS, user_code: str = ""
) -> list[dict]:
    """
    Search Gmail across ALL folders and labels (including Sent, Archive, Spam).
    Returns list of {subject, from, date, snippet}.
    Results are cached in Redis for _GMAIL_CACHE_TTL seconds.
    """
    if not is_google_available(user_code or None):
        return []

    max_results = max(1, min(max_results, _GMAIL_MAX_RESULTS))
    cache_key = _cache_key("gmail", user_code, query, str(max_results))
    cached = _cache_get(cache_key)
    if cached is not None:
        logger.debug("Gmail cache hit")
        return cached

    try:
        service = _get_gmail_service(user_code)

        # Default to INBOX unless the query already targets a specific folder/label.
        # Without this, Gmail returns messages from ALL folders including Sent.
        _in_keywords = ("in:", "label:", "is:", "from:", "to:")
        if not any(kw in query.lower() for kw in _in_keywords):
            query = f"in:inbox {query}".strip()

        list_resp = (
            service.users()
            .messages()
            .list(
                userId="me",
                q=query,
                maxResults=max_results,
                includeSpamTrash=False,
            )
            .execute()
        )

        message_refs = list_resp.get("messages", [])
        if not message_refs:
            _cache_set(cache_key, [], _GMAIL_CACHE_TTL)
            return []

        results = []
        for ref in message_refs:
            try:
                msg = (
                    service.users()
                    .messages()
                    .get(userId="me", id=ref["id"], format="full")
                    .execute()
                )
                headers = msg.get("payload", {}).get("headers", [])
                subject = _extract_header(headers, "Subject")[:_SUBJECT_MAX]
                sender_raw = _extract_header(headers, "From")
                _, sender_addr = parseaddr(sender_raw)
                date = _extract_header(headers, "Date")

                body = _decode_email_body(msg.get("payload", {}))
                snippet = (body or msg.get("snippet", ""))[:_EMAIL_BODY_MAX]

                results.append(
                    {
                        "subject": subject or "(sans objet)",
                        "from": sender_addr or sender_raw,
                        "date": date,
                        "snippet": snippet,
                    }
                )
            except HttpError as exc:
                logger.warning("Gmail message fetch skipped (HTTP %s)", exc.status_code)
                continue

        _cache_set(cache_key, results, _GMAIL_CACHE_TTL)
        logger.info("Gmail: %d messages retrieved", len(results))
        return results

    except HttpError as exc:
        logger.error("Gmail list error (HTTP %s)", exc.status_code)
        return []
    except RuntimeError:
        # Auth failure already logged in _get_credentials
        return []
    except RefreshError as exc:
        logger.error(
            "Gmail list: OAuth token refresh failed for %s: %s", user_code, exc
        )
        with _creds_lock:
            _credentials_cache.pop(user_code, None)
            _gmail_service_cache.pop(user_code, None)
        return []
    except Exception as exc:
        logger.error("Gmail unexpected error: %s: %s", type(exc).__name__, exc)
        return []


# ══════════════════════════════════════════════════
#  CALENDAR
# ══════════════════════════════════════════════════


def _localize_event_dt(raw: str, tz_name: str) -> str:
    """
    Ensure a Google Calendar dateTime string is timezone-aware.

    Google can return naive dateTime values (no UTC offset, no Z) when the
    event carries a separate "timeZone" field — e.g.:
        {"dateTime": "2026-04-20T11:30:00", "timeZone": "Europe/Paris"}
    Without this function, fromisoformat() would produce a naive datetime that
    helpers.fmt_event_time() silently treats as UTC, adding a spurious +2h
    offset for European users (CET/CEST).

    - If raw is date-only ("2026-04-20") or empty → returned unchanged.
    - If already offset-aware ("...+02:00" or "...Z") → returned unchanged.
    - If naive → localized to tz_name using pytz.localize() (DST-correct).
    """
    if not raw or len(raw) <= 10:
        return raw  # date-only all-day event
    if "Z" in raw or ("+" in raw[10:]) or ("-" in raw[10:]):
        return raw  # already offset-aware
    try:
        tz = pytz.timezone(tz_name)
        dt = datetime.fromisoformat(raw)
        return tz.localize(dt).isoformat()
    except Exception:
        return raw


def fetch_calendar_events(
    days: int = 7,
    date: date_type | None = None,
    tz_name: str | None = None,
    user_code: str = "",
) -> list[dict]:
    """
    Fetch calendar events across all calendars.

    date=None       → from now to now+days (rolling window, default)
    date=<date>     → full day midnight-to-midnight in tz_name (used by briefing)
    tz_name=None    → falls back to BRIEFING_TIMEZONE
    Results cached in Redis for _CALENDAR_CACHE_TTL seconds.
    """
    if not is_google_available(user_code or None):
        return []

    effective_tz = tz_name or BRIEFING_TIMEZONE

    if date is not None:
        cache_key = _cache_key(
            "calendar_today", user_code, date.strftime("%Y-%m-%d"), effective_tz
        )
    else:
        days = max(1, min(days, 90))
        cache_key = _cache_key("calendar", user_code, str(days))

    cached = _cache_get(cache_key)
    if cached is not None:
        logger.debug("Calendar cache hit (%s)", cache_key)
        return cached

    try:
        service = _get_calendar_service(user_code)

        if date is not None:
            tz = pytz.timezone(effective_tz)
            # tz.localize() is required with pytz — datetime(..., tzinfo=tz) uses
            # the historical LMT offset (+00:09:21 for Paris) instead of CET/CEST.
            start = tz.localize(datetime(date.year, date.month, date.day))
            time_min = start.isoformat()
            time_max = (start + timedelta(days=1)).isoformat()
        else:
            now = datetime.now(timezone.utc)
            time_min = now.isoformat()
            time_max = (now + timedelta(days=days)).isoformat()

        # Fetch all calendars the account has access to
        cal_list = service.calendarList().list().execute()
        calendar_ids = [c["id"] for c in cal_list.get("items", [])] or [
            GOOGLE_CALENDAR_ID
        ]

        seen_ids: set = set()
        results = []
        for cal_id in calendar_ids:
            resp = (
                service.events()
                .list(
                    calendarId=cal_id,
                    timeMin=time_min,
                    timeMax=time_max,
                    maxResults=_CALENDAR_MAX_RESULTS,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
            for event in resp.get("items", []):
                event_id = event.get("id", "")
                if event_id and event_id in seen_ids:
                    continue  # shared event already added from another calendar
                if event_id:
                    seen_ids.add(event_id)
                start = event.get("start", {})
                end = event.get("end", {})
                # Google can return naive dateTime (no offset) when the event
                # has a separate "timeZone" field. Localize using that timezone
                # (fallback: effective_tz) so fmt_event_time doesn't interpret
                # them as UTC and add a spurious +2h offset.
                event_tz = start.get("timeZone") or end.get("timeZone") or effective_tz
                results.append(
                    {
                        "summary": event.get("summary", "(sans titre)"),
                        "start": _localize_event_dt(
                            start.get("dateTime", start.get("date", "")), event_tz
                        ),
                        "end": _localize_event_dt(
                            end.get("dateTime", end.get("date", "")), event_tz
                        ),
                        "location": event.get("location", ""),
                        "description": (event.get("description") or "")[:200],
                        "all_day": "dateTime" not in start,
                    }
                )

        results.sort(key=lambda e: e["start"])
        _cache_set(cache_key, results, _CALENDAR_CACHE_TTL)
        label = date.isoformat() if date is not None else f"next {days}d"
        logger.info(
            "Calendar: %d events (%s, across %d calendars)",
            len(results),
            label,
            len(calendar_ids),
        )
        return results

    except HttpError as exc:
        logger.error("Calendar API error (HTTP %s)", exc.status_code)
        return []
    except RuntimeError:
        return []
    except RefreshError as exc:
        logger.error("Calendar: OAuth token refresh failed for %s: %s", user_code, exc)
        with _creds_lock:
            _credentials_cache.pop(user_code, None)
            _calendar_service_cache.pop(user_code, None)
        return []
    except Exception as exc:
        logger.error("Calendar unexpected error: %s: %s", type(exc).__name__, exc)
        return []


# ══════════════════════════════════════════════════
#  GMAIL SEND
# ══════════════════════════════════════════════════


def send_gmail_message(
    to: str, subject: str, html_body: str, text_body: str = "", user_code: str = ""
) -> bool:
    """
    Send an email via the authenticated Gmail account of user_code.
    Returns True on success, False on any error.
    Sends a multipart/alternative message (plain text + HTML).
    """
    if not is_google_available(user_code or None):
        logger.warning("Gmail send skipped — Google not configured for %s", user_code)
        return False

    try:
        service = _get_gmail_service(user_code)

        msg = email.mime.multipart.MIMEMultipart("alternative")
        msg["To"] = to
        msg["Subject"] = subject

        if text_body:
            msg.attach(email.mime.text.MIMEText(text_body, "plain", "utf-8"))
        msg.attach(email.mime.text.MIMEText(html_body, "html", "utf-8"))

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        service.users().messages().send(userId="me", body={"raw": raw}).execute()
        logger.info("Gmail send: message sent to %s", to)
        return True

    except HttpError as exc:
        logger.error("Gmail send error (HTTP %s)", exc.status_code)
        return False
    except RuntimeError:
        return False
    except RefreshError as exc:
        # Token revoked or expired — evict cache so next call rebuilds credentials
        logger.error(
            "Gmail send: OAuth token refresh failed for %s: %s", user_code, exc
        )
        with _creds_lock:
            _credentials_cache.pop(user_code, None)
            _gmail_service_cache.pop(user_code, None)
        return False
    except Exception as exc:
        logger.error("Gmail send unexpected error: %s: %s", type(exc).__name__, exc)
        return False


# ══════════════════════════════════════════════════
#  CALENDAR WRITE — INTENT + EXTRACTION
# ══════════════════════════════════════════════════

_CALENDAR_WRITE_KEYWORDS = (
    # crée
    "crée un rendez-vous",
    "crée un rdv",
    "crée une réunion",
    # ajoute
    "ajoutes un rendez-vous",
    "ajoutes un rendez vous",
    "ajoutes un rdv",
    "ajoutes dans mon agenda",
    "ajoutes a mon agenda",
    "ajoutes à mon agenda",
    "ajoutes une réunion",
    "ajoute un rendez-vous",
    "ajoute un rendez vous",
    "ajoute un rdv",
    "ajoute dans mon agenda",
    "ajoute a mon agenda",
    "ajoute à mon agenda",
    "ajoute une réunion",
    # planifie
    "planifie une réunion",
    # mets
    "mets un rdv",
    "mets un rendez-vous",
)


def is_calendar_write(message: str) -> bool:
    msg = message.lower()
    return any(kw in msg for kw in _CALENDAR_WRITE_KEYWORDS)


# Regex that matches leading command phrases so they can be stripped from the title.
_CMD_PREFIX_RE = re.compile(
    r"^(ajoute|crée|crée|planifie|mets|programme|rappelle(?:-moi)?)"
    r"(\s+(un|une|le|la|les))?"
    r"(\s+(rendez-vous|rdv|réunion|reunion|event|événement|evenement|rappel|rendez vous))?",
    re.IGNORECASE,
)
# Words that are temporal context, not event subjects.
_TEMPORAL_WORDS = {
    "demain",
    "aujourd'hui",
    "lundi",
    "mardi",
    "mercredi",
    "jeudi",
    "vendredi",
    "samedi",
    "dimanche",
    "prochain",
    "prochaine",
    "matin",
    "soir",
    "midi",
    "après-midi",
    "bientôt",
}


def _sanitize_event_title(title: str) -> str:
    """Strip LLM command-phrase artefacts from an extracted event title.

    Returns '' if no meaningful subject remains after stripping.
    """
    t = _CMD_PREFIX_RE.sub("", title).strip(" ,-:;")
    # If only temporal / generic words remain → no real subject
    words = [w.lower() for w in t.split() if w]
    if not words or all(w in _TEMPORAL_WORDS for w in words):
        return ""
    return t


async def extract_calendar_event_llm(message: str) -> dict | None:
    """
    Extract event details from a user message using the router model.
    Returns a dict {title, start_date, end_date, start_time, end_time, location, description}.
    title may be '' when date/time were found but no event subject — caller must ask user.
    Returns None on hard failure (no date/time extractable or API error).
    """
    from datetime import date as _date

    from prompts import get_prompt

    today = _date.today().strftime("%Y-%m-%d")
    _model = PRIMARY_MODEL
    _api_url = PRIMARY_API_URL
    _api_key = PRIMARY_API_KEY
    try:
        raw = await call_llm_async(
            [
                {
                    "role": "user",
                    "content": get_prompt("CALENDAR_WRITE_EXTRACT").format(
                        message=message,
                        today=today,
                        timezone=BRIEFING_TIMEZONE,
                    ),
                }
            ],
            model=_model,
            api_url=_api_url,
            api_key=_api_key,
            temperature=0,
            max_tokens=MAX_TOKENS_SHORT,
            json_response=True,
            no_think=True,
            timeout=8.0,
        )
        parsed = extract_llm_json(raw)
        if not parsed.get("end_date"):
            parsed["end_date"] = parsed.get("start_date", "")
        if (
            "error" in parsed
            or not parsed.get("start_date")
            or not parsed.get("start_time")
        ):
            logger.warning("Calendar extraction incomplete (no date/time): %s", parsed)
            return None
        if not parsed.get("end_time"):
            parts = parsed["start_time"].replace("h", ":").split(":")
            h, m = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
            parsed["end_time"] = f"{(h + 1) % 24:02d}:{m:02d}"
        # Sanitize title — strip command artefacts added by the LLM.
        parsed["title"] = _sanitize_event_title(parsed.get("title", ""))
        return parsed
    except Exception as exc:
        logger.warning("Calendar event extraction failed: %s", type(exc).__name__)
        return None


# ══════════════════════════════════════════════════
#  CALENDAR WRITE — API CALLS
# ══════════════════════════════════════════════════
def _invalidate_calendar_cache() -> None:
    """Delete all Google calendar Redis cache entries so next read is fresh."""
    try:
        r = get_redis()
        cursor = 0
        deleted = 0
        while True:
            cursor, keys = r.scan(cursor, match="google:calendar*", count=100)
            if keys:
                r.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break
        if deleted:
            logger.info("Calendar cache invalidated (%d keys)", deleted)
    except Exception as exc:
        logger.warning("Calendar cache invalidation failed: %s", type(exc).__name__)


def create_calendar_event(
    title: str,
    start_dt: str,
    end_dt: str,
    description: str = "",
    location: str = "",
    calendar_id: str | None = None,
    user_code: str = "",
) -> str | None:
    """
    Create an event in Google Calendar of user_code.
    start_dt / end_dt: ISO 8601 with timezone, e.g. "2026-03-25T14:00:00+01:00".
    Returns the event ID on success, None on failure.
    """
    if not is_google_available(user_code or None):
        logger.warning(
            "Calendar write skipped — Google not configured for %s", user_code
        )
        return None

    cal_id = calendar_id or GOOGLE_CALENDAR_ID
    try:
        service = _get_calendar_service(user_code)
        body: dict = {
            "summary": title,
            "start": {"dateTime": start_dt},
            "end": {"dateTime": end_dt},
        }
        if description:
            body["description"] = description
        if location:
            body["location"] = location
        created = service.events().insert(calendarId=cal_id, body=body).execute()
        event_id = created.get("id")
        logger.info("Calendar event created (id=%s, title=%r)", event_id, title)
        _invalidate_calendar_cache()
        return event_id
    except HttpError as exc:
        logger.error("Calendar create error (HTTP %s)", exc.status_code)
        return None
    except RuntimeError:
        return None
    except Exception as exc:
        logger.error("Calendar create unexpected error: %s", type(exc).__name__)
        return None

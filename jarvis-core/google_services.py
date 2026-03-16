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
import logging
import threading
from datetime import datetime, timedelta, timezone
from email.utils import parseaddr

import redis as redis_lib
from google.auth.exceptions import GoogleAuthError, TransportError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config import (
    GOOGLE_CALENDAR_ID,
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_REFRESH_TOKEN,
    REDIS_URL,
)

logger = logging.getLogger("jarvis-google")

# ── OAuth scopes ──
_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar.readonly",
]

# ── Cache TTLs (seconds) ──
_GMAIL_CACHE_TTL = 300      # 5 min
_CALENDAR_CACHE_TTL = 300   # 5 min

# ── Result limits ──
_GMAIL_MAX_RESULTS = 10
_CALENDAR_MAX_RESULTS = 50
_EMAIL_BODY_MAX = 400   # chars per email
_SUBJECT_MAX = 120

# ── Singletons ──
_credentials: Credentials | None = None
_gmail_service = None
_calendar_service = None
_creds_lock = threading.Lock()
_service_lock = threading.Lock()
_redis = None


# ══════════════════════════════════════════════════
#  AVAILABILITY CHECK
# ══════════════════════════════════════════════════

def is_google_available() -> bool:
    """True only when all required credentials are present in the environment."""
    return bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET and GOOGLE_REFRESH_TOKEN)


# ══════════════════════════════════════════════════
#  CREDENTIALS  (thread-safe refresh)
# ══════════════════════════════════════════════════

def _get_credentials() -> Credentials:
    """
    Return a valid Credentials object, refreshing the access token when needed.
    Lock prevents concurrent refresh races in async/threaded contexts.
    Credential values are never written to logs.
    """
    global _credentials

    with _creds_lock:
        if _credentials is None:
            _credentials = Credentials(
                token=None,
                refresh_token=GOOGLE_REFRESH_TOKEN,
                client_id=GOOGLE_CLIENT_ID,
                client_secret=GOOGLE_CLIENT_SECRET,
                token_uri="https://oauth2.googleapis.com/token",
                scopes=_SCOPES,
            )

        if not _credentials.valid:
            try:
                _credentials.refresh(Request())
                logger.info("Google access token refreshed")
            except (GoogleAuthError, TransportError) as exc:
                # Log only the exception type — never the token or secret values
                logger.error("Google credential refresh failed: %s", type(exc).__name__)
                raise RuntimeError("Google authentication failed") from exc

        return _credentials


# ══════════════════════════════════════════════════
#  SERVICE SINGLETONS
# ══════════════════════════════════════════════════

def _get_gmail_service():
    global _gmail_service
    if _gmail_service is None:
        with _service_lock:
            if _gmail_service is None:
                creds = _get_credentials()
                _gmail_service = build(
                    "gmail", "v1", credentials=creds, cache_discovery=False
                )
    return _gmail_service


def _get_calendar_service():
    global _calendar_service
    if _calendar_service is None:
        with _service_lock:
            if _calendar_service is None:
                creds = _get_credentials()
                _calendar_service = build(
                    "calendar", "v3", credentials=creds, cache_discovery=False
                )
    return _calendar_service


# ══════════════════════════════════════════════════
#  REDIS CACHE
# ══════════════════════════════════════════════════

def _get_redis():
    global _redis
    if _redis is None:
        _redis = redis_lib.from_url(REDIS_URL, decode_responses=True)
    return _redis


def _cache_get(key: str):
    try:
        data = _get_redis().get(key)
        return json.loads(data) if data else None
    except Exception:
        return None


def _cache_set(key: str, data, ttl: int):
    try:
        _get_redis().setex(key, ttl, json.dumps(data, ensure_ascii=False))
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
            return base64.urlsafe_b64decode(body_data + "==").decode(
                "utf-8", errors="replace"
            ).strip()
        except Exception:
            return ""

    parts = payload.get("parts", [])
    plain, html_fallback = "", ""

    for part in parts:
        part_mime = part.get("mimeType", "")
        part_data = part.get("body", {}).get("data", "")

        if part_mime == "text/plain" and part_data:
            try:
                plain += base64.urlsafe_b64decode(part_data + "==").decode(
                    "utf-8", errors="replace"
                ).strip()
            except Exception:
                pass
        elif part_mime == "text/html" and part_data and not html_fallback:
            try:
                html_fallback = base64.urlsafe_b64decode(part_data + "==").decode(
                    "utf-8", errors="replace"
                ).strip()
            except Exception:
                pass
        elif part_mime.startswith("multipart/"):
            plain += _decode_email_body(part)

    return plain or html_fallback


def fetch_gmail_messages(query: str, max_results: int = _GMAIL_MAX_RESULTS) -> list[dict]:
    """
    Search Gmail across ALL folders and labels (including Sent, Archive, Spam).
    Returns list of {subject, from, date, snippet}.
    Results are cached in Redis for _GMAIL_CACHE_TTL seconds.
    """
    if not is_google_available():
        return []

    max_results = max(1, min(max_results, _GMAIL_MAX_RESULTS))
    cache_key = _cache_key("gmail", query, str(max_results))
    cached = _cache_get(cache_key)
    if cached is not None:
        logger.debug("Gmail cache hit")
        return cached

    try:
        service = _get_gmail_service()

        # includeSpamTrash=False + no label filter = ALL folders except Trash & spam
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

                results.append({
                    "subject": subject or "(sans objet)",
                    "from": sender_addr or sender_raw,
                    "date": date,
                    "snippet": snippet,
                })
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
    except Exception as exc:
        logger.error("Gmail unexpected error: %s", type(exc).__name__)
        return []


# ══════════════════════════════════════════════════
#  CALENDAR
# ══════════════════════════════════════════════════

def fetch_calendar_events(days: int = 7) -> list[dict]:
    """
    Fetch upcoming events for the next `days` days.
    days=7  → weekly view
    days=30 → monthly view
    Clamped to [1, 90]. Results cached in Redis for _CALENDAR_CACHE_TTL seconds.
    """
    if not is_google_available():
        return []

    days = max(1, min(days, 90))
    cache_key = _cache_key("calendar", str(days))
    cached = _cache_get(cache_key)
    if cached is not None:
        logger.debug("Calendar cache hit (days=%d)", days)
        return cached

    try:
        service = _get_calendar_service()
        now = datetime.now(timezone.utc)
        time_min = now.isoformat()
        time_max = (now + timedelta(days=days)).isoformat()

        resp = (
            service.events()
            .list(
                calendarId=GOOGLE_CALENDAR_ID,
                timeMin=time_min,
                timeMax=time_max,
                maxResults=_CALENDAR_MAX_RESULTS,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )

        results = []
        for event in resp.get("items", []):
            start = event.get("start", {})
            end = event.get("end", {})
            results.append({
                "summary": event.get("summary", "(sans titre)"),
                "start": start.get("dateTime", start.get("date", "")),
                "end": end.get("dateTime", end.get("date", "")),
                "location": event.get("location", ""),
                "description": (event.get("description") or "")[:200],
                "all_day": "dateTime" not in start,
            })

        _cache_set(cache_key, results, _CALENDAR_CACHE_TTL)
        logger.info("Calendar: %d events for next %d days", len(results), days)
        return results

    except HttpError as exc:
        logger.error("Calendar API error (HTTP %s)", exc.status_code)
        return []
    except RuntimeError:
        return []
    except Exception as exc:
        logger.error("Calendar unexpected error: %s", type(exc).__name__)
        return []


# ══════════════════════════════════════════════════
#  GMAIL SEND
# ══════════════════════════════════════════════════

def send_gmail_message(to: str, subject: str, html_body: str, text_body: str = "") -> bool:
    """
    Send an email via the authenticated Gmail account.
    Returns True on success, False on any error.
    Sends a multipart/alternative message (plain text + HTML).
    """
    if not is_google_available():
        logger.warning("Gmail send skipped — Google not configured")
        return False

    try:
        service = _get_gmail_service()

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
    except Exception as exc:
        logger.error("Gmail send unexpected error: %s", type(exc).__name__)
        return False


# ══════════════════════════════════════════════════=====
#  QUERY HELPERS - DEPRECATED BECAUSE USING LLM BUILDING
# ══════════════════════════════════════════════════=====

def build_gmail_query(message: str) -> str:
    """
    Convert a natural language message into a Gmail search query string.
    Detects common patterns (sender, subject, unread, attachments)
    and falls back to keyword search.
    """
    msg = message.lower()

    # Words that must NOT be treated as sender/subject values
    _FR_STOPS = {
        "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses",
        "les", "des", "une", "les", "aux", "ces", "cet", "cette",
        "hier", "demain", "semaine", "mois", "matin", "soir",
        "dernier", "dernière", "prochain", "prochaine",
        "nouveau", "nouvelle", "récent", "récente",
        "tout", "tous", "toutes",
    }

    # Sender detection — require explicit mail/email context before accepting "de "
    # Unambiguous prefixes (always mean sender)
    for prefix in ("de la part de ", "from:", "from "):
        if prefix in msg:
            after = msg.split(prefix, 1)[1].split()[0].strip(".,?!:()")
            if len(after) > 2 and after not in _FR_STOPS:
                return f"from:{after}"

    # "de " / "d'" — only accept when preceded by mail/email keywords
    _MAIL_TRIGGERS = ("mail de ", "email de ", "courriel de ", "message de ", "mails de ", "emails de ")
    for trigger in _MAIL_TRIGGERS:
        if trigger in msg:
            after = msg.split(trigger, 1)[1].split()[0].strip(".,?!:()")
            if len(after) > 2 and after not in _FR_STOPS:
                return f"from:{after}"

    # Subject search: explicit subject markers only ("concernant", "objet", "subject:")
    for prefix in ("concernant ", "subject:", "objet ", "about ", "sur le sujet "):
        if prefix in msg:
            after = msg.split(prefix, 1)[1].strip()
            keywords = " ".join(after.split()[:5])
            if keywords:
                return f"subject:{keywords}"

    # Unread
    if any(w in msg for w in ("non lu", "unread", "pas lu", "nouveau mail", "nouveaux mails")):
        return "is:unread"

    # Attachments
    if any(w in msg for w in ("pièce jointe", "attachment", "fichier joint", "avec fichier")):
        return "has:attachment"

    # Time-scoped recent queries
    _RECENT_WORDS = ("dernier", "derniers", "dernière", "dernières", "récent", "récents",
                     "récente", "récentes", "aujourd", "today", "recent", "latest", "last")
    if any(w in msg for w in _RECENT_WORDS):
        return "in:anywhere newer_than:7d"

    # Fallback: strip noise words, use remaining keywords
    _STOP = {
        "email", "emails", "mail", "mails", "courriel", "courriels",
        "regarde", "lis", "montre", "trouve", "cherche", "recherche",
        "mes", "mon", "ma", "les", "des", "le", "la", "un", "une",
        "dans", "sur", "avec", "pour", "est", "ce", "qui", "quoi",
        "boite", "boîte", "inbox", "jarvis", "regarde", "vois",
        "jai", "j'ai", "reçu", "recu", "si", "ai", "est-ce", "que",
        "nouveau", "nouvelle", "récent", "récents", "vérifie", "verifie",
        "consulte", "voir", "aujourd", "hui", "aujourd'hui",
    }
    words = [w for w in message.split() if w.lower() not in _STOP and len(w) > 2]
    query = " ".join(words[:6])
    return query if query.strip() else "in:anywhere newer_than:7d"


def detect_calendar_range(message: str) -> int:
    """
    Return the number of days to look ahead based on user message.
    Detects 'mois/month/30 jours' → 30 days, otherwise defaults to 7.
    """
    msg = message.lower()
    if any(w in msg for w in ("mois", "month", "mensuel", "30 jours", "ce mois", "30j")):
        return 30
    return 7

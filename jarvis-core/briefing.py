"""
Jarvis Morning Briefing
========================
Assembles a personalised daily briefing for each user and delivers it:
  - Stored in Redis (retrievable anytime in conversation)
  - Sent by email via Gmail (if USER_EMAILS[user_code] is set)

Data sources (gathered in parallel):
  - Google Calendar : today's events
  - Gmail           : important unread emails (last 24h)
  - Weather         : forecast for the user's city
  - News            : headlines filtered by user interests (from memory profile)
  - Projects        : ongoing projects/tasks recalled from vector memory

Entry points:
  gather_briefing(user_code)          -> BriefingResult
  store_briefing(user_code, result)   -> None  (Redis, TTL 28h)
  get_stored_briefing(user_code)      -> BriefingResult | None
  deliver_briefing(user_code, result) -> None  (email if configured)
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import httpx
import redis as redis_lib

import pytz

from config import (
    BRIEFING_TIMEZONE,
    PRIMARY_API_KEY,
    PRIMARY_API_URL,
    PRIMARY_MODEL,
    no_think_suffix,
    REDIS_URL,
    USER_CITIES,
    USER_CODES,
    USER_EMAILS,
    USERS,
)
from google_services import (
    fetch_calendar_events,
    fetch_gmail_messages,
    is_google_available,
    send_gmail_message,
)
from memory import get_user_profile, search_memory
from trading import get_portfolio_summary_text

logger = logging.getLogger("jarvis-briefing")

# ── Redis key helpers ─────────────────────────────────────────────────────
_BRIEFING_TTL = 28 * 3600   # 28h — survives until next morning


def _redis_key(user_code: str) -> str:
    return f"briefing:{user_code}:latest"


def _sent_key(user_code: str) -> str:
    return f"briefing:{user_code}:last_sent"


# ── Result dataclass ──────────────────────────────────────────────────────

@dataclass
class BriefingResult:
    user_code: str
    user_name: str
    generated_at: str                    # ISO 8601
    text: str                            # Conversational version for chat
    html: str                            # Rich HTML for email
    sections: dict = field(default_factory=dict)   # Raw section data


# ── Redis store/retrieve ──────────────────────────────────────────────────

_redis_client: redis_lib.Redis | None = None


def _get_redis() -> redis_lib.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis_lib.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def store_briefing(user_code: str, result: BriefingResult) -> None:
    try:
        payload = json.dumps({
            "user_code": result.user_code,
            "user_name": result.user_name,
            "generated_at": result.generated_at,
            "text": result.text,
            "html": result.html,
        }, ensure_ascii=False)
        _get_redis().setex(_redis_key(user_code), _BRIEFING_TTL, payload)
        logger.info("Briefing stored for %s", user_code)
    except Exception as exc:
        logger.error("Briefing store failed: %s", type(exc).__name__)


def get_stored_briefing(user_code: str) -> BriefingResult | None:
    try:
        raw = _get_redis().get(_redis_key(user_code))
        if not raw:
            return None
        d = json.loads(raw)
        return BriefingResult(
            user_code=d["user_code"],
            user_name=d["user_name"],
            generated_at=d["generated_at"],
            text=d["text"],
            html=d["html"],
        )
    except Exception:
        return None


# ── Data gathering ────────────────────────────────────────────────────────

_INTEREST_KEYS = (
    "interests", "interest", "hobbies", "topics", "passions",
    "work", "profession", "employer", "current_employer",
    "current_project", "expertise",
)
# Words too generic to be useful in a news query
_STOP_WORDS = {
    "informatique", "it", "the", "and", "or", "de", "du", "la", "le",
    "les", "des", "une", "pour", "avec", "dans", "sur", "par", "que",
    "qui", "est", "son", "ses", "mes", "mon", "notre", "votre",
}


def _get_user_interests(user_code: str) -> list[str]:
    """
    Extract interest keywords from the user's Redis profile.
    Looks at all profile keys likely to contain useful interests,
    not just a fixed whitelist.
    """
    profile = get_user_profile(user_code)
    interests = []

    for key in _INTEREST_KEYS:
        val = profile.get(key, "").strip()
        if not val:
            continue
        # Split on commas or spaces; take multi-word values as a phrase
        parts = [p.strip() for p in val.replace(",", ";").split(";") if p.strip()]
        for part in parts:
            words = part.split()
            if len(words) >= 2:
                # Keep multi-word phrases as-is (e.g. "cybersécurité réseau")
                phrase = part[:40]
                if phrase.lower() not in _STOP_WORDS:
                    interests.append(phrase)
            elif words:
                w = words[0].lower()
                if len(w) > 2 and w not in _STOP_WORDS:
                    interests.append(words[0])

    # Deduplicate, preserve order, cap at 6
    seen, unique = set(), []
    for item in interests:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
        if len(unique) >= 6:
            break

    return unique


def _get_active_projects(user_code: str) -> list[str]:
    """
    Recall ongoing projects / tasks from vector memory.
    Returns up to 4 relevant snippets.
    """
    try:
        memories = search_memory(user_code, "project task working on deadline objective", limit=4)
        snippets = []
        for m in memories:
            text = m.get("text", "").strip()
            if text:
                snippets.append(text[:200])
        return snippets
    except Exception:
        return []


_WX_CODES = {
    0: "Ciel dégagé", 1: "Principalement dégagé", 2: "Partiellement nuageux",
    3: "Couvert", 45: "Brouillard", 48: "Brouillard givrant",
    51: "Bruine légère", 53: "Bruine modérée", 55: "Bruine dense",
    61: "Pluie faible", 63: "Pluie modérée", 65: "Pluie forte",
    71: "Neige faible", 73: "Neige modérée", 75: "Neige forte",
    80: "Averses faibles", 81: "Averses modérées", 82: "Averses fortes",
    95: "Orage", 96: "Orage avec grêle", 99: "Orage violent",
}


async def _fetch_weather(city: str) -> str:
    """Fetch today's weather summary for the city via Open-Meteo (geocoding + forecast)."""
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            geo = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "fr", "format": "json"},
            )
            geo.raise_for_status()
            results = geo.json().get("results", [])
            if not results:
                logger.warning("Weather: city not found: %s", city)
                return ""
            loc = results[0]
            lat, lon = loc["latitude"], loc["longitude"]

            wx = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,weather_code",
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                    "timezone": BRIEFING_TIMEZONE,
                    "forecast_days": 1,
                },
            )
            wx.raise_for_status()
            data = wx.json()

            cur  = data.get("current", {})
            daily = data.get("daily", {})

            tmax  = (daily.get("temperature_2m_max") or [None])[0]
            tmin  = (daily.get("temperature_2m_min") or [None])[0]
            rain  = (daily.get("precipitation_sum") or [0])[0] or 0
            code  = cur.get("weather_code") or (daily.get("weather_code") or [-1])[0]
            t_now = cur.get("temperature_2m")

            if tmax is None:
                return ""

            condition = _WX_CODES.get(int(code), "")
            parts = [f"{city}"]
            if t_now is not None:
                parts.append(f"actuellement {t_now:.0f}°C")
            parts.append(f"min {tmin:.0f}°C / max {tmax:.0f}°C")
            if condition:
                parts.append(condition)
            if rain > 1:
                parts.append(f"pluie {rain:.0f}mm")

            return ", ".join(parts)
    except Exception as exc:
        logger.warning("Weather fetch failed: %s", type(exc).__name__)
        return ""


async def _fetch_news(interests: list[str]) -> list[str]:
    """
    Fetch top headlines personalised by user interests.
    Uses a space-separated keyword query (no OR — DDGS news is unreliable with operators).
    Falls back to 'actualités france' if no interests or first query fails.
    """
    from ddgs import DDGS

    # Build a simple keyword query from the top 3 interests
    queries = []
    if interests:
        queries.append(" ".join(interests[:3]))
    queries.append("actualités france")   # fallback

    def _ddg_news(query: str) -> list[dict]:
        with DDGS() as ddgs:
            return list(ddgs.news(query, region="fr-fr", max_results=5))

    for query in queries:
        try:
            results = await asyncio.to_thread(_ddg_news, query)
            if results:
                logger.info("News fetched (%d results) for query: %r", len(results), query)
                return [
                    f"{r['title']} — {r.get('source', '')}"
                    for r in results if r.get("title")
                ]
        except Exception as exc:
            logger.warning("News fetch failed for %r: %s", query, type(exc).__name__)

    return []


# ── LLM assembly ──────────────────────────────────────────────────────────

_BRIEFING_SYSTEM = """\
Tu es Jarvis, l'assistant personnel de {user_name}. Tu rédiges son briefing matinal.
Sois chaleureux, direct et concis. Utilise le prénom de l'utilisateur naturellement.
Pas de markdown excessif dans la version texte — elle sera lue à voix haute ou en chat.
La version HTML peut être structurée avec des sections."""

_BRIEFING_USER = """\
Génère le briefing matinal de {user_name} pour le {date}.

DONNÉES DISPONIBLES:

AGENDA DU JOUR:
{calendar}

EMAILS NON LUS (dernières 24h):
{gmail}

MÉTÉO:
{weather}

ACTUALITÉS (centres d'intérêt: {interests}):
{news}

PROJETS / TÂCHES EN COURS:
{projects}

PORTEFEUILLE BOURSIER:
{portfolio}

---
Génère deux versions:

1. VERSION TEXTE (clé "text"): briefing conversationnel naturel, 150-280 mots.
   Structure: accroche météo → agenda → emails importants → actu pertinente → portefeuille (si données) → rappel projet.
   Parle à la première personne de Jarvis ("J'ai regardé ton agenda...").
   Pour le portefeuille: mentionne uniquement les mouvements notables (>1% intraday) ou alertes thresholds.
   Si aucune donnée portefeuille disponible, omets cette section.

2. VERSION HTML (clé "html"): même contenu, formaté en HTML email propre.
   Utilise <h2> pour les sections, <ul> pour les listes, couleurs sobres inline.
   Sections: Agenda, Emails importants, Météo & Actu, Portefeuille (si données), À ne pas oublier.

Réponds en JSON uniquement: {{"text": "...", "html": "..."}}"""


async def _assemble_with_llm(
    user_name: str,
    sections: dict,
) -> tuple[str, str]:
    """Call the LLM to write the final briefing text and HTML."""
    date_str = datetime.now(timezone.utc).strftime("%A %d %B %Y")

    calendar_text = "\n".join(
        f"- {e['start'][:16]} : {e['summary']}" + (f" ({e['location']})" if e.get("location") else "")
        for e in sections.get("calendar", [])
    ) or "Aucun événement aujourd'hui."

    gmail_text = "\n".join(
        f"- De {m['from']} | {m['subject']}"
        for m in sections.get("gmail", [])[:5]
    ) or "Aucun email non lu."

    weather_text = sections.get("weather", "") or "Données météo indisponibles."
    news_items = sections.get("news", [])
    news_text = "\n".join(f"- {n}" for n in news_items) or "Aucune actualité disponible."
    interests_text = ", ".join(sections.get("interests", [])) or "généralistes"
    projects_text = "\n".join(f"- {p}" for p in sections.get("projects", [])) or "Aucun projet rappelé."

    portfolio_text = sections.get("portfolio") or "Aucune donnée de portefeuille disponible."

    prompt = _BRIEFING_USER.format(
        user_name=user_name,
        date=date_str,
        calendar=calendar_text,
        gmail=gmail_text,
        weather=weather_text,
        news=news_text,
        interests=interests_text,
        projects=projects_text,
        portfolio=portfolio_text,
    )

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{PRIMARY_API_URL}/chat/completions",
                headers={"Authorization": f"Bearer {PRIMARY_API_KEY}"},
                json={
                    "model": PRIMARY_MODEL,
                    "messages": [
                        {"role": "system", "content": _BRIEFING_SYSTEM.format(user_name=user_name) + no_think_suffix(PRIMARY_MODEL)},
                        {"role": "user", "content": prompt},
                    ],
                    "response_format": {"type": "json_object"},
                    "max_tokens": 1200,
                    "temperature": 0.6,
                },
            )
        result = json.loads(resp.json()["choices"][0]["message"]["content"])
        return result.get("text", ""), result.get("html", "")
    except Exception as exc:
        logger.error("Briefing LLM assembly failed: %s", type(exc).__name__)
        # Minimal fallback
        fallback = f"Bonjour {user_name} ! Voici ton briefing du {date_str}.\n\n"
        fallback += f"Agenda: {calendar_text}\n\nMétéo: {weather_text}"
        return fallback, f"<p>{fallback}</p>"


def _fetch_today_calendar() -> list[dict]:
    """
    Fetch today's full-day calendar events (midnight → midnight Paris time).
    Avoids the fetch_calendar_events(1) issue of starting from 'now' which
    misses morning events that already started.
    """
    from googleapiclient.errors import HttpError
    from google_services import _get_calendar_service, _cache_get, _cache_set, _cache_key, _CALENDAR_CACHE_TTL
    from config import GOOGLE_CALENDAR_ID

    tz = pytz.timezone(BRIEFING_TIMEZONE)
    now_local = datetime.now(tz)
    start_of_day = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day   = start_of_day + timedelta(days=1)

    cache_key = _cache_key("calendar_today", start_of_day.strftime("%Y-%m-%d"))
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        service = _get_calendar_service()
        resp = (
            service.events()
            .list(
                calendarId=GOOGLE_CALENDAR_ID,
                timeMin=start_of_day.isoformat(),
                timeMax=end_of_day.isoformat(),
                maxResults=20,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        results = []
        for event in resp.get("items", []):
            start = event.get("start", {})
            end   = event.get("end", {})
            results.append({
                "summary":     event.get("summary", "(sans titre)"),
                "start":       start.get("dateTime", start.get("date", "")),
                "end":         end.get("dateTime", end.get("date", "")),
                "location":    event.get("location", ""),
                "description": (event.get("description") or "")[:200],
                "all_day":     "dateTime" not in start,
            })
        _cache_set(cache_key, results, _CALENDAR_CACHE_TTL)
        logger.info("Today calendar: %d events", len(results))
        return results
    except Exception as exc:
        logger.warning("Today calendar fetch failed: %s", type(exc).__name__)
        return []


# ── Main entry point ──────────────────────────────────────────────────────

async def gather_briefing(user_code: str) -> BriefingResult:
    """
    Gather all data sources in parallel and assemble the briefing.
    Always returns a BriefingResult even if some sources fail.
    """
    user_name = USER_CODES.get(user_code, user_code)
    city = USER_CITIES.get(user_code, "Paris")
    logger.info("Generating briefing for %s (%s)", user_name, user_code)

    # Personalisation data (sync, fast)
    interests = _get_user_interests(user_code)
    projects = _get_active_projects(user_code)
    logger.info("Briefing interests for %s: %s", user_code, interests)

    # Parallel data gathering — calendar uses today-scoped fetch (midnight→midnight)
    calendar_task  = asyncio.to_thread(_fetch_today_calendar) if is_google_available() else asyncio.sleep(0, result=[])
    gmail_task     = asyncio.to_thread(fetch_gmail_messages, "is:unread newer_than:1d", 5) if is_google_available() else asyncio.sleep(0, result=[])
    weather_task   = _fetch_weather(city)
    news_task      = _fetch_news(interests)
    portfolio_task = asyncio.to_thread(get_portfolio_summary_text, user_code)

    calendar, gmail, weather, news, portfolio = await asyncio.gather(
        calendar_task, gmail_task, weather_task, news_task, portfolio_task
    )

    sections = {
        "calendar":  calendar,
        "gmail":     gmail,
        "weather":   weather,
        "news":      news,
        "interests": interests,
        "projects":  projects,
        "portfolio": portfolio,
    }

    text, html = await _assemble_with_llm(user_name, sections)

    result = BriefingResult(
        user_code=user_code,
        user_name=user_name,
        generated_at=datetime.now(timezone.utc).isoformat(),
        text=text,
        html=html,
        sections=sections,
    )

    logger.info("Briefing ready for %s (%d chars text, %d chars html)", user_name, len(text), len(html))
    return result


# ── Email delivery ────────────────────────────────────────────────────────

def deliver_briefing(user_code: str, result: BriefingResult) -> None:
    """Send the briefing by email if an address is configured for this user."""
    to = USER_EMAILS.get(user_code, "")
    if not to:
        logger.debug("No email configured for %s — skipping delivery", user_code)
        return

    # Avoid double-sending (e.g. on container restart)
    r = _get_redis()
    sent_key = _sent_key(user_code)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if r.get(sent_key) == today:
        logger.info("Briefing already sent today for %s — skipping", user_code)
        return

    date_label = datetime.now(timezone.utc).strftime("%d/%m/%Y")
    subject = f"Jarvis — Briefing du {date_label}"
    success = send_gmail_message(to=to, subject=subject, html_body=result.html, text_body=result.text)

    if success:
        r.setex(sent_key, _BRIEFING_TTL, today)
        logger.info("Briefing email delivered to %s for %s", to, user_code)

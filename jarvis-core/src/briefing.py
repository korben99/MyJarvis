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
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
from ddgs import DDGS

from config import (
    PRIMARY_API_KEY,
    PRIMARY_API_URL,
    PRIMARY_MODEL,
    USER_CITIES,
    USER_CODES,
    USER_EMAILS,
    USER_TIMEZONES,
)
from helpers import (
    WEATHER_CODES,
    call_llm_async,
    fmt_event_time,
    get_logger,
    get_redis,
    get_user_tz,
    now_user,
    today_user,
)
from google_services import (
    fetch_calendar_events,
    fetch_gmail_messages,
    is_google_available,
    send_gmail_message,
)
from memory import get_interest_weights, get_user_profile, get_user_projects
from prompts import get_prompt
from trading import get_portfolio_summary_text

logger = get_logger("jarvis-briefing")

# ── Redis key helpers ─────────────────────────────────────────────────────
_BRIEFING_TTL = 28 * 3600   # 28h — survives until next morning


def _briefing_key(user_code: str) -> str:
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

def store_briefing(user_code: str, result: BriefingResult) -> None:
    try:
        payload = json.dumps({
            "user_code": result.user_code,
            "user_name": result.user_name,
            "generated_at": result.generated_at,
            "text": result.text,
            "html": result.html,
        }, ensure_ascii=False)
        get_redis().setex(_briefing_key(user_code), _BRIEFING_TTL, payload)
        logger.info("Briefing stored for %s", user_code)
    except Exception as exc:
        logger.error("Briefing store failed: %s", type(exc).__name__)


def get_stored_briefing(user_code: str) -> BriefingResult | None:
    try:
        raw = get_redis().get(_briefing_key(user_code))
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

    # Deduplicate
    seen, unique = set(), []
    for item in interests:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)

    # Apply interest weights: filter out weight=0, sort by weight desc, cap at 6
    weights = get_interest_weights(user_code)
    unique = [t for t in unique if weights.get(t.lower(), 1.0) > 0]
    unique.sort(key=lambda t: weights.get(t.lower(), 1.0), reverse=True)

    return unique[:6]


def _get_active_projects(user_code: str) -> list[str]:
    """
    Recall ongoing projects / tasks from vector memory.
    Returns up to 4 relevant snippets if active or in_progress
    """
    projects = get_user_projects(user_code)
    return [
        f"{p['name']}: {p.get('description','')}"
        for p in projects
        if p.get("status") in ("active", "in_progress")
    ][:4]



async def _fetch_weather(city: str, tz_name: str = "Europe/Paris") -> str:
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
                    "timezone": tz_name,
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

            condition = WEATHER_CODES.get(int(code), "")
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


def _ddg_news_sync(query: str) -> list[dict]:
    """Synchronous DDG news fetch — run via asyncio.to_thread."""
    with DDGS() as ddgs:
        return list(ddgs.news(query, region="fr-fr", max_results=5))


def _unwrap(val, default):
    """Unwrap an asyncio.gather result, returning default on exception."""
    if isinstance(val, BaseException):
        logger.warning("Briefing data source failed: %s", type(val).__name__)
        return default
    return val


async def _fetch_news(interests: list[str]) -> list[str]:
    """
    Fetch top headlines personalised by user interests.
    Uses a space-separated keyword query (no OR — DDGS news is unreliable with operators).
    Falls back to 'actualités france' if no interests or first query fails.
    """
    # Build a simple keyword query from the top 3 interests
    queries = []
    if interests:
        queries.append(" ".join(interests[:3]))
    queries.append("actualités france")   # fallback

    for query in queries:
        try:
            results = await asyncio.to_thread(_ddg_news_sync, query)
            if results:
                logger.info("News fetched (%d results) for query: %r", len(results), query)
            return [
                f"{r['title']} — {r.get('body','')[:120]} ({r.get('source','')})"
                for r in results
            ]
        except Exception as exc:
            logger.warning("News fetch failed for %r: %s", query, type(exc).__name__)

    return []


# ── LLM assembly ──────────────────────────────────────────────────────────

async def _assemble_with_llm(
    user_name: str,
    user_code: str,
    sections: dict,
) -> tuple[str, str]:
    """Call the LLM to write the final briefing text and HTML."""
    date_str = now_user(user_code).strftime("%A %d %B %Y")

    calendar_text = "\n".join(
        (
            f"- {e['start']} : {e['summary']}" + (f" ({e['location']})" if e.get("location") else "")
            if e.get("all_day") else
            f"- {fmt_event_time(e['start'], user_code, '%H:%M')} : {e['summary']}" + (f" ({e['location']})" if e.get("location") else "")
        )
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

    prompt = get_prompt("BRIEFING_USER").format(
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
        content = await call_llm_async(
            [
                {"role": "system", "content": get_prompt("BRIEFING_SYSTEM").format(user_name=user_name)},
                {"role": "user", "content": prompt},
            ],
            model=PRIMARY_MODEL, api_url=PRIMARY_API_URL, api_key=PRIMARY_API_KEY,
            temperature=0.6, max_tokens=1200, json_response=True, no_think=True, timeout=30.0,
        )
        result = json.loads(content)
        return result.get("text", ""), result.get("html", "")
    except Exception as exc:
        logger.error("Briefing LLM assembly failed: %s", type(exc).__name__)
        # Minimal fallback
        fallback = f"Bonjour {user_name} ! Voici ton briefing du {date_str}.\n\n"
        fallback += f"Agenda: {calendar_text}\n\nMétéo: {weather_text}"
        return fallback, f"<p>{fallback}</p>"


# ── Main entry point ──────────────────────────────────────────────────────

async def gather_briefing(user_code: str) -> BriefingResult:
    """
    Gather all data sources in parallel and assemble the briefing.
    Always returns a BriefingResult even if some sources fail.
    """
    user_name = USER_CODES.get(user_code, user_code)
    city = USER_CITIES.get(user_code, "Paris")
    tz_name = USER_TIMEZONES.get(user_code, "Europe/Paris")
    logger.info("Generating briefing for %s (%s, tz=%s)", user_name, user_code, tz_name)

    # Personalisation data (sync, fast)
    interests = _get_user_interests(user_code)
    projects = _get_active_projects(user_code)
    logger.info("Briefing interests for %s: %s", user_code, interests)

    # Parallel data gathering — date= gives midnight→midnight fetch across all calendars
    today = today_user(user_code)
    has_google = is_google_available(user_code)
    calendar_task  = asyncio.to_thread(fetch_calendar_events, 1, today, tz_name, user_code) if has_google else asyncio.sleep(0, result=[])
    gmail_task     = asyncio.to_thread(fetch_gmail_messages, "is:unread newer_than:1d", 5, user_code) if has_google else asyncio.sleep(0, result=[])
    weather_task   = _fetch_weather(city, tz_name)
    news_task      = _fetch_news(interests)
    portfolio_task = asyncio.to_thread(get_portfolio_summary_text, user_code)

    results = await asyncio.gather(
        calendar_task, gmail_task, weather_task, news_task, portfolio_task,
        return_exceptions=True,
    )

    calendar  = _unwrap(results[0], [])
    gmail     = _unwrap(results[1], [])
    weather   = _unwrap(results[2], "")
    news      = _unwrap(results[3], [])
    portfolio = _unwrap(results[4], "")

    sections = {
        "calendar":  calendar,
        "gmail":     gmail,
        "weather":   weather,
        "news":      news,
        "interests": interests,
        "projects":  projects,
        "portfolio": portfolio,
    }

    text, html = await _assemble_with_llm(user_name, user_code, sections)

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
    r = get_redis()
    sent_key = _sent_key(user_code)
    today = now_user(user_code).strftime("%Y-%m-%d")
    if r.get(sent_key) == today:
        logger.info("Briefing already sent today for %s — skipping", user_code)
        return

    date_label = now_user(user_code).strftime("%d/%m/%Y")
    subject = f"Jarvis — Briefing du {date_label}"
    success = send_gmail_message(to=to, subject=subject, html_body=result.html, text_body=result.text, user_code=user_code)

    if success:
        r.setex(sent_key, _BRIEFING_TTL, today)
        logger.info("Briefing email delivered to %s for %s", to, user_code)

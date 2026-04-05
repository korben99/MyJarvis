"""
Jarvis Web Search
=================
All external search backends in one place, keeping main.py clean.

Backends:
  - Weather   : Open-Meteo (geocoding + forecast, no API key)
  - News      : DuckDuckGo news
  - General   : DuckDuckGo text  →  3-stage deep pipeline

Deep pipeline (general text queries only):
  Stage 1 — DDG snippets + LLM relevance judge
             Sufficient → done.
  Stage 2 — Fetch actual pages in parallel + LLM relevance judge
             Sufficient → done.
  Stage 3 — LLM generates a refined query → fresh DDG search → done.

LLM calls use the router model (fast, no_think) — same tier as intent
classification.  Falls back to the primary model if ROUTER_MODEL is unset.
"""

import asyncio
import re
import unicodedata
from urllib.parse import quote

import httpx
from ddgs import DDGS

from config import (
    PRIMARY_API_KEY,
    PRIMARY_API_URL,
    PRIMARY_MODEL,
    ROUTER_API_KEY,
    ROUTER_API_URL,
    ROUTER_MODEL,
)
from helpers import WEATHER_CODES, call_llm_async, extract_llm_json, get_logger
from prompts import get_prompt

logger = get_logger("jarvis-web")

# ── Internet-error sentinel ────────────────────────────────────────────────
# Returned by search_web() when the network is unreachable so that the caller
# (main.py) can inject a clear context block into the LLM prompt.
INTERNET_ERROR: list[dict] = [{"title": "__INTERNET_ERROR__", "body": "", "url": ""}]


def _is_network_error(exc: Exception) -> bool:
    """True when the exception is almost certainly a connectivity issue."""
    if isinstance(exc, (httpx.NetworkError, httpx.ConnectTimeout, httpx.ConnectError)):
        return True
    msg = str(exc).lower()
    return any(k in msg for k in ("connect", "network", "unreachable", "name or service not known", "nodename nor servname"))


# ── Shared HTTP client ─────────────────────────────────────────────────────
_HTTP = httpx.AsyncClient(
    timeout=10.0,
    follow_redirects=True,
    headers={
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
    },
)

# ── Deep search tuning ─────────────────────────────────────────────────────
_PAGE_FETCH_TIMEOUT = 8.0    # per-page HTTP timeout (seconds)
_PAGE_MAX_CHARS     = 3000   # max extracted chars kept per fetched page
_MAX_FETCH_PAGES    = 3      # pages fetched in parallel per deep round


# ══════════════════════════════════════════════════════════════════════════
#  QUERY OPTIMISER
# ══════════════════════════════════════════════════════════════════════════

def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def optimize_web_query(message: str) -> str:
    """Strip conversational filler and truncate to a search-engine query."""
    msg = message.lower()
    for filler in (
        "est ce que", "peux tu", "dis moi", "explique", "pourquoi",
        "comment", "tell me", "what is", "why", "how",
    ):
        msg = msg.replace(filler, "")
    msg = msg.replace("?", "").replace("!", "").strip()
    return " ".join(msg.split()[:10])


# ══════════════════════════════════════════════════════════════════════════
#  WEATHER  (Open-Meteo)
# ══════════════════════════════════════════════════════════════════════════

_WEATHER_KEYWORDS = {
    "météo", "meteo", "weather", "forecast", "prévision",
    "température", "temperature", "degrés", "degré",
    "pluie", "rain", "neige", "snow", "grêle",
    "vent", "wind", "rafale", "soleil", "sun",
    "nuage", "nuageux", "couvert", "ensoleillé", "orage",
    "brouillard", "brume", "humidité", "précipitation",
}


def _is_weather_query(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in _WEATHER_KEYWORDS)


def _extract_location(query: str) -> str:
    """Heuristic location extractor — fallback for the embedding-router path."""
    stop = {
        "météo", "meteo", "weather", "forecast", "prévision", "prévisions",
        "température", "temperature", "temps", "climat",
        "pluie", "vent", "neige", "soleil", "nuage", "nuageux", "nuageuse",
        "orage", "brouillard", "brume", "humidité", "degrés", "degré",
        "mini", "maxi", "minimum", "maximum", "min", "max",
        "aujourd'hui", "aujourdhui", "aujourd", "today", "demain", "tomorrow", "semaine", "week",
        "matin", "soir", "après-midi", "nuit", "maintenant", "now",
        "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche",
        "quelle", "quel", "quels", "quelles", "comment", "est-ce", "est",
        "qu'est", "que", "quoi",
        "à", "a", "de", "du", "pour", "en", "sur", "le", "la", "les", "dans",
        "il", "y", "va", "fait", "aura", "t", "avoir", "au", "aux",
        "merci", "svp", "stp", "bonjour", "bonsoir", "s'il", "vous", "plaît",
        "moi", "dis", "donne", "montre", "cherche", "dites",
        "?", "!", ".",
    }
    words = [
        w.strip("?!.,;:")
        for w in query.split()
        if w.strip("?!.,;:").lower() not in stop
        and not w.strip("?!.,;:").isdigit()
    ]
    return " ".join(words).strip()


async def search_weather(query: str) -> list[dict]:
    """Fetch current conditions + 3-day forecast from Open-Meteo.

    Retries up to 3 times with a 3 s delay on network errors (ConnectError).
    This handles the Mac Mini wake-from-sleep race where the network interface
    isn't ready yet when the 07:30 briefing job fires.
    """
    location = _extract_location(query)
    if not location:
        logger.warning("Weather: no location extracted")
        return []

    _MAX_ATTEMPTS = 3
    _RETRY_DELAY  = 3.0  # seconds between attempts

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            geo = await _HTTP.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1, "language": "fr"},
            )
            geo.raise_for_status()
            places = geo.json().get("results")
            if not places:
                logger.warning("Weather: location not found for '%s'", location)
                return []
            place   = places[0]
            lat     = place["latitude"]
            lon     = place["longitude"]
            name    = place.get("name", location)
            country = place.get("country", "")

            wx = await _HTTP.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude":  lat,
                    "longitude": lon,
                    "current":   "temperature_2m,apparent_temperature,weather_code,"
                                 "wind_speed_10m,relative_humidity_2m",
                    "daily":     "weather_code,temperature_2m_max,temperature_2m_min,"
                                 "precipitation_sum,wind_speed_10m_max",
                    "timezone":  "auto",
                    "forecast_days": 3,
                },
            )
            wx.raise_for_status()
            data  = wx.json()
            cur   = data.get("current", {})
            daily = data.get("daily",   {})

            condition = WEATHER_CODES.get(cur.get("weather_code", -1), "")
            now_body  = (
                f"Actuellement à {name} ({country}) : {cur.get('temperature_2m')}°C "
                f"(ressenti {cur.get('apparent_temperature')}°C), {condition}. "
                f"Vent {cur.get('wind_speed_10m')} km/h, "
                f"Humidité {cur.get('relative_humidity_2m')}%."
            )
            days = [
                f"{date}: {WEATHER_CODES.get(daily['weather_code'][i], '')} "
                f"{daily['temperature_2m_min'][i]}–{daily['temperature_2m_max'][i]}°C, "
                f"précip. {daily['precipitation_sum'][i]} mm, "
                f"vent max {daily['wind_speed_10m_max'][i]} km/h"
                for i, date in enumerate(daily.get("time", [])[:3])
            ]
            base_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            logger.info("Weather: fetched forecast for %s (attempt %d)", name, attempt)
            return [
                {"title": f"Météo actuelle — {name}",     "body": now_body,          "url": base_url},
                {"title": f"Prévisions 3 jours — {name}", "body": " | ".join(days),  "url": base_url},
            ]

        except Exception as exc:
            retriable = _is_network_error(exc) or isinstance(exc, httpx.TimeoutException)
            if retriable:
                if attempt < _MAX_ATTEMPTS:
                    logger.warning(
                        "Weather: %s (attempt %d/%d), retrying in %.0fs",
                        type(exc).__name__, attempt, _MAX_ATTEMPTS, _RETRY_DELAY,
                    )
                    await asyncio.sleep(_RETRY_DELAY)
                    continue
                logger.error(
                    "Weather: all %d attempts failed — %s",
                    _MAX_ATTEMPTS, type(exc).__name__,
                )
                return []
            logger.error("Weather fetch error (%s): %s", type(exc).__name__, exc)
            return []

    return []  # unreachable but satisfies type checker


# ══════════════════════════════════════════════════════════════════════════
#  NEWS  (DDG news)
# ══════════════════════════════════════════════════════════════════════════

_NEWS_KEYWORDS = {
    "news", "actualité", "actualites", "actu", "dernière", "dernieres",
    "latest", "recent", "aujourd'hui", "today", "breaking",
    "que se passe", "what is happening", "en ce moment",
}


def _is_news_query(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in _NEWS_KEYWORDS)


def _ddg_news_sync(query: str, max_results: int, region: str = "", timelimit: str = "") -> list[dict]:
    results = []
    kwargs: dict = {"max_results": max_results}
    if region:
        kwargs["region"] = region
    if timelimit:
        kwargs["timelimit"] = timelimit
    with DDGS() as ddgs:
        for r in ddgs.news(query, **kwargs):
            body   = r.get("body", "")
            date   = r.get("date", "")
            source = r.get("source", "")
            prefix = " | ".join(filter(None, [date, source]))
            results.append({
                "title": r.get("title", ""),
                "body":  f"[{prefix}] {body}" if prefix else body,
                "url":   r.get("url", ""),
            })
    return results


async def search_news(query: str, max_results: int = 5, region: str = "", timelimit: str = "") -> list[dict]:
    try:
        loop    = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, _ddg_news_sync, query, max_results, region, timelimit)
        logger.info("News: %d articles for: %s", len(results), query[:50])
        return results
    except Exception as exc:
        if _is_network_error(exc):
            raise
        logger.warning("News search error (will retry if caller allows): %s", exc)
        return []


# ══════════════════════════════════════════════════════════════════════════
#  WIKIPEDIA  (last-resort fallback)
# ══════════════════════════════════════════════════════════════════════════

async def search_wikipedia(query: str) -> list[dict]:
    url = f"https://fr.wikipedia.org/api/rest_v1/page/summary/{quote(query)}"
    try:
        r = await _HTTP.get(url)
        if r.status_code != 200:
            return []
        data = r.json()
        return [{
            "title": data.get("title"),
            "body":  data.get("extract"),
            "url":   data.get("content_urls", {}).get("desktop", {}).get("page"),
        }]
    except Exception as exc:
        logger.warning("Wikipedia search failed: %s", exc)
        return []


# ══════════════════════════════════════════════════════════════════════════
#  DEEP TEXT SEARCH  —  3-stage pipeline
# ══════════════════════════════════════════════════════════════════════════

def _ddg_text_sync(query: str, max_results: int) -> list[dict]:
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "body":  r.get("body",  ""),
                "url":   r.get("href",  ""),
            })
    return results


def _extract_text_from_html(html: str, max_chars: int = _PAGE_MAX_CHARS) -> str:
    """Strip HTML noise and return readable plain text."""
    for tag in ("script", "style", "nav", "footer", "header", "aside", "form"):
        html = re.sub(
            rf"<{tag}[^>]*>.*?</{tag}>", " ", html,
            flags=re.DOTALL | re.IGNORECASE,
        )
    text = re.sub(r"<[^>]+>", " ", html)
    for ent, ch in (
        ("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"),
        ("&gt;", ">"), ("&#39;", "'"), ("&quot;", '"'),
    ):
        text = text.replace(ent, ch)
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


async def _fetch_page_text(url: str) -> str:
    """Fetch a URL and return extracted plain text. Returns '' on any error."""
    if not url or not url.startswith("http"):
        return ""
    try:
        resp = await _HTTP.get(url, timeout=_PAGE_FETCH_TIMEOUT)
        if resp.status_code != 200:
            logger.debug("fetch_page_text: HTTP %d for %s", resp.status_code, url)
            return ""
        ct = resp.headers.get("content-type", "").lower()
        if not any(x in ct for x in ("html", "text/plain")):
            logger.debug("fetch_page_text: unsupported content-type %r for %s", ct, url)
            return ""
        return _extract_text_from_html(resp.text)
    except Exception as exc:
        logger.debug("fetch_page_text: error fetching %s — %s: %s", url, type(exc).__name__, exc)
        return ""


async def fetch_user_urls(urls: list[str], max_urls: int = 3) -> list[dict]:
    """Fetch user-provided URLs and return them as web-result dicts.

    Called when the user pastes a direct URL in their message.
    Fetches up to *max_urls* in parallel and returns non-empty results only.
    Result format matches the web_results dicts used throughout the pipeline.

    On failure (403, timeout, unsupported content-type…) an error sentinel dict
    is returned so the LLM can explain the situation to the user rather than
    silently ignoring the URL.
    """
    urls = [u for u in urls if u.startswith("http")][:max_urls]
    if not urls:
        return []

    async def _fetch_with_status(url: str) -> tuple[str, str]:
        """Return (text, error_reason). One of them is always empty."""
        if not url.startswith("http"):
            return "", "URL invalide"
        try:
            resp = await _HTTP.get(url, timeout=_PAGE_FETCH_TIMEOUT)
            if resp.status_code == 403:
                return "", f"accès refusé (HTTP 403) — le site bloque les requêtes automatiques"
            if resp.status_code != 200:
                return "", f"HTTP {resp.status_code}"
            ct = resp.headers.get("content-type", "").lower()
            if not any(x in ct for x in ("html", "text/plain")):
                return "", f"type de contenu non supporté ({ct})"
            text = _extract_text_from_html(resp.text)
            if not text:
                return "", "page vide ou contenu non extractible (JavaScript requis ?)"
            return text, ""
        except Exception as exc:
            return "", f"{type(exc).__name__}: {exc}"

    fetches = await asyncio.gather(*[_fetch_with_status(u) for u in urls])
    results = []
    for url, (text, error) in zip(urls, fetches):
        if text:
            first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), url)
            results.append({"title": first_line[:120], "body": text, "url": url})
            logger.info("Fetched user-provided URL: %s (%d chars)", url, len(text))
        else:
            logger.warning("Could not fetch user-provided URL: %s — %s", url, error)
            # Inject an error sentinel so the LLM informs the user instead of ignoring the URL
            results.append({
                "title": f"[Erreur fetch] {url}",
                "body": (
                    f"Impossible de lire la page : {error}. "
                    f"Tu peux copier-coller le contenu directement dans le chat."
                ),
                "url": url,
            })
    return results


# ── Router-model LLM calls (fast, no_think) ────────────────────────────────

def _router_llm_params() -> dict:
    """Return the model/url/key for the router tier, falling back to primary."""
    if ROUTER_MODEL:
        return {"model": ROUTER_MODEL, "api_url": ROUTER_API_URL, "api_key": ROUTER_API_KEY}
    return {"model": PRIMARY_MODEL, "api_url": PRIMARY_API_URL, "api_key": PRIMARY_API_KEY}


async def _llm_judge_relevance(question: str, results: list[dict]) -> bool:
    """
    Ask the router model: do these results sufficiently answer the question?
    Returns True (sufficient) or False (need more).
    Fails open — returns True on any error so the pipeline is never blocked.
    """
    snippets = "\n".join(
        f"[{i+1}] {r['title']}: {r['body'][:300]}"
        for i, r in enumerate(results[:5])
    )
    try:
        raw = await call_llm_async(
            [{"role": "user", "content": get_prompt("WEB_RELEVANCE_JUDGE").format(
                question=question,
                snippets=snippets,
            )}],
            **_router_llm_params(),
            temperature=0,
            max_tokens=80,
            json_response=True,
            no_think=True,
            timeout=8.0,
        )
        parsed     = extract_llm_json(raw)
        sufficient = bool(parsed.get("sufficient", True))
        logger.info("Web judge: sufficient=%s — %s", sufficient, parsed.get("reason", "")[:80])
        return sufficient
    except Exception as exc:
        logger.warning("Web judge failed (%s) — assuming sufficient", type(exc).__name__)
        return True  # fail-open


async def _refine_web_query(question: str, current_query: str, results: list[dict]) -> str:
    """Generate a better search query when the judge deems results insufficient."""
    snippets = "\n".join(f"- {r['title']}: {r['body'][:120]}" for r in results[:3])
    try:
        refined = await call_llm_async(
            [{"role": "user", "content": (
                f"User question: {question}\n"
                f"Search query tried: {current_query}\n"
                f"Insufficient results:\n{snippets}\n\n"
                f"Write ONE better search query. Same language as the question. "
                f"Output only the query, nothing else."
            )}],
            **_router_llm_params(),
            temperature=0,
            max_tokens=60,
            json_response=False,
            no_think=True,
            timeout=8.0,
        )
        refined = refined.strip().strip("\"'")
        return refined if refined and refined != current_query else ""
    except Exception:
        return ""


async def _ddg_text_deep(
    query: str,
    original_message: str,
    max_results: int = 5,
) -> list[dict]:
    """
    3-stage deep DDG text search:

      Stage 1 — DDG snippets
                LLM judge → sufficient? yes → return / no → Stage 2

      Stage 2 — Fetch actual pages (parallel, _MAX_FETCH_PAGES at once)
                LLM judge → sufficient? yes → return / no → Stage 3

      Stage 3 — LLM refines the query → fresh DDG search → return

    Weather and news bypass this function entirely.
    """
    question  = original_message or query
    seen_urls: set[str] = set()

    async def _run_ddg(q: str) -> list[dict]:
        loop = asyncio.get_running_loop()
        raw  = await loop.run_in_executor(None, _ddg_text_sync, q, max_results)
        unique = []
        for r in raw:
            if r["url"] and r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                unique.append(r)
        return unique

    # ── Stage 1: DDG snippets ─────────────────────────────────────────────
    results = await _run_ddg(query)
    if not results:
        return []

    if await _llm_judge_relevance(question, results):
        logger.info("Web deep[1]: %d results — sufficient", len(results))
        return results

    logger.info("Web deep[1]: insufficient — fetching pages")

    # ── Stage 2: Fetch page content (parallel) ────────────────────────────
    to_fetch   = [r for r in results if r["url"]][:_MAX_FETCH_PAGES]
    page_texts = await asyncio.gather(*[_fetch_page_text(r["url"]) for r in to_fetch])

    enriched = [
        {**r, "body": page_texts[i] if i < len(page_texts) and len(page_texts[i]) > len(r["body"]) else r["body"]}
        for i, r in enumerate(results)
    ]

    if await _llm_judge_relevance(question, enriched):
        logger.info("Web deep[2]: enriched pages — sufficient")
        return enriched

    logger.info("Web deep[2]: still insufficient — refining query")

    # ── Stage 3: Query refinement ─────────────────────────────────────────
    refined_query = await _refine_web_query(question, query, enriched)
    if not refined_query:
        logger.info("Web deep[3]: no refined query generated — returning best effort")
        return enriched

    refined_results = await _run_ddg(refined_query)
    if refined_results:
        logger.info("Web deep[3]: '%s' → %d new results", refined_query[:50], len(refined_results))
        # Refined results first; pad with non-duplicate enriched ones
        refined_urls = {r["url"] for r in refined_results}
        merged = refined_results + [r for r in enriched if r["url"] not in refined_urls]
        return merged[:max_results]

    logger.info("Web deep[3]: refined search empty — returning enriched")
    return enriched


# ══════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

async def search_web(
    query: str,
    max_results: int = 3,
    original_message: str = "",
) -> list[dict]:
    """
    Main search entry point.

    Routes:
      - Weather keywords  → Open-Meteo
      - News keywords     → DDG news
      - Everything else   → 3-stage deep DDG text

    original_message is the raw user question, used by the LLM judge and
    query refiner to understand intent beyond the optimised query string.
    """
    if _is_weather_query(query):
        results = await search_weather(query)
        if results:
            return results

    if _is_news_query(query):
        results = await search_news(query, max_results=max_results)
        if results:
            return results

    try:
        results = await _ddg_text_deep(
            query, original_message, max_results=max(max_results, 5),
        )
        if not results:
            return await search_wikipedia(query)
        return results
    except Exception as exc:
        if _is_network_error(exc):
            logger.warning("Internet unavailable: %s", type(exc).__name__)
            return INTERNET_ERROR
        logger.error("Web search error: %s", exc)
        return []

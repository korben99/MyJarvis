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
from datetime import date as _date
from urllib.parse import quote

import httpx
from ddgs import DDGS

from config import (
    MAX_TOKENS_SHORT,
    MAX_TOKENS_TINY,
    PRIMARY_API_KEY,
    PRIMARY_API_URL,
    PRIMARY_MODEL,
    ROUTER_API_KEY,
    ROUTER_API_URL,
    ROUTER_MODEL,
    TAVILY_API_KEY,
)
from helpers import WEATHER_CODES, call_llm_async, extract_llm_json, fmt_date_fr, get_logger
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
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    },
)

# ── Deep search tuning ─────────────────────────────────────────────────────
_PAGE_FETCH_TIMEOUT = 8.0    # per-page HTTP timeout (seconds)
_PAGE_MAX_CHARS     = 6000   # max extracted chars kept per fetched page
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
    """Strip conversational filler and truncate to a search-engine query.

    Case is preserved — lowercasing destroys proper nouns and acronyms
    (e.g. "IA" → "ia" causes DDG to return unrelated results).
    Filler words are matched case-insensitively via re.sub.
    """
    msg = message
    for filler in (
        "est ce que", "peux tu", "dis moi", "explique", "pourquoi",
        "comment", "tell me", "what is", "why", "how",
    ):
        msg = re.sub(re.escape(filler), "", msg, flags=re.IGNORECASE)
    msg = msg.replace("?", "").replace("!", "").replace(".", "").strip()
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
            # Today's daily forecast (index 0) merged into the current-conditions entry
            # so the LLM can't skip it when it sees current conditions describe the same day.
            today_outlook = ""
            daily_times = daily.get("time", [])
            if daily_times:
                today_condition = WEATHER_CODES.get(daily["weather_code"][0], "")
                today_outlook = (
                    f" Prévisions du jour : {today_condition} "
                    f"{daily['temperature_2m_min'][0]}–{daily['temperature_2m_max'][0]}°C, "
                    f"précip. {daily['precipitation_sum'][0]} mm, "
                    f"vent max {daily['wind_speed_10m_max'][0]} km/h."
                )
            now_body = (
                f"Actuellement à {name} ({country}) : {cur.get('temperature_2m')}°C "
                f"(ressenti {cur.get('apparent_temperature')}°C), {condition}. "
                f"Vent {cur.get('wind_speed_10m')} km/h, "
                f"Humidité {cur.get('relative_humidity_2m')}%.{today_outlook}"
            )
            # Days 1 and 2 only (day 0 = today is already in now_body)
            days = [
                f"{fmt_date_fr(_date.fromisoformat(date))}: {WEATHER_CODES.get(daily['weather_code'][i], '')} "
                f"{daily['temperature_2m_min'][i]}–{daily['temperature_2m_max'][i]}°C, "
                f"précip. {daily['precipitation_sum'][i]} mm, "
                f"vent max {daily['wind_speed_10m_max'][i]} km/h"
                for i, date in enumerate(daily_times[1:3], start=1)
            ]
            base_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            logger.info("Weather: fetched forecast for %s (attempt %d)", name, attempt)
            return [
                {"title": f"Météo actuelle — {name}",        "body": now_body,         "url": base_url},
                {"title": f"Prévisions J+1/J+2 — {name}",   "body": " | ".join(days), "url": base_url},
            ]

        except Exception as exc:
            is_server_error = (
                isinstance(exc, httpx.HTTPStatusError)
                and exc.response.status_code >= 500
            )
            retriable = (
                _is_network_error(exc)
                or isinstance(exc, httpx.TimeoutException)
                or is_server_error
            )
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


async def _fetch_via_jina(url: str) -> str:
    """Fallback fetch via Jina reader (r.jina.ai).

    Handles Cloudflare-protected pages, JS-rendered content, and most 403s.
    Returns plain text (Jina returns clean Markdown), empty string on failure.
    """
    jina_url = f"https://r.jina.ai/{url}"
    try:
        resp = await _HTTP.get(
            jina_url,
            timeout=25.0,
            headers={
                "Accept": "text/plain, text/markdown",
                # Only remove structural chrome — avoid class selectors that may
                # accidentally match product/article content containers.
                "X-Remove-Selector": "header, nav, footer, aside, .cookie-banner, #cookie-banner, #cookieBanner",
                "X-With-Images-Summary": "false",
                # X-Timeout tells Jina's headless browser how long to wait for JS
                # rendering before returning — important for SPAs (Qwen, etc.)
                "X-Timeout": "20",
            },
        )
        if resp.status_code == 200:
            text = resp.text.strip()
            logger.debug("Jina fetch %s → %d chars", url, len(text))
            if len(text) > 100:  # Jina sometimes returns tiny error pages
                return text[:_PAGE_MAX_CHARS]
    except Exception as exc:
        logger.debug("Jina fetch failed for %s: %s", url, exc)
    return ""


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


_DATE_META_PROPS = re.compile(
    r"article:published_time|publishedDate|datePublished|publish[-_]?date|"
    r"article:modified_time|og:updated_time",
    re.IGNORECASE,
)
_DATE_YYYY_MM_DD = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _extract_pub_date(html: str) -> str | None:
    """Extract article publication date from raw HTML. Returns 'YYYY-MM-DD' or None.

    Checks (in order): Open Graph / schema.org meta tags, <time datetime>, JSON-LD.
    Attribute order in <meta> tags varies by CMS — this handles both orderings.
    """
    for tag in re.findall(r"<meta[^>]+>", html, re.IGNORECASE):
        if _DATE_META_PROPS.search(tag):
            m = _DATE_YYYY_MM_DD.search(tag)
            if m:
                return m.group(1)
    m = re.search(r'<time[^>]+datetime=["\'](\d{4}-\d{2}-\d{2})', html, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r'"datePublished"\s*:\s*"(\d{4}-\d{2}-\d{2})', html)
    if m:
        return m.group(1)
    return None


async def _fetch_page(url: str) -> tuple[str, str | None]:
    """Fetch a URL and return (extracted text, publication date | None).

    Falls back to Jina reader on 403 or empty content.
    Publication date is extracted from raw HTML before tag stripping.
    """
    if not url or not url.startswith("http"):
        return "", None
    try:
        resp = await _HTTP.get(url, timeout=_PAGE_FETCH_TIMEOUT)
        if resp.status_code == 403:
            return await _fetch_via_jina(url), None
        if resp.status_code != 200:
            logger.debug("_fetch_page: HTTP %d for %s", resp.status_code, url)
            return "", None
        ct = resp.headers.get("content-type", "").lower()
        if not any(x in ct for x in ("html", "text/plain")):
            logger.debug("_fetch_page: unsupported content-type %r for %s", ct, url)
            return "", None
        raw = resp.text
        pub_date = _extract_pub_date(raw)
        text = _extract_text_from_html(raw)
        if not text:
            return await _fetch_via_jina(url), pub_date
        return text, pub_date
    except Exception as exc:
        logger.debug("_fetch_page: error fetching %s — %s: %s", url, type(exc).__name__, exc)
        return "", None


async def _fetch_page_text(url: str) -> str:
    """Compatibility wrapper — returns only the text part of _fetch_page()."""
    text, _ = await _fetch_page(url)
    return text


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
                text = await _fetch_via_jina(url)
                if text:
                    return text, ""
                return "", "accès refusé (HTTP 403) — le site bloque les requêtes automatiques"
            if resp.status_code != 200:
                return "", f"HTTP {resp.status_code}"
            ct = resp.headers.get("content-type", "").lower()
            if not any(x in ct for x in ("html", "text/plain")):
                return "", f"type de contenu non supporté ({ct})"
            text = _extract_text_from_html(resp.text)
            if not text:
                text = await _fetch_via_jina(url)
                if text:
                    return text, ""
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
            max_tokens=MAX_TOKENS_SHORT,
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


def _background_discard(task: asyncio.Task) -> None:
    """Done-callback that suppresses 'Task exception was never retrieved' for cancelled tasks."""
    if not task.cancelled():
        try:
            task.exception()
        except Exception:
            pass


async def _generate_optimized_query(question: str, current_query: str) -> str:
    """LLM-generated search query — runs concurrently with Stage-0 DDG (zero extra latency).

    Returns an improved query string, or '' if the LLM produces nothing new.
    """
    try:
        result = await call_llm_async(
            [{"role": "user", "content": (
                f"User question: {question}\n"
                f"Current search query: {current_query}\n\n"
                f"Write ONE optimized search engine query (4-8 words max). "
                f"Remove filler words. Keep proper nouns, technical terms, dates. "
                f"Same language as the question. Output only the query."
            )}],
            **_router_llm_params(),
            temperature=0,
            max_tokens=MAX_TOKENS_TINY,
            json_response=False,
            no_think=True,
            timeout=6.0,
        )
        result = result.strip().strip("\"'")
        return result if result and result.lower() != current_query.lower() else ""
    except Exception:
        return ""


async def _refine_web_queries(
    question: str, current_query: str, results: list[dict]
) -> tuple[str, str]:
    """Generate TWO alternative search queries when results are still insufficient.

    One LLM call returns both queries (Q1/Q2 format) — run DDG on each in parallel.
    """
    snippets = "\n".join(f"- {r['title']}: {r['body'][:120]}" for r in results[:3])
    try:
        raw = await call_llm_async(
            [{"role": "user", "content": (
                f"User question: {question}\n"
                f"Search query tried: {current_query}\n"
                f"Insufficient results:\n{snippets}\n\n"
                f"Write TWO different search queries (5-8 words each) to find better results. "
                f"Try different angles, synonyms, or more specific terms. "
                f"Same language as the question.\n"
                f"Output exactly:\nQ1: <first query>\nQ2: <second query>"
            )}],
            **_router_llm_params(),
            temperature=0,
            max_tokens=MAX_TOKENS_TINY,
            json_response=False,
            no_think=True,
            timeout=8.0,
        )
        q1, q2 = "", ""
        for line in raw.strip().splitlines():
            line = line.strip()
            if line.upper().startswith("Q1:"):
                q1 = line[3:].strip().strip("\"'")
            elif line.upper().startswith("Q2:"):
                q2 = line[3:].strip().strip("\"'")
        return (
            q1 if q1 and q1 != current_query else "",
            q2 if q2 and q2 != current_query and q2 != q1 else "",
        )
    except Exception:
        return "", ""


_DDG_EXECUTOR_TIMEOUT = 12.0   # max time for a single DDG executor call
_PIPELINE_TIMEOUT     = 25.0   # hard cap for the entire deep pipeline (raised for parallel stages)


async def _ddg_text_deep(
    query: str,
    original_message: str,
    max_results: int = 5,
) -> list[dict]:
    """
    Parallel 4-stage deep DDG text search:

      Stage 0 — DDG(original_query)  +  LLM generates optimized query  [concurrent]

      Stage 1 — Judge DDG snippets
                Speculatively launch: page fetch tasks + LLM-query DDG task
                Sufficient → cancel speculative tasks, return snippets
                Insufficient → Stage 2

      Stage 2 — Await speculative tasks (already ~2 s into execution)
                Enrich results with page content + publication dates (extracted from HTML)
                Merge LLM-query DDG results (new URLs only)
                Judge enriched+merged results
                Sufficient → return
                Insufficient → Stage 3

      Stage 3 — LLM generates 2 refined queries in one call  [concurrent DDG on both]
                Merge new URLs → return

    Hard cap: _PIPELINE_TIMEOUT seconds.  On timeout, best_so_far is returned.
    """
    question = original_message or query
    seen_urls: set[str] = set()
    best_so_far: list[dict] = []

    async def _run_ddg(q: str) -> list[dict]:
        loop = asyncio.get_running_loop()
        try:
            raw = await asyncio.wait_for(
                loop.run_in_executor(None, _ddg_text_sync, q, max_results),
                timeout=_DDG_EXECUTOR_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("DDG executor timed out after %.0fs for: %s", _DDG_EXECUTOR_TIMEOUT, q[:60])
            return []
        unique = []
        for r in raw:
            if r["url"] and r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                unique.append(r)
        return unique

    async def _pipeline() -> list[dict]:
        nonlocal best_so_far

        # ── Stage 0: DDG + LLM query optimization (concurrent) ───────────
        # _generate_optimized_query overlaps with the DDG executor call → zero extra latency.
        results, llm_query = await asyncio.gather(
            _run_ddg(query),
            _generate_optimized_query(question, query),
        )
        if not results:
            return []
        best_so_far = results
        if llm_query:
            logger.debug("Web deep[0]: LLM query = '%s'", llm_query[:60])

        # ── Stage 1: Judge snippets — speculatively launch page fetch + LLM-query DDG ──
        to_fetch = [r for r in results if r["url"]][:_MAX_FETCH_PAGES]
        page_tasks = [asyncio.create_task(_fetch_page(r["url"])) for r in to_fetch]
        llm_ddg_task: asyncio.Task | None = None
        if llm_query:
            llm_ddg_task = asyncio.create_task(_run_ddg(llm_query))

        if await _llm_judge_relevance(question, results):
            logger.info("Web deep[1]: %d snippets — sufficient (cancelling speculative tasks)", len(results))
            spec_tasks = page_tasks + ([llm_ddg_task] if llm_ddg_task else [])
            for t in spec_tasks:
                t.add_done_callback(_background_discard)
                t.cancel()
            return results

        logger.info("Web deep[1]: insufficient — awaiting page fetch + LLM-query DDG")

        # ── Stage 2: Enrich with pages + dates + LLM-query results ────────
        # page_tasks and llm_ddg_task have been running for the ~2 s judge took.
        spec_tasks = page_tasks + ([llm_ddg_task] if llm_ddg_task else [])
        gathered = await asyncio.gather(*spec_tasks, return_exceptions=True)
        page_raw = gathered[: len(page_tasks)]
        llm_ddg_results: list[dict] = []
        if llm_ddg_task:
            lr = gathered[len(page_tasks)]
            if isinstance(lr, list):
                llm_ddg_results = lr

        # Build enriched list: Stage-0 snippets → replace body with full page + add date
        url_to_page: dict[str, tuple[str, str | None]] = {}
        for i, res in enumerate(page_raw):
            if isinstance(res, tuple):
                text, pub_date = res
                if text:
                    url_to_page[to_fetch[i]["url"]] = (text, pub_date)

        enriched: list[dict] = []
        for r in results:
            url = r.get("url", "")
            if url in url_to_page:
                text, pub_date = url_to_page[url]
                entry: dict = {**r, "body": text if len(text) > len(r["body"]) else r["body"]}
                if pub_date:
                    entry["date"] = pub_date
            else:
                entry = dict(r)
            enriched.append(entry)

        # Append LLM-query DDG results (new URLs only, no page fetch for them yet)
        if llm_ddg_results:
            for r in llm_ddg_results:
                if r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    enriched.append(r)
            logger.info(
                "Web deep[2]: +%d results from LLM query '%s'",
                len(llm_ddg_results), llm_query[:50],
            )

        best_so_far = enriched

        if await _llm_judge_relevance(question, enriched):
            logger.info("Web deep[2]: enriched+merged — sufficient")
            return enriched[:max_results]

        logger.info("Web deep[2]: still insufficient — dual refined queries")

        # ── Stage 3: 2 refined queries → DDG in parallel ─────────────────
        q1, q2 = await _refine_web_queries(question, query, enriched)
        ddg_coros = [_run_ddg(q) for q in (q1, q2) if q]
        if not ddg_coros:
            logger.info("Web deep[3]: no refined queries generated — returning best effort")
            return enriched[:max_results]

        refined_batches = await asyncio.gather(*ddg_coros)
        new_results: list[dict] = []
        for batch in refined_batches:
            new_results.extend(batch)  # _run_ddg already deduplicates via seen_urls

        if new_results:
            logger.info("Web deep[3]: +%d new results from dual refinement", len(new_results))
            return (new_results + enriched)[:max_results]

        logger.info("Web deep[3]: refined searches empty — returning enriched")
        return enriched[:max_results]

    try:
        return await asyncio.wait_for(_pipeline(), timeout=_PIPELINE_TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning(
            "Web pipeline timeout (%.0fs) — returning %d partial results",
            _PIPELINE_TIMEOUT, len(best_so_far),
        )
        return best_so_far


# ══════════════════════════════════════════════════════════════════════════
#  TAVILY  (primary backend — designed for LLM agents)
# ══════════════════════════════════════════════════════════════════════════

_TAVILY_URL = "https://api.tavily.com/search"


async def search_tavily(
    query: str,
    original_message: str = "",
    max_results: int = 5,
) -> list[dict]:
    """Search via Tavily API. Returns results in the same {title, body, url, date?} format.

    Routing:
      - News queries  → topic="news", search_depth="basic", days=7
      - General       → topic="general", search_depth="advanced" (Tavily crawls full pages)

    include_answer=True: if Tavily generates a synthesised answer it is prepended
    as a "Synthèse" entry so the LLM sees the direct answer first.

    Returns [] on quota/API errors (caller falls back to DDG).
    Raises the exception on network errors so the caller can return INTERNET_ERROR.
    """
    is_news = _is_news_query(query)
    # Tavily understands natural language — send the full user question when available.
    # Truncate at 400 chars: beyond that, conversational preamble hurts precision.
    tavily_query = (original_message or query)[:400].strip()
    payload: dict = {
        "api_key": TAVILY_API_KEY,
        "query": tavily_query,
        "search_depth": "basic" if is_news else "advanced",
        "topic": "news" if is_news else "general",
        "max_results": max_results,
        "include_answer": True,
    }
    if is_news:
        payload["days"] = 7

    resp = await _HTTP.post(_TAVILY_URL, json=payload, timeout=25.0)

    if resp.status_code == 429:
        logger.warning("Tavily: quota exceeded (429) — falling back to DDG")
        return []
    if resp.status_code != 200:
        logger.warning("Tavily: HTTP %d — falling back to DDG", resp.status_code)
        return []

    data = resp.json()
    raw = data.get("results", [])
    answer = (data.get("answer") or "").strip()

    results: list[dict] = []

    # Synthesised answer — prepend only when substantial (avoids trivial "I don't know" entries)
    if len(answer) > 40:
        results.append({
            "title": "Synthèse",
            "body": answer,
            "url": raw[0]["url"] if raw else "",
            "date": None,
        })

    for r in raw:
        content = (r.get("content") or "")[:_PAGE_MAX_CHARS]
        if not content:
            continue
        entry: dict = {
            "title": r.get("title", ""),
            "body": content,
            "url": r.get("url", ""),
        }
        pub = r.get("published_date") or ""
        if pub:
            entry["date"] = pub[:10]  # normalise to YYYY-MM-DD
        results.append(entry)

    logger.info(
        "Tavily: %d results (synth=%s, depth=%s, topic=%s) for: %s",
        len(results),
        bool(answer),
        payload["search_depth"],
        payload["topic"],
        query[:50],
    )
    return results


# ══════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

_SEARCH_WEB_TIMEOUT = 30.0   # hard cap for the entire search_web call (> _PIPELINE_TIMEOUT)


async def search_web(
    query: str,
    max_results: int = 3,
    original_message: str = "",
) -> list[dict]:
    """
    Main search entry point.

    Routing priority:
      1. Weather keywords → Open-Meteo (structured data, no search engine needed)
      2. Tavily           → primary backend (TAVILY_API_KEY set)
                           handles both news and general queries natively
      3. DDG              → fallback when Tavily is unconfigured or fails
                           news → DDG news | general → 4-stage deep pipeline

    original_message is the raw user question passed to Tavily / DDG pipeline
    for intent-aware query refinement.

    Hard cap: _SEARCH_WEB_TIMEOUT seconds — returns INTERNET_ERROR on timeout.
    """
    async def _inner() -> list[dict]:
        # ── 1. Weather — always Open-Meteo ───────────────────────────────
        if _is_weather_query(query):
            results = await search_weather(query)
            if results:
                return results

        # ── 2. Tavily (primary) ───────────────────────────────────────────
        if TAVILY_API_KEY:
            try:
                results = await search_tavily(query, original_message, max(max_results, 5))
                if results:
                    return results
                logger.info("Tavily returned no results — falling back to DDG")
            except Exception as exc:
                if _is_network_error(exc):
                    logger.warning("Tavily: network unavailable (%s)", type(exc).__name__)
                    return INTERNET_ERROR
                logger.warning("Tavily error (%s) — falling back to DDG: %s", type(exc).__name__, exc)

        # ── 3. DDG fallback ───────────────────────────────────────────────
        if _is_news_query(query):
            try:
                results = await search_news(query, max_results=max_results)
            except Exception as exc:
                if _is_network_error(exc):
                    logger.warning("Internet unavailable (DDG news): %s", type(exc).__name__)
                    return INTERNET_ERROR
                results = []
            if results:
                return results

        try:
            results = await _ddg_text_deep(
                query, original_message, max_results=max(max_results, 5),
            )
            if not results:
                wiki = await search_wikipedia(query)
                if wiki:
                    return wiki
                logger.warning("All web backends returned empty — returning INTERNET_ERROR")
                return INTERNET_ERROR
            return results
        except Exception as exc:
            if _is_network_error(exc):
                logger.warning("Internet unavailable: %s", type(exc).__name__)
                return INTERNET_ERROR
            logger.error("Web search error: %s", exc)
            return INTERNET_ERROR

    try:
        return await asyncio.wait_for(_inner(), timeout=_SEARCH_WEB_TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning("search_web hard timeout (%.0fs) for query: %s", _SEARCH_WEB_TIMEOUT, query[:60])
        return INTERNET_ERROR

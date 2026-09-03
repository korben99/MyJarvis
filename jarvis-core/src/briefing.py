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
import html as _html
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

from config import (
    DEFAULT_TEMP,
    MAX_TOKENS_BRIEFING,
    PRIMARY_API_KEY,
    PRIMARY_API_URL,
    PRIMARY_MODEL,
    USER_ADMINS,
    USER_CITIES,
    llm_timeout,
    USER_CODES,
    USER_EMAILS,
    USERS,
    USER_TIMEZONES,
    USER_TRADING,
)
from google_services import (
    fetch_calendar_events,
    fetch_gmail_messages,
    is_google_available,
    send_gmail_message,
)
from helpers import (
    call_llm_async,
    extract_llm_json,
    fmt_event_time,
    get_logger,
    get_redis,
    now_user,
    today_user,
)
from memory import get_interest_weights, get_user_profile, get_user_projects
from prompts import get_prompt
from trading import get_portfolio_summary_text
from trading.market import render_briefing_block as render_market_block
from web_search import search_news, search_weather

logger = get_logger("jarvis-briefing")

# ── Redis key helpers ─────────────────────────────────────────────────────
_BRIEFING_TTL = 28 * 3600  # 28h — survives until next morning


def _briefing_key(user_code: str) -> str:
    return f"briefing:{user_code}:latest"


def _sent_key(user_code: str) -> str:
    return f"briefing:{user_code}:last_sent"


# ── Result dataclass ──────────────────────────────────────────────────────


@dataclass
class BriefingResult:
    user_code: str
    user_name: str
    generated_at: str  # ISO 8601
    text: str  # Conversational version for chat
    html: str  # Rich HTML for email
    sections: dict = field(default_factory=dict)  # Raw section data


# ── Redis store/retrieve ──────────────────────────────────────────────────


def store_briefing(user_code: str, result: BriefingResult) -> None:
    try:
        payload = json.dumps(
            {
                "user_code": result.user_code,
                "user_name": result.user_name,
                "generated_at": result.generated_at,
                "text": result.text,
                "html": result.html,
            },
            ensure_ascii=False,
        )
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
    "interests",
    "interest",
    "hobbies",
    "hobby",
    "topics",
    "passions",
    "sport",
    "loisir",
    "loisirs",
    "work",
    "profession",
    "expertise",
)
# Words too generic to be useful in a news query
_STOP_WORDS = {
    "informatique",
    "it",
    "the",
    "and",
    "or",
    "de",
    "du",
    "la",
    "le",
    "les",
    "des",
    "une",
    "pour",
    "avec",
    "dans",
    "sur",
    "par",
    "que",
    "qui",
    "est",
    "son",
    "ses",
    "mes",
    "mon",
    "notre",
    "votre",
}


def _split_top_level(raw: str) -> list[str]:
    """Split on commas that are OUTSIDE parentheses.

    Le champ `intérêts` du profil contient des listes imbriquées
    ("sports (vtt, footing), voyages (îles)") : un split brut sur la virgule
    coupait à l'intérieur des parenthèses et produisait les termes de recherche
    'sports (vtt' et 'footing)'. La parenthèse est conservée telle quelle, le
    moteur de news la traite comme du texte.
    """
    parts, buf, depth = [], [], 0
    for ch in raw:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf).strip())
    return [p for p in parts if p]


def _get_user_interests(user_code: str) -> list[str]:
    """
    Extract interest keywords from the user's Redis profile + stable profile.

    Strategy:
    - Stable profile `intérêts` field (clean comma-separated keywords) seeded first.
    - Redis keys with format "category:subcategory": the subcategory is used as the
      search term (e.g. "loisir:equitation" → "equitation"). Profile values can be
      verbose AI descriptions unsuitable for news queries.
    - Plain Redis keys (no subcategory): value parsed for short terms as before.
    """
    interests: list[str] = []

    # 1. Stable profile interests — always clean, comma-separated keywords
    stable = USERS.get(user_code, {}).get("profile", {})
    stable_interests_raw = stable.get("intérêts", "") or stable.get("interests", "")
    for term in _split_top_level(stable_interests_raw):
        if len(term) > 2 and term.lower() not in _STOP_WORDS:
            interests.append(term)

    # 2. Redis dynamic profile — use subcategory name, not verbose values
    profile = get_user_profile(user_code)
    _interest_key_set = set(_INTEREST_KEYS)
    matched_keys = [k for k in profile if k.split(":")[0] in _interest_key_set]
    for key in matched_keys:
        parts = key.split(":", 1)
        if len(parts) == 2:
            # Use the subcategory as the search term (clean, topic-level keyword)
            subcategory = parts[1].strip().replace("-", " ").replace("_", " ")
            if len(subcategory) > 2 and subcategory.lower() not in _STOP_WORDS:
                interests.append(subcategory)
        else:
            # Plain key — fall back to value parsing for short single-word terms
            val = profile.get(key, "").strip()
            for part in val.replace(",", ";").split(";"):
                part = part.strip()
                words = part.split()
                if len(words) == 1:
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
        p["name"] if not p.get("description") else f"{p['name']}: {p['description']}"
        for p in projects
        if p.get("status") in ("active", "in_progress")
    ][:4]


def _unwrap(val, default):
    """Unwrap an asyncio.gather result, returning default on exception."""
    if isinstance(val, BaseException):
        logger.warning("Briefing data source failed: %s", type(val).__name__)
        return default
    return val


_NEWS_GARBAGE = (
    "fil info", "en direct", "direct live", "retrouver", "abonnez",
    "flux rss", "rss feed", "regarder en", "nous suivre", "newsletter",
)


def _is_quality_article(r: dict) -> bool:
    """Rejette les résultats DDG qui sont des pages RSS/homepage plutôt que des articles."""
    title = r.get("title", "").lower()
    body = r.get("body", "")
    # Le body DDG a le format "[date | source] contenu" — extraire le contenu réel
    body_content = body.split("] ", 1)[1] if body.startswith("[") and "] " in body else body
    if any(m in title for m in _NEWS_GARBAGE):
        return False
    if len(body_content.strip()) < 40:  # snippet vide ou stub sans contenu
        return False
    return True


def _news_snippet(body: str) -> str:
    """Strip the [date | source] DDG prefix and return up to 160 chars of content."""
    content = body.split("] ", 1)[1] if body.startswith("[") and "] " in body else body
    return content[:160]


async def _fetch_news(interests: list[str]) -> list[dict]:
    """
    Fetch recent headlines in parallel: one query per interest (up to 3) +
    one general French news query.

    Returns list of dicts {title, snippet, url} — URL kept for briefing links.
    Interest results appear first (priority), then general news fills the rest.
    """
    general_query = "actualités france"

    async def _fetch_one(query: str, n: int) -> list[dict]:
        try:
            results = await search_news(query, n, region="fr-fr", timelimit="d")
            if not results:
                results = await search_news(query, n, region="fr-fr", timelimit="w")
            return [r for r in results if _is_quality_article(r)]
        except Exception as exc:
            logger.warning("News fetch failed for %r: %s", query, type(exc).__name__)
            return []

    # Gabarit fixe. Une reformulation LLM a été essayée puis abandonnée le
    # à partir d'un intérêt nu ("Horlogerie"), le modèle n'a aucun
    # contexte à exploiter et se contente de rembourrer avec des synonymes de
    # domaine ("horlogerie horloge horloger"). Sans injecter beaucoup de contexte
    # utilisateur, la reformulation n'apporte rien qu'un gabarit ne donne déjà.
    tasks = [_fetch_one(general_query, 4)] + [
        _fetch_one(f"info importantes {kw} ", 2) for kw in interests[:3]
    ]
    results_lists = await asyncio.gather(*tasks)
    gen_results = results_lists[0]
    interest_results = [r for sub in results_lists[1:] for r in sub]

    # Interest articles first, then general — deduplicate by title
    seen_titles: set = set()
    merged = []
    for r in interest_results + gen_results:
        t = r.get("title", "").strip()
        if t and t not in seen_titles:
            seen_titles.add(t)
            merged.append(r)

    return [
        {
            "title": r["title"],
            "snippet": _news_snippet(r.get("body", "")),
            "url": r.get("url", ""),
        }
        for r in merged[:7]
    ]


# ── LLM assembly ──────────────────────────────────────────────────────────


async def _assemble_with_llm(
    user_name: str,
    user_code: str,
    sections: dict,
) -> tuple[str, str]:
    """Call the LLM to write the final briefing text and HTML."""
    date_str = now_user(user_code).strftime("%A %d %B %Y")

    def _esc(s: str) -> str:
        """Escape braces in external data so str.format() doesn't crash on {word}."""
        return s.replace("{", "{{").replace("}", "}}")

    calendar_text = (
        "\n".join(
            (
                f"- {e['start']} : {_esc(e['summary'])}"
                + (f" ({_esc(e['location'])})" if e.get("location") else "")
                if e.get("all_day")
                else f"- {fmt_event_time(e['start'], user_code, '%H:%M')} : {_esc(e['summary'])}"
                + (f" ({_esc(e['location'])})" if e.get("location") else "")
            )
            for e in sections.get("calendar", [])
        )
        or "Aucun événement aujourd'hui."
    )

    gmail_text = (
        "\n".join(
            f"- De {_esc(m['from'])} | {_esc(m['subject'])}"
            for m in sections.get("gmail", [])[:5]
        )
        or "Aucun email non lu."
    )

    weather_text = _esc(sections.get("weather", "") or "Données météo indisponibles.")
    news_items = sections.get("news", [])
    news_text = (
        "\n".join(
            f"- {_esc(n['title'])} — {_esc(n['snippet'])} [URL: {n['url']}]"
            if isinstance(n, dict)
            else f"- {_esc(str(n))}"
            for n in news_items
        )
        or "Aucune actualité disponible."
    )
    interests_text = _esc(", ".join(sections.get("interests", [])) or "généralistes")
    projects_text = (
        "\n".join(f"- {_esc(p)}" for p in sections.get("projects", []))
        or "Aucun projet rappelé."
    )

    portfolio_text = _esc(
        sections.get("portfolio") or "Aucune donnée de portefeuille disponible."
    )
    market_text = _esc(
        sections.get("market") or "Aucune donnée de marché disponible."
    )

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
        market=market_text,
    )

    try:
        content = await call_llm_async(
            [
                {
                    "role": "system",
                    "content": get_prompt("BRIEFING_SYSTEM").format(
                        user_name=user_name
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=PRIMARY_MODEL,
            api_url=PRIMARY_API_URL,
            api_key=PRIMARY_API_KEY,
            temperature=DEFAULT_TEMP,
            max_tokens=MAX_TOKENS_BRIEFING,
            json_response=True,
            no_think=True,
            timeout=llm_timeout(MAX_TOKENS_BRIEFING),
        )
        result = extract_llm_json(content)
        return result.get("text", ""), result.get("html", "")
    except Exception as exc:
        logger.error("Briefing LLM assembly failed: %s: %s", type(exc).__name__, exc)
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
    calendar_task = (
        asyncio.to_thread(fetch_calendar_events, 1, today, tz_name, user_code)
        if has_google
        else asyncio.sleep(0, result=[])
    )
    gmail_task = (
        asyncio.to_thread(fetch_gmail_messages, "is:unread newer_than:1d", 5, user_code)
        if has_google
        else asyncio.sleep(0, result=[])
    )
    weather_task = search_weather(city)
    news_task = _fetch_news(interests)
    # Portefeuille et marché : réservés aux comptes dont le trading est activé, sur le
    # modèle de `has_google` ci-dessus. `USER_TRADING` est le drapeau qui décide qu'un
    # compte a un portefeuille ; sans cette garde le briefing sert un bloc boursier à
    # quelqu'un qui n'en a pas — le bloc marché est global, donc il se remplit même pour
    # un compte sans la moindre position.
    has_trading = user_code in USER_TRADING
    portfolio_task = (
        asyncio.to_thread(get_portfolio_summary_text, user_code)
        if has_trading
        else asyncio.sleep(0, result="")
    )
    # Perspectives de marché : un an d'historique par ligne + les grands indices. Lent la
    # première fois (~5 s, une quinzaine de téléchargements), instantané ensuite — le cache
    # Redis tient 20 h, donc le briefing du matin paie le coût et personne d'autre.
    # C'est aussi pour ça que ça vit ici et pas dans la boucle horaire de trading.core.
    market_task = (
        asyncio.to_thread(render_market_block, user_code)
        if has_trading
        else asyncio.sleep(0, result="")
    )

    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                calendar_task,
                gmail_task,
                weather_task,
                news_task,
                portfolio_task,
                market_task,
                return_exceptions=True,
            ),
            timeout=35.0,
        )
    except asyncio.TimeoutError:
        logger.warning("Briefing data gather timed out after 35s — using empty sources")
        results = [[], [], "", [], "", ""]

    calendar = _unwrap(results[0], [])
    gmail = _unwrap(results[1], [])
    weather_hits = _unwrap(results[2], [])
    # weather_hits[0] = conditions actuelles, weather_hits[1] = prévisions 3 jours.
    # Passer les deux au LLM pour éviter de reporter la température du matin (07:30)
    # comme température de la journée.
    weather = "\n".join(r["body"] for r in weather_hits) if weather_hits else ""
    news = _unwrap(results[3], [])
    portfolio = _unwrap(results[4], "")
    market = _unwrap(results[5], "")

    sections = {
        "calendar": calendar,
        "gmail": gmail,
        "weather": weather,
        "news": news,
        "interests": interests,
        "projects": projects,
        "portfolio": portfolio,
        "market": market,
    }

    text, html = await _assemble_with_llm(user_name, user_code, sections)

    # Pending prompt proposals — appended after the LLM assembly, admins only, and
    # deliberately routed around the model. Anything handed to it as context gets absorbed:
    # the in-chat reminder was injected 166 times and relayed to the user once. The creation
    # email fires only at creation, so a proposal sat pending 78 days unnoticed. The briefing
    # already lands in the admin's inbox every morning — it carries the reminder for free.
    if user_code in USER_ADMINS:
        from self import list_pending_proposals  # local import: self.py is heavy

        pending = list_pending_proposals()
        if pending:
            now = datetime.now(timezone.utc)
            lines = [
                f"{p['prompt_name']} — {p['topic']} "
                f"(en attente depuis {(now - datetime.fromisoformat(p['created_at'])).days} j, "
                f"id {p['id']})"
                for p in pending
            ]
            text += "\n\nÀ valider — propositions de prompt :\n" + "\n".join(
                f"- {line}" for line in lines
            )
            html += (
                "<p><b>À valider — propositions de prompt</b><br>"
                + "<br>".join(_html.escape(line) for line in lines)
                + "</p>"
            )

    result = BriefingResult(
        user_code=user_code,
        user_name=user_name,
        generated_at=datetime.now(timezone.utc).isoformat(),
        text=text,
        html=html,
        sections=sections,
    )

    logger.info(
        "Briefing ready for %s (%d chars text, %d chars html)",
        user_name,
        len(text),
        len(html),
    )
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
    success = send_gmail_message(
        to=to,
        subject=subject,
        html_body=result.html,
        text_body=result.text,
        user_code=user_code,
    )

    if success:
        r.setex(sent_key, _BRIEFING_TTL, today)
        logger.info("Briefing email delivered to %s for %s", to, user_code)

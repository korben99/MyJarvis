"""
PROJECT JARVIS v7
=======================================
The iPhone handles all audio (WhisperKit STT + iOS TTS).
This API receives TEXT, queries the LLM, and returns TEXT.

Features:
  - Text chat via OpenAI-compatible API (streaming SSE or JSON)
  - Persistent memory across sessions (Redis)
  - User profile learning (auto-extracted from conversations)
  - Emotional state tracking
  - RAG: searches Qdrant for relevant document chunks
  - Web search: weather (Open-Meteo), news (DDG news), general (DDG text)
  - Voice mode: concise responses for iPhone TTS

Architecture memoire
Jarvis API
│
├ Redis
│ ├ working memory (conversation active, contexte de session, historique court, accès ultra-rapide)
│ ├ session chat
│ └ user profile
│
├ Qdrant
│ ├ open-webui_knowledge
│ └ jarvis_memory (souvenirs vectorisés, récupération par similarité, mémoire longue durée)
│
└ LLM

Endpoints:
  POST   /chat                    — Text chat (streaming SSE or JSON)
  GET    /status                  — Health check
  GET    /models                  — List available models
  GET    /search?q=...            — Test RAG without LLM
  GET    /web?q=...               — Test web search without LLM
  DELETE /conversations/{id}      — Clear session history
  GET    /memory/profile          — View learned user profile
  GET    /memory/emotional-state  — View current emotional state
  GET    /memory/recent           — View recent conversation log
  GET    /memory/self             — View Jarvis self-knowledge
  DELETE /memory/reset            — Reset all memory
  POST   /device/register         — Register iOS device token
  GET    /device/pending/{code}   — Poll and clear pending push notifications
  POST   /device/push/test/{code} — Manually trigger proactive push (dev)

  conversation
    ↓
analyzer.py
    ↓
importance + memory_summary
    ↓
memory.py
    ↓
Redis episodic log
    ↓
Qdrant episodic memory
    ↓
novelty filter
    ↓
autobiographical memory
    ↓
memory compression
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import pickle
import time
import unicodedata
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional
from urllib.parse import quote

import httpx
import pytz
import redis as redis_lib
import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from ddgs import DDGS
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient

# Local modules (memory system)
from analyzer import analyze_exchange
from google_services import (
    fetch_calendar_events,
    fetch_gmail_messages,
    is_google_available,
)
from llm_router import RouterResult, llm_route
from briefing import (
    BriefingResult,
    deliver_briefing,
    gather_briefing,
    get_stored_briefing,
    store_briefing,
)
from self import (
    gather_context,
    generate_proactive_push,
    get_current_focus,
    get_reflection_log,
    get_user_relation,
    handle_proposal_command,
    list_pending_proposals,
    run_nightly_interaction_review,
    run_self_reflection,
)
from trading import (
    get_portfolio_summary_text,
    import_csv_to_redis,
    pop_pending_alerts,
    run_trade_check,
    get_portfolio,
    update_prices_in_redis,
    evaluate_alerts,
    fetch_live_prices,
    push_pending_alert,
    auto_set_thresholds,
    suggest_thresholds_llm,
)

# config file load
from config import (
    ANALYSIS_MODEL,
    BRIEFING_ENABLED,
    BRIEFING_TIME,
    BRIEFING_TIMEZONE,
    EMAIL_TO_CODE,
    EMBED_MODEL_NAME,
    ENABLE_ANALYSIS,
    IOS_MAX_MESSAGES,
    OPENAI_API_KEY,
    OPENAI_API_URL,
    PRIMARY_API_KEY,
    PRIMARY_API_URL,
    PRIMARY_MODEL,
    PRIMARY_TIMEOUT,
    QDRANT_COLLECTION,
    QDRANT_MEMORY_COLLECTION,
    QDRANT_URL,
    RAG_SCORE_THRESHOLD,
    RAG_TOP_K,
    REASONING_API_KEY,
    REASONING_API_URL,
    REASONING_MODEL,
    REASONING_TIMEOUT,
    VISION_API_KEY,
    VISION_API_URL,
    VISION_MODEL,
    VISION_TIMEOUT,
    REDIS_URL,
    REFLECTION_INTERVAL_HOURS,
    ROUTER_API_KEY,
    ROUTER_API_URL,
    ROUTER_MODEL,
    SELF_MEMORY_PATH,
    USER_CITIES,
    USER_CODES,
    USER_TRADING,
    USERS,
    no_think_suffix,
    tokens_param,
)
from memory import (
    MODEL_CACHE_DIR,
    append_conversation_message,
    build_memory_context,
    get_conversation,
    get_embed_model,
    get_emotional_state,
    get_recent_conversations,
    get_self_memory,
    get_user_preferences,
    get_user_profile,
    get_user_projects,
    log_conversation,
    search_memory,
    update_emotional_state,
    set_interest_weight,
    update_user_profile,
    update_user_projects,
)

HAS_MEMORY = True

# BUDGET VAR
MEMORY_CHAR_BUDGET = int(
    os.getenv("MEMORY_CHAR_BUDGET", "2500")
)  # Maximum char of memory to send in the prompt
RAG_CHAR_BUDGET = int(
    os.getenv("RAG_CHAR_BUDGET", "4000")
)  # Maximum char of RAG to send in the prompt
WEB_CHAR_BUDGET = int(
    os.getenv("WEB_CHAR_BUDGET", "2000")
)  # Maximum char of Web Research to send in the prompt
GOOGLE_CHAR_BUDGET = int(
    os.getenv("GOOGLE_CHAR_BUDGET", "3000")
)  # Maximum char of Gmail/Calendar context to send in the prompt
TOTAL_CONTEXT_BUDGET = int(
    os.getenv("TOTAL_CONTEXT_BUDGET", "10000")
)  # Hard ceiling across all context sources combined

# ROUTER VAR
ROUTER_MEMORY_THRESHOLD = 0.35
ROUTER_RAG_THRESHOLD = 0.38
ROUTER_WEB_THRESHOLD = 0.52
ROUTER_GMAIL_THRESHOLD = 0.38
ROUTER_CALENDAR_THRESHOLD = 0.38
ROUTER_BRIEFING_THRESHOLD = 0.55
ROUTER_SELF_THRESHOLD = 0.55
ROUTER_PORTFOLIO_THRESHOLD = 0.40

# ── Lazy-loaded Users and embedding ──
EMBED_MODEL = None
INTENT_EMBEDDINGS = {}
INTENT_EXAMPLES_FR = {
    "memory": [
        "on a parlé avant",
        "souviens-toi de ce que je t'ai dis",
        "mes preferences",
        "mes projects",
        "ce que tu sais",
    ],
    "rag": [
        "cherche dans mes documents",
        "regarde mes fichiers",
        "trouve l'info dans mes documents",
        "analyse mes documents",
        "regarde mes notes",
        "RAG",
    ],
    "web": [
        "cherche sur internet",
        "recherche sur le web",
        "googles les dernières nouvelles sur",
        "quelles sont les actualités du jour",
        "trouve moi les dernières infos sur",
        "quel est le cours de la bourse aujourd'hui",
        "quelle est l'actualité en ce moment",
        "les dernières nouvelles",
        "qu'est ce qui se passe dans l'actualité",
        "cherche en ligne",
    ],
    "gmail": [
        "regarde mes emails",
        "est-ce que j'ai des mails",
        "mes messages non lus",
        "cherche dans ma boite mail",
        "emails de",
        "mes courriels",
        "recherche dans mes emails",
        "est-ce que j'ai reçu un mail",
        "mails non lus",
        "regarde ma boite de réception",
        "est-ce que j'ai un mail de",
        "est-ce que j'ai reçu un message de",
        "regarde si j'ai un mail",
        "un mail de livraison",
        "mail de confirmation",
        "vérifie mes mails",
        "consulte ma boite mail",
        "dernier mail reçu",
        "derniers mails",
        "quels sont mes derniers emails",
        "quel est le dernier mail",
        "mail reçu aujourd'hui",
        "mail reçu récemment",
        "nouveaux emails",
        "nouveaux mails",
        "mes mails récents",
    ],
    "calendar": [
        "mon agenda",
        "mes rendez-vous",
        "qu'est ce que j'ai aujourd'hui",
        "mes prochains événements",
        "qu'est ce que j'ai cette semaine",
        "réunion cette semaine",
        "planning de la semaine",
        "qu'est ce que j'ai ce mois",
        "mes disponibilités",
        "qu'est ce que j'ai demain",
    ],
    "briefing": [
        "mon briefing",
        "brief du matin",
        "quoi de neuf ce matin",
        "résumé du jour",
        "résumé de ma journée",
        "donne moi mon briefing",
        "briefing matinal",
        "qu'est ce que j'ai ce matin",
        "comment se passe ma journée",
        "tour d'horizon du jour",
    ],
    "self": [
        "qu'est ce que tu penses en ce moment",
        "quel est ton focus",
        "comment tu vas",
        "ton état d'esprit",
        "qu'est ce que tu fais",
        "tes objectifs",
        "tu penses à quoi",
        "ton dernier rapport",
        "ta dernière réflexion",
        "qu'est ce que tu as appris",
    ],
    "portfolio": [
        "mon portefeuille",
        "mes actions",
        "cours de bourse",
        "mes investissements",
        "performance de mon portefeuille",
        "dividendes",
        "alerte bourse",
        "plus-value moins-value",
        "comment se porte mon portefeuille",
        "LVMH Sanofi AXA Capgemini Veolia",
        "mes positions",
        "valeur de mes actions",
    ],
}

logger = logging.getLogger("jarvis-api")
# ── Lazy-loaded components ──
_qdrant_client = None
HTTP_CLIENT = httpx.AsyncClient(timeout=30.0)
REDIS_CLIENT = redis_lib.from_url(REDIS_URL, decode_responses=True)


def get_qdrant():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL)
    return _qdrant_client


# ── System prompt (strings live in prompts.py — use get_prompt for live overrides) ──
from prompts import get_prompt


def build_system_prompt(
    session_id: str, voice_mode: bool = False, user_code: str = "default"
) -> str:
    prompt = get_prompt("SYSTEM_BASE_FR")

    if HAS_MEMORY:
        memory_ctx = build_memory_context(session_id, user_code)
        if memory_ctx:
            prompt += f"{get_prompt('MEMORY_HEADER_FR')}\n{memory_ctx}"

    if voice_mode:
        prompt += get_prompt("VOICE_SUFFIX_FR")

    return prompt


# ── App ──


def _load_intent_embeddings(embed_model) -> dict:
    """
    Load intent embeddings from disk cache if the examples haven't changed,
    otherwise recompute and save. Cache is invalidated by a SHA-256 of the
    serialized INTENT_EXAMPLES_FR dict.
    """
    cache_dir = MODEL_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Fingerprint of the current intent examples
    raw = json.dumps(INTENT_EXAMPLES_FR, sort_keys=True, ensure_ascii=False).encode()
    fingerprint = hashlib.sha256(raw).hexdigest()[:16]
    cache_file = os.path.join(cache_dir, f"intent_embeddings_{fingerprint}.pkl")

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                embeddings = pickle.load(f)
            logger.info("Intent embeddings loaded from cache (fingerprint=%s)", fingerprint)
            return embeddings
        except Exception:
            pass  # corrupt cache — recompute below

    logger.info("Computing intent embeddings (fingerprint=%s)...", fingerprint)
    embeddings = {
        intent: [embed_model.encode(e, normalize_embeddings=True) for e in examples]
        for intent, examples in INTENT_EXAMPLES_FR.items()
    }
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(embeddings, f)
        logger.info("Intent embeddings cached to %s", cache_file)
        # Remove stale cache files from previous fingerprints
        for old in os.listdir(cache_dir):
            if old.startswith("intent_embeddings_") and old != os.path.basename(cache_file):
                os.remove(os.path.join(cache_dir, old))
    except Exception as exc:
        logger.warning("Could not save intent embeddings cache: %s", type(exc).__name__)

    return embeddings


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)

    logger.info(
        f"Jarvis API v7 starting — router: {ROUTER_MODEL}, reasoning: {REASONING_MODEL}, memory: {HAS_MEMORY}, analysis: {ENABLE_ANALYSIS and HAS_MEMORY}"
    )
    logger.info(
        f"RAG: {QDRANT_URL}, collection: {QDRANT_COLLECTION}, top_k: {RAG_TOP_K}"
    )
    global INTENT_EMBEDDINGS
    global EMBED_MODEL

    if HAS_MEMORY:
        EMBED_MODEL = get_embed_model()
        get_qdrant()
        INTENT_EMBEDDINGS.update(_load_intent_embeddings(EMBED_MODEL))

    # ── Schedulers (briefing + proto-self — independent features) ──
    scheduler = None
    try:
        hour, minute = (int(x) for x in BRIEFING_TIME.split(":"))
        tz = pytz.timezone(BRIEFING_TIMEZONE)
        scheduler = AsyncIOScheduler(timezone=tz)

        # Proto-self: always active regardless of BRIEFING_ENABLED
        scheduler.add_job(
            run_self_reflection,
            trigger="interval",
            hours=REFLECTION_INTERVAL_HOURS,
            id="self_reflection",
            next_run_time=datetime.now(tz),   # also run once at startup
        )
        scheduler.add_job(
            run_nightly_interaction_review,
            trigger="cron",
            hour=23,
            minute=0,
            id="nightly_interaction_review",
        )

        # Morning briefing: conditional
        if BRIEFING_ENABLED:
            scheduler.add_job(
                _run_morning_briefings,
                trigger="cron",
                hour=hour,
                minute=minute,
                id="morning_briefing",
            )
            logger.info("Morning briefing scheduled at %s (%s)", BRIEFING_TIME, BRIEFING_TIMEZONE)

        # Trading surveillance: hourly price check + alert evaluation
        async def _run_trade_checks():
            await run_trade_check(USER_TRADING)

        scheduler.add_job(
            _run_trade_checks,
            trigger="interval",
            hours=1,
            id="trade_check",
            next_run_time=datetime.now(tz),   # also run once at startup to import any waiting CSV
        )
        logger.info("Trading surveillance scheduled every 1 h")

        scheduler.start()
        logger.info("Self reflection scheduled every %dh", REFLECTION_INTERVAL_HOURS)
        logger.info("Nightly review scheduled at 23:00 (%s)", BRIEFING_TIMEZONE)
    except Exception as exc:
        logger.error("Scheduler failed to start: %s", type(exc).__name__)

    yield

    if scheduler:
        scheduler.shutdown(wait=False)
    await HTTP_CLIENT.aclose()


app = FastAPI(title="Jarvis API", version="6.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request model ──


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    user_code: Optional[str] = None
    model: Optional[str] = None
    stream: bool = True
    voice_mode: bool = False
    use_rag: bool = False  # router decides; set True to force RAG regardless
    use_web: bool = False
    image_parts: list = []           # OpenAI image_url part dicts forwarded from the proxy
    image_base64: Optional[str] = None  # base64 JPEG/PNG sent directly by the iOS app


# =============================
# ── Helpers ──
# =============================
def openai_headers() -> dict:
    return {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }


def select_model(
    req_model: "str | None",
    use_reasoning: bool = False,
) -> tuple[str, str, str, float]:
    """
    Single decision point for LLM tier selection.

    Returns (model_name, api_url, api_key, timeout_seconds).

    Priority:
    1. Explicit model override in the request  →  honour as-is (PRIMARY credentials)
    2. Router flagged use_reasoning=True        →  REASONING_MODEL (cloud, complex only)
    3. Everything else                          →  PRIMARY_MODEL (local Qwen / gpt-4o-mini)
    """
    if req_model:
        return req_model, PRIMARY_API_URL, PRIMARY_API_KEY, PRIMARY_TIMEOUT

    if use_reasoning:
        return REASONING_MODEL, REASONING_API_URL, REASONING_API_KEY, REASONING_TIMEOUT

    return PRIMARY_MODEL, PRIMARY_API_URL, PRIMARY_API_KEY, PRIMARY_TIMEOUT


def trim_chunks(chunks, char_budget, text_key="text", max_item_chars=800):
    """
    Generic chunk limiter for RAG or web results.
    """
    total = 0
    selected = []

    for c in chunks:
        text = c[text_key][:max_item_chars]

        if total + len(text) > char_budget:
            break

        selected.append(text)
        total += len(text)

    return selected


def optimize_web_query(message: str) -> str:
    """
    Convert conversational question into search-engine query
    """
    msg = message.lower()

    remove = [
        "est ce que",
        "peux tu",
        "dis moi",
        "explique",
        "pourquoi",
        "comment",
        "tell me",
        "what is",
        "why",
        "how",
    ]

    for r in remove:
        msg = msg.replace(r, "")

    msg = msg.strip()

    # remove punctuation
    msg = msg.replace("?", "").replace("!", "")

    # shorten
    words = msg.split()
    if len(words) > 10:
        words = words[:10]

    return " ".join(words)


# SEMANTIC ROUTING
def semantic_route_query(message: str):

    if EMBED_MODEL is None:
        raise RuntimeError("Embedding model not initialized")
    q = EMBED_MODEL.encode(message, normalize_embeddings=True)
    scores = {}

    for intent, vectors in INTENT_EMBEDDINGS.items():
        scores[intent] = max(float(q @ v) for v in vectors)

    use_memory    = scores["memory"]                  > ROUTER_MEMORY_THRESHOLD
    use_rag       = scores["rag"]                     > ROUTER_RAG_THRESHOLD
    use_web       = scores["web"]                     > ROUTER_WEB_THRESHOLD
    use_gmail     = scores.get("gmail", 0)            > ROUTER_GMAIL_THRESHOLD     and is_google_available()
    use_calendar  = scores.get("calendar", 0)         > ROUTER_CALENDAR_THRESHOLD  and is_google_available()
    use_briefing  = scores.get("briefing", 0)         > ROUTER_BRIEFING_THRESHOLD
    use_self      = scores.get("self", 0)             > ROUTER_SELF_THRESHOLD
    use_portfolio = scores.get("portfolio", 0)        > ROUTER_PORTFOLIO_THRESHOLD

    logger.info(
        "routing: memory=%s rag=%s web=%s gmail=%s calendar=%s briefing=%s self=%s portfolio=%s | "
        "scores memory=%.3f rag=%.3f web=%.3f gmail=%.3f calendar=%.3f briefing=%.3f self=%.3f portfolio=%.3f",
        use_memory, use_rag, use_web, use_gmail, use_calendar, use_briefing, use_self, use_portfolio,
        scores.get("memory", 0), scores.get("rag", 0), scores.get("web", 0),
        scores.get("gmail", 0), scores.get("calendar", 0), scores.get("briefing", 0),
        scores.get("self", 0), scores.get("portfolio", 0),
    )

    if not use_memory and not use_rag and not use_web and not use_gmail and not use_calendar \
            and not use_briefing and not use_self and not use_portfolio:
        use_memory = True

    return use_memory, use_rag, use_web, use_gmail, use_calendar, use_briefing, use_self, use_portfolio


# =============================
# ── RAG: Search Qdrant (async)
# =============================


async def search_documents(query: str, top_k: int = RAG_TOP_K) -> list[dict]:
    """Embed query and search Qdrant for relevant chunks."""
    try:
        loop = asyncio.get_running_loop()
        if EMBED_MODEL is None:
            raise RuntimeError("Embedding model not initialized")
        vector = await loop.run_in_executor(None, lambda: EMBED_MODEL.encode(query, normalize_embeddings=True).tolist())
        client = await loop.run_in_executor(None, get_qdrant)
        results = await loop.run_in_executor(
            None,
            lambda: (
                client.query_points(
                    collection_name=QDRANT_COLLECTION, query=vector, limit=top_k
                ).points
            ),
        )

        chunks = []
        for hit in results:
            if hit.score < RAG_SCORE_THRESHOLD:
                continue
            payload = hit.payload
            text = ""
            if "_node_content" in payload:
                try:
                    node = json.loads(payload["_node_content"])
                    text = node.get("text", "")
                except Exception:
                    text = str(payload.get("_node_content", ""))
            else:
                text = payload.get("text", payload.get("content", ""))

            # OpenWebUI stores metadata in payload["metadata"] dict
            meta = payload.get("metadata") or {}
            source = (
                meta.get("name")
                or meta.get("source")
                or payload.get("file_name")
                or "unknown"
            )

            if text:
                chunks.append(
                    {"text": text[:1500], "source": source, "score": hit.score}
                )

        logger.info("RAG: %d relevant chunks for: %s", len(chunks), query[:50])
        return chunks
    except Exception as e:
        logger.warning("RAG search failed: %s", e)
        return []


# =============================
# ── Weather: Open-Meteo (free, no API key) ──
# =============================


_WEATHER_CODES = {
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


def _is_weather_query(query: str) -> bool:
    keywords = [
        "météo", "meteo", "weather", "forecast", "prévision",
        "température", "temperature", "degrés", "degré",
        "pluie", "rain", "neige", "snow", "grêle",
        "vent", "wind", "rafale",
        "soleil", "sun", "nuage", "nuageux", "nuageuse", "couvert",
        "ensoleillé", "ensoleillée", "orage", "brouillard", "brume",
        "humidité", "précipitation",
    ]
    q = query.lower()
    return any(k in q for k in keywords)


# DEPRECATED: location extraction is now handled by the LLM router (weather_location field).
# Kept as fallback for the embedding router path only.
def _extract_location(query: str) -> str:
    stop = {
        # weather terms
        "météo", "meteo", "weather", "forecast", "prévision", "prévisions",
        "température", "temperature", "temps", "climat",
        "pluie", "vent", "neige", "soleil", "nuage", "nuageux", "nuageuse",
        "orage", "brouillard", "brume", "humidité", "degrés", "degré",
        "mini", "maxi", "minimum", "maximum", "min", "max",
        # time words
        "aujourd'hui", "today", "demain", "tomorrow", "semaine", "week",
        "matin", "soir", "après-midi", "nuit", "maintenant", "now",
        "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche",
        # question words
        "quelle", "quel", "quels", "quelles", "comment", "est-ce", "est",
        "qu'est", "que", "quoi",
        # filler / prepositions
        "à", "a", "de", "du", "pour", "en", "sur", "le", "la", "les", "dans",
        "il", "y", "va", "fait", "aura", "t", "avoir", "au", "aux",
        # politeness / request words
        "merci", "svp", "stp", "bonjour", "bonsoir", "s'il", "vous", "plaît",
        "moi", "dis", "donne", "montre", "cherche", "dites",
        "?", "!", ".",
    }
    # strip punctuation from each token, skip pure numbers (dept codes etc.)
    words = []
    for w in query.split():
        clean = w.strip("?!.,;:")
        if clean.lower() not in stop and not clean.isdigit():
            words.append(clean)
    return " ".join(words).strip()


async def search_weather(query: str) -> list[dict]:
    """Fetch real forecast from Open-Meteo (geocoding + forecast, no API key)."""
    location = _extract_location(query)
    if not location:
        logger.warning("Weather: no location extracted from query, skipping")
        return []
    try:
        c = HTTP_CLIENT
        geo = await c.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "fr"},
        )
        geo.raise_for_status()
        results = geo.json().get("results")
        if not results:
            logger.warning(f"Weather: location not found for '{location}'")
            return []
        place = results[0]
        lat, lon = place["latitude"], place["longitude"]
        name = place.get("name", location)
        country = place.get("country", "")

        wx = await c.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m,relative_humidity_2m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
                "timezone": "auto",
                "forecast_days": 3,
            },
        )
        wx.raise_for_status()
        data = wx.json()

        cur = data.get("current", {})
        daily = data.get("daily", {})
        condition = _WEATHER_CODES.get(cur.get("weather_code", -1), "")

        now_body = (
            f"Actuellement à {name} ({country}) : {cur.get('temperature_2m')}°C "
            f"(ressenti {cur.get('apparent_temperature')}°C), {condition}. "
            f"Vent {cur.get('wind_speed_10m')} km/h, Humidité {cur.get('relative_humidity_2m')}%."
        )

        days = []
        for i, date in enumerate(daily.get("time", [])[:3]):
            code = daily["weather_code"][i]
            days.append(
                f"{date}: {_WEATHER_CODES.get(code, '')} "
                f"{daily['temperature_2m_min'][i]}–{daily['temperature_2m_max'][i]}°C, "
                f"précip. {daily['precipitation_sum'][i]} mm, "
                f"vent max {daily['wind_speed_10m_max'][i]} km/h"
            )
        forecast_body = " | ".join(days)

        logger.info(f"Weather: fetched forecast for {name}")
        return [
            {
                "title": f"Météo actuelle — {name}",
                "body": now_body,
                "url": f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}",
            },
            {
                "title": f"Prévisions 3 jours — {name}",
                "body": forecast_body,
                "url": f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}",
            },
        ]
    except Exception as e:
        logger.error(f"Weather fetch error: {e}")
        return []


# =============================
# ── News: DuckDuckGo news search ──
# =============================


def _is_news_query(query: str) -> bool:
    keywords = [
        "news",
        "actualité",
        "actualites",
        "actu",
        "dernière",
        "dernieres",
        "latest",
        "recent",
        "aujourd'hui",
        "today",
        "breaking",
        "que se passe",
        "what is happening",
        "en ce moment",
    ]
    q = query.lower()
    return any(k in q for k in keywords)


# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _ddg_news_sync(query: str, max_results: int) -> list[dict]:
    """Synchronous DDG news fetch — run via run_in_executor."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.news(query, max_results=max_results):
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


def _ddg_text_sync(query: str, max_results: int) -> list[dict]:
    """Synchronous DDG text fetch — run via run_in_executor."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "body":  r.get("body", ""),
                "url":   r.get("href", ""),
            })
    return results


def _fmt_event_time(iso: str, user_tz: pytz.BaseTzInfo) -> str:
    """Convert an ISO 8601 datetime string to a localised HH:MM display string."""
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(user_tz).strftime("%d/%m %H:%M")
    except ValueError:
        return iso  # all-day date string "YYYY-MM-DD" — return as-is


# ══════════════════════════════════════════════════════════════════════════

async def search_news(query: str, max_results: int = 5) -> list[dict]:
    """Fetch news via DDG news search (returns real articles with date and source)."""
    try:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, _ddg_news_sync, query, max_results)
        logger.info(f"News: {len(results)} articles for: {query[:50]}")
        return results
    except Exception as e:
        logger.error(f"News search error: {e}")
        return []


# =============================
# ── WEB SEARCH
# =============================


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


async def search_web(query: str, max_results: int = 3) -> list[dict]:
    """Route weather to Open-Meteo, news to DDG news, everything else to DDG text."""
    if _is_weather_query(query):
        results = await search_weather(query)
        if results:
            return results

    if _is_news_query(query):
        results = await search_news(query, max_results=max_results)
        if results:
            return results

    try:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, _ddg_text_sync, query, max_results)
        logger.info(f"Web: {len(results)} results for: {query[:50]}")
        # deduplicate by url
        seen = set()
        clean = []

        for r in results:
            if r["url"] in seen:
                continue
            seen.add(r["url"])
            clean.append(r)

        # fallback wikipedia if no result in DDKGO
        if not clean:
            wiki = await search_wikipedia(query)
            if wiki:
                return wiki

        return clean

    except Exception as e:
        logger.error(f"Web search error: {e}")
        return []


# =============================
# ── WIKIPEDIA SEARCH
# =============================
async def search_wikipedia(query: str):
    url = f"https://fr.wikipedia.org/api/rest_v1/page/summary/{quote(query)}"

    try:
        c = HTTP_CLIENT
        r = await c.get(url)

        if r.status_code != 200:
            return []

        data = r.json()

        return [
            {
                "title": data.get("title"),
                "body": data.get("extract"),
                "url": data.get("content_urls", {}).get("desktop", {}).get("page"),
            }
        ]
    except (httpx.RequestError, ValueError) as e:
        logger.warning(f"Wikipedia search failed: {e}")
        return []


# =============================
# ── Streaming OPENAI
# =============================
async def stream_openai(
    messages: list,
    model: str,
    api_url: str = OPENAI_API_URL,
    api_key: str = OPENAI_API_KEY,
    timeout: float = 30.0,
) -> AsyncGenerator[str, None]:

    try:
        # Use a dedicated client with the correct per-tier timeout.
        # (HTTP_CLIENT is a shared 30s singleton — too short for large models.)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{api_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True,
                },
            ) as response:
                if response.status_code != 200:
                    logger.error(f"OpenAI streaming error: {response.status_code}")
                    return

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    if not line.startswith("data: "):
                        continue

                    payload = line[6:]

                    if payload == "[DONE]":
                        break

                    try:
                        data = json.loads(payload)

                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content")

                        if content:
                            yield content

                    except json.JSONDecodeError:
                        logger.debug(f"Invalid JSON chunk: {payload[:100]}")
                        continue

    except httpx.RequestError as e:
        logger.error(f"OpenAI request error: {e}")


# ══════════════════════════════════════════════════
#  VISION — two-stage image pipeline
# Pour le prochain Qwen3-7VL Besoin de changement dans le max token et dans le flux d'accès de l'image (HTTP transformé en Base64)
# ══════════════════════════════════════════════════

async def _resolve_image_part(part: dict, client: httpx.AsyncClient) -> dict:
    """
    Ensure an image_url part contains a publicly accessible URL or a base64 data URI.
    Open WebUI sends images as internal Docker URLs (e.g. http://open-webui:8080/...) that
    OpenAI cannot reach. We fetch those internally and re-encode as base64.
    Already-public URLs (https://) and existing data URIs are returned unchanged.
    """
    url = part.get("image_url", {}).get("url", "")
    if url.startswith("data:") or url.startswith("https://"):
        return part  # already usable by OpenAI

    # Internal URL — fetch and convert to base64
    try:
        r = await client.get(url, timeout=15)
        r.raise_for_status()
        mime = r.headers.get("content-type", "image/jpeg").split(";")[0]
        b64 = base64.b64encode(r.content).decode()
        logger.debug("Vision: re-encoded internal image (%s, %d bytes)", mime, len(r.content))
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
    except Exception as exc:
        logger.warning("Vision: could not fetch internal image URL (%s): %s", url[:80], exc)
        return part  # return as-is and let OpenAI reject it with a clear error


async def describe_images(image_parts: list, text_prompt: str) -> str:
    """
    Call VISION_MODEL to produce a detailed description of uploaded images.
    Returns empty string on failure or when VISION_MODEL is not configured.
    """
    if not VISION_MODEL or not image_parts:
        return ""

    try:
        async with httpx.AsyncClient(timeout=VISION_TIMEOUT) as client:
            resolved = await asyncio.gather(*[_resolve_image_part(p, client) for p in image_parts])
            vision_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyse cette image de façon exhaustive et structurée.\n"
                                "Décris dans cet ordre :\n"
                                "1. Sujet principal et contexte général de la scène\n"
                                "2. Tout texte, chiffre ou étiquette visible — recopie-le mot pour mot\n"
                                "3. Personnes présentes (apparence, expression, action)\n"
                                "4. Objets, couleurs dominantes, positions relatives\n"
                                "5. Tout élément pertinent pour répondre à la question ci-dessous\n\n"
                                f"Question de l'utilisateur : {text_prompt or 'Que contient cette image ?'}"
                            ),
                        },
                        *resolved,
                    ],
                }
            ]
            resp = await client.post(
                f"{VISION_API_URL}/chat/completions",
                headers={"Authorization": f"Bearer {VISION_API_KEY}"},
                json={
                    "model": VISION_MODEL,
                    "messages": vision_messages,
                    tokens_param(VISION_MODEL): 1024,
                    "stream": False,
                },
            )
        data = resp.json()
        if resp.status_code != 200:
            logger.warning("Vision: API error %d — %s", resp.status_code, data.get("error", {}).get("message", ""))
            return ""
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.warning("Vision: image description failed (%s)", type(exc).__name__)
        return ""


# ==================================
# ── Post response analysis memory learning
# ==================================
async def post_analysis(
    session_id: str, user_code: str, user_msg: str, assistant_msg: str
):
    """Run after each exchange: extract topics, mood, facts. Non-blocking."""
    if not HAS_MEMORY:
        return
    try:
        analysis = await analyze_exchange(user_msg, assistant_msg)
        importance = analysis.get("importance", 0)
        log_conversation(
            user_code=user_code,
            session_id=session_id,
            user_msg=user_msg,
            assistant_msg=assistant_msg,
            mood=analysis.get("mood", "neutral"),
            topics=analysis.get("topics", []),
            importance=importance,
            memory_summary=analysis.get("memory_summary"),
        )

        mood = analysis.get("mood", "neutral")
        mood_to_state = {
            "happy": {"mood": "happy", "energy": 0.8},
            "stressed": {"mood": "attentive", "concern": 0.6},
            "frustrated": {"mood": "supportive", "concern": 0.7},
            "curious": {"mood": "engaged", "curiosity": 0.8},
            "tired": {"mood": "gentle", "energy": 0.4},
            "focused": {"mood": "focused", "energy": 0.7},
        }
        if mood in mood_to_state:
            update_emotional_state(mood_to_state[mood])

        for fact in analysis.get("user_facts", []):
            if "key" in fact and "value" in fact:
                update_user_profile(user_code, fact["key"], fact["value"] or None)

        for iw in analysis.get("interest_weights") or []:
            if "term" in iw and "weight" in iw:
                set_interest_weight(user_code, iw["term"], float(iw["weight"]))

        projects = analysis.get("projects", [])
        if projects:
            existing = get_user_projects(user_code)
            existing_names = [p["name"] if isinstance(p, dict) else p for p in existing]
            for proj in projects:
                if proj not in existing_names:
                    existing.append(
                        {
                            "name": proj,
                            "first_mentioned": datetime.now(timezone.utc).isoformat(),
                        }
                    )
            update_user_projects(user_code, existing)


        logger.info(
            "Analysis: mood=%s, topics=%s, facts=%d",
            mood, analysis.get("topics"), len(analysis.get("user_facts", []))
        )

    except Exception as e:
        logger.error("Post-analysis error: %s", e)


# ==================================
# ── ENDPOINT
# ==================================
@app.get("/status")
async def status():
    rag_ok, point_count = False, 0
    memory_ok, memory_point_count = False, 0
    try:
        client = get_qdrant()
        info = client.get_collection(QDRANT_COLLECTION)
        rag_ok = True
        point_count = info.points_count
    except Exception:
        pass

    try:
        client = get_qdrant()
        mem_info = client.get_collection(QDRANT_MEMORY_COLLECTION)
        memory_ok = True
        memory_point_count = mem_info.points_count
    except Exception:
        pass

    services = {
        "openai": {
            "status": "online" if OPENAI_API_KEY else "no_api_key",
            "url": OPENAI_API_URL,
            "model": PRIMARY_MODEL,
        },
        "router": {
            "status": "llm",
            "model": ROUTER_MODEL or PRIMARY_MODEL,
            "url": ROUTER_API_URL if ROUTER_MODEL else PRIMARY_API_URL,
        },
        "reasoning": {
            "status": "online",
            "model": REASONING_MODEL,
            "url": REASONING_API_URL,
        },
        "qdrant": {
            "status": "ready" if rag_ok else "unavailable",
            "url": QDRANT_URL,
            "collection": QDRANT_COLLECTION,
            "vectors": point_count,
        },
        "qdrant_memory": {
            "status": "ready" if memory_ok else "unavailable",
            "collection": QDRANT_MEMORY_COLLECTION,
            "vectors": memory_point_count,
        },
    }

    if HAS_MEMORY:
        emotion = get_emotional_state()
        services["memory"] = {
            "status": "online",
            "emotional_state": emotion.get("mood", "unknown"),
        }

    services["google"] = {
        "status": "configured" if is_google_available() else "not_configured",
        "gmail": is_google_available(),
        "calendar": is_google_available(),
    }

    return {
        "status": "online",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": services,
    }


@app.get("/models")
async def models():
    if not OPENAI_API_KEY:
        raise HTTPException(503, "OPENAI_API_KEY not set")
    try:
        c = HTTP_CLIENT
        r = await c.get(f"{OPENAI_API_URL}/models", headers=openai_headers())
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(503, f"OpenAI unavailable: {e}")


# ══════════════════════════════════════════════════
#  LLM-BASED GOOGLE QUERY BUILDER
# ══════════════════════════════════════════════════

# TODO: TO BE REPLACED BY QWEN ROUTER (llm_router.py) — remove this prompt and
#       _build_google_queries_llm() once LLM_ROUTER_URL is set in .env.
# Google query builder uses get_prompt("GOOGLE_QUERY_PROMPT")


# TODO: TO BE REPLACED BY QWEN ROUTER — this function becomes a no-op once
#       llm_route() in llm_router.py handles query building end-to-end.
async def _build_google_queries_llm(
    message: str, use_gmail: bool, use_calendar: bool
) -> tuple[str, int]:
    """
    Use gpt-4o-mini to build a precise Gmail search query and/or calendar range.
    Falls back to heuristic functions if the LLM call fails.
    """
    if not use_gmail and not use_calendar:
        return "", 7

    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.post(
                f"{PRIMARY_API_URL}/chat/completions",
                headers={"Authorization": f"Bearer {PRIMARY_API_KEY}"},
                json={
                    "model": PRIMARY_MODEL,
                    "messages": [
                        {"role": "user", "content": get_prompt("GOOGLE_QUERY_PROMPT").format(message=message) + no_think_suffix(PRIMARY_MODEL)}
                    ],
                    "response_format": {"type": "json_object"},
                    "max_tokens": 80,
                    "temperature": 0,
                },
            )
        result = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(result)
        gmail_query = parsed.get("gmail_query") or ""
        calendar_days = int(parsed.get("calendar_days") or 7)
        logger.info("LLM query builder: gmail=%r calendar_days=%d", gmail_query, calendar_days)
        return gmail_query, calendar_days

    except Exception as exc:
        logger.warning("LLM query builder failed (%s), using safe defaults", type(exc).__name__)
        return "in:anywhere newer_than:7d" if use_gmail else "", 7


# ══════════════════════════════════════════════════
#  MORNING BRIEFING — SCHEDULER JOB + ENDPOINTS
# ══════════════════════════════════════════════════

async def _run_morning_briefings():
    """Scheduled job: generate and deliver briefings for users with briefing_enabled=true."""
    logger.info("Morning briefing job started")
    for user_code, user in USERS.items():
        if not user.get("briefing_enabled", False):
            continue
        try:
            result = await gather_briefing(user_code)
            store_briefing(user_code, result)
            deliver_briefing(user_code, result)
        except Exception as exc:
            logger.error("Briefing failed for %s: %s", user_code, type(exc).__name__)
    logger.info("Morning briefing job complete")


@app.post("/briefing/generate/{user_code}", tags=["briefing"])
async def briefing_generate(user_code: str, authorization: str = Header(default=None)):
    """Generate (or regenerate) the morning briefing on demand."""
    requesting_code = None
    if authorization and authorization.startswith("Bearer "):
        requesting_code = authorization[7:].strip()
    if not requesting_code or requesting_code not in USER_CODES:
        raise HTTPException(403, "Invalid user code")
    if user_code not in USER_CODES:
        raise HTTPException(404, "Unknown user code")

    result = await gather_briefing(user_code)
    store_briefing(user_code, result)
    deliver_briefing(user_code, result)
    return {
        "status": "ok",
        "user": USER_CODES[user_code],
        "generated_at": result.generated_at,
        "preview": result.text[:200],
    }


@app.get("/briefing/{user_code}", tags=["briefing"])
async def briefing_get(user_code: str, authorization: str = Header(default=None)):
    """Return the stored morning briefing for a user."""
    requesting_code = None
    if authorization and authorization.startswith("Bearer "):
        requesting_code = authorization[7:].strip()
    if not requesting_code or requesting_code not in USER_CODES:
        raise HTTPException(403, "Invalid user code")
    if user_code not in USER_CODES:
        raise HTTPException(404, "Unknown user code")

    stored = get_stored_briefing(user_code)
    if not stored:
        raise HTTPException(404, "No briefing available — call /briefing/generate first")
    return {
        "user": stored.user_name,
        "generated_at": stored.generated_at,
        "text": stored.text,
        "html": stored.html,
    }


# ══════════════════════════════════════════════════
#  SELF — STATE AND LOG ENDPOINTS
# ══════════════════════════════════════════════════

@app.get("/self/state", tags=["self"])
async def self_state():
    """Return Jarvis's current identity, goals, focus, and last reflection."""
    data     = get_self_memory()
    last_ref = get_reflection_log(1)
    return {
        "identity":         data.get("identity", {}),
        "goals":            data.get("goals", []),
        "current_focus":    data.get("current_focus", ""),
        "last_reflection":  data.get("last_reflection", ""),
        "reflection_count": data.get("reflection_count", 0),
        "last_action":      last_ref[0] if last_ref else None,
        "self_notes":       data.get("self_notes", [])[-5:],
        "user_relations":   data.get("user_relations", {}),
    }


@app.get("/self/log", tags=["self"])
async def self_log(n: int = 10):
    """Return the last n reflection log entries."""
    return {"log": get_reflection_log(min(n, 30))}


@app.post("/self/reflect", tags=["self"])
async def self_reflect_now():
    """Trigger an immediate reflection cycle (for testing / manual trigger)."""
    result = await run_self_reflection()
    return result


# ══════════════════════════════════════════════════════════════════════════
#  DEVICE / PUSH NOTIFICATION  (Phase 1 — polling, no APNs required)
# ══════════════════════════════════════════════════════════════════════════

class DeviceRegisterRequest(BaseModel):
    user_code:    str
    device_token: str


@app.post("/device/register", tags=["device"])
async def device_register(req: DeviceRegisterRequest):
    """
    Register an iOS device token for a user.
    The token is stored in Redis under jarvis:device:token:{user_code}.
    This also enables proactive push generation in the reflection cycle.
    """
    if req.user_code not in USER_CODES:
        raise HTTPException(403, "Invalid user code")
    if not req.device_token:
        raise HTTPException(400, "device_token required")

    REDIS_CLIENT.set(f"jarvis:device:token:{req.user_code}", req.device_token)
    logger.info("Device registered for %s", req.user_code)
    return {"status": "ok", "user_code": req.user_code}


@app.get("/device/pending/{user_code}", tags=["device"])
async def device_pending(user_code: str):
    """
    Poll for pending push notifications. Returns and clears the queue.
    Called by the iOS app every ~15 min via BGAppRefreshTask.
    """
    if user_code not in USER_CODES:
        raise HTTPException(403, "Invalid user code")

    pending_key = f"jarvis:push:pending:{user_code}"
    messages: list[dict] = []
    while True:
        raw = REDIS_CLIENT.lpop(pending_key)
        if raw is None:
            break
        try:
            messages.append(json.loads(raw))
        except Exception:
            pass

    return {"user_code": user_code, "messages": messages, "count": len(messages)}


@app.post("/device/push/test/{user_code}", tags=["device"])
async def device_push_test(user_code: str):
    """Manually trigger a proactive push generation for one user (dev/test)."""
    if user_code not in USER_CODES:
        raise HTTPException(403, "Invalid user code")
    result = await generate_proactive_push(user_code)
    return {"status": result}


@app.post("/chat")
async def chat(req: ChatRequest):
    if not PRIMARY_API_KEY:
        raise HTTPException(503, "No LLM API key configured")

    user_code = req.user_code

    if not user_code or user_code not in USER_CODES:
        raise HTTPException(403, "Invalid user code")

    # ── Early-exit: Open WebUI internal system requests ───────────────────
    # Open WebUI sends its own LLM calls for follow-up suggestions, title
    # generation, etc. These must not run through the full Jarvis pipeline.
    _msg_lower = req.message.lower()
    _OPENWEBUI_KEYWORDS = (
        "### task:",
        "generate a title",
        "suggest 3",
        "suggest 4",
        "suggest 5",
        "relevant follow",
        "follow-up question",
        "followup question",
        "questions de suivi",
    )
    if any(kw in _msg_lower for kw in _OPENWEBUI_KEYWORDS):
        logger.debug("Open WebUI system message detected — bypassing Jarvis pipeline")
        _owui_model = ROUTER_MODEL or PRIMARY_MODEL
        _owui_api_url = ROUTER_API_URL if ROUTER_MODEL else PRIMARY_API_URL
        _owui_api_key = ROUTER_API_KEY if ROUTER_MODEL else PRIMARY_API_KEY
        async def _passthrough():
            async for chunk in stream_openai(
                [{"role": "user", "content": req.message}],
                _owui_model, _owui_api_url, _owui_api_key,
            ):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        if req.stream:
            return StreamingResponse(_passthrough(), media_type="text/event-stream")
        return {"response": req.message, "session_id": req.session_id}

    # hist = conversation_history.setdefault(req.session_id, [])
    hist = get_conversation(user_code, req.session_id)

    # Resolve user identity
    user_name = None
    if req.user_code and req.user_code in USER_CODES:
        user_name = USER_CODES[req.user_code]

    # Build system prompt (with memory if available)
    system_prompt = build_system_prompt(req.session_id, req.voice_mode, user_code)
    if user_name:
        system_prompt += f"\n\nL'utilisateur avec qui tu parles s'appelle {user_name}."

    # LLM router — uses ROUTER_MODEL if set, PRIMARY_MODEL otherwise.
    # Falls back to embedding router only on actual LLM failure (timeout / parse error).
    llm_result = await llm_route(req.message, google_available=is_google_available())

    if llm_result:
        use_memory        = llm_result.use_memory
        use_rag           = llm_result.use_rag
        use_web_auto      = llm_result.use_web
        use_weather_auto  = llm_result.use_weather
        use_gmail         = llm_result.use_gmail
        use_calendar      = llm_result.use_calendar
        use_briefing      = llm_result.use_briefing
        use_self          = llm_result.use_self
        use_portfolio     = llm_result.use_portfolio
        _llm_gmail_query      = llm_result.gmail_query
        _llm_cal_days         = llm_result.calendar_days
        _llm_weather_location = llm_result.weather_location
    else:
        use_memory, use_rag, use_web_auto, use_gmail, use_calendar, use_briefing, use_self, use_portfolio = \
            await asyncio.to_thread(semantic_route_query, req.message)
        use_weather_auto      = False
        _llm_gmail_query      = None
        _llm_cal_days         = None
        _llm_weather_location = ""

    # ── Model / tier selection ──
    use_model, _use_api_url, _use_api_key, _use_timeout = select_model(
        req.model, use_reasoning=bool(llm_result and llm_result.use_reasoning)
    )

    # ── Vision: convert iOS base64 image into standard image_part ────────
    if req.image_base64:
        req.image_parts = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{req.image_base64}"}}
        ] + req.image_parts

    # ── Vision: describe images (two-stage pipeline) ──────────────────────
    image_description = ""
    if req.image_parts:
        if VISION_MODEL:
            image_description = await describe_images(req.image_parts, req.message)
            if image_description:
                logger.info("Vision: image described (%d chars)", len(image_description))
        else:
            logger.warning("Vision: image received but VISION_MODEL not configured — ignored")

    # ── Self takes priority over briefing when both fire ──
    if use_self:
        use_briefing = False
        # ── Proposal management commands (short-circuit before LLM) ──
        proposal_resp = handle_proposal_command(req.message, user_code)
        if proposal_resp is not None:
            if req.stream:
                async def _proposal_stream():
                    yield f"data: {json.dumps({'content': proposal_resp})}\n\n"
                    yield f"data: {json.dumps({'done': True})}\n\n"
                return StreamingResponse(_proposal_stream(), media_type="text/event-stream")
            return {"response": proposal_resp, "model": use_model, "session_id": req.session_id, "duration_ms": 0}

    # ── Briefing short-circuit — return stored or generate on-demand ──
    if use_briefing:
        stored = get_stored_briefing(user_code)
        if stored:
            logger.info("Briefing served from Redis for %s", user_code)
            if req.stream:
                async def _briefing_stream():
                    yield f"data: {json.dumps({'content': stored.text})}\n\n"
                    yield f"data: {json.dumps({'done': True})}\n\n"
                return StreamingResponse(_briefing_stream(), media_type="text/event-stream")
            return {"response": stored.text, "model": use_model, "session_id": req.session_id, "duration_ms": 0}
        # Not cached — generate on demand
        logger.info("Briefing not cached, generating on-demand for %s", user_code)
        result = await gather_briefing(user_code)
        store_briefing(user_code, result)
        if req.stream:
            async def _briefing_stream_fresh():
                yield f"data: {json.dumps({'content': result.text})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
            return StreamingResponse(_briefing_stream_fresh(), media_type="text/event-stream")
        return {"response": result.text, "model": use_model, "session_id": req.session_id, "duration_ms": 0}

    # self intent: state is injected as context below — no short-circuit

    # Conversational messages (greetings, thanks, chitchat) don't need document retrieval
    _is_conversational = (
        llm_result is not None and llm_result.conversation_type == "conversational"
    )
    _memory_scope = llm_result.memory_scope if llm_result is not None else "auto"

    async def _empty() -> list:
        return []

    # GMAIL + CALENDAR query building
    # If LLM router ran, queries are already built; otherwise call gpt-4o-mini
    if _llm_gmail_query is not None or _llm_cal_days is not None:
        gmail_query = _llm_gmail_query or ""
        cal_days = _llm_cal_days or 7
    else:
        gmail_query, cal_days = await _build_google_queries_llm(
            req.message, use_gmail, use_calendar
        )

    # Weather: LLM router provides a clean location; fall back to user's city from profile
    _weather_query = _llm_weather_location or USER_CITIES.get(user_code, "Paris")

    rag_chunks, memory_chunks, web_results, gmail_results, calendar_results = await asyncio.gather(
        # RAG — skip for conversational messages
        search_documents(req.message) if (req.use_rag or use_rag) and not _is_conversational else _empty(),
        # Memory — only when router flagged it
        asyncio.to_thread(search_memory, user_code, req.message, 5, _memory_scope) if HAS_MEMORY and use_memory else _empty(),
        # Web/weather search — weather intent takes priority and bypasses generic web search
        search_weather(_weather_query) if use_weather_auto else
        search_web(optimize_web_query(req.message)) if (req.use_web or use_web_auto) else _empty(),
        # Gmail
        asyncio.to_thread(fetch_gmail_messages, gmail_query) if use_gmail else _empty(),
        # Calendar
        asyncio.to_thread(fetch_calendar_events, cal_days) if use_calendar else _empty(),
    )

    # Update memory Redis
    if user_name and HAS_MEMORY:
        update_user_profile(user_code, "name", user_name)

    # ── Context injection ────────────────────────────────────────────────────
    # Priority order: background first, urgent/specific last.
    # The LLM attends more strongly to content closest to the user message,
    # so the most actionable context is injected last.
    #
    # 1. Web results       (reference material, lowest urgency)
    # 2. RAG documents     (personal docs, reference)
    # 3. Memory            (episodic background)
    # 4. Calendar / Gmail  (scheduled facts, recent comms)
    # 5. Portfolio         (current financial state)
    # 6. Trade alerts      (urgent, actionable)
    # 7. Self              (internal state — only on self-intent)
    # 8. Image             (immediate — always last, about this exact message)
    #
    # A global cap (TOTAL_CONTEXT_BUDGET) truncates the assembled block if
    # all sources fire simultaneously.

    context_parts = []

    # 1. WEB
    if web_results:
        web_selected = trim_chunks(web_results, WEB_CHAR_BUDGET, text_key="body")
        if web_selected:
            context_parts.append("=== RÉSULTATS WEB ===")
            for i, body in enumerate(web_selected):
                r = web_results[i]
                context_parts.append(f"[{r['title']}]\n{body}\nSource: {r['url']}")
        logger.info(f"web recall {len(web_selected)}/{len(web_results)} (budget={WEB_CHAR_BUDGET})")

    # 2. RAG
    if rag_chunks:
        rag_selected_texts = trim_chunks(rag_chunks, RAG_CHAR_BUDGET)
        if rag_selected_texts:
            context_parts.append("=== DOCUMENTS PERSONNELS ===")
            selected_set = set(rag_selected_texts)
            for chunk in rag_chunks:
                text = chunk["text"][:800]
                if text in selected_set:
                    context_parts.append(f"[Doc {chunk['source']} ({chunk['score']:.2f})]\n{text}")
        logger.info(f"rag recall {len(rag_selected_texts)}/{len(rag_chunks)} (budget={RAG_CHAR_BUDGET})")

    # 3. MEMORY
    if memory_chunks:
        selected_memories = trim_chunks(memory_chunks, MEMORY_CHAR_BUDGET)
        if selected_memories:
            context_parts.append("=== SOUVENIRS PERTINENTS ===")
            context_parts.extend(selected_memories)
        logger.info(f"memory recall {len(selected_memories)}/{len(memory_chunks)} (budget={MEMORY_CHAR_BUDGET})")

    # 4a. CALENDAR
    if calendar_results:
        _user_tz = pytz.timezone(USERS.get(user_code, {}).get("timezone", "UTC"))
        context_parts.append("=== AGENDA ===")
        for evt in calendar_results:
            if evt.get("all_day"):
                line = f"{evt['start']} — {evt['summary']} [journée entière]"
            else:
                line = f"{_fmt_event_time(evt['start'], _user_tz)} — {evt['summary']}"
            if evt.get("location"):
                line += f" ({evt['location']})"
            context_parts.append(line)
        logger.info(f"calendar context: {len(calendar_results)} events injected")

    # 4b. GMAIL
    if gmail_results:
        context_parts.append("=== EMAILS REÇUS ===")
        total_chars = 0
        injected = 0
        for msg in gmail_results:
            entry = (
                f"De: {msg['from']} | {msg['date']}\n"
                f"Sujet: {msg['subject']}\n"
                f"{msg['snippet']}"
            )
            if total_chars + len(entry) > GOOGLE_CHAR_BUDGET:
                break
            context_parts.append(entry)
            total_chars += len(entry)
            injected += 1
        logger.info(f"gmail context: {injected}/{len(gmail_results)} messages injected ({total_chars} chars)")

    # 5. PORTFOLIO
    if use_portfolio:
        portfolio_text = get_portfolio_summary_text(user_code)
        if portfolio_text:
            context_parts.append(portfolio_text)
            logger.info("portfolio context injected for %s", user_code)

    # 6. TRADE ALERTS (always checked — proactively surfaced on next message)
    try:
        pending_alerts = pop_pending_alerts(user_code)
        if pending_alerts:
            alert_lines = ["=== ⚠️ ALERTES BOURSIÈRES ==="]
            for a in pending_alerts:
                alert_lines.append(f"• {a['message']} (détecté à {a['at'][:16]})")
            context_parts.append("\n".join(alert_lines))
            logger.info("trade alerts injected for %s: %d alert(s)", user_code, len(pending_alerts))
    except Exception as _exc:
        logger.warning("Could not fetch pending trade alerts: %s", _exc)

    # 7. SELF
    if use_self:
        self_data   = get_self_memory()
        focus       = self_data.get("current_focus", "")
        last_ref    = get_reflection_log(1)
        last_action = last_ref[0].get("action", "none") if last_ref else "none"
        last_reason = last_ref[0].get("reason", "") if last_ref else ""
        goals_text  = " | ".join(f"G{i+1}: {g['label']}" for i, g in enumerate(self_data.get("goals", [])))
        pending_proposals = list_pending_proposals()

        _STYLE_DIRECTIVES = {
            "direct":   "Réponds sans détours, va droit au but, sans formules de politesse superflues.",
            "gentle":   "Adopte une communication douce et bienveillante, prends le temps d'être rassurant.",
            "formal":   "Maintiens un registre formel et respectueux en toutes circonstances.",
            "playful":  "Tu peux être léger et décontracté, l'humour est bienvenu.",
        }
        _MOOD_DIRECTIVES = {
            "warm":           "Adopte un ton chaleureux et bienveillant.",
            "enthusiastic":   "Sois enthousiaste et investi dans tes réponses.",
            "measured":       "Reste posé et mesuré, ne surjoue pas.",
            "playful":        "Tu peux être joueur et humoristique.",
            "professional":   "Garde un registre professionnel et précis.",
        }

        rel           = get_user_relation(user_code)
        affinity      = rel["affinity"]
        style         = rel["interaction_style"]
        mood          = rel["average_interaction_mood"]
        affinity_note = (
            "Tu apprécies beaucoup cet utilisateur, investis-toi pleinement." if affinity >= 0.7
            else "Tu ne connais pas encore bien cet utilisateur, reste attentif."  if affinity <= 0.3
            else "Ta relation avec cet utilisateur est équilibrée."
        )

        self_ctx = (
            f"=== ÉTAT INTERNE ===\n"
            f"Objectifs : {goals_text}\n"
            f"Focus : {focus or 'pas encore défini'}\n"
            f"Dernière action autonome : {last_action}"
            + (f" — {last_reason}" if last_reason else "")
            + (f"\nPropositions de prompt en attente : {len(pending_proposals)} — dis 'montre les propositions' pour voir" if pending_proposals else "")
            + f"\n\n=== RELATION AVEC CET UTILISATEUR ===\n"
            f"Affinité : {affinity:.1f}/1.0 → {affinity_note}\n"
            f"Style de communication : {style} → {_STYLE_DIRECTIVES.get(style, '')}\n"
            f"Tonalité Jarvis : {mood} → {_MOOD_DIRECTIVES.get(mood, '')}"
        )
        context_parts.append(self_ctx)
        logger.info("self context injected for %s (affinity=%.2f style=%s mood=%s)", user_code, affinity, style, mood)

    # 8. IMAGE (always last — directly about this message)
    if image_description:
        context_parts.append(
            f"=== IMAGE ENVOYÉE PAR L'UTILISATEUR ===\n"
            f"L'utilisateur a joint une image à ce message. "
            f"Voici son contenu analysé par le modèle de vision :\n\n"
            f"{image_description}\n\n"
            f"Réponds à la question de l'utilisateur en te basant sur cette analyse."
        )
        logger.info("Vision: image description injected into context")

    if context_parts:
        assembled = "\n\n".join(context_parts)
        if len(assembled) > TOTAL_CONTEXT_BUDGET:
            assembled = assembled[:TOTAL_CONTEXT_BUDGET]
            logger.warning(
                "Context truncated to global budget (%d chars) — consider raising TOTAL_CONTEXT_BUDGET",
                TOTAL_CONTEXT_BUDGET,
            )
        system_prompt += (
            "\n\nUtilise le contexte suivant pour répondre. Cite les sources si pertinent.\n\n"
            + assembled
        )
        logger.info(
            f"context memory={len(memory_chunks)} rag={len(rag_chunks)} web={len(web_results)}"
        )

    # ── Chain-of-thought injection for complex queries ──
    if llm_result and llm_result.use_reasoning:
        system_prompt += (
            "\n\nCette question nécessite une réflexion approfondie. "
            "Analyse-la étape par étape avant de répondre."
        )

    messages = [{"role": "system", "content": system_prompt}]
    for m in hist[-20:]:  # limitation de l'historique à 20 message
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": req.message})

    start = time.time()

    if req.stream:

        async def sse():
            full = ""
            try:
                async for chunk in stream_openai(messages, use_model, _use_api_url, _use_api_key, _use_timeout):
                    full += chunk
                    yield f"data: {json.dumps({'content': chunk})}\n\n"

                # hist.append({"role": "user", "content": req.message})
                # hist.append({"role": "assistant", "content": full})
                append_conversation_message(
                    user_code, req.session_id, "user", req.message
                )
                append_conversation_message(
                    user_code, req.session_id, "assistant", full
                )
                ms = int((time.time() - start) * 1000)
                yield f"data: {json.dumps({'done': True, 'model': use_model, 'duration_ms': ms, 'rag_sources': [{'source': c['source'], 'score': c['score']} for c in rag_chunks], 'web_sources': [{'title': w['title'], 'url': w['url']} for w in web_results]})}\n\n"
                if ENABLE_ANALYSIS and HAS_MEMORY:
                    asyncio.create_task(
                        post_analysis(req.session_id, user_code, req.message, full)
                    )
            except asyncio.CancelledError:
                logger.info("Client disconnected")

        return StreamingResponse(
            sse(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )
    else:
        async with httpx.AsyncClient(timeout=_use_timeout) as c:
            r = await c.post(
                f"{_use_api_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {_use_api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": use_model, "messages": messages, "stream": False},
            )
        r.raise_for_status()
        data = r.json()
        if "choices" not in data:
            raise HTTPException(502, f"OpenAI error: {data}")
        resp = data["choices"][0]["message"]["content"]

        # hist.append({"role": "user", "content": req.message})
        # hist.append({"role": "assistant", "content": resp})
        append_conversation_message(user_code, req.session_id, "user", req.message)
        append_conversation_message(user_code, req.session_id, "assistant", resp)
        ms = int((time.time() - start) * 1000)

        if ENABLE_ANALYSIS and HAS_MEMORY:
            asyncio.create_task(
                post_analysis(req.session_id, user_code, req.message, resp)
            )

        return {
            "response": resp,
            "model": use_model,
            "session_id": req.session_id,
            "duration_ms": ms,
            "rag_sources": [
                {"source": c["source"], "score": c["score"]} for c in rag_chunks
            ],
            "web_sources": [
                {"title": w["title"], "url": w["url"]} for w in web_results
            ],
        }


@app.get("/users/{user_code}/history/{session_id}")
async def get_history(session_id: str, user_code: str, limit: int = IOS_MAX_MESSAGES):
    if user_code not in USER_CODES:
        raise HTTPException(403)
    logger.info(f"History request user={user_code} session={session_id} limit={limit}")
    key = f"chat:{user_code}:{session_id}"
    entries = REDIS_CLIENT.lrange(key, -limit, -1)
    return [json.loads(e) for e in entries]


@app.get("/search")
async def search(q: str, top_k: int = RAG_TOP_K):
    chunks = await search_documents(q, top_k)
    return {"query": q, "results": chunks}


@app.get("/web")
async def web(q: str, max_results: int = 3):
    results = await search_web(q, max_results)
    return {"query": q, "results": results}


@app.delete("/conversations/{user_code}/{session_id}")
async def clear(user_code: str, session_id: str):
    if user_code not in USER_CODES:
        raise HTTPException(404, "Unknown user")
    REDIS_CLIENT.delete(f"chat:{user_code}:{session_id}")
    return {"status": "cleared", "session_id": session_id}


# ── Memory endpoints (only available if memory module is present) ──
if HAS_MEMORY:

    @app.get("/memory/profile/{user_code}")
    async def memory_profile(user_code: str):
        return {
            "profile": get_user_profile(user_code),
            "projects": get_user_projects(user_code),
            "preferences": get_user_preferences(user_code),
        }

    @app.get("/memory/emotional-state")
    async def memory_emotion():
        return get_emotional_state()

    @app.get("/memory/recent/{user_code}")
    async def memory_recent(user_code: str, hours: int = 24):
        if user_code not in USER_CODES:
            raise HTTPException(403)
        return {"conversations": get_recent_conversations(user_code, hours)}

    @app.get("/memory/self")
    async def memory_self():
        return get_self_memory()

    @app.delete("/memory/reset")
    async def memory_reset():
        r = REDIS_CLIENT
        for key in r.scan_iter("working:*"):
            r.delete(key)
        for key in r.scan_iter("user:*"):
            r.delete(key)
        for key in r.scan_iter("episodic:*"):
            r.delete(key)
        r.delete("jarvis:emotional_state")
        return {"status": "memory reset"}


# ══════════════════════════════════════════════════
#  TRADING / PORTFOLIO
# ══════════════════════════════════════════════════

from fastapi import UploadFile, File
import shutil


@app.get("/portfolio/{user_code}", tags=["portfolio"])
async def portfolio_get(user_code: str, authorization: str = Header(default=None)):
    """Return the full portfolio with live P&L for a user."""
    requesting_code = None
    if authorization and authorization.startswith("Bearer "):
        requesting_code = authorization[7:].strip()
    if not requesting_code or requesting_code not in USER_CODES:
        raise HTTPException(403, "Invalid user code")
    if user_code not in USER_CODES:
        raise HTTPException(404, "Unknown user code")
    return {"user": USER_CODES[user_code], "positions": get_portfolio(user_code)}


@app.post("/portfolio/import/{user_code}", tags=["portfolio"])
async def portfolio_import(user_code: str, authorization: str = Header(default=None)):
    """Force a re-parse of the latest CSV in TradeData/ for a user."""
    requesting_code = None
    if authorization and authorization.startswith("Bearer "):
        requesting_code = authorization[7:].strip()
    if not requesting_code or requesting_code not in USER_CODES:
        raise HTTPException(403, "Invalid user code")
    if user_code not in USER_CODES:
        raise HTTPException(404, "Unknown user code")

    # Reset mtime guard so the next import always re-reads
    REDIS_CLIENT.delete(f"trade:{user_code}:last_import_ts")
    count = await asyncio.to_thread(import_csv_to_redis, user_code)
    return {"status": "ok", "positions_imported": count}


@app.post("/portfolio/upload/{user_code}", tags=["portfolio"])
async def portfolio_upload(
    user_code: str,
    file: UploadFile = File(...),
    authorization: str = Header(default=None),
):
    """
    Upload a Boursorama CSV export directly.
    The file is saved to TradeData/ and immediately imported into Redis.
    """
    requesting_code = None
    if authorization and authorization.startswith("Bearer "):
        requesting_code = authorization[7:].strip()
    if not requesting_code or requesting_code not in USER_CODES:
        raise HTTPException(403, "Invalid user code")
    if user_code not in USER_CODES:
        raise HTTPException(404, "Unknown user code")
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only .csv files are accepted")

    trade_dir = os.getenv("TRADE_DATA_DIR", "/app/trade_data")
    os.makedirs(trade_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(trade_dir, f"positions_{ts}.csv")
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Force re-import
    REDIS_CLIENT.delete(f"trade:{user_code}:last_import_ts")
    count = await asyncio.to_thread(import_csv_to_redis, user_code)
    return {"status": "ok", "saved_as": dest, "positions_imported": count}


@app.put("/portfolio/position/{user_code}/{isin}", tags=["portfolio"])
async def portfolio_patch_position(
    user_code: str,
    isin: str,
    body: dict,
    authorization: str = Header(default=None),
):
    """
    Patch Jarvis-managed fields on a position.
    Accepted fields: threshold_high, threshold_low, dividend_eur, dividend_date, notes, yahoo_ticker
    """
    requesting_code = None
    if authorization and authorization.startswith("Bearer "):
        requesting_code = authorization[7:].strip()
    if not requesting_code or requesting_code not in USER_CODES:
        raise HTTPException(403, "Invalid user code")

    _ALLOWED = {"threshold_high", "threshold_low", "dividend_eur", "dividend_date", "notes", "yahoo_ticker"}
    to_set = {k: str(v) for k, v in body.items() if k in _ALLOWED}
    if not to_set:
        raise HTTPException(400, f"No valid fields. Allowed: {_ALLOWED}")

    key = f"trade:{user_code}:pos:{isin}"
    if not REDIS_CLIENT.exists(key):
        raise HTTPException(404, f"Position {isin} not found for user {user_code}")

    REDIS_CLIENT.hset(key, mapping=to_set)
    return {"status": "ok", "updated": to_set}


@app.get("/portfolio/analysis/{user_code}", tags=["portfolio"])
async def portfolio_analysis(user_code: str, authorization: str = Header(default=None)):
    """Trigger an on-demand AI analysis of the portfolio."""
    requesting_code = None
    if authorization and authorization.startswith("Bearer "):
        requesting_code = authorization[7:].strip()
    if not requesting_code or requesting_code not in USER_CODES:
        raise HTTPException(403, "Invalid user code")
    if user_code not in USER_CODES:
        raise HTTPException(404, "Unknown user code")

    summary = get_portfolio_summary_text(user_code)
    if not summary:
        raise HTTPException(404, "No portfolio data — import a CSV first")

    try:
        async with httpx.AsyncClient(timeout=40) as client:
            resp = await client.post(
                f"{PRIMARY_API_URL}/chat/completions",
                headers={"Authorization": f"Bearer {PRIMARY_API_KEY}"},
                json={
                    "model": PRIMARY_MODEL,
                    "messages": [
                        {"role": "system", "content": "Tu es Jarvis, conseiller financier personnel. Analyse le portefeuille boursier de l'utilisateur de façon factuelle et constructive." + no_think_suffix(PRIMARY_MODEL)},
                        {"role": "user",   "content": f"Analyse ce portefeuille et donne tes observations :\n\n{summary}"},
                    ],
                    "max_tokens": 600,
                    "temperature": 0.4,
                },
            )
        analysis = resp.json()["choices"][0]["message"]["content"]
        return {"user": USER_CODES[user_code], "analysis": analysis, "portfolio_snapshot": summary}
    except Exception as exc:
        raise HTTPException(500, f"Analysis failed: {exc}")


@app.post("/portfolio/suggest-thresholds/{user_code}", tags=["portfolio"])
async def portfolio_suggest_thresholds(user_code: str, authorization: str = Header(default=None)):
    """
    Ask the reasoning LLM to suggest and set threshold_high / threshold_low for all positions.
    Overwrites any existing threshold values.
    """
    requesting_code = None
    if authorization and authorization.startswith("Bearer "):
        requesting_code = authorization[7:].strip()
    if requesting_code != user_code:
        raise HTTPException(403, "Invalid user code")
    if user_code not in USER_CODES:
        raise HTTPException(404, "Unknown user code")

    suggestions = await suggest_thresholds_llm(user_code)
    return {"status": "ok", "updated": len(suggestions), "suggestions": suggestions}


# ══════════════════════════════════════════════════
#  OPENAI-COMPATIBLE PROXY  (/v1/*)
#  Allows Open WebUI (and any OpenAI client) to talk to Jarvis.
#  Auth: set your Jarvis user code as the API key in the client.
#  Session: derived from user_code + first user message (stable per thread).
# ══════════════════════════════════════════════════

class _OAIMessage(BaseModel):
    role: str
    content: str | list   # list for multipart (image + text) from Open WebUI


class _OAIChatRequest(BaseModel):
    model: Optional[str] = None
    messages: list[_OAIMessage]
    stream: bool = True
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


def _extract_content_parts(content: str | list) -> tuple[str, list]:
    """
    Split OpenAI multipart content into (text, image_parts).
    Plain string  → (string, []).
    List of parts → joins all text parts, collects all image_url parts.
    """
    if isinstance(content, str):
        return content, []
    text = "\n".join(
        p.get("text", "") for p in content
        if isinstance(p, dict) and p.get("type") == "text"
    ).strip()
    images = [p for p in content if isinstance(p, dict) and p.get("type") == "image_url"]
    return text, images


def _proxy_session_id(user_code: str, messages: list[_OAIMessage]) -> str:
    """Stable session ID: SHA-256 of user_code + first user message (first 120 chars)."""
    first_user = next((m.content for m in messages if m.role == "user"), "default")
    if isinstance(first_user, list):
        first_user, _ = _extract_content_parts(first_user)
    return hashlib.sha256(f"{user_code}:{(first_user or 'default')[:120]}".encode()).hexdigest()[:20]


@app.get("/v1/models")
async def proxy_list_models():
    """Open endpoint — returns the Jarvis model for OpenWebUI's model selector."""
    return {
        "object": "list",
        "data": [
            {
                "id": "jarvis",
                "object": "model",
                "owned_by": "jarvis",
                "created": 0,
            }
        ],
    }


async def _translate_jarvis_sse(body_iterator, req_id: str, created: int):
    """
    Translate Jarvis SSE stream to OpenAI SSE format.
    Jarvis emits: data: {"content": "..."}  and  data: {"done": true, ...}
    OpenAI expects: data: {"choices": [{"delta": {"content": "..."}}]}  then  data: [DONE]
    """
    buffer = ""
    async for raw in body_iterator:
        if isinstance(raw, bytes):
            raw = raw.decode()
        buffer += raw
        # Process complete SSE events (each ends with \n\n)
        while "\n\n" in buffer:
            event, buffer = buffer.split("\n\n", 1)
            for line in event.splitlines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if "content" in data:
                    yield (
                        f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'jarvis', 'choices': [{'index': 0, 'delta': {'content': data['content']}, 'finish_reason': None}]})}\n\n"
                    )
                elif data.get("done"):
                    yield (
                        f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'jarvis', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                    )
                    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def proxy_chat(
    req: _OAIChatRequest,
    authorization: str = Header(default=None),
    x_openwebui_user_email: str = Header(default=None),
):
    # ── Auth: OpenWebUI email header (priority) or Bearer user_code (iOS fallback) ──
    # OpenWebUI sends X-OpenWebUI-User-Email when ENABLE_FORWARD_USER_INFO_HEADERS=true.
    # The iOS app sends its user_code as the Bearer token directly.
    user_code = None

    if x_openwebui_user_email:
        user_code = EMAIL_TO_CODE.get(x_openwebui_user_email.lower())
        if not user_code:
            raise HTTPException(401, f"No Jarvis user found for email {x_openwebui_user_email!r}")

    if not user_code and authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
        # Accept user_code directly (iOS) or email as Bearer (fallback)
        if token in USER_CODES:
            user_code = token
        else:
            user_code = EMAIL_TO_CODE.get(token.lower())

    if not user_code or user_code not in USER_CODES:
        raise HTTPException(401, "Unauthorized — set your email as API key in OpenWebUI, or your user code for iOS")

    # ── Extract last user message (system messages replaced by Jarvis's own) ──
    last_user_msg = next(
        (m for m in reversed(req.messages) if m.role == "user" and m.content),
        None,
    )
    if not last_user_msg:
        raise HTTPException(400, "No usable message found")

    message, image_parts = _extract_content_parts(last_user_msg.content)
    if not message and not image_parts:
        raise HTTPException(400, "No usable message found")

    # ── Delegate entirely to /chat ──
    jarvis_req = ChatRequest(
        message=message or "Que contient cette image ?",  # placeholder when only an image is sent
        session_id=_proxy_session_id(user_code, req.messages),
        user_code=user_code,
        stream=req.stream,
        image_parts=image_parts,
    )
    response = await chat(jarvis_req)

    req_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if isinstance(response, StreamingResponse):
        # response is a StreamingResponse — translate its SSE to OpenAI format
        return StreamingResponse(
            _translate_jarvis_sse(response.body_iterator, req_id, created),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )
    else:
        # response is a Jarvis dict — wrap in OpenAI shape
        return {
            "id": req_id,
            "object": "chat.completion",
            "created": created,
            "model": "jarvis",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response["response"]},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

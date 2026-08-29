"""
PROJECT JARVIS v10 — Bootstrap
==============================
Endpoints découpés en modules dans routes/ :
  routes/chat.py          POST /chat, GET /users/{code}/history/{session}
  routes/briefing.py      POST/GET /briefing/*
  routes/self_routes.py   GET/POST /self/*
  routes/device.py        POST/GET /device/*
  routes/memory_routes.py GET/DELETE /memory/*
  routes/portfolio.py     GET/POST/PUT /portfolio/*
  routes/proxy.py         GET/POST /v1/*
"""

import asyncio
import os
from contextlib import asynccontextmanager

# Prevent HuggingFace tokenizers from spawning loky worker processes.
# Without this, a hard kill leaves orphaned semaphores that trigger
# resource_tracker warnings on the next startup.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from datetime import datetime, timedelta, timezone

import deps
import httpx
import pytz
import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from config import (
    BRIEFING_ENABLED,
    BRIEFING_TIME,
    BRIEFING_TIMEZONE,
    CONV_ANALYSIS_INTERVAL_MINUTES,
    LLM_LOCAL,
    OPENAI_API_KEY,
    OPENAI_API_URL,
    PRIMARY_API_URL,
    PRIMARY_MODEL,
    QDRANT_COLLECTION,
    QDRANT_MEMORY_COLLECTION,
    QDRANT_URL,
    RAG_TOP_K,
    REASONING_API_URL,
    REASONING_MODEL,
    REFLECTION_INTERVAL_HOURS,
    ROUTER_API_URL,
    ROUTER_MODEL,
    USER_ADMINS,
    USER_CODES,
    USER_TRADING,
)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client.models import Distance, HnswConfigDiff, VectorParams

if LLM_LOCAL:
    from llm.local import preload_models
    from pipeline import build_system_prompt
from agent import start_worker, stop_worker
from analyzer import analyse_recent_conversations
from deps import _STREAM_CLIENTS, HTTP_CLIENT, QDRANT_CLIENT
from llm.embed_router import preload_embed_router
from google_services import is_google_available
from helpers import get_logger, setup_logging
from llm.client import openai_headers
import emotional_state
from memory import get_embed_model
from rag import search_documents
from routes.agent_routes import router as agent_router
from routes.briefing_routes import router as briefing_router
from routes.briefing_routes import run_morning_briefings
from routes.chat import router as chat_router
from routes.device import router as device_router
from routes.memory_routes import router as memory_router
from routes.portfolio import router as portfolio_router
from routes.proxy import router as proxy_router
from routes.self_routes import router as self_router
from self import run_nightly_interaction_review, run_self_reflection
from trading import run_trade_check
from web_search import search_web

logger = get_logger("jarvis-api")


# ── Application lifespan ───────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()

    logger.info(
        "Jarvis API v9 starting — router: %s, reasoning: %s",
        ROUTER_MODEL,
        REASONING_MODEL,
    )

    # Pendant de mark_shutdown : lit la trace de l'arrêt PRÉCÉDENT et, si la coupure a été
    # notable, l'enregistre comme incident. À faire tôt, avant qu'un nouvel arrêt ne l'écrase.
    try:
        from vitals import note_boot

        note_boot()
    except Exception as exc:
        logger.debug("vitals: note_boot ignoré (%s)", exc)

    if LLM_LOCAL:
        _warmup_code = next(iter(USER_ADMINS), "")
        await asyncio.to_thread(preload_models, build_system_prompt(_warmup_code))
    logger.info(
        "RAG: %s, collection: %s, top_k: %d", QDRANT_URL, QDRANT_COLLECTION, RAG_TOP_K
    )

    deps.EMBED_MODEL = get_embed_model()
    await asyncio.to_thread(preload_embed_router)

    try:
        QDRANT_CLIENT.get_collection(QDRANT_MEMORY_COLLECTION)
    except Exception:
        logger.info(
            "Qdrant collection %r missing — creating it (fresh install)",
            QDRANT_MEMORY_COLLECTION,
        )
        QDRANT_CLIENT.create_collection(
            collection_name=QDRANT_MEMORY_COLLECTION,
            vectors_config=VectorParams(
                size=deps.EMBED_MODEL.get_sentence_embedding_dimension(),
                distance=Distance.DOT,
                hnsw_config=HnswConfigDiff(m=24, ef_construct=256, payload_m=24),
            ),
        )

    # Index de payload — posés à CHAQUE démarrage, pas seulement à la création. Ils
    # n'existaient nulle part dans le code : ceux en place avaient été créés à la main, et
    # `timestamp` l'avait été en `integer` alors que le payload y écrit un flottant
    # (time.time()). Un index integer n'indexe aucune valeur flottante : il contenait 0
    # point sur 281, et `scroll(order_by="timestamp")` rendait une liste VIDE sans lever
    # d'erreur. C'est ce que fait `_consolidate_user_memories` — la consolidation
    # mensuelle épisodique → autobiographique n'a donc jamais rien consolidé.
    # Constaté et corrigé. La création est idempotente.
    for _champ, _schema in (
        ("user_code", "keyword"),
        ("memory_type", "keyword"),
        ("status", "keyword"),
        ("timestamp", "float"),
    ):
        try:
            QDRANT_CLIENT.create_payload_index(
                collection_name=QDRANT_MEMORY_COLLECTION,
                field_name=_champ,
                field_schema=_schema,
            )
        except Exception as exc:
            # Déjà présent avec le bon schéma : Qdrant refuse, et c'est le cas nominal.
            logger.debug("Index de payload %s (%s) : %s", _champ, _schema, type(exc).__name__)

    scheduler = None
    try:
        hour, minute = (int(x) for x in BRIEFING_TIME.split(":"))
        tz = pytz.timezone(BRIEFING_TIMEZONE)
        scheduler = AsyncIOScheduler(timezone=tz)

        scheduler.add_job(
            run_self_reflection,
            trigger="interval",
            hours=REFLECTION_INTERVAL_HOURS,
            id="self_reflection",
            next_run_time=datetime.now(tz) + timedelta(minutes=5),
        )
        scheduler.add_job(
            run_nightly_interaction_review,
            trigger="cron",
            hour=23,
            minute=0,
            id="nightly_interaction_review",
        )
        scheduler.add_job(
            analyse_recent_conversations,
            trigger="interval",
            minutes=CONV_ANALYSIS_INTERVAL_MINUTES,
            id="conversation_analysis",
            next_run_time=datetime.now(tz) + timedelta(minutes=10),
        )
        logger.info(
            "Conversation analysis scheduled every %d min",
            CONV_ANALYSIS_INTERVAL_MINUTES,
        )

        if BRIEFING_ENABLED:
            scheduler.add_job(
                run_morning_briefings,
                trigger="cron",
                hour=hour,
                minute=minute,
                id="morning_briefing",
            )
            logger.info(
                "Morning briefing scheduled at %s (%s)",
                BRIEFING_TIME,
                BRIEFING_TIMEZONE,
            )

        async def _run_trade_checks():
            await run_trade_check(USER_TRADING)

        scheduler.add_job(
            _run_trade_checks,
            trigger="interval",
            hours=2,
            id="trade_check",
            next_run_time=datetime.now(tz) + timedelta(minutes=8),
        )
        logger.info("Trading surveillance scheduled every 2 h")

        async def _run_cve_scan():
            # SBOM venv + images conteneurs → grype. Lent (~15-20 s) et gourmand CPU :
            # hors boucle de requête, une fois par jour. vitals ne lit ensuite que le cache.
            from cve import scan
            await asyncio.to_thread(scan)

        scheduler.add_job(
            _run_cve_scan,
            trigger="cron",
            hour=4,
            minute=30,
            id="cve_scan",
            next_run_time=datetime.now(tz) + timedelta(minutes=4),
        )
        logger.info("CVE scan scheduled daily at 04:30")

        scheduler.start()
        logger.info("Self reflection scheduled every %d h", REFLECTION_INTERVAL_HOURS)
        logger.info("Nightly review scheduled at 23:00 (%s)", BRIEFING_TIMEZONE)
    except Exception as exc:
        logger.error("Scheduler failed to start: %s", type(exc).__name__)

    # Worker agentique — hors scheduler : il n'est pas périodique, il consomme une file.
    # No-op si AGENT_ENABLED=false.
    start_worker()

    yield

    await stop_worker()

    # Trace l'heure d'arrêt : c'est elle qui permet au démarrage suivant de mesurer la
    # durée de coupure (vitals, famille « discontinuité »). Sans cette trace, le champ
    # est simplement absent — jamais inventé.
    try:
        from vitals import mark_shutdown

        mark_shutdown()
    except Exception as exc:
        logger.debug("vitals: trace d'arrêt non posée (%s)", exc)

    if scheduler:
        scheduler.shutdown(wait=False)
    await HTTP_CLIENT.aclose()
    for _sc in _STREAM_CLIENTS.values():
        await _sc.aclose()


# ── App + routers ──────────────────────────────────────────────────────────────

app = FastAPI(title="Jarvis API", version="8.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(agent_router)
app.include_router(briefing_router)
app.include_router(self_router)
app.include_router(device_router)
app.include_router(memory_router)
app.include_router(portfolio_router)
app.include_router(proxy_router)


# ── Core utility endpoints ─────────────────────────────────────────────────────


@app.get("/status")
async def status():
    rag_ok, point_count = False, 0
    memory_ok, memory_point_count = False, 0
    try:
        info = QDRANT_CLIENT.get_collection(QDRANT_COLLECTION)
        rag_ok, point_count = True, info.points_count
    except Exception:
        pass
    try:
        mem_info = QDRANT_CLIENT.get_collection(QDRANT_MEMORY_COLLECTION)
        memory_ok, memory_point_count = True, mem_info.points_count
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

    services["memory"] = {
        "status": "online",
        "emotional_state": emotional_state.describe(),
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
        r = await HTTP_CLIENT.get(f"{OPENAI_API_URL}/models", headers=openai_headers())
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(503, f"OpenAI unavailable: {e}")


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
    from deps import REDIS_CLIENT

    if user_code not in USER_CODES:
        raise HTTPException(404, "Unknown user")
    REDIS_CLIENT.delete(
        f"chat:{user_code}:{session_id}",
        f"session:summary:{user_code}:{session_id}",
        f"jarvis:sticky_rag:{user_code}:{session_id}",
    )
    return {"status": "cleared", "session_id": session_id}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

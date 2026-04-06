"""
routes/chat.py — POST /chat and GET /users/{user_code}/history/{session_id}
============================================================================
The main Jarvis pipeline: routing → context gathering → LLM → streaming SSE.
"""

import asyncio
import json
import re
import time
from typing import Optional

from briefing import gather_briefing, get_stored_briefing, store_briefing
from config import (
    BRIEFING_TIMEZONE,
    IOS_MAX_MESSAGES,
    PRIMARY_API_KEY,
    PRIMARY_API_URL,
    PRIMARY_MODEL,
    PRIMARY_TIMEOUT,
    ROUTER_API_KEY,
    ROUTER_API_URL,
    ROUTER_MODEL,
    USER_CITIES,
    USER_CODES,
    USER_TIMEZONES,
    VISION_MODEL,
)
from deps import REDIS_CLIENT
from embed_router import embed_route
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from google_services import (
    create_calendar_event,
    extract_calendar_event_llm,
    fetch_calendar_events,
    fetch_gmail_messages,
    is_calendar_write,
    is_google_available,
)
from helpers import build_iso_dt, call_llm_async, get_logger
from llm_client import describe_images, stream_openai
from llm_router import llm_route
from memory import (
    append_conversation_message,
    get_conversation,
    search_memory,
    update_user_profile,
)
from pipeline import (
    augment_user_message,
    build_context,
    build_dynamic_prefix,
    build_system_prompt,
    post_analysis,
)
from pydantic import BaseModel
from rag import search_documents
from self import handle_proposal_command
from web_search import (
    INTERNET_ERROR,
    fetch_user_urls,
    optimize_web_query,
    search_weather,
    search_web,
)

logger = get_logger("jarvis-chat")

router = APIRouter()
_HIST_WINDOW = 8  # HISTORIQUE DES MESSAGES DANS LE PROMPT INJECTE

# Compiled once at module level — used in STEP 5 URL detection.
_URL_RE = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]{5,}')

# ── Web auto-trigger signals ────────────────────────────────────────────────────
# Factual patterns that strongly suggest external info is needed even when the
# router classified the message as "memory" (conversational phrasing).
_AUTO_WEB_RE = re.compile(
    r'\bprix\b|\btarif\b|\bcombien\b|\bcoûte?\b'
    r'|\bversion\b|\bmodèle\b'
    r'|\bsorti\b|\bdisponible\b|\blancé\b'
    r'|\bactuel\b|\bactuelle\b|\bactuellem?ent\b'
    r'|\bdernier\b|\bdernière\b|\bnouveau\b|\bnouvelle\b'
    r'|\bspecs?\b|\bcaractéristique\b'
    r'|\bqui est\b|\bc\'est qui\b'
    r'|\bcompar[e-]\b|\bregarde\b',
    re.IGNORECASE,
)

def _auto_web_needed(message: str, memory_chunks: list) -> bool:
    """True if the message looks factual but the context is thin.

    Conditions:
    - At least one factual signal in the message (price, version, current, etc.)
    - No memory hit with strong relevance (score > 0.70) — if memory already has
      a relevant answer, web would be redundant noise.
    - Message is long enough to be a real question (not a greeting).
    """
    if len(message) <= 20:
        return False
    if memory_chunks and any(m.get("score", 0) > 0.70 for m in memory_chunks):
        return False
    return bool(_AUTO_WEB_RE.search(message))

# ── Request model ──────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    user_code: Optional[str] = None
    model: Optional[str] = None
    stream: bool = True
    voice_mode: bool = False
    use_rag: bool = False  # router decides; set True to force RAG regardless
    use_web: bool = False
    image_parts: list = []  # OpenAI image_url part dicts forwarded from the proxy
    image_base64: Optional[str] = None  # base64 JPEG/PNG sent directly by the iOS app


# ── Module-level coroutine helpers (no closure dependency) ────────────────────


async def _empty() -> list:
    return []


async def _timed_thread(fn, *args, timeout: float = 15.0) -> list:
    """Run a sync function in a thread with a timeout; returns [] on timeout."""
    try:
        return await asyncio.wait_for(asyncio.to_thread(fn, *args), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("Google API call timed out: %s", fn.__name__)
        return []


# ── Pipeline helpers ───────────────────────────────────────────────────────────

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


def _openwebui_passthrough(req: ChatRequest) -> "StreamingResponse | dict | None":
    """
    Détecte les requêtes système d'Open WebUI (suggestions, titres, follow-ups)
    et les renvoie directement au modèle léger, sans passer par le pipeline Jarvis.
    Retourne None si le message n'est pas une requête Open WebUI.
    """
    if not any(kw in req.message.lower() for kw in _OPENWEBUI_KEYWORDS):
        return None

    logger.debug("Open WebUI system message detected — bypassing Jarvis pipeline")
    _owui_model = ROUTER_MODEL or PRIMARY_MODEL
    _owui_api_url = ROUTER_API_URL if ROUTER_MODEL else PRIMARY_API_URL
    _owui_api_key = ROUTER_API_KEY if ROUTER_MODEL else PRIMARY_API_KEY

    async def _passthrough():
        async for chunk in stream_openai(
            [{"role": "user", "content": req.message}],
            _owui_model,
            _owui_api_url,
            _owui_api_key,
            no_think=True,
        ):
            yield f"data: {json.dumps({'content': chunk})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    if req.stream:
        return StreamingResponse(_passthrough(), media_type="text/event-stream")
    return {"response": req.message, "session_id": req.session_id}


def _instant_reply(
    req: ChatRequest,
    user_code: str,
    text: str,
    model: str = PRIMARY_MODEL,
) -> "StreamingResponse | dict":
    """
    Enregistre l'échange dans l'historique et retourne une réponse immédiate
    sans appel LLM. Utilisé pour le keyword-dispatch, le briefing, les proposals.
    """
    append_conversation_message(user_code, req.session_id, "user", req.message)
    append_conversation_message(user_code, req.session_id, "assistant", text)

    if req.stream:
        _text = text  # capture pour la closure

        async def _stream():
            yield f"data: {json.dumps({'content': _text})}\n\n"
            yield f"data: {json.dumps({'done': True, 'model': model, 'duration_ms': 0})}\n\n"

        return StreamingResponse(_stream(), media_type="text/event-stream")

    return {
        "response": text,
        "model": model,
        "session_id": req.session_id,
        "duration_ms": 0,
    }


async def _handle_calendar_pending(
    req: ChatRequest,
    user_code: str,
    pending_raw: bytes,
) -> "StreamingResponse | dict | None":
    """
    Gère la confirmation ou l'annulation d'un événement calendrier en attente
    (clé Redis jarvis:{user_code}:pending_calendar_action).
    Retourne None si le message n'est ni "confirme" ni "annule" → pipeline normal.
    """
    words = set(req.message.lower().split())

    if words & {"confirme"}:
        pending = json.loads(pending_raw)
        REDIS_CLIENT.delete(f"jarvis:{user_code}:pending_calendar_action")
        try:
            event_id = await asyncio.wait_for(
                asyncio.to_thread(
                    create_calendar_event,
                    pending["title"],
                    pending["start_dt"],
                    pending["end_dt"],
                    pending.get("description", ""),
                    pending.get("location", ""),
                    None,
                    user_code,
                ),
                timeout=15.0,
            )
        except asyncio.TimeoutError:
            logger.warning("create_calendar_event timed out for %s", user_code)
            event_id = None
        reply = (
            f"C'est fait ! J'ai ajouté « {pending['title']} » à ton agenda."
            if event_id
            else "Désolé, je n'ai pas pu créer l'événement. Vérifie les droits d'accès au calendrier."
        )
        return _instant_reply(req, user_code, reply)

    if words & {"non", "annule", "annuler"}:
        REDIS_CLIENT.delete(f"jarvis:{user_code}:pending_calendar_action")
        return _instant_reply(
            req, user_code, "D'accord, j'annule. L'événement n'a pas été créé."
        )

    return None  # mot non reconnu → fall-through vers le pipeline


async def _handle_calendar_write(
    req: ChatRequest,
    user_code: str,
    google_available: bool,
) -> "StreamingResponse | dict | None":
    """
    Détecte les requêtes d'ajout au calendrier, extrait l'événement via LLM
    et stocke une action en attente de confirmation.
    Retourne None si la détection échoue ou si Google n'est pas disponible.
    """
    if not is_calendar_write(req.message) or not google_available:
        return None

    event = await extract_calendar_event_llm(req.message)
    if not event:
        # Date ou heure manquante — demander sans passer par le LLM principal
        return _instant_reply(
            req,
            user_code,
            "Je n'ai pas trouvé la date ou l'heure du rendez-vous. Peux-tu préciser ?",
        )

    try:
        tz = USER_TIMEZONES.get(user_code, BRIEFING_TIMEZONE)
        start_dt = build_iso_dt(event["start_date"], event["start_time"], tz)
        end_dt = build_iso_dt(event["end_date"], event["end_time"], tz)
        pending_data = json.dumps(
            {
                "title": event["title"],
                "start_dt": start_dt,
                "end_dt": end_dt,
                "description": event.get("description", ""),
                "location": event.get("location", ""),
            }
        )
        REDIS_CLIENT.setex(
            f"jarvis:{user_code}:pending_calendar_action", 600, pending_data
        )
        loc_line = f"\n📍 {event['location']}" if event.get("location") else ""
        multi = event["start_date"] != event["end_date"]
        date_line = (
            f"📅 {event['start_date']} {event['start_time']} → {event['end_date']} {event['end_time']}"
            if multi
            else f"📅 {event['start_date']} · {event['start_time']} → {event['end_time']}"
        )
        confirm_msg = (
            f"Je vais créer : **{event['title']}**\n"
            f"{date_line}"
            f'{loc_line}\n\nConfirmes ? (réponds "confirme" ou "annule")'
        )
        return _instant_reply(req, user_code, confirm_msg)
    except Exception as exc:
        logger.warning("Calendar write prep failed: %s", type(exc).__name__)
        return None  # fall-through vers le pipeline


def _handle_proposal(
    req: ChatRequest,
    user_code: str,
    use_model: str,
) -> "StreamingResponse | dict | None":
    """
    Exécute une commande de proposition (optimisation de prompt, etc.).
    Retourne None si aucune commande ne correspond au message.
    """
    proposal_resp = handle_proposal_command(req.message, user_code)
    if proposal_resp is None:
        return None
    return _instant_reply(req, user_code, proposal_resp, model=use_model)


async def _handle_briefing(
    req: ChatRequest,
    user_code: str,
    use_model: str,
) -> "StreamingResponse | dict":
    """
    Sert le briefing matinal depuis Redis ou le génère à la demande.
    Toujours appelé avec use_briefing=True — ne retourne jamais None.
    """
    stored = get_stored_briefing(user_code)
    if stored:
        logger.info("Briefing served from Redis for %s", user_code)
        return _instant_reply(req, user_code, stored.text, model=use_model)

    logger.info("Briefing not cached, generating on-demand for %s", user_code)
    result = await gather_briefing(user_code)
    store_briefing(user_code, result)
    return _instant_reply(req, user_code, result.text, model=use_model)


# ── Endpoints ──────────────────────────────────────────────────────────────────


@router.post("/chat")
async def chat(req: ChatRequest):
    if not PRIMARY_API_KEY:
        raise HTTPException(503, "No LLM API key configured")

    user_code = req.user_code
    if not user_code or user_code not in USER_CODES:
        raise HTTPException(403, "Invalid user code")

    # Timer starts here — before any processing — so all TTFT logs are accurate.
    _t0 = time.time()
    logger.debug("[TTFT] request received — user=%s msg=%r", user_code, req.message[:60])

    # Compute once — reused in embed router, LLM router, calendar write, context gather.
    _google_available = is_google_available(user_code)

    # ── Early-exit: Open WebUI internal system requests ─────────────────────
    # Open WebUI envoie ses propres appels LLM (suggestions, titres…).
    # Ces messages doivent être renvoyés directement au modèle léger.
    if (result := _openwebui_passthrough(req)) is not None:
        return result

    # ════════════════════════════════════════════════════════════════════════
    # STEP 1 — KEYWORD DISPATCH — fast paths, no LLM router cost
    # Checks keyword-triggered actions before any embedding or LLM call.
    # ════════════════════════════════════════════════════════════════════════

    # ── 1a. Pending calendar action: confirm or cancel ──────────────────────
    # If a pending action exists and the user confirms/cancels → act and return.
    # If the word is unrecognised → skip calendar write check (avoids overwriting
    # the existing pending event with a brand new one).
    _pending_raw = REDIS_CLIENT.get(f"jarvis:{user_code}:pending_calendar_action")
    if _pending_raw:
        if (result := await _handle_calendar_pending(req, user_code, _pending_raw)) is not None:
            return result
        # Unrecognised word while pending → fall through to router, skip 1b.
    else:
        # ── 1b. Calendar write (keyword → LLM extraction, no router needed) ─
        # Only checked when there is no pending action to avoid overwriting it.
        if (result := await _handle_calendar_write(req, user_code, _google_available)) is not None:
            return result

    # ════════════════════════════════════════════════════════════════════════
    # STEP 2 — EMBEDDING ROUTER — fast cosine-similarity intent classifier
    # ~2-5 ms. Skips the LLM router (~1.3 s) when confident.
    # Returns None when score is low or ambiguous → LLM router takes over.
    # ════════════════════════════════════════════════════════════════════════
    _embed_result = embed_route(req.message, google_available=_google_available)

    # ════════════════════════════════════════════════════════════════════════
    # STEP 3 — LLM ROUTER + dynamic prefix + conversation history (parallel)
    # get_conversation is gathered here so it overlaps with the LLM router call
    # instead of running sequentially after it.
    # ════════════════════════════════════════════════════════════════════════
    user_name = USER_CODES.get(user_code)
    system_prompt = build_system_prompt()

    # Speculative memory search — started immediately, in parallel with routing.
    # Memory is the most common intent (~80 % of requests); the embedding call
    # (~2–3 s on CPU) was previously sequential with routing, adding 2–3 s to TTFT.
    # If routing decides use_memory=False the result is discarded (cost: one
    # embedding call, ~2–3 s CPU, no GPU impact).
    _spec_mem_task: asyncio.Task = asyncio.ensure_future(
        asyncio.to_thread(search_memory, user_code, req.message, 5)
    )

    if _embed_result is not None:
        # Fast-path: no LLM router — load prefix, history, memory in parallel.
        _gather_ep = await asyncio.gather(
            asyncio.to_thread(
                build_dynamic_prefix,
                req.session_id, user_code, user_name or "", req.voice_mode,
            ),
            asyncio.to_thread(get_conversation, user_code, req.session_id, _HIST_WINDOW),
            _spec_mem_task,
            return_exceptions=True,
        )
        _pfx = _gather_ep[0]
        if isinstance(_pfx, BaseException):
            logger.error("build_dynamic_prefix failed: %s", _pfx)
            dynamic_prefix, _self_mem = "", {}
        else:
            dynamic_prefix, _self_mem = _pfx
        hist = _gather_ep[1] if not isinstance(_gather_ep[1], BaseException) else []
        if isinstance(_gather_ep[1], BaseException):
            logger.error("get_conversation failed: %s", _gather_ep[1])
        _prefetched_memory = _gather_ep[2] if not isinstance(_gather_ep[2], BaseException) else None
        if isinstance(_gather_ep[2], BaseException):
            logger.warning("speculative memory search failed: %s", _gather_ep[2])
        llm_result = _embed_result
        logger.debug(
            "[TTFT] embed router hit — LLM router skipped — %.3fs", time.time() - _t0
        )
    else:
        # Fallback: LLM router 3B + dynamic prefix + history + memory in parallel.
        _gather1 = await asyncio.gather(
            asyncio.to_thread(
                build_dynamic_prefix,
                req.session_id, user_code, user_name or "", req.voice_mode,
            ),
            llm_route(req.message, google_available=_google_available),
            asyncio.to_thread(get_conversation, user_code, req.session_id, _HIST_WINDOW),
            _spec_mem_task,
            return_exceptions=True,
        )
        _pfx = _gather1[0]
        if isinstance(_pfx, BaseException):
            logger.error("build_dynamic_prefix failed: %s", _pfx)
            dynamic_prefix, _self_mem = "", {}
        else:
            dynamic_prefix, _self_mem = _pfx
        llm_result = _gather1[1] if not isinstance(_gather1[1], BaseException) else None
        hist = _gather1[2] if not isinstance(_gather1[2], BaseException) else []
        _prefetched_memory = _gather1[3] if not isinstance(_gather1[3], BaseException) else None
        if isinstance(_gather1[1], BaseException):
            logger.error("llm_route failed: %s", _gather1[1])
        if isinstance(_gather1[2], BaseException):
            logger.error("get_conversation failed: %s", _gather1[2])
        if isinstance(_gather1[3], BaseException):
            logger.warning("speculative memory search failed: %s", _gather1[3])
        logger.debug(
            "[TTFT] gather1 done (LLM router+dynamic_prefix+hist+mem) — %.3fs", time.time() - _t0
        )

    # ── Router result extraction ────────────────────────────────────────────
    if llm_result:
        use_memory = llm_result.use_memory
        use_rag = llm_result.use_rag
        use_web_auto = llm_result.use_web
        use_weather_auto = llm_result.use_weather
        use_gmail = llm_result.use_gmail
        use_calendar = llm_result.use_calendar
        use_briefing = llm_result.use_briefing
        use_self = llm_result.use_self
        use_portfolio = llm_result.use_portfolio
        _llm_gmail_query = llm_result.gmail_query
        _llm_cal_days = llm_result.calendar_days
        _llm_weather_location = llm_result.weather_location
    else:
        use_memory = use_rag = use_web_auto = use_gmail = use_calendar = False
        use_briefing = use_self = use_portfolio = use_weather_auto = False
        _llm_gmail_query = None
        _llm_cal_days = None
        _llm_weather_location = ""

    # ── Model selection ─────────────────────────────────────────────────────
    # Always PRIMARY infrastructure — reasoning is handled via no_think flag only.
    use_model = req.model or PRIMARY_MODEL
    _use_api_url = PRIMARY_API_URL
    _use_api_key = PRIMARY_API_KEY
    _use_timeout = PRIMARY_TIMEOUT

    # ── no_think for simple intents (memory/conversation) ──────────────────
    # Complex intents (web, RAG, reasoning) keep chain-of-thought.
    # Typical saving: ~4 s of TTFT on conversational exchanges.
    _complex_intents = use_rag or use_web_auto
    chat_no_think = (
        False if (llm_result and llm_result.use_reasoning) else not _complex_intents
    )

    # ── Vision: convert iOS base64 image into standard image_part ──────────
    if req.image_base64:
        req.image_parts = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{req.image_base64}"},
            }
        ] + req.image_parts

    # ── Vision: describe images (two-stage pipeline) ────────────────────────
    image_description = ""
    if req.image_parts:
        if VISION_MODEL:
            image_description = await describe_images(req.image_parts, req.message)
            if image_description:
                logger.info("Vision: image described (%d chars)", len(image_description))
        else:
            logger.warning(
                "Vision: image received but VISION_MODEL not configured — ignored"
            )

    # ════════════════════════════════════════════════════════════════════════
    # STEP 4 — INTENT SHORT-CIRCUITS — self / briefing
    # These intents return immediately, without going through context gather.
    # ════════════════════════════════════════════════════════════════════════

    # ── 4a. Self: état interne Jarvis (proposals take priority over briefing) ─
    if use_self:
        use_briefing = False
        if (result := _handle_proposal(req, user_code, use_model)) is not None:
            return result
        # No proposal match → self context injected below via build_context

    # ── 4b. Briefing: sert depuis Redis ou génère à la demande ─────────────
    if use_briefing:
        return await _handle_briefing(req, user_code, use_model)

    # ════════════════════════════════════════════════════════════════════════
    # STEP 5 — CONTEXT GATHER — all external sources fetched in parallel
    # ════════════════════════════════════════════════════════════════════════
    gmail_query = _llm_gmail_query or ""
    cal_days = _llm_cal_days or 7
    _weather_query = _llm_weather_location or USER_CITIES.get(user_code, "Paris")
    _inline_urls = _URL_RE.findall(req.message)

    logger.debug(
        "[TTFT] gather2 start (rag=%s memory=%s web=%s gmail=%s cal=%s urls=%d) — %.3fs",
        use_rag,
        use_memory,
        use_web_auto or req.use_web,
        use_gmail,
        use_calendar,
        len(_inline_urls),
        time.time() - _t0,
    )

    async def _resolved_memory():
        """Retourne la recherche mémoire préchargée (gather1) ou [] si non requise."""
        return _prefetched_memory if (_prefetched_memory is not None) else []

    _gather2 = await asyncio.gather(
        search_documents(req.message) if (req.use_rag or use_rag) else _empty(),
        _resolved_memory() if use_memory else _empty(),
        search_weather(_weather_query)
        if use_weather_auto
        else search_web(optimize_web_query(req.message), original_message=req.message)
        if (req.use_web or use_web_auto)
        else _empty(),
        _timed_thread(fetch_gmail_messages, gmail_query, 10, user_code)
        if use_gmail and _google_available
        else _empty(),
        _timed_thread(fetch_calendar_events, cal_days, None, None, user_code)
        if use_calendar and _google_available
        else _empty(),
        fetch_user_urls(_inline_urls) if _inline_urls else _empty(),
        return_exceptions=True,
    )
    _ctx_names = ("rag", "memory", "web", "gmail", "calendar", "inline_urls")
    (
        rag_chunks,
        memory_chunks,
        web_results,
        gmail_results,
        calendar_results,
        inline_url_results,
    ) = [
        (logger.error("Context source '%s' failed: %s", _ctx_names[i], v) or [])
        if isinstance(v, BaseException)
        else v
        for i, v in enumerate(_gather2)
    ]

    # Inline URL results take priority — prepend so the LLM sees them first.
    if inline_url_results:
        _seen_urls = {r.get("url") for r in inline_url_results}
        web_results = inline_url_results + [
            r for r in web_results if r.get("url") not in _seen_urls
        ]

    logger.debug(
        "[TTFT] gather2 done (all context sources resolved) — %.3fs", time.time() - _t0
    )

    # ── Web auto-trigger — fallback when context thin + factual question ────────
    # Fires ONLY when: no web was planned, no RAG, no inline URL, and the question
    # contains factual signals (price, version, current state, …).
    # Memory score guard avoids firing when a relevant memory hit already exists.
    if (
        not web_results
        and not rag_chunks
        and not inline_url_results
        and not (use_web_auto or req.use_web)
        and _auto_web_needed(req.message, memory_chunks)
    ):
        logger.info("chat: auto-web fallback — factual query, thin context")
        web_results = await search_web(
            optimize_web_query(req.message),
            original_message=req.message,
            max_results=3,
        )
        chat_no_think = False  # research query → enable thinking

    # ════════════════════════════════════════════════════════════════════════
    # STEP 6 — MESSAGE ASSEMBLY — context + prefix + history → final prompt
    # ════════════════════════════════════════════════════════════════════════

    # Write user name once — avoids HKEYS + LLM normalisation on every request
    if user_name and not REDIS_CLIENT.hget(f"user:{user_code}:profile", "name"):
        update_user_profile(user_code, "name", user_name)

    assembled = build_context(
        rag_chunks,
        memory_chunks,
        web_results,
        gmail_results,
        calendar_results,
        use_portfolio,
        use_self,
        user_code,
        self_mem=_self_mem,  # reuse from build_dynamic_prefix — avoids second Redis call
    )
    if assembled:
        logger.info(
            "context memory=%d rag=%d web=%d",
            len(memory_chunks),
            len(rag_chunks),
            len(web_results),
        )

    # Chain-of-thought hint injected into user message (not system prompt)
    # to keep the static system prefix token-identical → KV cache valid.
    reasoning_hint = ""
    if llm_result and llm_result.use_reasoning:
        reasoning_hint = (
            "\n\nCette question nécessite une réflexion approfondie. "
            "Analyse-la étape par étape avant de répondre."
        )

    extra_ctx_parts = []
    if dynamic_prefix:
        extra_ctx_parts.append(dynamic_prefix)
    if assembled:
        extra_ctx_parts.append(
            "Utilise le contexte suivant pour répondre. Cite les sources si pertinent.\n\n"
            + assembled
        )
    if reasoning_hint:
        extra_ctx_parts.append(reasoning_hint.strip())

    raw_user_content = req.message
    if image_description:
        raw_user_content = (
            f"{req.message}\n\n"
            f"--- Image jointe, décrite ci-dessous ---\n"
            f"{image_description}"
        )

    full_prefix = "\n\n".join(extra_ctx_parts)
    user_content = augment_user_message(full_prefix, raw_user_content)

    # Window: last 8 messages (4 exchanges).
    # Raw messages (no dynamic prefix) are stored in Redis → history stays compact.
    hist_slice = hist[-_HIST_WINDOW:]
    messages = [{"role": "system", "content": system_prompt}]
    for m in hist_slice:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_content})

    logger.debug(
        "[TTFT] messages built (%d msgs, sysprompt=%d chars) — handing off to LLM — %.3fs",
        len(messages),
        len(system_prompt),
        time.time() - _t0,
    )

    start = time.time()

    # Pre-compute safe web sources (used in both streaming and JSON paths).
    _safe_web = [] if web_results == INTERNET_ERROR else web_results

    # ════════════════════════════════════════════════════════════════════════
    # STEP 7 — LLM CALL — streaming SSE or blocking JSON
    # ════════════════════════════════════════════════════════════════════════
    if req.stream:

        async def sse():
            full = ""
            try:
                in_think = False
                first_chunk = True

                async for chunk in stream_openai(
                    messages,
                    use_model,
                    _use_api_url,
                    _use_api_key,
                    _use_timeout,
                    no_think=chat_no_think,
                    session_id=req.session_id,
                ):
                    full += chunk

                    # ── Think filtering ─────────────────────────────────────
                    # Special case: <think> and </think> in the same chunk
                    # (e.g. Qwen3.5 no-think mode → "<think>\n\n</think>\n\n").
                    # Doing `continue` on <think> alone would miss the </think>
                    # → in_think stays True → the entire response is swallowed.
                    if "<think>" in chunk:
                        in_think = True
                        if "</think>" in chunk:
                            in_think = False  # empty block entirely in this chunk
                        continue

                    if "</think>" in chunk:
                        in_think = False
                        continue

                    if in_think:
                        continue

                    clean = chunk

                    if first_chunk:
                        clean = clean.lstrip("\n")
                        logger.debug(
                            "[TTFT] first visible token yielded — %.3fs since request",
                            time.time() - _t0,
                        )
                        first_chunk = False

                    if clean:
                        yield f"data: {json.dumps({'content': clean})}\n\n"

                # Strip complete think blocks, then any truncated open <think>
                # (e.g. model hit token budget mid-reasoning — no closing </think>).
                full_clean = re.sub(r"<think>.*?</think>", "", full, flags=re.DOTALL)
                full_clean = re.sub(
                    r"<think>.*$", "", full_clean, flags=re.DOTALL
                ).strip()
                append_conversation_message(
                    user_code, req.session_id, "user", raw_user_content
                )
                append_conversation_message(
                    user_code, req.session_id, "assistant", full_clean
                )
                ms = int((time.time() - start) * 1000)
                yield f"data: {json.dumps({'done': True, 'model': use_model, 'duration_ms': ms, 'rag_sources': [{'source': c['source'], 'score': c['score']} for c in rag_chunks], 'web_sources': [{'title': w['title'], 'url': w['url']} for w in _safe_web]})}\n\n"
                asyncio.create_task(
                    post_analysis(req.session_id, user_code, req.message, full_clean)
                )
            except asyncio.CancelledError:
                logger.info("Client disconnected")

        return StreamingResponse(
            sse(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    # ── JSON response (non-streaming) ───────────────────────────────────────
    resp = await call_llm_async(
        messages,
        model=use_model,
        api_url=_use_api_url,
        api_key=_use_api_key,
        timeout=_use_timeout,
        no_think=chat_no_think,
    )

    append_conversation_message(user_code, req.session_id, "user", raw_user_content)
    append_conversation_message(user_code, req.session_id, "assistant", resp)
    ms = int((time.time() - start) * 1000)

    asyncio.create_task(post_analysis(req.session_id, user_code, req.message, resp))

    return {
        "response": resp,
        "model": use_model,
        "session_id": req.session_id,
        "duration_ms": ms,
        "rag_sources": [
            {"source": c["source"], "score": c["score"]} for c in rag_chunks
        ],
        "web_sources": [
            {"title": w["title"], "url": w["url"]} for w in _safe_web
        ],
    }


@router.get("/users/{user_code}/history/{session_id}")
async def get_history(session_id: str, user_code: str, limit: int = IOS_MAX_MESSAGES):
    if user_code not in USER_CODES:
        raise HTTPException(403)
    logger.debug(
        "History request user=%s session=%s limit=%d", user_code, session_id, limit
    )
    key = f"chat:{user_code}:{session_id}"
    entries = REDIS_CLIENT.lrange(key, -limit, -1)
    result = []
    for e in entries:
        try:
            msg = json.loads(e)
            result.append(msg)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Skipping corrupted history entry session=%s", session_id)
    return result

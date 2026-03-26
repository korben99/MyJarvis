"""
routes/chat.py — POST /chat and GET /users/{user_code}/history/{session_id}
============================================================================
The main Jarvis pipeline: routing → context gathering → LLM → streaming SSE.
"""

import asyncio
import json
import time
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from briefing import gather_briefing, get_stored_briefing, store_briefing
from config import (
    BRIEFING_TIMEZONE,
    IOS_MAX_MESSAGES,
    PRIMARY_API_KEY,
    PRIMARY_API_URL,
    PRIMARY_MODEL,
    ROUTER_API_KEY,
    ROUTER_API_URL,
    ROUTER_MODEL,
    USER_CITIES,
    USER_CODES,
    USER_TIMEZONES,
    VISION_MODEL,
)
from deps import REDIS_CLIENT
from google_services import (
    create_calendar_event,
    extract_calendar_event_llm,
    fetch_calendar_events,
    fetch_gmail_messages,
    is_calendar_write,
    is_google_available,
)
from helpers import build_iso_dt, get_logger
from llm_client import describe_images, select_model, stream_openai
from llm_router import llm_route
from memory import (
    append_conversation_message,
    get_conversation,
    search_memory,
    update_user_profile,
)
from pipeline import build_context, build_system_prompt, post_analysis
from rag import search_documents
from self import handle_proposal_command
from web_search import INTERNET_ERROR, optimize_web_query, search_weather, search_web

logger = get_logger("jarvis-chat")

router = APIRouter()


# ── Request model ──────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    user_code: Optional[str] = None
    model: Optional[str] = None
    stream: bool = True
    voice_mode: bool = False
    use_rag: bool = False        # router decides; set True to force RAG regardless
    use_web: bool = False
    image_parts: list = []       # OpenAI image_url part dicts forwarded from the proxy
    image_base64: Optional[str] = None  # base64 JPEG/PNG sent directly by the iOS app


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/chat")
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
        _owui_model   = ROUTER_MODEL or PRIMARY_MODEL
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

    hist = get_conversation(user_code, req.session_id)

    # ════════════════════════════════════════════════════════════════════════
    # KEYWORD DISPATCH — fast paths, no LLM router cost
    # Order: pure-keyword checks first, then router for everything else.
    # ════════════════════════════════════════════════════════════════════════

    # ── 1. Pending calendar action: confirm or cancel ─────────────────────
    _pending_key = f"jarvis:{user_code}:pending_calendar_action"
    _pending_raw = REDIS_CLIENT.get(_pending_key)
    if _pending_raw:
        _words = set(req.message.lower().split())
        if _words & {"confirme"}:
            _pending = json.loads(_pending_raw)
            REDIS_CLIENT.delete(_pending_key)
            try:
                _event_id = await asyncio.wait_for(
                    asyncio.to_thread(
                        create_calendar_event,
                        _pending["title"], _pending["start_dt"], _pending["end_dt"],
                        _pending.get("description", ""), _pending.get("location", ""), None, user_code,
                    ),
                    timeout=15.0,
                )
            except asyncio.TimeoutError:
                logger.warning("create_calendar_event timed out for %s", user_code)
                _event_id = None
            _cal_reply = (
                f"C'est fait ! J'ai ajouté « {_pending['title']} » à ton agenda."
                if _event_id else
                "Désolé, je n'ai pas pu créer l'événement. Vérifie les droits d'accès au calendrier."
            )
            append_conversation_message(user_code, req.session_id, "user", req.message)
            append_conversation_message(user_code, req.session_id, "assistant", _cal_reply)
            if req.stream:
                async def _cal_confirm_stream():
                    yield f"data: {json.dumps({'content': _cal_reply})}\n\n"
                    yield f"data: {json.dumps({'done': True, 'model': PRIMARY_MODEL, 'duration_ms': 0})}\n\n"
                return StreamingResponse(_cal_confirm_stream(), media_type="text/event-stream")
            return {"response": _cal_reply, "model": PRIMARY_MODEL, "session_id": req.session_id, "duration_ms": 0}

        elif _words & {"non", "annule", "annuler"}:
            REDIS_CLIENT.delete(_pending_key)
            _cancel_reply = "D'accord, j'annule. L'événement n'a pas été créé."
            append_conversation_message(user_code, req.session_id, "user", req.message)
            append_conversation_message(user_code, req.session_id, "assistant", _cancel_reply)
            if req.stream:
                async def _cal_cancel_stream():
                    yield f"data: {json.dumps({'content': _cancel_reply})}\n\n"
                    yield f"data: {json.dumps({'done': True, 'model': PRIMARY_MODEL, 'duration_ms': 0})}\n\n"
                return StreamingResponse(_cal_cancel_stream(), media_type="text/event-stream")
            return {"response": _cancel_reply, "model": PRIMARY_MODEL, "session_id": req.session_id, "duration_ms": 0}
        # neither confirm nor cancel → fall through to router

    # ── 2. Calendar write (keyword → LLM extraction, no router needed) ────
    if is_calendar_write(req.message) and is_google_available(user_code):
        _event = await extract_calendar_event_llm(req.message)
        if _event:
            try:
                _tz = USER_TIMEZONES.get(user_code, BRIEFING_TIMEZONE)
                _start_dt = build_iso_dt(_event["start_date"], _event["start_time"], _tz)
                _end_dt   = build_iso_dt(_event["end_date"],   _event["end_time"],   _tz)
                _pending_data = json.dumps({
                    "title":       _event["title"],
                    "start_dt":    _start_dt,
                    "end_dt":      _end_dt,
                    "description": _event.get("description", ""),
                    "location":    _event.get("location", ""),
                })
                REDIS_CLIENT.setex(f"jarvis:{user_code}:pending_calendar_action", 600, _pending_data)
                _loc_line = f"\n📍 {_event['location']}" if _event.get("location") else ""
                _multi = _event["start_date"] != _event["end_date"]
                _date_line = (
                    f"📅 {_event['start_date']} {_event['start_time']} → {_event['end_date']} {_event['end_time']}"
                    if _multi else
                    f"📅 {_event['start_date']} · {_event['start_time']} → {_event['end_time']}"
                )
                _confirm_msg = (
                    f"Je vais créer : **{_event['title']}**\n"
                    f"{_date_line}"
                    f"{_loc_line}\n\nConfirmes ? (réponds \"confirme\" ou \"annule\")"
                )
                append_conversation_message(user_code, req.session_id, "user", req.message)
                append_conversation_message(user_code, req.session_id, "assistant", _confirm_msg)
                if req.stream:
                    async def _cal_write_stream():
                        yield f"data: {json.dumps({'content': _confirm_msg})}\n\n"
                        yield f"data: {json.dumps({'done': True, 'model': PRIMARY_MODEL, 'duration_ms': 0})}\n\n"
                    return StreamingResponse(_cal_write_stream(), media_type="text/event-stream")
                return {"response": _confirm_msg, "model": PRIMARY_MODEL, "session_id": req.session_id, "duration_ms": 0}
            except Exception as exc:
                logger.warning("Calendar write prep failed: %s", type(exc).__name__)
        else:
            # Missing date/time — ask without going to LLM or router
            _missing_msg = "Je n'ai pas trouvé la date ou l'heure du rendez-vous. Peux-tu préciser ?"
            append_conversation_message(user_code, req.session_id, "user", req.message)
            append_conversation_message(user_code, req.session_id, "assistant", _missing_msg)
            if req.stream:
                async def _cal_missing_stream():
                    yield f"data: {json.dumps({'content': _missing_msg})}\n\n"
                    yield f"data: {json.dumps({'done': True, 'model': PRIMARY_MODEL, 'duration_ms': 0})}\n\n"
                return StreamingResponse(_cal_missing_stream(), media_type="text/event-stream")
            return {"response": _missing_msg, "model": PRIMARY_MODEL, "session_id": req.session_id, "duration_ms": 0}

    # ════════════════════════════════════════════════════════════════════════
    # LLM ROUTER — parallel with system prompt build
    # ════════════════════════════════════════════════════════════════════════

    user_name = USER_CODES.get(user_code)

    system_prompt, llm_result = await asyncio.gather(
        asyncio.to_thread(build_system_prompt, req.session_id, req.voice_mode, user_code),
        llm_route(req.message, google_available=is_google_available(user_code)),
    )
    if user_name:
        system_prompt += f"\n\nL'utilisateur avec qui tu parles s'appelle {user_name}."

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
        use_memory = use_rag = use_web_auto = use_gmail = use_calendar = False
        use_briefing = use_self = use_portfolio = use_weather_auto = False
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
        proposal_resp = handle_proposal_command(req.message, user_code)
        if proposal_resp is not None:
            append_conversation_message(user_code, req.session_id, "user", req.message)
            append_conversation_message(user_code, req.session_id, "assistant", proposal_resp)
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

    # ── Parallel context fetch ─────────────────────────────────────────────
    # memory_scope/conversation_type removed — router intents handle routing decisions.

    async def _empty() -> list:
        return []

    gmail_query    = _llm_gmail_query or ""
    cal_days       = _llm_cal_days or 7
    _weather_query = _llm_weather_location or USER_CITIES.get(user_code, "Paris")

    _google_available = is_google_available(user_code)

    async def _timed_thread(fn, *args, timeout=15.0):
        try:
            return await asyncio.wait_for(asyncio.to_thread(fn, *args), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Google API call timed out: %s", fn.__name__)
            return []

    rag_chunks, memory_chunks, web_results, gmail_results, calendar_results = await asyncio.gather(
        search_documents(req.message) if (req.use_rag or use_rag) else _empty(),
        asyncio.to_thread(search_memory, user_code, req.message, 5) if use_memory else _empty(),
        search_weather(_weather_query) if use_weather_auto else
        search_web(optimize_web_query(req.message), original_message=req.message) if (req.use_web or use_web_auto) else _empty(),
        _timed_thread(fetch_gmail_messages, gmail_query, 10, user_code) if use_gmail and _google_available else _empty(),
        _timed_thread(fetch_calendar_events, cal_days, None, None, user_code) if use_calendar and _google_available else _empty(),
    )

    if user_name:
        update_user_profile(user_code, "name", user_name)

    # ── Context assembly ──────────────────────────────────────────────────
    assembled = build_context(
        rag_chunks, memory_chunks, web_results, gmail_results, calendar_results,
        use_portfolio, use_self, user_code,
    )
    if assembled:
        system_prompt += (
            "\n\nUtilise le contexte suivant pour répondre. Cite les sources si pertinent.\n\n"
            + assembled
        )
        logger.info("context memory=%d rag=%d web=%d", len(memory_chunks), len(rag_chunks), len(web_results))

    # ── Chain-of-thought for complex queries ──────────────────────────────
    if llm_result and llm_result.use_reasoning:
        system_prompt += (
            "\n\nCette question nécessite une réflexion approfondie. "
            "Analyse-la étape par étape avant de répondre."
        )

    # ── Build message list ────────────────────────────────────────────────
    messages = [{"role": "system", "content": system_prompt}]
    for m in hist[-20:]:
        messages.append({"role": m["role"], "content": m["content"]})

    user_content = req.message
    if image_description:
        user_content = (
            f"{req.message}\n\n"
            f"--- Image jointe, décrite ci-dessous ---\n"
            f"{image_description}"
        )
    messages.append({"role": "user", "content": user_content})

    start = time.time()

    # ── Streaming response ────────────────────────────────────────────────
    if req.stream:
        async def sse():
            full = ""
            try:
                async for chunk in stream_openai(messages, use_model, _use_api_url, _use_api_key, _use_timeout):
                    full += chunk
                    yield f"data: {json.dumps({'content': chunk})}\n\n"

                append_conversation_message(user_code, req.session_id, "user", req.message)
                append_conversation_message(user_code, req.session_id, "assistant", full)
                ms = int((time.time() - start) * 1000)
                _safe_web = [] if web_results == INTERNET_ERROR else web_results
                yield f"data: {json.dumps({'done': True, 'model': use_model, 'duration_ms': ms, 'rag_sources': [{'source': c['source'], 'score': c['score']} for c in rag_chunks], 'web_sources': [{'title': w['title'], 'url': w['url']} for w in _safe_web]})}\n\n"
                asyncio.create_task(post_analysis(req.session_id, user_code, req.message, full))
            except asyncio.CancelledError:
                logger.info("Client disconnected")

        return StreamingResponse(sse(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    # ── JSON response ─────────────────────────────────────────────────────
    async with httpx.AsyncClient(timeout=_use_timeout) as c:
        r = await c.post(
            f"{_use_api_url}/chat/completions",
            headers={"Authorization": f"Bearer {_use_api_key}", "Content-Type": "application/json"},
            json={"model": use_model, "messages": messages, "stream": False},
        )
    r.raise_for_status()
    data = r.json()
    if "choices" not in data:
        raise HTTPException(502, f"OpenAI error: {data}")
    resp = data["choices"][0]["message"]["content"]

    append_conversation_message(user_code, req.session_id, "user", req.message)
    append_conversation_message(user_code, req.session_id, "assistant", resp)
    ms = int((time.time() - start) * 1000)

    asyncio.create_task(post_analysis(req.session_id, user_code, req.message, resp))

    return {
        "response":    resp,
        "model":       use_model,
        "session_id":  req.session_id,
        "duration_ms": ms,
        "rag_sources": [{"source": c["source"], "score": c["score"]} for c in rag_chunks],
        "web_sources": [{"title": w["title"], "url": w["url"]} for w in web_results if w != INTERNET_ERROR[0]],
    }


@router.get("/users/{user_code}/history/{session_id}")
async def get_history(session_id: str, user_code: str, limit: int = IOS_MAX_MESSAGES):
    if user_code not in USER_CODES:
        raise HTTPException(403)
    logger.debug("History request user=%s session=%s limit=%d", user_code, session_id, limit)
    key = f"chat:{user_code}:{session_id}"
    entries = REDIS_CLIENT.lrange(key, -limit, -1)
    return [json.loads(e) for e in entries]

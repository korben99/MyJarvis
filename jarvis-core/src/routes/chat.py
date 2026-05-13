"""
routes/chat.py — POST /chat and GET /users/{user_code}/history/{session_id}
============================================================================
The main Jarvis pipeline: routing → context gathering → LLM → streaming SSE.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Optional

from briefing import gather_briefing, get_stored_briefing, store_briefing
from config import (
    BRIEFING_TIMEZONE,
    HIST_CONV_SUMMARIZE_THRESHOLD,
    HIST_CONV_TOKEN_BUDGET,
    IOS_MAX_MESSAGES,
    LLM_LOCAL,
    MAX_TOKENS_NO_THINK,
    MAX_TOKENS_REASONING,
    MAX_TOKENS_SYNTHESIS,
    SESSION_SUMMARY_TOKENS,
    THINKING_BUDGET_DEEP,
    THINKING_BUDGET_MEDIUM,
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
    is_qwen3,
    llm_timeout,
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
from helpers import (
    build_iso_dt,
    call_llm_async,
    filter_think_chunk,
    fmt_now_fr,
    get_logger,
    get_session_summary_data,
    rel_time_fr,
    set_session_summary_data,
)
from llm_client import describe_images, stream_openai
from llm_router import llm_route
from memory import (
    append_conversation_message,
    async_search_memory,
    get_conversation,
    get_project_detail,
    get_project_timeline_text,
    update_user_profile,
)
from pipeline import (
    build_context,
    build_dynamic_prefix,
    build_system_prompt,
    post_analysis,
)
from prompts import get_prompt
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
def _trim_history_to_budget(hist: list[dict], budget_tokens: int) -> list[dict]:
    """Keep the most recent messages that fit within the token budget (4 chars ≈ 1 token)."""
    budget_chars = budget_tokens * 4
    selected: list[dict] = []
    used = 0
    for msg in reversed(hist):
        cost = len(msg.get("content", ""))
        if used + cost > budget_chars and selected:
            break
        selected.append(msg)
        used += cost
    return list(reversed(selected))


async def _update_session_summary(user_code: str, session_id: str) -> None:
    """Post-response background task: compress conversation history into a rolling summary.

    Self-contained: fetches state from Redis, checks threshold, generates if needed.
    Uses the ROUTER model (warm in VRAM, 3B) — runs after response is sent, no GPU conflict.
    Trigger: uncovered messages (since last summary) exceed HIST_CONV_SUMMARIZE_THRESHOLD.
    """
    try:
        summary_data = get_session_summary_data(user_code, session_id)
        covered_count = summary_data["msg_count"] if summary_data else 0
        existing_text = summary_data["text"] if summary_data else ""

        total_count = REDIS_CLIENT.llen(f"chat:{user_code}:{session_id}")
        uncovered_n = max(0, int(total_count) - covered_count)
        if uncovered_n == 0:
            return

        uncovered = get_conversation(user_code, session_id, limit=uncovered_n)
        uncovered_chars = sum(len(m.get("content", "")) for m in uncovered)
        if uncovered_chars <= HIST_CONV_SUMMARIZE_THRESHOLD * 4:
            return

        dropped_text = "\n".join(
            f"{m['role']}: {m.get('content', '')[:400]}" for m in uncovered
        )
        existing_block = f"Résumé précédent :\n{existing_text}\n\n" if existing_text else ""
        prompt = get_prompt("SESSION_SUMMARY_PROMPT").format(
            existing_block=existing_block,
            dropped_text=dropped_text,
        )
        content = await call_llm_async(
            [{"role": "user", "content": prompt}],
            model=ROUTER_MODEL,
            api_url=ROUTER_API_URL,
            api_key=ROUTER_API_KEY,
            temperature=0.0,
            max_tokens=SESSION_SUMMARY_TOKENS,
            no_think=True,
            timeout=llm_timeout(SESSION_SUMMARY_TOKENS),
        )
        if content and content.strip():
            set_session_summary_data(user_code, session_id, content.strip(), int(total_count))
            logger.debug(
                "session summary updated: %s/%s (covers %d msgs)", user_code, session_id, total_count
            )
    except Exception as exc:
        logger.warning("session summary update failed: %s", exc)


# Derived fetch limit: enough messages for injection + one threshold's worth of summarization.
# Not a config variable — purely an implementation detail derived from existing constants.
_HIST_FETCH_N = max(HIST_CONV_TOKEN_BUDGET // 50, HIST_CONV_SUMMARIZE_THRESHOLD // 50, 10)

# Think banner throttle — emit one SSE event every N tokens (≈8/s at 120 tok/s).
# iOS ThinkingBanner already shows suffix(120) so sending the last 200 chars is enough.
_THINK_EMIT_EVERY = 15
_THINK_PREVIEW_CHARS = 200

# user_codes whose Redis profile has been initialised this process lifetime.
# Avoids a Redis hget on every request — populated on first message per user.
_profile_initialised: set[str] = set()

# Compiled once at module level — used in STEP 5 URL detection.
_URL_RE = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]{5,}')

# ── Web auto-trigger signals ────────────────────────────────────────────────────
# Factual patterns that strongly suggest external info is needed even when the
# router classified the message as "memory" (conversational phrasing).
_AUTO_WEB_RE = re.compile(
    r"\bprix\b|\btarif\b|\bcombien\b|\bcoûte?\b"
    r"|\bversion\b|\bmodèle\b"
    r"|\bsorti\b|\bdisponible\b|\blancé\b"
    r"|\bactuel\b|\bactuelle\b|\bactuellem?ent\b"
    r"|\bdernier\b|\bdernière\b|\bnouveau\b|\bnouvelle\b"
    r"|\bspecs?\b|\bcaractéristique\b"
    r"|\bqui est\b|\bc\'est qui\b"
    r"|\bcompar[e-]\b",
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


async def _prefetched_or_empty(value) -> list:
    """Retourne la valeur préchargée si disponible, sinon liste vide.

    Remplace la closure _resolved_memory() qui était définie inline dans chat()
    pour accéder à _prefetched_memory. Rend la coroutine compatible avec
    asyncio.gather() sans fermeture sur des variables locales.
    """
    return value if value is not None else []


# ── SSE streaming helpers ──────────────────────────────────────────────────────


def _strip_sse_response(raw: str, started_in_think: bool) -> str:
    """Retire le bloc de réflexion Qwen3 de la réponse brute accumulée.

    Deux cas :
    - started_in_think=True  : le template a injecté <think> dans le prompt,
      l'output commence DANS le bloc (pas de tag ouvrant). On coupe à </think>.
    - started_in_think=False : flux normal. On retire les blocs <think>…</think>
      complets, puis tout <think> ouvert non fermé (troncature budget).
    """
    if started_in_think:
        if "</think>" in raw:
            return raw.split("</think>", 1)[1].strip()
        # Truncation mid-reasoning : aucune réponse visible
        return ""
    # Flux normal (no_think ou modèle cloud)
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    return re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL).strip()


@dataclass
class _SseCtx:
    """Contexte transmis à _sse_stream() — regroupe toutes les variables de chat()
    nécessaires au générateur SSE. Évite une fermeture sur les locals de chat()."""

    messages: list
    use_model: str
    api_url: str
    api_key: str
    timeout: float
    no_think: bool
    max_tokens: int
    session_id: str
    t0: float  # timer global requête (pour les logs TTFT)
    start: float  # timer démarrage appel LLM (pour duration_ms)
    rag_chunks: list
    safe_web: list
    user_code: str
    raw_user_content: str
    original_message: str


async def _sse_stream(ctx: _SseCtx):
    """Générateur SSE pour le streaming Qwen3/cloud.

    Extrait de la closure sse() qui était définie inline dans chat().
    Reçoit toutes ses dépendances via _SseCtx — aucune fermeture sur chat().

    Comportement :
    - Filtre les blocs <think> en temps réel (filter_think_chunk).
    - Envoie les fragments de réflexion via {"think": …} pour l'iOS.
    - Sauvegarde la réponse nettoyée dans Redis (historique) en fin de flux.
    - Gère la déconnexion client (CancelledError) : sauvegarde partielle.
    """
    full_parts: list[str] = []
    try:
        # With Qwen3 local + enable_thinking=True, apply_chat_template appends
        # <think>\n to the prompt (not to the output).  The model's first output
        # tokens are already INSIDE the think block — no opening <think> tag ever
        # arrives in the stream.  Starting in_think=True ensures filter_think_chunk
        # correctly routes those tokens as {"think":…} SSE events to the iOS banner.
        # For no_think=True or remote models, in_think stays False (normal path).
        in_think = LLM_LOCAL and is_qwen3(ctx.use_model) and not ctx.no_think
        in_think_started = (
            in_think  # snapshot avant la boucle (in_think mute dans le loop)
        )
        first_chunk = True
        # Think throttle: accumulate fragments, emit a sliding window every
        # _THINK_EMIT_EVERY tokens so the iOS banner updates at ~8/s instead
        # of ~120/s — readable scroll without changing the iOS side.
        _think_accum: list[str] = []
        _think_since_emit: int = 0

        async for chunk in stream_openai(
            ctx.messages,
            ctx.use_model,
            ctx.api_url,
            ctx.api_key,
            ctx.timeout,
            no_think=ctx.no_think,
            session_id=ctx.session_id,
            max_tokens=ctx.max_tokens,
            thinking_budget=THINKING_BUDGET_MEDIUM if not ctx.no_think else 0,
        ):
            full_parts.append(chunk)

            # ── Think filtering ─────────────────────────────────────────────
            # filter_think_chunk splits each chunk into visible text and
            # think-block content. Think fragments are forwarded as a
            # separate SSE event so the iOS client can display them as a
            # live ticker without mixing them into the chat bubble.
            clean, think_frag, in_think = filter_think_chunk(chunk, in_think)

            if think_frag:
                _think_accum.append(think_frag)
                _think_since_emit += 1
                if _think_since_emit >= _THINK_EMIT_EVERY:
                    snapshot = "".join(_think_accum)
                    yield f"data: {json.dumps({'think': snapshot[-_THINK_PREVIEW_CHARS:]})}\n\n"
                    _think_since_emit = 0
                    if len(snapshot) > _THINK_PREVIEW_CHARS * 2:
                        _think_accum = [snapshot[-_THINK_PREVIEW_CHARS:]]

            if first_chunk:
                clean = clean.lstrip("\n")
                if clean:
                    logger.debug(
                        "[TTFT] first visible token yielded — %.3fs since request",
                        time.time() - ctx.t0,
                    )
                    first_chunk = False

            if clean:
                yield f"data: {json.dumps({'content': clean})}\n\n"

        # Strip thinking block before saving to Redis.
        # in_think_started=True → output began INSIDE think block (no opening tag).
        full_clean = _strip_sse_response("".join(full_parts), in_think_started)
        if full_clean:
            append_conversation_message(
                ctx.user_code, ctx.session_id, "user", ctx.raw_user_content
            )
            append_conversation_message(
                ctx.user_code, ctx.session_id, "assistant", full_clean
            )
        ms = int((time.time() - ctx.start) * 1000)
        _done_payload = json.dumps(
            {
                "done": True,
                "model": ctx.use_model,
                "duration_ms": ms,
                "rag_sources": [
                    {"source": c["source"], "score": c["score"]} for c in ctx.rag_chunks
                ],
                "web_sources": [
                    {"title": w["title"], "url": w["url"]} for w in ctx.safe_web
                ],
            }
        )
        yield f"data: {_done_payload}\n\n"
        asyncio.create_task(
            post_analysis(
                ctx.session_id, ctx.user_code, ctx.original_message, full_clean
            )
        )
        asyncio.create_task(_update_session_summary(ctx.user_code, ctx.session_id))
    except asyncio.CancelledError:
        logger.info("Client disconnected")
        if full_parts:
            try:
                full_clean = _strip_sse_response("".join(full_parts), in_think_started)
                if full_clean:
                    append_conversation_message(
                        ctx.user_code, ctx.session_id, "user", ctx.raw_user_content
                    )
                    append_conversation_message(
                        ctx.user_code, ctx.session_id, "assistant", full_clean
                    )
                    asyncio.create_task(
                        post_analysis(
                            ctx.session_id,
                            ctx.user_code,
                            ctx.original_message,
                            full_clean,
                        )
                    )
                    logger.info(
                        "Saved response to Redis after disconnect (%d chars)",
                        len(full_clean),
                    )
            except Exception as _save_err:
                logger.warning("Failed to save on disconnect: %s", _save_err)


# ── Pipeline helpers ───────────────────────────────────────────────────────────


async def _instant_stream(text: str, model: str):
    """Générateur SSE minimal pour les réponses immédiates sans appel LLM.

    Extrait de la closure _stream() qui était définie inline dans _instant_reply().
    Toutes les dépendances sont passées explicitement.
    """
    yield f"data: {json.dumps({'content': text})}\n\n"
    yield f"data: {json.dumps({'done': True, 'model': model, 'duration_ms': 0})}\n\n"


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
        return StreamingResponse(
            _instant_stream(text, model),
            media_type="text/event-stream",
        )

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
    msg_lower = req.message.lower().strip()
    if re.match(r"^(oui[,\s]*)?(confirme[sz]?|ok|yes)\s*[!.]?$", msg_lower):
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

    if re.match(r"^(non[,\s]*)?(annule[r]?)\s*[!.]?$", msg_lower):
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
        return _instant_reply(
            req,
            user_code,
            "Je n'ai pas trouvé la date ou l'heure du rendez-vous. Peux-tu préciser ?",
        )
    if not event.get("title"):
        return _instant_reply(
            req,
            user_code,
            "Pour quel événement ? Donne-moi le nom du rendez-vous.",
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
        return _instant_reply(
            req,
            user_code,
            "Je n'ai pas pu préparer l'événement (erreur interne). Peux-tu réessayer ?",
        )


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
    logger.debug(
        "[TTFT] request received — user=%s msg=%r", user_code, req.message[:60]
    )

    # Compute once — reused in embed router, LLM router, calendar write, context gather.
    _google_available = is_google_available(user_code)

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
        if (
            result := await _handle_calendar_pending(req, user_code, _pending_raw)
        ) is not None:
            return result
        # Unrecognised word while pending → fall through to router, skip 1b.
    else:
        # ── 1b. Calendar write (keyword → LLM extraction, no router needed) ─
        # Only checked when there is no pending action to avoid overwriting it.
        if (
            result := await _handle_calendar_write(req, user_code, _google_available)
        ) is not None:
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
    # Guard: skip for very short messages (< 15 chars) — almost always small-talk
    # ("ok", "merci", "oui"). asyncio cancel() doesn't stop the underlying thread,
    # so avoiding the launch entirely is the only way to prevent wasted CPU.
    _spec_mem_task: asyncio.Task = asyncio.ensure_future(
        async_search_memory(user_code, req.message, 5)
        if len(req.message.strip()) >= 15
        else _empty()
    )

    if _embed_result is not None:
        # Fast-path: no LLM router — load prefix, history, memory in parallel.
        # Opinions and tomorrow_suggestions are only useful for conversational intents.
        _rich_intent = bool(
            _embed_result.use_memory
            or _embed_result.use_rag
            or _embed_result.use_web
            or _embed_result.use_self
        )
        if not _embed_result.use_memory:
            _spec_mem_task.cancel()
            _prefetched_memory_coro = _empty()
        else:
            _prefetched_memory_coro = _spec_mem_task

        if _embed_result.use_small_talk:
            # Small talk (acquiescements purs) — pas de profil, pas de recall mémoire.
            # Seul l'historique de conversation suffit.
            _spec_mem_task.cancel()
            tz = USER_TIMEZONES.get(user_code, "Europe/Paris")
            _name_part = f" Tu parles avec {user_name}." if user_name else ""
            dynamic_prefix = f"Date : {fmt_now_fr(tz)}.{_name_part}"
            _self_mem = {}
            hist = await asyncio.to_thread(
                get_conversation, user_code, req.session_id, _HIST_FETCH_N
            )
            _prefetched_memory = None
            logger.debug(
                "[TTFT] small talk — prefix minimal, no memory — %.3fs",
                time.time() - _t0,
            )
        else:
            _gather_ep = await asyncio.gather(
                asyncio.to_thread(
                    build_dynamic_prefix,
                    req.session_id,
                    user_code,
                    user_name or "",
                    req.voice_mode,
                    _rich_intent,
                    _rich_intent,  # include_opinions, include_suggestions
                ),
                asyncio.to_thread(
                    get_conversation, user_code, req.session_id, _HIST_FETCH_N
                ),
                _prefetched_memory_coro,
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
            _prefetched_memory = (
                _gather_ep[2] if not isinstance(_gather_ep[2], BaseException) else None
            )
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
                req.session_id,
                user_code,
                user_name or "",
                req.voice_mode,
            ),
            llm_route(req.message, google_available=_google_available),
            asyncio.to_thread(
                get_conversation, user_code, req.session_id, _HIST_FETCH_N
            ),
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
        _prefetched_memory = (
            _gather1[3] if not isinstance(_gather1[3], BaseException) else None
        )
        if isinstance(_gather1[1], BaseException):
            logger.error("llm_route failed: %s", _gather1[1])
        if isinstance(_gather1[2], BaseException):
            logger.error("get_conversation failed: %s", _gather1[2])
        if isinstance(_gather1[3], BaseException):
            logger.warning("speculative memory search failed: %s", _gather1[3])
        logger.debug(
            "[TTFT] gather1 done (LLM router+dynamic_prefix+hist+mem) — %.3fs",
            time.time() - _t0,
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
        _llm_rag_query = llm_result.rag_query
    else:
        use_memory = use_rag = use_web_auto = use_gmail = use_calendar = False
        use_briefing = use_self = use_portfolio = use_weather_auto = False
        _llm_gmail_query = None
        _llm_cal_days = None
        _llm_weather_location = ""
        _llm_rag_query = ""

    # ── Model selection ─────────────────────────────────────────────────────
    # Always PRIMARY infrastructure — reasoning is handled via no_think flag only.
    use_model = req.model or PRIMARY_MODEL
    _use_api_url = PRIMARY_API_URL
    _use_api_key = PRIMARY_API_KEY
    _use_timeout = PRIMARY_TIMEOUT

    # ── no_think for simple intents (memory/conversation) ──────────────────
    # Complex intents (web, RAG, reasoning) keep chain-of-thought.
    # Typical saving: ~4 s of TTFT on conversational exchanges.
    _complex_intents = use_rag or use_web_auto or req.use_web or req.use_rag
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
                logger.info(
                    "Vision: image described (%d chars)", len(image_description)
                )
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
        "[TTFT] gather2 start (rag=%s memory=%s web_auto=%s web_req=%s gmail=%s cal=%s urls=%d) — %.3fs",
        use_rag,
        use_memory,
        use_web_auto,
        req.use_web,
        use_gmail,
        use_calendar,
        len(_inline_urls),
        time.time() - _t0,
    )

    # _prefetched_or_empty() remplace la closure _resolved_memory() — voir module level.
    _gather2 = await asyncio.gather(
        search_documents(_llm_rag_query or req.message)
        if (req.use_rag or use_rag)
        else _empty(),
        _prefetched_or_empty(_prefetched_memory) if use_memory else _empty(),
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
    # Guard: don't iterate INTERNET_ERROR sentinel (it's a list with a magic dict).
    if inline_url_results:
        _seen_urls = {r.get("url") for r in inline_url_results}
        _web_to_merge = [] if web_results == INTERNET_ERROR else web_results
        web_results = inline_url_results + [
            r for r in _web_to_merge if r.get("url") not in _seen_urls
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
        and _auto_web_needed(req.message, memory_chunks)
    ):
        logger.info("chat: auto-web fallback — factual query, thin context")
        web_results = await search_web(
            optimize_web_query(req.message),
            original_message=req.message,
            max_results=3,
        )
        chat_no_think = False  # research query → enable thinking

    # Inject session-gap: timestamps are stripped when building the messages
    # list, so the model has no way to infer temporal distance from history
    # entries alone. This phrase prevents greeting again mid-conversation.
    if hist:
        _last_ts = hist[-1].get("ts")
        if _last_ts:
            _gap = time.time() - _last_ts
            _gap_txt = "moins d'une minute" if _gap < 60 else rel_time_fr(_last_ts)
            _gap_line = f"Dernier message : {_gap_txt}."
            dynamic_prefix = (dynamic_prefix + f"\n\n{_gap_line}") if dynamic_prefix else _gap_line

    # ════════════════════════════════════════════════════════════════════════
    # STEP 6 — MESSAGE ASSEMBLY — context + prefix + history → final prompt
    # ════════════════════════════════════════════════════════════════════════

    # Write user name once per process lifetime — skips Redis hget on every request
    if user_name and user_code not in _profile_initialised:
        if not REDIS_CLIENT.hget(f"user:{user_code}:profile", "name"):
            update_user_profile(user_code, "name", user_name)
        _profile_initialised.add(user_code)

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
        # Short hint — "étape par étape" triggers verbose formal checklists in the think block.
        reasoning_hint = "\n\nRéfléchis avant de répondre."
    elif not chat_no_think:
        # Web/RAG synthesis — encourage brief thinking to organise context.
        # No budget tag available for Qwen3.6 (causes garbled output); text hint instead.
        reasoning_hint = "\n\nSynthétise brièvement le contexte ci-dessus avant de répondre."

    raw_user_content = req.message
    if image_description:
        raw_user_content = (
            f"{req.message}\n\n"
            f"<image_analysis>\n"
            f"L'utilisateur a joint une image. Voici son analyse détaillée par le modèle vision "
            f"— traite ces informations comme si tu avais vu l'image toi-même et réponds directement "
            f"à la question sans demander de photo :\n\n"
            f"{image_description}\n"
            f"</image_analysis>"
        )

    # Project detail — injected once on first mention; history carries it forward.
    _project_name = (llm_result.project_name if llm_result else "") or ""
    _project_detail_block = ""
    if _project_name:
        _proj = await asyncio.to_thread(get_project_detail, user_code, _project_name)
        if _proj:
            _project_detail_block = (
                "<project_detail>\n"
                + get_project_timeline_text(_proj)
                + "\n</project_detail>"
            )
            logger.info("Project detail injected: %s", _proj["name"])

    _summary_data = get_session_summary_data(user_code, req.session_id)
    _session_summary = _summary_data["text"] if _summary_data else ""

    # Build the user message: [dynamic_prefix] → [summary] → [context] → [project_detail] → <user_message>
    # XML tag clearly delimits the actual question from all injected context above.
    msg_parts = []
    if dynamic_prefix:
        msg_parts.append(dynamic_prefix)
    if _session_summary:
        msg_parts.append(
            "<conversation_summary>\n" + _session_summary + "\n</conversation_summary>"
        )
    if assembled:
        msg_parts.append(assembled)
    if _project_detail_block:
        msg_parts.append(_project_detail_block)
    if reasoning_hint:
        msg_parts.append(reasoning_hint.strip())
    msg_parts.append(
        "<user_message>\n" + raw_user_content + "\n</user_message>"
    )
    user_content = "\n\n".join(msg_parts)

    # When a session summary exists, inject only the messages NOT yet covered by it.
    # The summary replaces everything up to msg_count; new messages are still injected raw.
    if _session_summary:
        _total = int(REDIS_CLIENT.llen(f"chat:{user_code}:{req.session_id}"))
        _uncovered_n = max(0, _total - _summary_data["msg_count"])
        _uncovered_hist = hist[-_uncovered_n:] if _uncovered_n > 0 else []
        hist_slice = _trim_history_to_budget(_uncovered_hist, HIST_CONV_TOKEN_BUDGET)
    else:
        hist_slice = _trim_history_to_budget(hist, HIST_CONV_TOKEN_BUDGET)
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

    # max_tokens budget: no-think / web-rag synthesis / deep reasoning (see config.py)
    _use_reasoning = bool(llm_result and llm_result.use_reasoning)
    _max_tokens = (
        MAX_TOKENS_NO_THINK  if chat_no_think
        else MAX_TOKENS_REASONING if _use_reasoning
        else MAX_TOKENS_SYNTHESIS
    )

    # ════════════════════════════════════════════════════════════════════════
    # STEP 7 — LLM CALL — streaming SSE or blocking JSON
    # ════════════════════════════════════════════════════════════════════════
    if req.stream:
        ctx = _SseCtx(
            messages=messages,
            use_model=use_model,
            api_url=_use_api_url,
            api_key=_use_api_key,
            timeout=_use_timeout,
            no_think=chat_no_think,
            max_tokens=_max_tokens,
            session_id=req.session_id,
            t0=_t0,
            start=start,
            rag_chunks=rag_chunks,
            safe_web=_safe_web,
            user_code=user_code,
            raw_user_content=raw_user_content,
            original_message=req.message,
        )
        return StreamingResponse(
            _sse_stream(ctx),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    # ── JSON response (non-streaming) ───────────────────────────────────────
    resp = await call_llm_async(
        messages,
        model=use_model,
        api_url=_use_api_url,
        api_key=_use_api_key,
        timeout=_use_timeout,
        no_think=chat_no_think,
        max_tokens=_max_tokens,
        thinking_budget=(THINKING_BUDGET_DEEP if _use_reasoning else THINKING_BUDGET_MEDIUM) if not chat_no_think else 0,
    )

    append_conversation_message(user_code, req.session_id, "user", raw_user_content)
    append_conversation_message(user_code, req.session_id, "assistant", resp)
    ms = int((time.time() - start) * 1000)

    asyncio.create_task(post_analysis(req.session_id, user_code, req.message, resp))
    asyncio.create_task(_update_session_summary(user_code, req.session_id))

    return {
        "response": resp,
        "model": use_model,
        "session_id": req.session_id,
        "duration_ms": ms,
        "rag_sources": [
            {"source": c["source"], "score": c["score"]} for c in rag_chunks
        ],
        "web_sources": [{"title": w["title"], "url": w["url"]} for w in _safe_web],
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

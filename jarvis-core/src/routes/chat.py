"""
routes/chat.py — POST /chat and GET /users/{user_code}/history/{session_id}
============================================================================
The main Jarvis pipeline: routing → context gathering → LLM → streaming SSE.
"""

import asyncio
import contextlib
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Optional

from briefing import gather_briefing, get_stored_briefing, store_briefing
from config import (
    AGENT_ENABLED,
    BRIEFING_TIMEZONE,
    DEFAULT_TEMP,
    HIST_CONV_SUMMARIZE_THRESHOLD,
    HIST_CONV_TOKEN_BUDGET,
    IOS_MAX_MESSAGES,
    LLM_LOCAL,
    MAX_TOKENS_NO_THINK,
    MAX_TOKENS_REASONING,
    MAX_TOKENS_SYNTHESIS,
    PRIMARY_API_KEY,
    PRIMARY_API_URL,
    PRIMARY_MODEL,
    PRIMARY_TIMEOUT,
    SESSION_SUMMARY_TOKENS,
    THINKING_BUDGET_COMPACT,
    THINKING_BUDGET_MEDIUM,
    USER_ADMINS,
    USER_CITIES,
    USER_CODES,
    USER_TIMEZONES,
    VISION_MODEL,
    is_qwen3,
    llm_timeout,
)
from deps import REDIS_CLIENT
from llm.embed_router import embed_route
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
    call_llm_async_bg,
    filter_think_chunk,
    fmt_now_fr,
    get_logger,
    get_session_summary_data,
    get_sticky_rag,
    rel_time_fr,
    set_session_summary_data,
    set_sticky_rag,
)
from apns import send_apns_push
from llm.client import describe_images, stream_openai
from llm.router import llm_route
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
    search_weather,
    search_web,
)

logger = get_logger("jarvis-chat")

router = APIRouter()

# Fire-and-forget background tasks must be kept alive until completion.
# Without a strong reference, the GC can collect the Task before it runs.
_background_tasks: set = set()


def _spawn_bg(coro) -> None:
    """Fire-and-forget task with a strong reference until completion.

    Every background asyncio.create_task in this module MUST go through here —
    a bare create_task can be garbage-collected before running (see comment above),
    silently losing post_analysis / session-summary work."""
    t = asyncio.create_task(coro)
    _background_tasks.add(t)
    t.add_done_callback(_background_tasks.discard)


def _trim_history_to_budget(hist: list[dict], budget_tokens: int) -> list[dict]:
    """Keep the most recent messages within token budget, always preserving the last exchange."""
    if not hist:
        return []
    budget_chars = budget_tokens * 4
    # Always keep the last exchange (last 2 messages = preceding user + last assistant).
    min_keep = min(2, len(hist))
    guaranteed = hist[-min_keep:]
    used = sum(len(m.get("content", "")) for m in guaranteed)
    older: list[dict] = []
    for msg in reversed(hist[:-min_keep]):
        cost = len(msg.get("content", ""))
        if used + cost > budget_chars:
            break
        older.append(msg)
        used += cost
    return list(reversed(older)) + guaranteed


async def _update_session_summary(user_code: str, session_id: str) -> None:
    """Post-response background task: compress conversation history into a rolling summary.

    Self-contained: fetches state from Redis, checks threshold, generates if needed.
    Uses the PRIMARY model — runs after response is sent, no GPU conflict.
    Trigger: uncovered messages (ts > last_ts) exceed HIST_CONV_SUMMARIZE_THRESHOLD chars.
    Tracks coverage by timestamp so it works correctly with LTRIM-capped lists.
    """
    try:
        summary_data = get_session_summary_data(user_code, session_id)
        # last_ts=0.0 covers legacy summaries (msg_count-based) — treats all as uncovered,
        # which forces a fresh summary with correct last_ts on the next run.
        last_ts = summary_data.get("last_ts", 0.0) if summary_data else 0.0
        existing_text = summary_data["text"] if summary_data else ""

        all_messages = get_conversation(user_code, session_id)
        uncovered = [m for m in all_messages if m.get("ts", 0.0) > last_ts]
        if not uncovered:
            return

        uncovered_chars = sum(len(m.get("content", "")) for m in uncovered)
        # Trigger as soon as the uncovered text exceeds the INJECTION budget:
        # anything beyond HIST_CONV_TOKEN_BUDGET is dropped by _trim_history_to_budget,
        # so it must be summarized or it becomes invisible to the model (context hole
        # between the 4000-char injection cap and the old 6000-char trigger).
        _trigger_chars = min(HIST_CONV_SUMMARIZE_THRESHOLD, HIST_CONV_TOKEN_BUDGET) * 4
        if uncovered_chars <= _trigger_chars:
            return

        dropped_text = "\n".join(
            f"{'Utilisateur' if m['role'] == 'user' else 'Jarvis'} : {m.get('content', '')[:300]}"
            for m in uncovered
        )
        if not dropped_text.strip():
            return
        existing_block = (
            f"Résumé précédent :\n{existing_text}\n\n" if existing_text else ""
        )
        prompt = get_prompt("SESSION_SUMMARY_PROMPT").format(
            existing_block=existing_block,
            dropped_text=dropped_text,
        )
        # call_llm_async_bg : cède le GPU si l'utilisateur enchaîne un message
        # (call_llm_async prenait le lock en priorité chat malgré le commentaire
        # « no GPU conflict »). json_response=False : prose attendue — le défaut
        # True active l'early-stop au premier objet {...} et tronquait le résumé
        # dès que la conversation contenait du JSON ou du code.
        content = await call_llm_async_bg(
            [{"role": "user", "content": prompt}],
            model=PRIMARY_MODEL,
            api_url=PRIMARY_API_URL,
            api_key=PRIMARY_API_KEY,
            temperature=DEFAULT_TEMP,
            max_tokens=SESSION_SUMMARY_TOKENS,
            json_response=False,
            no_think=True,
            timeout=llm_timeout(SESSION_SUMMARY_TOKENS),
        )
        if content and content.strip():
            new_last_ts = uncovered[-1].get("ts", 0.0)
            set_session_summary_data(user_code, session_id, content.strip(), new_last_ts)
            logger.debug(
                "session summary updated: %s/%s (%d uncovered msgs, last_ts=%.3f)",
                user_code,
                session_id,
                len(uncovered),
                new_last_ts,
            )
    except Exception as exc:
        logger.warning("session summary update failed: %s", exc)


# Derived fetch limit: enough messages for injection + one threshold's worth of summarization.
# Not a config variable — purely an implementation detail derived from existing constants.
_HIST_FETCH_N = max(
    HIST_CONV_TOKEN_BUDGET // 50, HIST_CONV_SUMMARIZE_THRESHOLD // 50, 10
)


# user_codes whose Redis profile has been initialised this process lifetime.
# Avoids a Redis hget on every request — populated on first message per user.
_profile_initialised: set[str] = set()

# Compiled once at module level — used in STEP 5 URL detection.
_URL_RE = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]{5,}')

# ── Web auto-trigger signals ────────────────────────────────────────────────────
# Factual patterns that strongly suggest external info is needed even when the
# router classified the message as "memory" (conversational phrasing).

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
    raw = raw.replace("</think >", "</think>")  # Qwen3.6 hallucination normalization
    if started_in_think:
        if "</think>" in raw:
            return raw.rsplit("</think>", 1)[1].strip()
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
    thinking_budget: int
    session_id: str
    t0: float  # timer global requête (pour les logs TTFT)
    start: float  # timer démarrage appel LLM (pour duration_ms)
    rag_chunks: list
    safe_web: list
    user_code: str
    raw_user_content: str
    original_message: str
    history_user_content: str  # req.message stripped of injected doc — saved to Redis/convlog
    image_parts: list  # non-empty when vision processing is deferred to _sse_stream


async def _complete_after_disconnect(ctx: _SseCtx) -> None:
    """LLM completion après déconnexion client (ex. écran veille pendant vision).

    Cas : client déconnecté avant le premier token SSE (raw_chunks=[]).
    On relance un appel LLM non-streaming, sauvegarde user+assistant dans Redis,
    puis push APNS pour réveiller l'app iOS.
    """
    try:
        logger.info(
            "post-disconnect completion: user=%s session=%s", ctx.user_code, ctx.session_id
        )
        if ctx.image_parts:
            logger.info("post-disconnect: processing deferred vision (%d parts)", len(ctx.image_parts))
            _img_desc = await describe_images(ctx.image_parts, ctx.original_message)
            ctx.image_parts = []
            if _img_desc:
                _new_raw = (
                    f"{ctx.original_message}\n\n"
                    f"<image_analysis>\n"
                    f"L'utilisateur a joint une image. Voici son analyse détaillée par le modèle vision "
                    f"— traite ces informations comme si tu avais vu l'image toi-même et réponds directement "
                    f"à la question sans demander de photo :\n\n"
                    f"{_img_desc}\n"
                    f"</image_analysis>"
                )
                _old_block = f"<user_message>\n{ctx.raw_user_content}\n</user_message>"
                _new_block = f"<user_message>\n{_new_raw}\n</user_message>"
                ctx.messages[-1]["content"] = ctx.messages[-1]["content"].replace(
                    _old_block, _new_block, 1
                )
                ctx.raw_user_content = _new_raw

        content = await call_llm_async(
            ctx.messages,
            model=ctx.use_model,
            api_url=ctx.api_url,
            api_key=ctx.api_key,
            timeout=ctx.timeout,
            no_think=ctx.no_think,
            max_tokens=ctx.max_tokens,
            thinking_budget=ctx.thinking_budget,
            json_response=False,
        )
        if not content or not content.strip():
            logger.warning("post-disconnect: LLM returned empty response, nothing saved")
            return
        # Strip think block (même logique que le path streaming)
        _in_think_started = LLM_LOCAL and is_qwen3(ctx.use_model) and not ctx.no_think
        clean = _strip_sse_response(content, _in_think_started) or content.strip()
        append_conversation_message(ctx.user_code, ctx.session_id, "user", ctx.history_user_content)
        append_conversation_message(ctx.user_code, ctx.session_id, "assistant", clean)
        logger.info("post-disconnect: saved to history (%d chars)", len(clean))
        # APNS push — réveille l'app iOS
        device_token = REDIS_CLIENT.get(f"jarvis:device:token:{ctx.user_code}")
        if device_token:
            preview = clean[:120].replace("\n", " ")
            if len(clean) > 120:
                preview += "…"
            await send_apns_push(device_token, body=preview, title="Jarvis")
        _spawn_bg(
            post_analysis(ctx.session_id, ctx.user_code, ctx.history_user_content, clean)
        )
        _spawn_bg(_update_session_summary(ctx.user_code, ctx.session_id))
    except Exception as exc:
        logger.warning("post-disconnect completion failed: %s", exc)


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
    raw_chunks: list[str] = []
    try:
        # ── Deferred vision processing ──────────────────────────────────────
        # describe_images can take 2+ min for large images. Running it here,
        # AFTER the StreamingResponse is returned, lets iOS receive this first
        # SSE event and reset its connection timeout before processing starts.
        if ctx.image_parts:
            yield "data: " + json.dumps({"think": "📷 Analyse de l'image en cours…"}) + "\n\n"
            _vision_task = asyncio.create_task(
                describe_images(ctx.image_parts, ctx.original_message)
            )
            try:
                # Send keepalives every 15 s so iOS doesn't close the connection
                # (vision can take 2+ min for large images).
                while not _vision_task.done():
                    try:
                        await asyncio.wait_for(asyncio.shield(_vision_task), timeout=15)
                    except asyncio.TimeoutError:
                        yield "data: " + json.dumps({"think": "…"}) + "\n\n"
                _img_desc = _vision_task.result()
            except asyncio.CancelledError:
                _vision_task.cancel()
                raise  # ctx.image_parts still set → _complete_after_disconnect handles it
            ctx.image_parts = []
            if _img_desc:
                logger.info("vision: image described (%d chars) [deferred]", len(_img_desc))
                _new_raw = (
                    f"{ctx.original_message}\n\n"
                    f"<image_analysis>\n"
                    f"L'utilisateur a joint une image. Voici son analyse détaillée par le modèle vision "
                    f"— traite ces informations comme si tu avais vu l'image toi-même et réponds directement "
                    f"à la question sans demander de photo :\n\n"
                    f"{_img_desc}\n"
                    f"</image_analysis>"
                )
                _old_block = f"<user_message>\n{ctx.raw_user_content}\n</user_message>"
                _new_block = f"<user_message>\n{_new_raw}\n</user_message>"
                ctx.messages[-1]["content"] = ctx.messages[-1]["content"].replace(
                    _old_block, _new_block, 1
                )
                ctx.raw_user_content = _new_raw
            else:
                logger.warning("vision: describe_images returned empty [deferred]")

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
        # When </think> arrives as a lone single-token chunk (after=""), we can't
        # tell real vs false close until the next chunk reveals what follows.
        _pending_close = False

        async for chunk in stream_openai(
            ctx.messages,
            ctx.use_model,
            ctx.api_url,
            ctx.api_key,
            ctx.timeout,
            no_think=ctx.no_think,
            session_id=ctx.session_id,
            max_tokens=ctx.max_tokens,
            thinking_budget=ctx.thinking_budget,
        ):
            raw_chunks.append(chunk)

            # ── Think filtering ─────────────────────────────────────────────
            # filter_think_chunk splits each chunk into visible text and
            # think-block content. Think fragments are forwarded as a
            # separate SSE event so the iOS client can display them as a
            # live ticker without mixing them into the chat bubble.
            if _pending_close:
                _pending_close = False
                if chunk and chunk[0] != "\n":
                    # False close confirmed: model used </think> as notation.
                    # Retroactively route the deferred tag to think and stay in.
                    in_think = True
                    yield f"data: {json.dumps({'think': '</think>'})}\n\n"
                    clean, think_frag, in_think = filter_think_chunk(chunk, True)
                else:
                    # Real close confirmed by following \n (or empty next chunk).
                    clean, think_frag, in_think = filter_think_chunk(chunk, False)
            else:
                prev_in_think = in_think
                clean, think_frag, in_think = filter_think_chunk(chunk, in_think)
                # Single-token </think> boundary: filter returned ("", "", False)
                # with no content on either side — defer until next chunk decides.
                if prev_in_think and not in_think and not clean and not think_frag:
                    _pending_close = True
                    continue

            if think_frag:
                yield f"data: {json.dumps({'think': think_frag})}\n\n"

            if first_chunk and clean:
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
        response_text = _strip_sse_response("".join(raw_chunks), in_think_started)
        if not response_text:
            logger.warning(
                "sse_stream: empty visible response — session=%s started_in_think=%s "
                "raw_len=%d (thinking truncation or empty generation — turn NOT saved)",
                ctx.session_id,
                in_think_started,
                len("".join(raw_chunks)),
            )
            # Budget-forced </think> mid-sentence can cause Qwen3.6 to emit EOS
            # immediately — give the iOS client a visible fallback instead of silence.
            # Zero chunks received means the LLM call itself failed (API/infra error),
            # not a thinking truncation — don't blame the reasoning budget for it.
            if raw_chunks:
                _fallback = "⚠️ Réponse incomplète — budget de réflexion dépassé. Reformule ta question."
            else:
                _fallback = "⚠️ Aucune réponse du modèle (erreur d'inférence ou d'API). Réessaie dans un instant."
            yield f"data: {json.dumps({'content': _fallback})}\n\n"
        if response_text:
            append_conversation_message(
                ctx.user_code, ctx.session_id, "user", ctx.history_user_content
            )
            append_conversation_message(
                ctx.user_code, ctx.session_id, "assistant", response_text
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
        # Schedule background tasks BEFORE the final yield — after it, the consumer
        # (e.g. OpenWebUI) may close the connection, which sends GeneratorExit into
        # this generator and skips any code that follows the yield.
        _spawn_bg(
            post_analysis(
                ctx.session_id, ctx.user_code, ctx.history_user_content, response_text
            )
        )
        _spawn_bg(_update_session_summary(ctx.user_code, ctx.session_id))
        yield f"data: {_done_payload}\n\n"
    except asyncio.CancelledError:
        logger.info("Client disconnected")
        _saved = False
        if raw_chunks:
            try:
                response_text = _strip_sse_response("".join(raw_chunks), in_think_started)
                if response_text:
                    append_conversation_message(
                        ctx.user_code, ctx.session_id, "user", ctx.history_user_content
                    )
                    append_conversation_message(
                        ctx.user_code, ctx.session_id, "assistant", response_text
                    )
                    _spawn_bg(
                        post_analysis(
                            ctx.session_id,
                            ctx.user_code,
                            ctx.history_user_content,
                            response_text,
                        )
                    )
                    logger.info(
                        "Saved response to Redis after disconnect (%d chars)",
                        len(response_text),
                    )
                    _saved = True
            except Exception as _save_err:
                logger.warning("Failed to save on disconnect: %s", _save_err)
        if not _saved:
            # Two cases: no tokens at all (zero chunks), or chunks received but all
            # inside the <think> block with no visible response extracted yet
            # (client went to sleep mid-reasoning on a long thinking request).
            # _complete_after_disconnect re-runs the LLM non-streaming and pushes
            # the result via APNS to wake the iOS app.
            reason = "mid-think disconnect" if raw_chunks else "no tokens received"
            logger.info("post-disconnect completion triggered (%s)", reason)
            _spawn_bg(_complete_after_disconnect(ctx))


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


def _handle_agent_task(
    req: ChatRequest,
    user_code: str,
) -> "StreamingResponse | dict | None":
    """Fast-track : « tâche agent: <objectif> » met une tâche en file, sans appel LLM.

    Déclenchement par préfixe explicite, et NON par le routeur d'intentions. C'est
    délibéré : un faux positif du routeur lancerait une tâche autonome de plusieurs
    minutes sur une phrase anodine. Le préfixe, lui, ne se déclenche jamais par accident.
    Le passage par l'intent (embed_router) est prévu en Phase 4, une fois la fiabilité de
    la boucle mesurée.

    Retourne None si le message n'est pas une commande agent — le pipeline normal suit.
    """
    # NFC d'abord : l'objectif est ensuite découpé sur `stripped` avec la longueur du
    # préfixe DÉSACCENTUÉ. En NFD, « tâche » compte 6 caractères et non 5 — le découpage
    # serait décalé d'un cran. macOS produit du NFD ; on ne peut pas supposer la forme.
    stripped = unicodedata.normalize("NFC", req.message.strip())
    lowered = unicodedata.normalize("NFD", stripped.lower())
    lowered = "".join(c for c in lowered if unicodedata.category(c) != "Mn")

    # Les préfixes anglais sont acceptés QUELLE QUE SOIT `JARVIS_LANG`, en plus des
    # français. Une commande n'est pas de la prose : reconnaître les deux ne coûte rien,
    # évite un lexique par langue pour quatre motifs, et rend le fast-track utilisable par
    # un anglophone sur une instance française comme l'inverse.
    for prefix in (
        "tache agent:", "tache agent :", "agent:", "agent :",
        "agent task:", "agent task :", "task:", "task :",
    ):
        if lowered.startswith(prefix):
            objective = stripped[len(prefix):].strip()
            break
    else:
        return None

    if not AGENT_ENABLED:
        return _instant_reply(req, user_code, "Le mode agent est désactivé (AGENT_ENABLED=false).")

    from agent import create_task, list_tasks

    # Consultation depuis l'iPhone : sans ça la fonctionnalité serait à sens unique,
    # curl n'étant pas une option sur un téléphone.
    if objective.lower() in (
        "statut", "status", "etat", "état", "où en es-tu", "ou en es-tu",
        "state", "where are you", "how is it going",
    ):
        tasks = list_tasks(5, user_code=user_code)
        if not tasks:
            return _instant_reply(req, user_code, "Tu n'as aucune tâche agent enregistrée.")
        lines = [
            f"· {t['objective'][:60]} — {t['status']} ({t['steps']} pas)"
            + (f" → {t['result'][:100]}" if t.get("result") else "")
            for t in tasks
        ]
        return _instant_reply(req, user_code, "Tes dernières tâches :\n" + "\n".join(lines))

    if user_code not in USER_ADMINS:
        return _instant_reply(req, user_code, "Le mode agent est réservé aux administrateurs.")

    if len(objective) < 10:
        return _instant_reply(
            req, user_code,
            "Il me faut un objectif un peu plus explicite pour partir seul là-dessus.",
        )

    task = create_task(user_code, objective)
    return _instant_reply(
        req, user_code,
        f"C'est parti — je m'en occupe en arrière-plan (tâche {task['id'][:8]}). "
        f"Je te préviens quand j'ai terminé. « tâche agent: statut » pour suivre.",
    )


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
    # "oui" seul confirme aussi — réponse naturelle à « Confirmes ? » (la regex
    # d'origine exigeait confirme/ok/yes et laissait "oui" retomber dans le pipeline).
    if re.match(
        r"^(oui|yes|yep)[,\s]*(confirme[sz]?|confirm|ok|okay)?\s*[!.]?$"
        r"|^(confirme[sz]?|confirm|ok|okay|sure|go ahead)\s*[!.]?$",
        msg_lower,
    ):
        try:
            pending = json.loads(pending_raw)
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            logger.warning(
                "pending_calendar_action corrupted for %s — clearing", user_code
            )
            REDIS_CLIENT.delete(f"jarvis:{user_code}:pending_calendar_action")
            return None  # fall through to the normal pipeline
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

    if re.match(
        r"^(non|no|nope)[,\s]*(annule[r]?|cancel)?\s*[!.]?$"
        r"|^(annule[r]?|cancel|forget it|never mind)\s*[!.]?$",
        msg_lower,
    ):
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
    if not PRIMARY_API_KEY and not LLM_LOCAL:
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

    # ── 1a. Fast-track agent : « tâche agent: … » ──────────────────────────
    # Testé en PREMIER, avant même le calendrier en attente : c'est une commande
    # explicite, elle ne doit pas être avalée par un « oui/non » attendu ailleurs.
    if (result := _handle_agent_task(req, user_code)) is not None:
        return result

    # ── 1b. Pending calendar action: confirm or cancel ──────────────────────
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
        # ── 1c. Calendar write (keyword → LLM extraction, no router needed) ─
        # Only checked when there is no pending action to avoid overwriting it.
        if (
            result := await _handle_calendar_write(req, user_code, _google_available)
        ) is not None:
            return result

    # ── Inline document injection — detect early so routers see only the query ─
    _has_injected_doc = "[Document injecté" in req.message
    _history_user_msg = (
        req.message.split("\n\n[Document injecté")[0].strip()
        if _has_injected_doc
        else req.message
    )

    # ════════════════════════════════════════════════════════════════════════
    # STEP 2 — EMBEDDING ROUTER — fast cosine-similarity intent classifier
    # ~2-5 ms. Skips the LLM router (~1.3 s) when confident.
    # Returns None when score is low or ambiguous → LLM router takes over.
    # ════════════════════════════════════════════════════════════════════════
    # Dernier tour assistant, lu AVANT le fast-path (Redis local, < 1 ms). Il était
    # auparavant lu dans la branche `else`, donc absent du chemin embedding — or c'est
    # la moitié du trafic, et c'est cette donnée qui porte l'antécédent des messages
    # elliptiques (« confirme », « la couronne »). Sans elle dans les échantillons, le
    # professeur du prochain ré-étiquetage aura le même angle mort et rabattra ces
    # messages sur `web`, comme mesuré.
    _last_jarvis_for_router: str | None = None
    try:
        _tail = REDIS_CLIENT.lrange(f"chat:{user_code}:{req.session_id}", -4, -1)
        _last_jarvis_for_router = next(
            (json.loads(m).get("content", "")[:600]
             for m in reversed(_tail)
             if json.loads(m).get("role") == "assistant"),
            None,
        )
    except Exception:
        pass

    _embed_result = embed_route(_history_user_msg, google_available=_google_available,
                                last_jarvis=_last_jarvis_for_router)

    # ════════════════════════════════════════════════════════════════════════
    # STEP 3 — LLM ROUTER + dynamic prefix + conversation history (parallel)
    # get_conversation is gathered here so it overlaps with the LLM router call
    # instead of running sequentially after it.
    # ════════════════════════════════════════════════════════════════════════
    user_name = USER_CODES.get(user_code)
    system_prompt = build_system_prompt(user_code)

    # Speculative memory search — started immediately, in parallel with routing.
    # Memory is the most common intent (~80 % of requests); the embedding call
    # (~2–3 s on CPU) was previously sequential with routing, adding 2–3 s to TTFT.
    # If routing decides use_memory=False the result is discarded (cost: one
    # embedding call, ~2–3 s CPU, no GPU impact).
    # Guard: skip for very short messages (< 15 chars) — almost always small-talk
    # ("ok", "merci", "oui"). asyncio cancel() doesn't stop the underlying thread,
    # so avoiding the launch entirely is the only way to prevent wasted CPU.
    _spec_mem_skipped = len(_history_user_msg.strip()) < 15
    _spec_mem_task: asyncio.Task = asyncio.ensure_future(
        _empty() if _spec_mem_skipped
        else async_search_memory(user_code, _history_user_msg, 5)
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
        if _embed_result.use_small_talk:
            # Small talk (acquiescements purs) — pas de profil, pas de recall mémoire.
            # Seul l'historique de conversation suffit.
            _spec_mem_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await _spec_mem_task
            tz = USER_TIMEZONES.get(user_code, "Europe/Paris")
            dynamic_prefix = f"Date : {fmt_now_fr(tz)}."
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
            if not _embed_result.use_memory:
                _spec_mem_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await _spec_mem_task
                _prefetched_memory_coro = asyncio.create_task(_empty())
            else:
                _prefetched_memory_coro = _spec_mem_task
            _gather_ep = await asyncio.gather(
                asyncio.to_thread(
                    build_dynamic_prefix,
                    req.session_id,
                    user_code,
                    user_name or "",
                    req.voice_mode,
                    _rich_intent,
                    _rich_intent,  # include_opinions, include_suggestions
                    _history_user_msg,
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
        # _last_jarvis_for_router est lu plus haut, avant le fast-path.
        _gather1 = await asyncio.gather(
            asyncio.to_thread(
                build_dynamic_prefix,
                req.session_id,
                user_code,
                user_name or "",
                req.voice_mode,
                user_message=_history_user_msg,
            ),
            llm_route(_history_user_msg, google_available=_google_available, last_jarvis=_last_jarvis_for_router),
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

    # ── Inline document injection — freeze intents ─────────────────────────
    # _has_injected_doc / _history_user_msg already computed before STEP 2 so
    # embed_route / llm_route / async_search_memory see only the user query.
    if _has_injected_doc:
        use_rag = False
        use_web_auto = False
        use_gmail = False
        use_calendar = False
        use_memory = True

    # ── Model selection ─────────────────────────────────────────────────────
    # Always PRIMARY infrastructure — reasoning is handled via no_think flag only.
    use_model = req.model or PRIMARY_MODEL
    _use_api_url = PRIMARY_API_URL
    _use_api_key = PRIMARY_API_KEY
    _use_timeout = PRIMARY_TIMEOUT

    # ── no_think for simple intents (memory/conversation) ──────────────────
    # Complex intents (web, RAG, reasoning) keep chain-of-thought.
    # Typical saving: ~4 s of TTFT on conversational exchanges.
    _complex_intents = use_rag or use_web_auto or req.use_web or req.use_rag or _has_injected_doc
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

    # ── Vision: placeholder for _sse_stream ────────────────────────────────
    # describe_images runs inside _sse_stream (keepalive path) — raw_user_content
    # gets the "non traitée" fallback here; _sse_stream replaces it after vision.

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
    # Ne pas scanner les URLs dans un document injecté — elles appartiennent au
    # code source, pas à la requête utilisateur (évite les fetches localhost/internes).
    _inline_urls = [] if _has_injected_doc else _URL_RE.findall(req.message)

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
        # Si la recherche spéculative a été sautée (message < 15 chars) mais que le
        # routeur veut la mémoire (ex. « et mon vélo ? »), on la lance ici — sinon
        # le recall était silencieusement perdu pour les messages courts.
        async_search_memory(user_code, _history_user_msg, 5)
        if (use_memory and _spec_mem_skipped)
        else _prefetched_or_empty(_prefetched_memory)
        if use_memory
        else _empty(),
        search_weather(_weather_query)
        if use_weather_auto
        else search_web(req.message, original_message=req.message)
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

    # ── Sticky RAG: store fresh results / inject cached on memory-only turns ─
    if rag_chunks:
        # Fresh RAG call succeeded → persist for subsequent turns.
        set_sticky_rag(user_code, req.session_id, rag_chunks)
    elif use_memory and not use_rag:
        # Memory turn, no fresh RAG → re-inject last stored document context.
        _sticky = get_sticky_rag(user_code, req.session_id)
        if _sticky:
            rag_chunks = _sticky
            logger.info(
                "chat: sticky RAG re-injected (%d chunks) for %s/%s",
                len(rag_chunks),
                user_code,
                req.session_id,
            )

    logger.debug(
        "[TTFT] gather2 done (all context sources resolved) — %.3fs", time.time() - _t0
    )


    # Inject session-gap: timestamps are stripped when building the messages
    # list, so the model has no way to infer temporal distance from history
    # entries alone. This phrase prevents greeting again mid-conversation.
    if hist:
        _last_ts = hist[-1].get("ts")
        if _last_ts:
            _gap = time.time() - _last_ts
            _gap_txt = "moins d'une minute" if _gap < 60 else rel_time_fr(_last_ts)
            _gap_line = f"Dernier message : {_gap_txt}."
            dynamic_prefix = (
                (dynamic_prefix + f"\n\n{_gap_line}") if dynamic_prefix else _gap_line
            )

    # ════════════════════════════════════════════════════════════════════════
    # STEP 6 — MESSAGE ASSEMBLY — context + prefix + history → final prompt
    # ════════════════════════════════════════════════════════════════════════

    # Write user name once per process lifetime — skips Redis hget on every request.
    # to_thread : update_user_profile peut déclencher un appel LLM sync de dedup de
    # clé — exécuté inline il bloquerait l'event loop (et donc le streaming en cours).
    if user_name and user_code not in _profile_initialised:
        if not REDIS_CLIENT.hget(f"user:{user_code}:profile", "name"):
            await asyncio.to_thread(update_user_profile, user_code, "name", user_name)
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
    # Only for deep reasoning — web/RAG context is self-explanatory.
    reasoning_hint = ""
    if llm_result and llm_result.use_reasoning:
        reasoning_hint = "\n\nAnalyse en profondeur — pèse les options et les risques avant de conclure."

    # Placeholder when images are present — _sse_stream replaces this block with
    # the actual <image_analysis> once describe_images completes.
    raw_user_content = (
        f"{req.message}\n\n[Image jointe reçue mais non traitée par le modèle vision"
        f" — ne demande pas de photo, réponds sans l'avoir vue.]"
        if req.image_parts
        else req.message
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

    # When a session summary exists, inject only the messages NOT yet covered by it.
    # Coverage tracked by timestamp (last_ts) — immune to LTRIM-capped list rotation.
    if _session_summary:
        _last_ts = _summary_data.get("last_ts", 0.0)
        _uncovered_hist = [m for m in hist if m.get("ts", 0.0) > _last_ts]
        # Prepend the last user+assistant exchange from covered history if not already
        # present, so the model sees the full prior turn, not just its own last response.
        has_assistant = any(m["role"] == "assistant" for m in _uncovered_hist)
        if not has_assistant:
            _covered = [m for m in hist if m.get("ts", 0.0) <= _last_ts]
            last_asst_idx = next(
                (i for i, m in enumerate(reversed(_covered)) if m["role"] == "assistant"),
                None,
            )
            if last_asst_idx is not None:
                asst_pos = len(_covered) - 1 - last_asst_idx
                anchor = _covered[asst_pos - 1 : asst_pos + 1] if asst_pos > 0 else [_covered[asst_pos]]
                _uncovered_hist = anchor + _uncovered_hist
        hist_slice = _trim_history_to_budget(_uncovered_hist, HIST_CONV_TOKEN_BUDGET)
    else:
        hist_slice = _trim_history_to_budget(hist, HIST_CONV_TOKEN_BUDGET)

    # Build the user message.
    # Without summary: standard path — hist_slice as separate messages keeps the chat format
    # intact and maximises LRU cache reuse.
    # With summary: inline hist_slice as text so the model reads in chronological order —
    # summary (old) → recent exchanges → context → question — with no fake assistant turn.
    msg_parts = []
    if dynamic_prefix:
        msg_parts.append(dynamic_prefix)
    if _session_summary:
        # Balises en français, comme toutes les autres du contexte injecté — la règle est
        # posée dans prompts_en.py (« The XML tag names stay French ») et ces deux-là y
        # échappaient. Elles ne sont citées par aucun prompt, seulement écrites ici, donc
        # renommables sans rien casser. Ce n'est pas de la cosmétique : deux balises
        # anglaises au milieu d'un contexte français sont deux repères de moins pour le
        # modèle. (`<context>`, lui, reste tel quel : SYSTEM_BASE le cite par son nom.)
        msg_parts.append(
            "<resume_conversation>\n" + _session_summary + "\n</resume_conversation>"
        )
        if hist_slice:
            # Étiquettes de locuteur : deux noms, pas deux pronoms. Dans une transcription
            # verbatim, « Utilisateur » et « Jarvis » désignent qui a parlé sans ambiguïté,
            # là où un « moi »/« toi » se heurterait au fait que ce bloc arrive dans le tour
            # utilisateur. Le point de vue à la première personne vit dans le résumé
            # (SESSION_SUMMARY_PROMPT), qui est un souvenir ; ceci est un relevé.
            _role_label = {"user": "Utilisateur", "assistant": "Jarvis"}
            _hist_lines = "\n".join(
                f"{_role_label.get(m['role'], m['role'])} : {m['content']}"
                for m in hist_slice
            )
            msg_parts.append("<echanges_recents>\n" + _hist_lines + "\n</echanges_recents>")
    if assembled:
        msg_parts.append(assembled)
    if _project_detail_block:
        msg_parts.append(_project_detail_block)
    if reasoning_hint:
        msg_parts.append(reasoning_hint.strip())
    msg_parts.append("<user_message>\n" + raw_user_content + "\n</user_message>")
    user_content = "\n\n".join(msg_parts)

    messages = [{"role": "system", "content": system_prompt}]
    if not _session_summary:
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
        MAX_TOKENS_NO_THINK
        if chat_no_think
        else MAX_TOKENS_REASONING
        if _use_reasoning
        else MAX_TOKENS_SYNTHESIS
    )
    _thinking_budget = (
        0 if chat_no_think
        else THINKING_BUDGET_MEDIUM if _use_reasoning
        else THINKING_BUDGET_COMPACT
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
            thinking_budget=_thinking_budget,
            session_id=req.session_id,
            t0=_t0,
            start=start,
            rag_chunks=rag_chunks,
            safe_web=_safe_web,
            user_code=user_code,
            raw_user_content=raw_user_content,
            original_message=req.message,
            history_user_content=_history_user_msg,
            image_parts=req.image_parts if VISION_MODEL else [],
        )
        return StreamingResponse(
            _sse_stream(ctx),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    # ── JSON response (non-streaming) ───────────────────────────────────────
    # Image analysis requires SSE streaming (keepalive during vision processing).
    # Non-streaming + image is not a supported combination — reject explicitly.
    if req.image_parts:
        raise HTTPException(400, "Image analysis requires stream=true")

    resp = await call_llm_async(
        messages,
        model=use_model,
        api_url=_use_api_url,
        api_key=_use_api_key,
        timeout=_use_timeout,
        no_think=chat_no_think,
        max_tokens=_max_tokens,
        thinking_budget=_thinking_budget,
    )

    append_conversation_message(user_code, req.session_id, "user", _history_user_msg)
    append_conversation_message(user_code, req.session_id, "assistant", resp)
    ms = int((time.time() - start) * 1000)

    _spawn_bg(post_analysis(req.session_id, user_code, _history_user_msg, resp))
    _spawn_bg(_update_session_summary(user_code, req.session_id))

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

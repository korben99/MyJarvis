"""
routes/proxy.py — OpenAI-compatible proxy (/v1/*)
==================================================
Allows Open WebUI (and any OpenAI client) to talk to Jarvis.
Auth: set Jarvis user code as API key, or use X-OpenWebUI-User-Email header.
Session: derived from user_code + first user message (stable per thread).
"""

import hashlib
import json
import os
import re
import time
import uuid
from typing import Optional

from config import (
    EMAIL_TO_CODE,
    LLM_LOCAL,
    OWUI_MAX_DOC_CHARS,
    PRIMARY_API_KEY,
    PRIMARY_API_URL,
    PRIMARY_MODEL,
    ROUTER_API_KEY,
    ROUTER_API_URL,
    ROUTER_MODEL,
    USER_CODES,
)
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from helpers import get_logger
from llm.client import stream_openai
from pydantic import BaseModel

from routes.chat import ChatRequest, chat

logger = get_logger("jarvis-proxy")
router = APIRouter()


class _OAIMessage(BaseModel):
    role: str
    # content optionnel : un message assistant porteur de tool_calls a content=null, et
    # un message role="tool" porte son résultat + tool_call_id. Sans ça, tout agent de
    # code se prend un 422 dès son second tour.
    content: str | list | None = None  # list for multipart (image + text) from Open WebUI
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


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
        p.get("text", "")
        for p in content
        if isinstance(p, dict) and p.get("type") == "text"
    ).strip()
    images = [
        p for p in content if isinstance(p, dict) and p.get("type") == "image_url"
    ]
    return text, images


# ── OpenWebUI system-request detection ────────────────────────────────────────
# OpenWebUI fires its own LLM calls for UI tasks (title generation, follow-up
# suggestions…).  These never go through the Jarvis pipeline — handled here at
# the proxy layer so chat.py stays OpenWebUI-agnostic.
_OWUI_SYSTEM_KEYWORDS = (
    "### task:",
    "generate a title",
    "relevant follow",
    "follow-up question",
    "followup question",
    "questions de suivi",
)


async def _owui_system_stream(message: str, req_id: str, created: int):
    """
    Respond to an OpenWebUI system request (title, suggestions…) using the
    router model, emitting OpenAI SSE directly — no Jarvis-SSE intermediate.
    """
    model = ROUTER_MODEL or PRIMARY_MODEL
    api_url = ROUTER_API_URL if ROUTER_MODEL else PRIMARY_API_URL
    api_key = ROUTER_API_KEY if ROUTER_MODEL else PRIMARY_API_KEY

    async for chunk in stream_openai(
        [{"role": "user", "content": message}],
        model,
        api_url,
        api_key,
        no_think=True,
    ):
        yield (
            "data: "
            + json.dumps(
                {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "jarvis",
                    "choices": [
                        {"index": 0, "delta": {"content": chunk}, "finish_reason": None}
                    ],
                }
            )
            + "\n\n"
        )
    yield (
        "data: "
        + json.dumps(
            {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "jarvis",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        + "\n\n"
    )
    yield "data: [DONE]\n\n"


# ── OpenWebUI RAG-template stripping ──────────────────────────────────────────
_OWUI_RAG_MARKER = "respond to the user query"


def _strip_owui_rag(message: str) -> str:
    """
    Detect OpenWebUI document-injection templates and convert them to a clean
    Jarvis message:  user question  +  inline document context.

    OpenWebUI injects documents via the `/v1/chat/completions` endpoint with
    this template:
        ### Task:
        Respond to the user query using the provided context...

        ### Context:
        <context>
        <source id="1">...</source>
        </context>

        ### Query:
        [actual user question]

    Without this cleaning the message matches _OPENWEBUI_KEYWORDS ("### task:")
    and gets routed to Hermes without a system prompt — causing vouvoiement and
    poor quality.

    Returns the original message unchanged for all non-matching inputs.
    """
    if _OWUI_RAG_MARKER not in message.lower():
        return message

    # Extract the real user question from ### Query: section
    query_match = re.search(
        r"### Query:\s*\n(.+?)(?:\n###|\Z)", message, re.DOTALL | re.IGNORECASE
    )
    if not query_match:
        # Template detected but malformed — pass through as-is (safe fallback)
        logger.warning("_strip_owui_rag: ### Query: section not found, passing through")
        return message

    query = query_match.group(1).strip()
    # Strip OWUI default-template <query>…</query> wrapper (present when no custom template)
    query = re.sub(r"^\s*<query>\s*|\s*</query>\s*$", "", query, flags=re.DOTALL).strip()

    # Extract <source>…</source> blocks with their attributes (OpenWebUI may inject
    # multiple chunks). Name and content are paired in a single pass: a <source> without a
    # name="…" attribute would otherwise shift the two lists out of alignment.
    _blocks = re.findall(r"<source([^>]*)>(.*?)</source>", message, re.DOTALL)
    _names = []
    for _attrs, _ in _blocks:
        _m = re.search(r'\bname="([^"]+)"', _attrs)
        _names.append(_m.group(1) if _m else "")

    # Drop sources whose extracted text is blank. OpenWebUI emits a <source> carrying the
    # filename even when its extractor found nothing — a PDF made of images has no text
    # layer and yields whitespace, with status "completed" and no error anywhere. Without
    # this guard the prompt still carried "[Document injecté — fichier.pdf]" followed by
    # nothing, so the model believed it had the document and answered confidently from the
    # rest of the message. Observed 2026-08-07 on 'state OT cyber small.pdf' (14 spaces).
    _blank = [n for (_, c), n in zip(_blocks, _names) if not c.strip()]
    sources = [c for _, c in _blocks if c.strip()]
    source_names = [n for (_, c), n in zip(_blocks, _names) if c.strip()]
    if _blank:
        logger.warning(
            "_strip_owui_rag: %d source(s) sans texte exploitable — non injectée(s) : %s. "
            "PDF sans couche texte ? (OCR désactivé côté OpenWebUI)",
            len(_blank),
            ", ".join(n or "sans nom" for n in _blank),
        )

    if sources:
        n_total = len(sources)
        _MAX_DOC_CHARS = OWUI_MAX_DOC_CHARS
        selected, used_chars = [], 0
        for s in sources:
            remaining = _MAX_DOC_CHARS - used_chars
            if len(s) > remaining:
                # Truncate rather than drop — critical for Full Context Mode (single large source)
                if not selected:
                    selected.append(s[:remaining])
                    used_chars = _MAX_DOC_CHARS
                break
            selected.append(s)
            used_chars += len(s)
        n = len(selected)

        header = source_names[0] if source_names else "fichier"
        if n == 1:
            original_len = len(sources[0])
            doc_body = selected[0].strip()
            truncated = original_len > _MAX_DOC_CHARS
            trunc_note = f", tronqué à {_MAX_DOC_CHARS // 1000}k chars" if truncated else ""
            doc_label = f"[Document injecté — {header}{trunc_note}]"
        else:
            truncated = n < n_total
            parts = [f"# extrait {i + 1}/{n}\n{s.strip()}" for i, s in enumerate(selected)]
            doc_body = "\n\n".join(parts)
            trunc_note = f", tronqué {n}/{n_total} extraits" if truncated else ", ordre non garanti"
            doc_label = f"[Document injecté — {header}, {n} extraits{trunc_note}]"

        clean = f"{query}\n\n{doc_label}\n{doc_body}"
        logger.debug(
            "_strip_owui_rag: stripped template → query=%r, sources=%d/%d names=%s truncated=%s",
            query[:80],
            n,
            n_total,
            source_names[:3] or [],
            truncated if n == 1 else n < n_total,
        )
        return clean

    # No <source> blocks found — return just the query
    logger.debug("_strip_owui_rag: no sources found, returning query only")
    return query


def _proxy_session_id(user_code: str, messages: list[_OAIMessage]) -> str:
    """Stable session ID: SHA-256 of user_code + first user message (first 120 chars)."""
    first_user = next((m.content for m in messages if m.role == "user"), "default")
    if isinstance(first_user, list):
        first_user, _ = _extract_content_parts(first_user)
    return hashlib.sha256(
        f"{user_code}:{(first_user or 'default')[:120]}".encode()
    ).hexdigest()[:20]


async def _translate_jarvis_sse(body_iterator, req_id: str, created: int):
    """
    Translate Jarvis SSE stream to OpenAI SSE format.
    Jarvis: data: {"content": "..."}  /  {"think": "..."}  /  {"done": true, ...}
    OpenAI: data: {"choices": [{"delta": {"content": "..."}}]}  /  data: [DONE]

    Think events are wrapped in <think>…</think> so OpenWebUI 0.4+ renders them
    as a collapsible "Thinking" panel (same mechanism as DeepSeek-R1 / QwQ).
    State machine: open <think> on first think event, close </think> on transition
    to visible content or done.
    """
    buffer = ""
    in_think = False

    def _delta(text: str) -> str:
        return f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'jarvis', 'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': None}]})}\n\n"

    async for raw in body_iterator:
        if isinstance(raw, bytes):
            raw = raw.decode()
        buffer += raw
        while "\n\n" in buffer:
            event, buffer = buffer.split("\n\n", 1)
            for line in event.splitlines():
                if not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                if "think" in data and data["think"]:
                    if not in_think:
                        yield _delta("<think>")
                        in_think = True
                    yield _delta(data["think"])

                elif "content" in data:
                    if in_think:
                        yield _delta("</think>")
                        in_think = False
                    if data["content"]:
                        yield _delta(data["content"])

                elif data.get("done"):
                    if in_think:
                        yield _delta("</think>")
                        in_think = False
                    yield (
                        f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'jarvis', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                    )
                    yield "data: [DONE]\n\n"
                    return

    # Guard: upstream ended without a done event — close think block and signal stop.
    if in_think:
        yield _delta("</think>")
    yield (
        f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'jarvis', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    )
    yield "data: [DONE]\n\n"


# Paliers d'effort exposés comme des modèles distincts — équivalent du /effort de Claude
# Code. Un client OpenAI-compatible n'a aucun champ standard pour demander plus ou moins de
# raisonnement ; le champ `model`, lui, est envoyé à chaque requête et sélectionnable à la
# volée dans OpenCode (/models, ou -m jarvis/jarvis-deep). C'est donc le seul canal propre,
# sans extension propriétaire côté client.
#   nom → (no_think, thinking_budget)
RAW_EFFORT_MODELS: dict[str, tuple[bool, int]] = {
    "jarvis-fast": (True, 0),      # aucun raisonnement — édition triviale, question factuelle
    "jarvis": (False, 3000),       # défaut
    "jarvis-deep": (False, 8000),  # refactor, diagnostic, enchaînement d'outils long
}


class _RawChatRequest(_OAIChatRequest):
    no_think: Optional[bool] = None          # override explicite du client
    thinking_budget: Optional[int] = None    # niveau d'effort demandé par le client
    tools: Optional[list] = None             # schémas d'outils au format OpenAI
    tool_choice: Optional[str | dict] = None  # accepté, non contraint côté template
    priority: Optional[str] = None           # "bg" (défaut de la route) | "chat"


@router.post("/v1/raw/chat/completions")
async def raw_chat(req: _RawChatRequest):
    """
    Bypass endpoint — appel direct à stream_local() sur PRIMARY_MODEL.
    Aucun routage Jarvis, aucune injection mémoire/RAG/état émotionnel, aucune écriture
    en mémoire. C'est l'endpoint des agents de code externes (OpenCode) : contrairement à
    /v1/chat/completions, il conserve TOUS les messages et respecte le rôle system.
    Uniquement disponible quand LLM_LOCAL=True.
    Les blocs <think>…</think> sont strippés avant d'être envoyés au client.

    Function calling : passer `tools` (schémas OpenAI) active le format natif du modèle.
    Les appels sont retraduits en `tool_calls` OpenAI (voir tool_calls.py).

    Contrôle du thinking (priorité décroissante) :
      1. no_think + thinking_budget dans le body
      2. Variable d'env RAW_NO_THINK (défaut=true)
    """
    if not LLM_LOCAL:
        raise HTTPException(503, "LLM_LOCAL non activé — endpoint raw indisponible")

    # import tardif : mlx non chargé si LLM_LOCAL=False
    from llm.local import _RAW_PROMPTS_LOG_PATH, stream_local as _stream_local
    from tool_calls import normalise_messages_for_template, parse_tool_calls

    # Messages transmis intégralement : le template a besoin de tool_calls (tours passés
    # de l'assistant) et de tool_call_id (résultats d'outil), pas seulement role+content.
    messages = []
    for m in req.messages:
        content = m.content
        if not isinstance(content, str):
            content = _extract_content_parts(content)[0] if content else ""
        message: dict = {"role": m.role, "content": content}
        if m.tool_calls:
            message["tool_calls"] = m.tool_calls
        if m.tool_call_id:
            message["tool_call_id"] = m.tool_call_id
        if m.name:
            message["name"] = m.name
        messages.append(message)
    messages = normalise_messages_for_template(messages)

    # Priorité : body explicite > palier d'effort choisi via `model` > env var.
    # Défaut sans thinking passé à false : le raisonnement est le principal
    # levier de qualité pour un agent de code (choix de l'outil, enchaînement lecture →
    # édition), et aucun client de cette route ne sait envoyer no_think.
    # "jarvis/jarvis-deep" → on ne garde que le nom du modèle.
    effort = RAW_EFFORT_MODELS.get((req.model or "").split("/")[-1].strip())

    if req.no_think is not None:
        no_think = req.no_think
    elif effort is not None:
        no_think = effort[0]
    else:
        no_think = os.getenv("RAW_NO_THINK", "false").lower() in ("yes", "true", "1")

    # Budget de réflexion OBLIGATOIRE dès que le thinking est actif : ThinkingBudgetProcessor
    # (llm_local.py) ne borne rien à 0, et le raisonnement partagerait alors max_tokens avec
    # la réponse — le modèle peut épuiser son budget en réflexion sans jamais émettre
    # l'appel d'outil. Les clients OpenAI-compatibles n'envoient pas thinking_budget.
    thinking_budget = req.thinking_budget or 0
    if not no_think and thinking_budget <= 0:
        thinking_budget = (
            effort[1] if effort and effort[1] > 0
            else int(os.getenv("RAW_THINKING_BUDGET", "3000"))
        )

    # 16000 (et non 8000) car réflexion et réponse se partagent ce budget.
    max_tokens = req.max_tokens or int(os.getenv("RAW_MAX_TOKENS", "16000"))

    # Priorité GPU basse par défaut sur CETTE route : /v1/raw est par définition le trafic
    # des agents (OpenCode), jamais celui de l'assistant — le chat passe par /chat et
    # /v1/chat/completions. L'info est donc portée par la route, aucun réglage à faire côté
    # client. Sans ça, un refactor OpenCode concurrence le chat à égalité et dégrade son TTFT.
    # Coût quand personne ne chatte : nul (aucun chat_waiter → lock pris immédiatement).
    # Attente non bornée en cas de chat soutenu : assumé, l'agent patiente.
    priority = (req.priority or os.getenv("RAW_PRIORITY", "bg")).strip().lower()
    if priority not in ("bg", "chat"):
        priority = "bg"

    req_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def _sse(text: str) -> str:
        return (
            f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'created': created, 'model': PRIMARY_MODEL, 'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': None}]})}\n\n"
        )

    def _final_chunk(finish_reason: str, delta: dict | None = None) -> str:
        return (
            f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'created': created, 'model': PRIMARY_MODEL, 'choices': [{'index': 0, 'delta': delta or {}, 'finish_reason': finish_reason}]})}\n\n"
        )

    async def _generate_with_tools():
        """Streaming bufferisé, utilisé uniquement quand des outils sont demandés.

        Un appel d'outil ne peut pas être diffusé au fil de l'eau : tant que le bloc
        <tool_call> n'est pas clos, on ne sait pas si le texte en cours est de la prose ou
        le début d'un appel. On accumule donc tout, puis on émet le contenu et les
        tool_calls en une fois. Un agent de code consomme la réponse complète de toute
        façon — la perte de réactivité est sans effet, contrairement au chat.
        """
        full = ""
        async for chunk in _stream_local(
            messages,
            model=PRIMARY_MODEL,
            no_think=no_think,
            max_tokens=max_tokens,
            temperature=req.temperature,
            thinking_budget=thinking_budget,
            skip_debug_log=False,
            debug_log_path=_RAW_PROMPTS_LOG_PATH,
            tools=req.tools,
            priority=priority,
        ):
            full += chunk

        if not no_think and "</think>" in full:
            full = full.split("</think>", 1)[1]
        text, tool_calls = parse_tool_calls(full.strip(), req.tools)

        if text:
            yield _sse(text)
        if tool_calls:
            yield _final_chunk("tool_calls", {"tool_calls": tool_calls})
        else:
            yield _final_chunk("stop")
        yield "data: [DONE]\n\n"

    async def _generate():
        buffer = ""
        think_done = no_think  # True → stream direct sans buffering

        async for chunk in _stream_local(
            messages,
            model=PRIMARY_MODEL,
            no_think=no_think,
            max_tokens=max_tokens,
            temperature=req.temperature,
            thinking_budget=thinking_budget,
            skip_debug_log=False,
            debug_log_path=_RAW_PROMPTS_LOG_PATH,
            priority=priority,
        ):
            if think_done:
                yield _sse(chunk)
                continue

            buffer += chunk

            if "</think>" in buffer:
                after = buffer.split("</think>", 1)[1]
                think_done = True
                buffer = ""
                if after:
                    yield _sse(after)
            elif len(buffer) > 100_000:  # garde-fou si </think> n'arrive jamais
                think_done = True
                yield _sse(buffer)
                buffer = ""

        yield _final_chunk("stop")
        yield "data: [DONE]\n\n"

    if req.stream:
        return StreamingResponse(
            _generate_with_tools() if req.tools else _generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming : collecte puis strip thinking
    full = ""
    async for chunk in _stream_local(
        messages, model=PRIMARY_MODEL, no_think=no_think,
        max_tokens=max_tokens, temperature=req.temperature,
        thinking_budget=thinking_budget,
        skip_debug_log=False,
        debug_log_path=_RAW_PROMPTS_LOG_PATH,
        tools=req.tools,
        priority=priority,
    ):
        full += chunk
    if not no_think and "</think>" in full:
        full = full.split("</think>", 1)[1].strip()

    text, tool_calls = parse_tool_calls(full.strip(), req.tools)
    # content=null uniquement s'il y a des tool_calls (convention OpenAI). Sans outil, on
    # renvoie la chaîne telle quelle, y compris vide : comportement inchangé pour les
    # clients existants.
    message: dict = {"role": "assistant", "content": (text or None) if tool_calls else text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": req_id, "object": "chat.completion", "created": created,
        "model": PRIMARY_MODEL,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if tool_calls else "stop",
        }],
        "usage": {},
    }


@router.get("/v1/models")
async def proxy_list_models():
    """Open endpoint — returns the Jarvis model for OpenWebUI's model selector."""
    return {
        "object": "list",
        "data": [
            {"id": "jarvis", "object": "model", "owned_by": "jarvis", "created": 0}
        ],
    }


@router.post("/v1/chat/completions")
async def proxy_chat(
    req: _OAIChatRequest,
    authorization: str = Header(default=None),
    x_openwebui_user_email: str = Header(default=None),
    x_openwebui_chat_id: str = Header(default=None),
    x_openwebui_user_id: str = Header(default=None),
):
    logger.debug(
        "proxy headers: chat_id=%r user_id=%r email=%r",
        x_openwebui_chat_id,
        x_openwebui_user_id,
        x_openwebui_user_email,
    )

    # ── Auth: OpenWebUI email header (priority) or Bearer user_code ──
    user_code = None

    if x_openwebui_user_email:
        user_code = EMAIL_TO_CODE.get(x_openwebui_user_email.lower())
        if not user_code:
            raise HTTPException(
                401, f"No Jarvis user found for email {x_openwebui_user_email!r}"
            )

    if not user_code and authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
        if token in USER_CODES:
            user_code = token
        else:
            user_code = EMAIL_TO_CODE.get(token.lower())

    if not user_code or user_code not in USER_CODES:
        raise HTTPException(
            401,
            "Unauthorized — set your email as API key in OpenWebUI, or your user code for iOS",
        )

    # ── Extract last user message ──
    last_user_msg = next(
        (m for m in reversed(req.messages) if m.role == "user" and m.content), None
    )
    if not last_user_msg:
        raise HTTPException(400, "No usable message found")

    message, image_parts = _extract_content_parts(last_user_msg.content)
    logger.info(
        "proxy: content type=%s image_parts=%d extra_keys=%s",
        type(last_user_msg.content).__name__,
        len(image_parts),
        [
            k
            for k in (last_user_msg.model_fields_set or set())
            if k not in ("role", "content")
        ],
    )
    if isinstance(last_user_msg.content, list):
        logger.info(
            "proxy: content parts=%s",
            [
                p.get("type") if isinstance(p, dict) else type(p).__name__
                for p in last_user_msg.content
            ],
        )
    if not message and not image_parts:
        raise HTTPException(400, "No usable message found")

    req_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    # ── Strip OpenWebUI RAG template (document injection via '+' button) ──
    # Converts "### Task: Respond to the user query…" templates into a clean
    # question + inline document context that goes through the full Jarvis pipeline.
    if message:
        message = _strip_owui_rag(message)

    # ── OpenWebUI system requests (title, suggestions…) — handled at proxy level ──
    # These never reach the Jarvis pipeline; chat.py stays OpenWebUI-agnostic.
    if message and any(kw in message.lower() for kw in _OWUI_SYSTEM_KEYWORDS):
        logger.debug(
            "OpenWebUI system request — handled at proxy, bypassing Jarvis pipeline"
        )
        if req.stream:
            return StreamingResponse(
                _owui_system_stream(message, req_id, created),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        return {
            "id": req_id,
            "object": "chat.completion",
            "created": created,
            "model": "jarvis",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": message},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }

    # ── Delegate to /chat ──
    # UUID natif OpenWebUI (X-OpenWebUI-Chat-Id) en priorité.
    # Fallback SHA-256 pour les clients OpenAI-compatibles tiers sans ce header.
    # iOS n'utilise pas ce proxy — il envoie son propre session_id directement via /chat.
    session_id = (
        f"owui-{x_openwebui_chat_id}"
        if x_openwebui_chat_id
        else _proxy_session_id(user_code, req.messages)
    )
    logger.info(
        "proxy session_id=%r (source=%s)",
        session_id,
        "owui-chat-id" if x_openwebui_chat_id else "sha256-first-msg",
    )
    jarvis_req = ChatRequest(
        message=message or "Que contient cette image ?",
        session_id=session_id,
        user_code=user_code,
        stream=req.stream,
        image_parts=image_parts,
    )
    response = await chat(jarvis_req)

    if isinstance(response, StreamingResponse):
        return StreamingResponse(
            _translate_jarvis_sse(response.body_iterator, req_id, created),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )
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

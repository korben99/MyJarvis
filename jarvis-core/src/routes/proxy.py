"""
routes/proxy.py — OpenAI-compatible proxy (/v1/*)
==================================================
Allows Open WebUI (and any OpenAI client) to talk to Jarvis.
Auth: set Jarvis user code as API key, or use X-OpenWebUI-User-Email header.
Session: derived from user_code + first user message (stable per thread).
"""

import hashlib
import json
import re
import time
import uuid
from typing import Optional

from config import (
    EMAIL_TO_CODE,
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
from llm_client import stream_openai
from pydantic import BaseModel

from routes.chat import ChatRequest, chat

logger = get_logger("jarvis-proxy")
router = APIRouter()


class _OAIMessage(BaseModel):
    role: str
    content: str | list  # list for multipart (image + text) from Open WebUI


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

    # Extract <source>…</source> blocks (OpenWebUI may inject multiple)
    sources = re.findall(r"<source[^>]*>(.*?)</source>", message, re.DOTALL)
    if sources:
        doc_body = "\n\n---\n".join(s.strip() for s in sources)
        clean = f"{query}\n\n[Document injecté par l'utilisateur]\n{doc_body}"
        logger.debug(
            "_strip_owui_rag: stripped template → query=%r, sources=%d",
            query[:80],
            len(sources),
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

                if "think" in data:
                    if not in_think:
                        yield _delta("<think>")
                        in_think = True
                    yield _delta(data["think"])

                elif "content" in data:
                    if in_think:
                        yield _delta("</think>")
                        in_think = False
                    yield _delta(data["content"])

                elif data.get("done"):
                    if in_think:
                        yield _delta("</think>")
                        in_think = False
                    yield (
                        f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'jarvis', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                    )
                    yield "data: [DONE]\n\n"

    # Guard: close unclosed think block if upstream stream ended without a done event.
    if in_think:
        yield _delta("</think>")


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

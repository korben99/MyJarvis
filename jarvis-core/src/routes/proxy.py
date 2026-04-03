"""
routes/proxy.py — OpenAI-compatible proxy (/v1/*)
==================================================
Allows Open WebUI (and any OpenAI client) to talk to Jarvis.
Auth: set Jarvis user code as API key, or use X-OpenWebUI-User-Email header.
Session: derived from user_code + first user message (stable per thread).
"""

import hashlib
import json
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import EMAIL_TO_CODE, USER_CODES
from helpers import get_logger
from routes.chat import ChatRequest, chat

logger = get_logger("jarvis-proxy")
router = APIRouter()


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


async def _translate_jarvis_sse(body_iterator, req_id: str, created: int):
    """
    Translate Jarvis SSE stream to OpenAI SSE format.
    Jarvis: data: {"content": "..."}  /  data: {"done": true, ...}
    OpenAI: data: {"choices": [{"delta": {"content": "..."}}]}  /  data: [DONE]
    """
    buffer = ""
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
                if "content" in data:
                    yield (
                        f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'jarvis', 'choices': [{'index': 0, 'delta': {'content': data['content']}, 'finish_reason': None}]})}\n\n"
                    )
                elif data.get("done"):
                    yield (
                        f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'jarvis', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                    )
                    yield "data: [DONE]\n\n"


@router.get("/v1/models")
async def proxy_list_models():
    """Open endpoint — returns the Jarvis model for OpenWebUI's model selector."""
    return {
        "object": "list",
        "data": [{"id": "jarvis", "object": "model", "owned_by": "jarvis", "created": 0}],
    }


@router.post("/v1/chat/completions")
async def proxy_chat(
    req: _OAIChatRequest,
    authorization: str = Header(default=None),
    x_openwebui_user_email: str = Header(default=None),
):
    # ── Auth: OpenWebUI email header (priority) or Bearer user_code ──
    user_code = None

    if x_openwebui_user_email:
        user_code = EMAIL_TO_CODE.get(x_openwebui_user_email.lower())
        if not user_code:
            raise HTTPException(401, f"No Jarvis user found for email {x_openwebui_user_email!r}")

    if not user_code and authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
        if token in USER_CODES:
            user_code = token
        else:
            user_code = EMAIL_TO_CODE.get(token.lower())

    if not user_code or user_code not in USER_CODES:
        raise HTTPException(401, "Unauthorized — set your email as API key in OpenWebUI, or your user code for iOS")

    # ── Extract last user message ──
    last_user_msg = next(
        (m for m in reversed(req.messages) if m.role == "user" and m.content), None
    )
    if not last_user_msg:
        raise HTTPException(400, "No usable message found")

    message, image_parts = _extract_content_parts(last_user_msg.content)
    if not message and not image_parts:
        raise HTTPException(400, "No usable message found")

    # ── Delegate to /chat ──
    jarvis_req = ChatRequest(
        message=message or "Que contient cette image ?",
        session_id=_proxy_session_id(user_code, req.messages),
        user_code=user_code,
        stream=req.stream,
        image_parts=image_parts,
    )
    response = await chat(jarvis_req)

    req_id  = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if isinstance(response, StreamingResponse):
        return StreamingResponse(
            _translate_jarvis_sse(response.body_iterator, req_id, created),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )
    return {
        "id":      req_id,
        "object":  "chat.completion",
        "created": created,
        "model":   "jarvis",
        "choices": [
            {
                "index":         0,
                "message":       {"role": "assistant", "content": response["response"]},
                "finish_reason": "stop",
            }
        ],
        "usage": {},
    }

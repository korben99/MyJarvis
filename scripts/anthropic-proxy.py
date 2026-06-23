#!/usr/bin/env python3
"""
anthropic-proxy.py  —  Anthropic API → Jarvis /v1/raw/chat/completions
=======================================================================
Claude Code envoie des requêtes format Anthropic (/v1/messages).
Ce proxy les traduit vers l'endpoint bypass de Jarvis qui appelle
stream_local() directement — aucun routage, mémoire ou analyse Jarvis.

Contrôle du thinking via Claude Code :
  /effort low    → thinking désactivé               (no_think=true)
  /effort medium → thinking actif, budget 2 048 tok (no_think=false)
  /effort high   → thinking actif, budget 4 000 tok
  /effort max    → thinking actif, budget 10 000 tok
  (mapping via thinking.budget_tokens ou output_config.effort dans le body)

Usage :
  source /opt/jarvis/venv/bin/activate
  python /opt/jarvis/scripts/anthropic-proxy.py

Puis dans un autre terminal :
  ANTHROPIC_BASE_URL=http://localhost:8090 ANTHROPIC_API_KEY=local claude

Env vars :
  PROXY_JARVIS_URL   URL de Jarvis          (défaut: http://localhost:8000)
  PROXY_PORT         Port d'écoute          (défaut: 8090)
"""

import json
import logging
import os
import time
import uuid
from typing import AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

JARVIS_URL   = os.getenv("PROXY_JARVIS_URL",  "http://localhost:8000")
PORT         = int(os.getenv("PROXY_PORT",        "8090"))
# ~32K tokens × 4 chars/token — garde une marge pour le system prompt et la réponse
MAX_CTX_CHARS = int(os.getenv("PROXY_MAX_CTX_CHARS", str(28_000 * 4)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [proxy] %(message)s")
log = logging.getLogger("anthropic-proxy")

app = FastAPI(title="Anthropic→Jarvis raw proxy")


# ── Mapping effort → thinking_budget ───────────────────────────────────────

_EFFORT_BUDGET: dict[str, int | None] = {
    "low":   None,   # no_think=True
    "medium": 2048,
    "high":  4000,
    "xhigh": 7000,
    "max":   10000,
}


def _resolve_thinking(body: dict) -> tuple[bool, int]:
    """
    Extrait les paramètres de thinking du body Anthropic et retourne
    (no_think, thinking_budget) pour l'endpoint Jarvis.

    Sources lues (priorité décroissante) :
      1. output_config.effort  (Claude Fable 5 / /effort dans Claude Code)
      2. thinking.budget_tokens (Anthropic extended thinking)
      3. thinking.type == "disabled"
    """
    # 1. output_config.effort (Claude Fable 5)
    effort = (body.get("output_config") or {}).get("effort")
    if effort and effort in _EFFORT_BUDGET:
        budget = _EFFORT_BUDGET[effort]
        if budget is None:
            return True, 0
        return False, budget

    # 2. thinking parameter (tous modèles Anthropic)
    thinking = body.get("thinking") or {}
    if thinking.get("type") == "disabled":
        return True, 0
    if thinking.get("type") in ("enabled", "adaptive"):
        budget = int(thinking.get("budget_tokens") or 4000)
        return False, budget

    # 3. Défaut : pas de thinking (rapide, adapté aux appels en rafale de Claude Code)
    return True, 0


# ── Conversion format Anthropic → OpenAI ───────────────────────────────────

def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            btype = block.get("type", "")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_result":
                inner = block.get("content", "")
                if isinstance(inner, list):
                    inner = " ".join(b.get("text", "") for b in inner if b.get("type") == "text")
                parts.append(f"[résultat outil: {inner}]")
            elif btype == "tool_use":
                parts.append(f"[appel outil {block.get('name','?')}: {json.dumps(block.get('input', {}))}]")
        return "\n".join(parts)
    return str(content)


def _truncate_messages(messages: list[dict]) -> list[dict]:
    """
    Tronque l'historique pour rester sous MAX_CTX_CHARS.
    Stratégie : garde toujours le system prompt + le dernier message user,
    puis remplit en partant de la fin (messages les plus récents en priorité).
    """
    if not messages:
        return messages

    total = sum(len(m["content"]) for m in messages)
    if total <= MAX_CTX_CHARS:
        return messages

    system = [m for m in messages if m["role"] == "system"]
    convs  = [m for m in messages if m["role"] != "system"]

    sys_chars  = sum(len(m["content"]) for m in system)
    budget     = MAX_CTX_CHARS - sys_chars
    kept       = []
    used       = 0

    # Dernier message obligatoire (la question courante)
    if convs:
        last = convs[-1]
        kept.append(last)
        used += len(last["content"])
        convs = convs[:-1]

    # Remplit l'historique en remontant du plus récent
    for msg in reversed(convs):
        if used + len(msg["content"]) > budget:
            break
        kept.insert(0, msg)
        used += len(msg["content"])

    dropped = len(convs) - (len(kept) - 1)
    if dropped > 0:
        log.warning("Contexte tronqué : %d message(s) anciens supprimés (%d → %d chars)", dropped, total, used + sys_chars)

    return system + kept


def _build_jarvis_payload(body: dict, no_think: bool, thinking_budget: int) -> dict:
    """Construit le payload pour /v1/raw/chat/completions."""
    messages = []

    if system := body.get("system"):
        if isinstance(system, list):
            system = " ".join(b.get("text", "") for b in system if b.get("type") == "text")
        messages.append({"role": "system", "content": system})

    for msg in body.get("messages", []):
        messages.append({"role": msg["role"], "content": _content_to_text(msg["content"])})

    messages = _truncate_messages(messages)

    payload: dict = {
        "messages":       messages,
        "stream":         body.get("stream", False),
        "no_think":       no_think,
        "thinking_budget": thinking_budget,
    }
    if max_tokens := body.get("max_tokens"):
        payload["max_tokens"] = max_tokens
    if temp := body.get("temperature"):
        payload["temperature"] = temp

    return payload


# ── Conversion SSE OpenAI → SSE Anthropic ──────────────────────────────────

async def _stream_anthropic(upstream_resp: httpx.Response) -> AsyncIterator[str]:
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    yield f"event: message_start\ndata: {json.dumps({'type':'message_start','message':{'id':msg_id,'type':'message','role':'assistant','content':[],'model':'local','stop_reason':None,'stop_sequence':None,'usage':{'input_tokens':0,'output_tokens':0}}})}\n\n"
    yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}})}\n\n"
    yield "event: ping\ndata: {\"type\":\"ping\"}\n\n"

    async for line in upstream_resp.aiter_lines():
        if not line or not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload.strip() == "[DONE]":
            break
        try:
            data   = json.loads(payload)
            choice = (data.get("choices") or [{}])[0]
            text   = choice.get("delta", {}).get("content") or choice.get("message", {}).get("content")
            if text:
                yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':text}})}\n\n"
        except (json.JSONDecodeError, KeyError, IndexError):
            continue

    yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':0})}\n\n"
    yield f"event: message_delta\ndata: {json.dumps({'type':'message_delta','delta':{'stop_reason':'end_turn','stop_sequence':None},'usage':{'output_tokens':0}})}\n\n"
    yield "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"


# ── Routes ──────────────────────────────────────────────────────────────────

@app.post("/v1/messages")
async def messages(request: Request):
    body      = await request.json()
    no_think, thinking_budget = _resolve_thinking(body)
    payload   = _build_jarvis_payload(body, no_think, thinking_budget)

    log.info(
        "→ stream=%s no_think=%s budget=%d msgs=%d",
        payload["stream"], no_think, thinking_budget, len(payload["messages"]),
    )

    client = httpx.AsyncClient(timeout=300)
    target = f"{JARVIS_URL}/v1/raw/chat/completions"

    if payload["stream"]:
        req  = client.build_request("POST", target, json=payload)
        resp = await client.send(req, stream=True)
        return StreamingResponse(
            _stream_anthropic(resp),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    resp    = await client.post(target, json=payload)
    data    = resp.json()
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    usage   = data.get("usage", {})

    return JSONResponse({
        "id":            f"msg_{uuid.uuid4().hex[:24]}",
        "type":          "message",
        "role":          "assistant",
        "content":       [{"type": "text", "text": content}],
        "model":         "local",
        "stop_reason":   "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens":  usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    })


@app.get("/v1/models")
async def list_models():
    """Aliases claude-* pour que Claude Code ne soit pas bloqué sur le nom du modèle."""
    now = int(time.time())
    return JSONResponse({"object": "list", "data": [
        {"id": "claude-opus-4-8",   "object": "model", "created": now, "owned_by": "proxy"},
        {"id": "claude-sonnet-4-6", "object": "model", "created": now, "owned_by": "proxy"},
        {"id": "claude-haiku-4-5",  "object": "model", "created": now, "owned_by": "proxy"},
    ]})


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    body   = await request.json()
    approx = sum(len(_content_to_text(m.get("content", ""))) // 4 for m in body.get("messages", []))
    return JSONResponse({"input_tokens": approx})


@app.get("/health")
async def health():
    return {"status": "ok", "jarvis": JARVIS_URL}


if __name__ == "__main__":
    log.info("Proxy démarré sur :%d → %s/v1/raw/chat/completions", PORT, JARVIS_URL)
    log.info("Usage : ANTHROPIC_BASE_URL=http://jarvis.local:%d ANTHROPIC_API_KEY=local claude", PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")

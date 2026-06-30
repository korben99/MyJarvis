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

import asyncio
import json
import logging
import os
import time
import uuid
from typing import AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

JARVIS_URL    = os.getenv("PROXY_JARVIS_URL",   "http://localhost:8000")
PORT          = int(os.getenv("PROXY_PORT",         "8090"))
MAX_CTX_CHARS = int(os.getenv("PROXY_MAX_CTX_CHARS", str(28_000 * 4)))
MAX_OUT_TOKENS = int(os.getenv("RAW_MAX_TOKENS",     "8000"))

LOG_FILE  = os.getenv("PROXY_LOG_FILE",  "/opt/jarvis/logs/anthropic-proxy.log")
LOG_DEBUG = os.getenv("PROXY_LOG_DEBUG", "/opt/jarvis/logs/anthropic-proxy-debug.log")

_fmt = logging.Formatter("%(asctime)s [proxy] %(levelname)s %(message)s")

log = logging.getLogger("anthropic-proxy")
log.setLevel(logging.DEBUG)
log.propagate = False  # évite la duplication via le root logger / LaunchAgent stderr

_file_info = logging.FileHandler(LOG_FILE,  encoding="utf-8")
_file_info.setLevel(logging.INFO)
_file_info.setFormatter(_fmt)
log.addHandler(_file_info)

_file_debug = logging.FileHandler(LOG_DEBUG, encoding="utf-8")
_file_debug.setLevel(logging.DEBUG)
_file_debug.setFormatter(_fmt)
log.addHandler(_file_debug)

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
        "messages":        messages,
        "stream":          True,  # toujours streamer vers Jarvis — évite le ReadTimeout sur les longues générations
        "no_think":        no_think,
        "thinking_budget": thinking_budget,
        # Cap : Claude Code envoie parfois 64 000 — Qwen ne peut pas suivre
        "max_tokens":      min(body.get("max_tokens") or MAX_OUT_TOKENS, MAX_OUT_TOKENS),
    }
    if temp := body.get("temperature"):
        payload["temperature"] = temp

    return payload


# ── Parsing tool calls ──────────────────────────────────────────────────────

import re as _re

def _parse_tool_calls(text: str) -> list[dict]:
    """
    Détecte les tool calls dans la réponse brute de Qwen.
    Supporte plusieurs formats que Qwen peut générer en imitant Claude.

    Retourne une liste de {"name": str, "input": dict, "span": (start, end)}.
    """
    results = []

    # Format 1 — JSON inline: <tool_use>{"type":"tool_use","name":"...","input":{...}}</tool_use>
    for m in _re.finditer(r'<tool_use>\s*(\{.*?\})\s*</tool_use>', text, _re.DOTALL):
        try:
            data = json.loads(m.group(1))
            results.append({"name": data.get("name", "unknown"),
                            "input": data.get("input", data.get("arguments", {})),
                            "span": m.span()})
        except json.JSONDecodeError:
            pass

    # Format 2 — Qwen natif: <tool_call>\n{"name":"...","arguments":{...}}\n</tool_call>
    for m in _re.finditer(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, _re.DOTALL):
        if any(m.start() == r["span"][0] for r in results):
            continue  # déjà capturé
        try:
            data = json.loads(m.group(1))
            results.append({"name": data.get("name", "unknown"),
                            "input": data.get("arguments", data.get("input", {})),
                            "span": m.span()})
        except json.JSONDecodeError:
            pass

    # Format 3 — XML structuré: <tool_use><name>...</name><input>...</input></tool_use>
    for m in _re.finditer(
        r'<tool_use>\s*<name>(.*?)</name>\s*<(?:input|parameters)>(.*?)</(?:input|parameters)>\s*</tool_use>',
        text, _re.DOTALL,
    ):
        if any(m.start() == r["span"][0] for r in results):
            continue
        try:
            inp = json.loads(m.group(2))
            results.append({"name": m.group(1).strip(), "input": inp, "span": m.span()})
        except json.JSONDecodeError:
            pass

    # Trie par position dans le texte
    results.sort(key=lambda r: r["span"][0])
    return results


def _split_text_and_tools(text: str, tool_calls: list[dict]) -> list[dict]:
    """
    Découpe le texte en segments alternant texte / tool_use.
    Retourne une liste de {"type": "text"|"tool_use", "content": ...}.
    """
    segments = []
    cursor = 0
    for tc in tool_calls:
        start, end = tc["span"]
        if start > cursor:
            pre = text[cursor:start].strip()
            if pre:
                segments.append({"type": "text", "content": pre})
        segments.append({"type": "tool_use", "name": tc["name"], "input": tc["input"]})
        cursor = end
    tail = text[cursor:].strip()
    if tail:
        segments.append({"type": "text", "content": tail})
    return segments


# ── Conversion SSE OpenAI → SSE Anthropic ──────────────────────────────────

async def _stream_anthropic(upstream_resp: httpx.Response) -> AsyncIterator[str]:
    """
    Collecte la réponse complète de Jarvis, parse les tool calls éventuels,
    puis émet les événements SSE Anthropic corrects.
    Bufférisé intentionnellement : Claude Code a besoin de blocs tool_use complets.
    """
    # ── 1. Collecter tous les chunks ────────────────────────────────────────
    full_text = ""
    async for line in upstream_resp.aiter_lines():
        if not line or not line.startswith("data: "):
            continue
        raw = line[6:]
        if raw.strip() == "[DONE]":
            break
        try:
            data   = json.loads(raw)
            choice = (data.get("choices") or [{}])[0]
            chunk  = choice.get("delta", {}).get("content") or choice.get("message", {}).get("content")
            if chunk:
                full_text += chunk
        except (json.JSONDecodeError, KeyError, IndexError):
            continue

    log.debug("Réponse brute Qwen (%d chars): %s", len(full_text), full_text[:800])

    # ── 2. Parser les tool calls ────────────────────────────────────────────
    tool_calls = _parse_tool_calls(full_text)
    segments   = _split_text_and_tools(full_text, tool_calls)
    has_tools  = any(s["type"] == "tool_use" for s in segments)
    stop_reason = "tool_use" if has_tools else "end_turn"

    if tool_calls:
        log.info("Tool calls détectés: %s", [tc["name"] for tc in tool_calls])

    # ── 3. Émettre les événements SSE Anthropic ─────────────────────────────
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    yield f"event: message_start\ndata: {json.dumps({'type':'message_start','message':{'id':msg_id,'type':'message','role':'assistant','content':[],'model':'local','stop_reason':None,'stop_sequence':None,'usage':{'input_tokens':0,'output_tokens':0}}})}\n\n"
    yield "event: ping\ndata: {\"type\":\"ping\"}\n\n"

    idx = 0
    for seg in segments:
        if seg["type"] == "text":
            yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':idx,'content_block':{'type':'text','text':''}})}\n\n"
            # Envoie le texte en chunks de 40 chars pour un rendu progressif
            text = seg["content"]
            for i in range(0, len(text), 40):
                chunk = text[i:i + 40]
                yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':idx,'delta':{'type':'text_delta','text':chunk}})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':idx})}\n\n"

        else:  # tool_use
            tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
            yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':idx,'content_block':{'type':'tool_use','id':tool_id,'name':seg['name'],'input':{}}})}\n\n"
            yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':idx,'delta':{'type':'input_json_delta','partial_json':json.dumps(seg['input'])}})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':idx})}\n\n"

        idx += 1

    yield f"event: message_delta\ndata: {json.dumps({'type':'message_delta','delta':{'stop_reason':stop_reason,'stop_sequence':None},'usage':{'output_tokens':len(full_text)//4}})}\n\n"
    yield "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"


# ── Routes ──────────────────────────────────────────────────────────────────

@app.post("/v1/messages")
async def messages(request: Request):
    body      = await request.json()
    no_think, thinking_budget = _resolve_thinking(body)
    payload   = _build_jarvis_payload(body, no_think, thinking_budget)

    client_streams = body.get("stream", False)  # ce que Claude Code veut recevoir
    log.info(
        "→ client_stream=%s no_think=%s budget=%d msgs=%d",
        client_streams, no_think, thinking_budget, len(payload["messages"]),
    )
    log.debug("body reçu: %s", json.dumps(body)[:500])
    log.debug("payload Jarvis: %s", json.dumps({**payload, "messages": f"[{len(payload['messages'])} msgs]"}))

    target = f"{JARVIS_URL}/v1/raw/chat/completions"

    if client_streams:
        # Le client doit rester ouvert pendant tout le streaming
        client = httpx.AsyncClient(timeout=300)
        try:
            req  = client.build_request("POST", target, json=payload)
            resp = await client.send(req, stream=True)
            log.debug("Jarvis stream status: %d", resp.status_code)
            if resp.status_code != 200:
                body_err = await resp.aread()
                log.error("Jarvis stream error %d: %s", resp.status_code, body_err[:300])
            return StreamingResponse(
                _stream_anthropic(resp),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        except httpx.ConnectError as e:
            await client.aclose()
            log.error("ConnectError (stream): %s", e)
            raise HTTPException(status_code=503, detail=f"Jarvis unavailable: {e}")

    # Non-streaming : on consomme le stream SSE de Jarvis (stream=True côté Jarvis pour
    # éviter les ReadTimeout sur longues générations) et on renvoie un JSON statique à Claude Code.
    # Retry 3× avec backoff — Jarvis peut être momentanément occupé (GPU MLX lock).
    full_text = ""
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream("POST", target, json=payload) as resp:
                    log.debug("Jarvis sync status: %d", resp.status_code)
                    if resp.status_code != 200:
                        body_err = await resp.aread()
                        log.error("Jarvis sync error %d: %s", resp.status_code, body_err[:300])
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        raw = line[6:]
                        if raw.strip() == "[DONE]":
                            break
                        try:
                            data   = json.loads(raw)
                            choice = (data.get("choices") or [{}])[0]
                            chunk  = (choice.get("delta", {}).get("content")
                                      or choice.get("message", {}).get("content"))
                            if chunk:
                                full_text += chunk
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
            break
        except httpx.ConnectError as e:
            full_text = ""
            log.warning("ConnectError tentative %d/3: %s", attempt + 1, e)
            if attempt == 2:
                raise HTTPException(status_code=503, detail=f"Jarvis unavailable: {e}")
            await asyncio.sleep(1.5 * (attempt + 1))

    log.debug("Réponse sync (%d chars): %s", len(full_text), full_text[:300])

    # Parse tool calls éventuels (mêmes règles que le chemin streaming)
    tool_calls = _parse_tool_calls(full_text)
    segments   = _split_text_and_tools(full_text, tool_calls)
    has_tools  = any(s["type"] == "tool_use" for s in segments)
    if tool_calls:
        log.info("Tool calls détectés (sync): %s", [tc["name"] for tc in tool_calls])

    content_blocks = []
    for seg in segments:
        if seg["type"] == "text":
            content_blocks.append({"type": "text", "text": seg["content"]})
        else:
            content_blocks.append({
                "type":  "tool_use",
                "id":    f"toolu_{uuid.uuid4().hex[:24]}",
                "name":  seg["name"],
                "input": seg["input"],
            })
    if not content_blocks:
        content_blocks = [{"type": "text", "text": full_text}]

    return JSONResponse({
        "id":            f"msg_{uuid.uuid4().hex[:24]}",
        "type":          "message",
        "role":          "assistant",
        "content":       content_blocks,
        "model":         "local",
        "stop_reason":   "tool_use" if has_tools else "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens":  0,
            "output_tokens": len(full_text) // 4,
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

"""
llm_client.py — LLM HTTP client (streaming + vision)
======================================================
Provides:
  openai_headers()          : build auth headers for OpenAI-compatible APIs
  select_model()            : tier selection (primary / reasoning)
  trim_chunks()             : cap RAG/web results by char budget
  stream_openai()           : async SSE streaming generator
  describe_images()         : two-stage vision pipeline (resolve URLs → call VISION_MODEL)
"""

import asyncio
import base64
import json
from typing import AsyncGenerator

import httpx

from config import (
    OPENAI_API_KEY,
    OPENAI_API_URL,
    PRIMARY_API_KEY,
    PRIMARY_API_URL,
    PRIMARY_MODEL,
    PRIMARY_TIMEOUT,
    REASONING_API_KEY,
    REASONING_API_URL,
    REASONING_MODEL,
    REASONING_TIMEOUT,
    VISION_API_KEY,
    VISION_API_URL,
    VISION_MODEL,
    VISION_TIMEOUT,
    tokens_param,
)
from deps import get_stream_client
from helpers import get_logger

logger = get_logger("jarvis-llm")


def openai_headers() -> dict:
    return {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }


def select_model(
    req_model: "str | None",
    use_reasoning: bool = False,
) -> tuple[str, str, str, float]:
    """
    Single decision point for LLM tier selection.
    Returns (model_name, api_url, api_key, timeout_seconds).

    Priority:
    1. Explicit model override in the request  →  PRIMARY credentials
    2. Router flagged use_reasoning=True        →  REASONING_MODEL (cloud, complex only)
    3. Everything else                          →  PRIMARY_MODEL
    """
    if req_model:
        return req_model, PRIMARY_API_URL, PRIMARY_API_KEY, PRIMARY_TIMEOUT
    if use_reasoning:
        return REASONING_MODEL, REASONING_API_URL, REASONING_API_KEY, REASONING_TIMEOUT
    return PRIMARY_MODEL, PRIMARY_API_URL, PRIMARY_API_KEY, PRIMARY_TIMEOUT


def trim_chunks(chunks, char_budget, text_key="text", max_item_chars=800):
    """Generic chunk limiter for RAG or web results."""
    total = 0
    selected = []
    for c in chunks:
        text = c[text_key][:max_item_chars]
        if total + len(text) > char_budget:
            break
        selected.append(text)
        total += len(text)
    return selected


async def stream_openai(
    messages: list,
    model: str,
    api_url: str = OPENAI_API_URL,
    api_key: str = OPENAI_API_KEY,
    timeout: float = 30.0,
) -> AsyncGenerator[str, None]:
    try:
        client = get_stream_client(timeout)
        async with client.stream(
            "POST",
            f"{api_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "stream": True,
            },
        ) as response:
            if response.status_code != 200:
                logger.error("OpenAI streaming error: %s", response.status_code)
                return

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                    content = data.get("choices", [{}])[0].get("delta", {}).get("content")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    logger.debug("Invalid JSON chunk: %s", payload[:100])
                    continue

    except httpx.RequestError as e:
        logger.error("OpenAI request error: %s", e)


async def _resolve_image_part(part: dict, client: httpx.AsyncClient) -> dict:
    """
    Ensure an image_url part contains a publicly accessible URL or a base64 data URI.
    Open WebUI sends images as internal Docker URLs that OpenAI cannot reach — we fetch
    those internally and re-encode as base64.
    """
    url = part.get("image_url", {}).get("url", "")
    if url.startswith("data:") or url.startswith("https://"):
        return part

    try:
        r = await client.get(url, timeout=15)
        r.raise_for_status()
        mime = (r.headers.get("content-type") or "image/jpeg").split(";")[0]
        b64 = base64.b64encode(r.content).decode()
        logger.debug("Vision: re-encoded internal image (%s, %d bytes)", mime, len(r.content))
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
    except Exception as exc:
        logger.warning("Vision: could not fetch internal image URL (%s): %s", url[:80], exc)
        return part


async def describe_images(image_parts: list, text_prompt: str) -> str:
    """
    Call VISION_MODEL to produce a detailed description of uploaded images.
    Returns empty string on failure or when VISION_MODEL is not configured.
    """
    if not VISION_MODEL or not image_parts:
        return ""

    try:
        async with httpx.AsyncClient(timeout=VISION_TIMEOUT) as client:
            resolved = await asyncio.gather(
                *[_resolve_image_part(p, client) for p in image_parts]
            )
            vision_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyse cette image de façon exhaustive et structurée.\n"
                                "Décris dans cet ordre :\n"
                                "1. Sujet principal et contexte général de la scène\n"
                                "2. Tout texte, chiffre ou étiquette visible — recopie-le mot pour mot\n"
                                "3. Personnes présentes (apparence, expression, action)\n"
                                "4. Objets, couleurs dominantes, positions relatives\n"
                                "5. Tout élément pertinent pour répondre à la question ci-dessous\n\n"
                                f"Question de l'utilisateur : {text_prompt or 'Que contient cette image ?'}"
                            ),
                        },
                        *resolved,
                    ],
                }
            ]
            resp = await client.post(
                f"{VISION_API_URL}/chat/completions",
                headers={"Authorization": f"Bearer {VISION_API_KEY}"},
                json={
                    "model": VISION_MODEL,
                    "messages": vision_messages,
                    tokens_param(VISION_MODEL): 1024,
                    "stream": False,
                },
            )
        data = resp.json()
        if resp.status_code != 200:
            logger.warning(
                "Vision: API error %d — %s",
                resp.status_code,
                data.get("error", {}).get("message", ""),
            )
            return ""
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.warning("Vision: image description failed (%s)", type(exc).__name__)
        return ""

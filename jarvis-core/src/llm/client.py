"""
llm_client.py — LLM HTTP client (streaming + vision)
======================================================
Provides:
  openai_headers()          : build auth headers for OpenAI-compatible APIs
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
    LLM_LOCAL,
    MAX_TOKENS_HARD_CAP,
    OPENAI_API_KEY,
    OPENAI_API_URL,
    PRIMARY_MODEL,
    REASONING_MODEL,
    ROUTER_MODEL,
    VISION_API_KEY,
    VISION_API_URL,
    VISION_LOCAL,
    VISION_MODEL,
    VISION_TIMEOUT,
    tokens_param,
)
from deps import get_stream_client
from helpers import get_logger
from .local import stream_local
from prompts import VISION_USER_PROMPT

_LOCAL_MODELS = {ROUTER_MODEL, PRIMARY_MODEL, REASONING_MODEL} if LLM_LOCAL else set()

logger = get_logger("jarvis-llm")


def openai_headers() -> dict:
    return {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }


def trim_chunks(chunks, char_budget, text_key="text", max_item_chars=800):
    """Generic chunk limiter for RAG or web results.

    Stops at the FIRST item that would overflow the budget (no skip-and-continue):
    callers in pipeline.build_context rely on the selection being a prefix of
    `chunks` to zip texts back with their metadata by index."""
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
    no_think: bool = False,
    session_id: str = "",
    max_tokens: int = MAX_TOKENS_HARD_CAP,
    thinking_budget: int = 0,
) -> AsyncGenerator[str, None]:

    # ── Local path (MLX) ───────────────────────────────────────────
    if LLM_LOCAL and model in _LOCAL_MODELS:
        async for chunk in stream_local(
            messages,
            model,
            no_think=no_think,
            session_id=session_id,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
        ):
            yield chunk
        return

    # ── Remote path (OpenAI-compatible APIs) ───────────────────────
    # NOTE: no_think / thinking_budget ne sont honorés que sur le chemin local
    # (contrôle au niveau du template MLX) — le chemin HTTP les ignore.
    try:
        client = get_stream_client(timeout)

        # ── Build payload ──────────────────────────────────────────
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        # ── Token param (mlx / compat providers) ──────────────────
        try:
            payload[tokens_param(model)] = max_tokens
        except Exception:
            pass

        logger.debug(
            "LLM call model=%s no_think=%s keys=%s",
            model,
            no_think,
            list(payload.keys()),
        )

        # ── HTTP streaming ────────────────────────────────────────
        async with client.stream(
            "POST",
            f"{api_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        ) as response:
            if response.status_code != 200:
                try:
                    err_body = (await response.aread())[:500]
                except Exception:
                    err_body = b""
                logger.error(
                    "OpenAI streaming error: %s — %s",
                    response.status_code,
                    err_body.decode("utf-8", errors="replace"),
                )
                return

            # ── Stream parsing ────────────────────────────────────
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue

                payload_str = line[6:]

                if payload_str == "[DONE]":
                    break

                try:
                    data = json.loads(payload_str)

                    choice = (data.get("choices") or [{}])[0]

                    delta = choice.get("delta") or {}
                    message = choice.get("message") or {}

                    content = delta.get("content") or message.get("content")

                    if content:
                        yield content

                except json.JSONDecodeError:
                    logger.debug(
                        "Invalid JSON chunk: %s",
                        payload_str[:100],
                    )
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
    # split(",", 1)[0] — never url.index(","): a malformed data: URL without comma
    # would raise ValueError here and abort resolution for ALL images of the message.
    _url_head = url.split(",", 1)[0] if url.startswith("data:") else (url[:80] or "(empty)")
    logger.info("Vision._resolve: url=%s", _url_head)
    if url.startswith("data:") or url.startswith("https://"):
        return part

    try:
        r = await client.get(url, timeout=15)
        r.raise_for_status()
        mime = (r.headers.get("content-type") or "image/jpeg").split(";")[0]
        b64 = base64.b64encode(r.content).decode()
        logger.info(
            "Vision: re-encoded internal image (%s, %d bytes)", mime, len(r.content)
        )
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
    except Exception as exc:
        logger.warning(
            "Vision: could not fetch internal image URL (%s): %s", url[:80], exc
        )
        return part


async def describe_images(image_parts: list, text_prompt: str) -> str:
    """
    Call VISION_MODEL to produce a detailed description of uploaded images.
    Routes to mlx_vlm (local) when VISION_LOCAL=True, otherwise calls the API.
    Returns empty string on failure or when VISION_MODEL is not configured.
    """
    if not VISION_MODEL or not image_parts:
        return ""

    # ── Resolve internal URLs (OpenWebUI Docker URLs → base64) ────────────
    try:
        async with httpx.AsyncClient(timeout=VISION_TIMEOUT) as client:
            resolved = await asyncio.gather(
                *[_resolve_image_part(p, client) for p in image_parts]
            )
    except Exception as exc:
        logger.warning("Vision: image resolution failed (%s)", exc)
        return ""

    # ── Local path (mlx_vlm) ──────────────────────────────────────────────
    if VISION_LOCAL:
        from .local import describe_images_local
        try:
            return await describe_images_local(list(resolved), text_prompt)
        except Exception as exc:
            logger.warning("Vision local: failed (%s) — %s", type(exc).__name__, exc)
            return ""

    # ── Remote path (OpenAI-compatible API) ───────────────────────────────
    try:
        vision_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": VISION_USER_PROMPT.format(
                            text_prompt=text_prompt or "Décris cette image dans son ensemble."
                        ),
                    },
                    *resolved,
                ],
            }
        ]
        async with httpx.AsyncClient(timeout=VISION_TIMEOUT) as api_client:
            resp = await api_client.post(
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

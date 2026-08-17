"""Clients HTTP LLM partagés (connection-pooled) + variantes call_llm.

Un client httpx persistant par mode (sync/async), pour éviter le handshake TCP à
chaque appel. En mode LLM_LOCAL, on route vers call_llm_local* (MLX direct) ; sinon
vers le serveur mlx-lm / cloud. Les clés d'API ne sont jamais loggées.
"""

from threading import Lock

import httpx
from config import (
    LLM_LOCAL,
    PRIMARY_MODEL,
    REASONING_MODEL,
    ROUTER_MODEL,
    tokens_param,
)
from llm_local import (
    call_llm_local,
    call_llm_local_async,
    call_llm_local_async_bg,
    call_llm_local_bg,
)

_LOCAL_MODELS = {ROUTER_MODEL, PRIMARY_MODEL, REASONING_MODEL} if LLM_LOCAL else set()

# Single persistent client per mode — avoids TCP handshake overhead on every call.
# Sync: used by memory.py background tasks (not in async event loop).
# Async: used by all FastAPI route handlers and background coroutines.
_LLM_LIMITS = httpx.Limits(max_connections=20, max_keepalive_connections=10)
_llm_sync_client: httpx.Client | None = None
_llm_sync_lock = Lock()
_llm_async_client: httpx.AsyncClient | None = None


def _get_llm_sync_client() -> httpx.Client:
    global _llm_sync_client
    if _llm_sync_client is None:
        with _llm_sync_lock:
            if _llm_sync_client is None:
                _llm_sync_client = httpx.Client(limits=_LLM_LIMITS)
    return _llm_sync_client


def _get_llm_async_client() -> httpx.AsyncClient:
    """No lock needed: asyncio is single-threaded, no concurrent init risk."""
    global _llm_async_client
    if _llm_async_client is None:
        _llm_async_client = httpx.AsyncClient(limits=_LLM_LIMITS)
    return _llm_async_client


def _llm_headers(api_key: str) -> dict:
    """Build auth headers. Key value is never stored in logs or tracebacks."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


_LOCAL_DEFAULT_MAX_TOKENS = (
    10000  # global ceiling — thinking + output are counted together in mlx-lm
)


def _llm_body(
    messages: list[dict],
    model: str,
    temperature: float | None,
    max_tokens: int | None,
    json_response: bool,
) -> dict:
    """
    Build the JSON body for a /chat/completions request.

    - Uses tokens_param() to pick max_tokens vs max_completion_tokens.
    - Sets response_format when json_response=True.
    - max_tokens=None → field omitted → API uses model default (no truncation risk).
    - Thinking control (no_think) is handled at the MLX prompt level for local
      models (_build_prompt via enable_thinking), not at the HTTP body level.
    """

    body: dict = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens is not None:
        body[tokens_param(model)] = max_tokens
    if json_response:
        body["response_format"] = {"type": "json_object"}
    return body


def call_llm(
    messages: list[dict],
    *,
    model: str,
    api_url: str,
    api_key: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    json_response: bool = True,
    no_think: bool = False,
    timeout: float = 30.0,
    thinking_budget: int = 0,
) -> str:
    """
    Synchronous LLM call — HTTP (cloud/mlx-lm server) ou MLX direct (LLM_LOCAL=yes).

    Returns the model's raw text content.
    API key is never logged.
    max_tokens=None → no explicit limit (model stops at EOS / closing JSON brace).
    thinking_budget > 0 : cap de tokens de thinking (local uniquement, ignoré pour HTTP).
    """
    if LLM_LOCAL and model in _LOCAL_MODELS:
        return call_llm_local(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens or _LOCAL_DEFAULT_MAX_TOKENS,
            no_think=no_think,
            json_response=json_response,
            thinking_budget=thinking_budget,
        )
    resp = _get_llm_sync_client().post(
        f"{api_url}/chat/completions",
        headers=_llm_headers(api_key),
        json=_llm_body(messages, model, temperature, max_tokens, json_response),
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_llm_bg(
    messages: list[dict],
    *,
    model: str,
    api_url: str,
    api_key: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    json_response: bool = True,
    no_think: bool = False,
    timeout: float = 30.0,
    thinking_budget: int = 0,
) -> str:
    """Synchronous background-priority LLM call — yields GPU to chat callers.

    Use from sync background tasks (analyzer dedup, nightly jobs) instead of
    call_llm, which takes the GPU lock at chat priority and can delay user
    requests. Falls back to the normal HTTP path for cloud models."""
    if LLM_LOCAL and model in _LOCAL_MODELS:
        return call_llm_local_bg(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens or _LOCAL_DEFAULT_MAX_TOKENS,
            no_think=no_think,
            json_response=json_response,
            thinking_budget=thinking_budget,
        )
    resp = _get_llm_sync_client().post(
        f"{api_url}/chat/completions",
        headers=_llm_headers(api_key),
        json=_llm_body(messages, model, temperature, max_tokens, json_response),
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def call_llm_async(
    messages: list[dict],
    *,
    model: str,
    api_url: str,
    api_key: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    json_response: bool = True,
    no_think: bool = False,
    timeout: float = 30.0,
    thinking_budget: int = 0,
) -> str:
    """
    Async LLM call — HTTP (cloud/mlx-lm server) ou MLX direct (LLM_LOCAL=yes).

    Returns the model's raw text content.
    API key is never logged.
    max_tokens=None → no explicit limit (model stops at EOS / closing JSON brace).
    thinking_budget > 0 : cap de tokens de thinking (local uniquement, ignoré pour HTTP).
    """
    if LLM_LOCAL and model in _LOCAL_MODELS:
        return await call_llm_local_async(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens or _LOCAL_DEFAULT_MAX_TOKENS,
            no_think=no_think,
            json_response=json_response,
            thinking_budget=thinking_budget,
        )
    resp = await _get_llm_async_client().post(
        f"{api_url}/chat/completions",
        headers=_llm_headers(api_key),
        json=_llm_body(messages, model, temperature, max_tokens, json_response),
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def call_llm_async_bg(
    messages: list[dict],
    *,
    model: str,
    api_url: str,
    api_key: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    json_response: bool = True,
    no_think: bool = False,
    timeout: float = 30.0,
    thinking_budget: int = 0,
) -> str:
    """Background-priority async LLM call — yields GPU to chat callers when waiting.
    Use for self-reflection and other background tasks instead of call_llm_async.
    Falls back to normal HTTP path for cloud models (no local lock contention there).
    """
    if LLM_LOCAL and model in _LOCAL_MODELS:
        return await call_llm_local_async_bg(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens or _LOCAL_DEFAULT_MAX_TOKENS,
            no_think=no_think,
            json_response=json_response,
            thinking_budget=thinking_budget,
        )
    resp = await _get_llm_async_client().post(
        f"{api_url}/chat/completions",
        headers=_llm_headers(api_key),
        json=_llm_body(messages, model, temperature, max_tokens, json_response),
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

"""
Jarvis LLM Router — two-tier edition
=====================================
Tier 1 (this file): fast intent classifier + complexity detector.
Tier 2 (main.py):   reasoning model selected when use_reasoning=True.

Router backend is fully OpenAI-compatible (/v1/chat/completions).
Swap from GPT-4.1-nano to Qwen2.5-7B via mlx-lm by changing three env vars:
    ROUTER_API_URL=http://<mac-ip>:8080/v1
    ROUTER_API_KEY=mlx        (mlx-lm ignores auth but httpx must send something)
    ROUTER_MODEL=Qwen/Qwen2.5-7B-Instruct-8bit

mlx-lm note: response_format is supported from mlx-lm ≥ 0.21. For older
versions the JSON is extracted from the raw text as a fallback.

If ROUTER_MODEL is empty or the call fails for any reason, returns None
and main.py falls back to the embedding router automatically.

RouterResult fields:
    use_memory, use_rag, use_web, use_gmail, use_calendar,
    use_briefing, use_self   — data-source flags
    use_reasoning            — True → route to Tier-2 reasoning model
    gmail_query              — Gmail search string (or "")
    calendar_days            — days ahead to fetch (default 7)
"""

import json
import logging
import re
from dataclasses import dataclass, field

import httpx

from config import (
    ROUTER_API_URL,
    ROUTER_API_KEY,
    ROUTER_MODEL,
    ROUTER_TIMEOUT,
    no_think_suffix,
)

logger = logging.getLogger("jarvis-llm-router")


def is_llm_router_available() -> bool:
    """True when a router model is configured (non-empty ROUTER_MODEL)."""
    return bool(ROUTER_MODEL)


# ── Prompts (from prompts.py) ─────────────────────────────────────────────
from prompts import ROUTER_SYSTEM as _ROUTER_SYSTEM, ROUTER_USER as _ROUTER_USER


# ── Result dataclass ──────────────────────────────────────────────────────

@dataclass
class RouterResult:
    use_memory:       bool
    use_rag:          bool
    use_web:          bool
    use_weather:      bool
    use_gmail:        bool
    use_calendar:     bool
    use_briefing:     bool
    use_self:         bool
    use_portfolio:    bool
    use_reasoning:    bool
    gmail_query:      str
    calendar_days:    int
    weather_location: str = field(default="")
    memory_scope:     str = field(default="auto")
    conversation_type: str = field(default="conversational")


# ── JSON extraction (mlx-lm resilience) ──────────────────────────────────

def _extract_json(raw: str) -> dict:
    """
    Parse JSON from a model response robustly.

    Strategy:
    1. Try direct parse (works when response_format is supported).
    2. Strip markdown code fences if present.
    3. Extract the first {...} block from the text (mlx-lm fallback for
       older versions that ignore response_format).

    Raises json.JSONDecodeError if nothing works.
    """
    raw = raw.strip()

    # 1. Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences
    if "```" in raw:
        inner = raw.split("```")[1]
        first_newline = inner.find("\n")
        if first_newline != -1 and not inner[:first_newline].strip().startswith("{"):
            inner = inner[first_newline:].strip()
        try:
            return json.loads(inner.strip())
        except json.JSONDecodeError:
            pass

    # 3. Extract first {...} block (handles prose-prefixed responses)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group())

    raise json.JSONDecodeError("No JSON found in router response", raw, 0)


# ── Core call ─────────────────────────────────────────────────────────────

async def llm_route(message: str, google_available: bool = True) -> RouterResult | None:
    """
    Call the router model (OpenAI-compatible /v1/chat/completions).

    Returns RouterResult on success, None on any failure → caller falls back
    to the embedding router automatically.

    Works with:
    - OpenAI  (GPT-4.1-nano now)
    - mlx-lm  (Qwen2.5-7B later) — same endpoint, same request shape
    """
    if not is_llm_router_available():
        return None

    prompt = _ROUTER_USER.format(message=message)

    try:
        async with httpx.AsyncClient(timeout=ROUTER_TIMEOUT) as client:
            resp = await client.post(
                f"{ROUTER_API_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {ROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": ROUTER_MODEL,
                    "messages": [
                        {"role": "system", "content": _ROUTER_SYSTEM + no_think_suffix(ROUTER_MODEL)},
                        {"role": "user",   "content": prompt},
                    ],
                    "temperature": 0,
                    "max_tokens": 250,
                    # response_format is supported by OpenAI and mlx-lm ≥ 0.21.
                    # Older mlx-lm versions ignore it — _extract_json() handles that.
                    "response_format": {"type": "json_object"},
                    "stream": False,
                },
            )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        parsed = _extract_json(raw)

    except httpx.TimeoutException:
        logger.warning(
            "LLM router timeout (%.1fs) — falling back to embedding router", ROUTER_TIMEOUT
        )
        return None
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning(
            "LLM router response parse error (%s) — falling back to embedding router",
            type(exc).__name__,
        )
        return None
    except Exception as exc:
        logger.warning(
            "LLM router error (%s) — falling back to embedding router", type(exc).__name__
        )
        return None

    # ── Extract and validate fields ──
    intents: list[str] = parsed.get("intents", [])
    if not isinstance(intents, list):
        intents = []
    if not intents:
        intents = ["memory"]

    gmail_query:      str  = parsed.get("gmail_query") or ""
    calendar_days:    int  = int(parsed.get("calendar_days") or 7)
    weather_location: str  = parsed.get("weather_location") or ""
    use_reasoning:    bool = bool(parsed.get("use_reasoning", False))

    calendar_days = max(1, min(calendar_days, 90))

    _VALID_SCOPES     = {"episodic", "autobiographical", "profile", "auto"}
    _VALID_CONV_TYPES = {"conversational", "task", "question"}

    raw_scope     = parsed.get("memory_scope", "auto")
    memory_scope: str = raw_scope if raw_scope in _VALID_SCOPES else "auto"

    raw_conv_type = parsed.get("conversation_type", "conversational")
    conversation_type: str = raw_conv_type if raw_conv_type in _VALID_CONV_TYPES else "conversational"

    result = RouterResult(
        use_memory        = "memory"    in intents,
        use_rag           = "rag"       in intents,
        use_web           = "web"       in intents,
        use_weather       = "weather"   in intents,
        use_gmail         = "gmail"     in intents and google_available,
        use_calendar      = "calendar"  in intents and google_available,
        use_briefing      = "briefing"  in intents,
        use_self          = "self"      in intents,
        use_portfolio     = "portfolio" in intents,
        use_reasoning     = use_reasoning,
        gmail_query       = gmail_query,
        calendar_days     = calendar_days,
        weather_location  = weather_location,
        memory_scope      = memory_scope,
        conversation_type = conversation_type,
    )

    logger.info(
        "LLM router [%s]: intents=%s weather_location=%r gmail_query=%r calendar_days=%d use_reasoning=%s memory_scope=%s conv_type=%s",
        ROUTER_MODEL, intents, weather_location, gmail_query, calendar_days, use_reasoning,
        memory_scope, conversation_type,
    )
    return result

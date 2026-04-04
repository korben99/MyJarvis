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
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime

import httpx
from config import (
    PRIMARY_API_KEY,
    PRIMARY_API_URL,
    PRIMARY_MODEL,
    ROUTER_API_KEY,
    ROUTER_API_URL,
    ROUTER_DATA_DIR,
    ROUTER_MODEL,
    ROUTER_TIMEOUT,
)
from helpers import call_llm_async, extract_llm_json, get_logger
from prompts import get_prompt

logger = get_logger("jarvis-llm-router")


# ── Result dataclass ──────────────────────────────────────────────────────
@dataclass
class RouterResult:
    use_memory: bool
    use_rag: bool
    use_web: bool
    use_weather: bool
    use_gmail: bool
    use_calendar: bool
    use_briefing: bool
    use_self: bool
    use_portfolio: bool
    use_reasoning: bool
    gmail_query: str
    calendar_days: int
    weather_location: str = field(default="")


# ── Training data collector ───────────────────────────────────────────────


def _log_routing_sample(
    message: str,
    result: "RouterResult",
    model: str,
) -> None:
    """Append one JSONL entry to the router training file.

    File: {ROUTER_DATA_DIR}/routing_samples.jsonl
    Each line is a self-contained JSON object with:
      - id / ts   : deduplication + timeline
      - message   : raw user input
      - routing   : the full RouterResult as a dict
      - model     : which router model produced this result
      - ok        : null = uncurated, true = validated, false = wrong
    """
    os.makedirs(ROUTER_DATA_DIR, exist_ok=True)
    sample = {
        "id": str(uuid.uuid4()),
        "ts": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        "message": message,
        "routing": {
            "intents": [
                intent
                for intent, flag in [
                    ("memory", result.use_memory),
                    ("rag", result.use_rag),
                    ("web", result.use_web),
                    ("weather", result.use_weather),
                    ("gmail", result.use_gmail),
                    ("calendar", result.use_calendar),
                    ("briefing", result.use_briefing),
                    ("self", result.use_self),
                    ("portfolio", result.use_portfolio),
                ]
                if flag
            ],
            "gmail_query": result.gmail_query,
            "calendar_days": result.calendar_days,
            "weather_location": result.weather_location,
            "use_reasoning": result.use_reasoning,
        },
        "model": model,
        "ok": None,  # null = not yet reviewed by human
    }
    path = os.path.join(ROUTER_DATA_DIR, "routing_samples.jsonl")
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("Could not write routing sample: %s", exc)


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
    api_url = ROUTER_API_URL if ROUTER_MODEL else PRIMARY_API_URL
    api_key = ROUTER_API_KEY if ROUTER_MODEL else PRIMARY_API_KEY
    model = ROUTER_MODEL if ROUTER_MODEL else PRIMARY_MODEL

    prompt = get_prompt("ROUTER_USER").format(message=message)

    try:
        # response_format is supported by OpenAI and mlx-lm ≥ 0.21.
        # Older mlx-lm versions ignore it — extract_llm_json() handles that.
        raw = await call_llm_async(
            [
                {"role": "system", "content": get_prompt("ROUTER_SYSTEM")},
                {"role": "user", "content": prompt},
            ],
            model=model,
            api_url=api_url,
            api_key=api_key,
            temperature=0,
            max_tokens=400,
            json_response=True,
            no_think=True,
            timeout=ROUTER_TIMEOUT,
        )
        parsed = extract_llm_json(raw)

    except httpx.TimeoutException:
        logger.warning(
            "LLM router timeout (%.1fs) — falling back to embedding router",
            ROUTER_TIMEOUT,
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
            "LLM router error (%s) — falling back to embedding router",
            type(exc).__name__,
        )
        return None

    # ── Extract and validate fields ──
    intents: list[str] = parsed.get("intents", [])
    if not isinstance(intents, list):
        intents = []
    if not intents:
        intents = ["memory"]

    gmail_query: str = parsed.get("gmail_query") or ""
    calendar_days: int = int(parsed.get("calendar_days") or 7)
    weather_location: str = parsed.get("weather_location") or ""
    use_reasoning: bool = bool(parsed.get("use_reasoning", False))

    # ── Guardrail: prevent over-triggering reasoning on simple queries ──
    _simple_query = (
        len(message) < 80
        and "?" not in message
        and not any(
            k in message.lower() for k in ["analyse", "explique", "compare", "pourquoi"]
        )
    )

    if _simple_query:
        use_reasoning = False

    calendar_days = max(1, min(calendar_days, 90))

    result = RouterResult(
        use_memory="memory" in intents,
        use_rag="rag" in intents,
        use_web="web" in intents,
        use_weather="weather" in intents,
        use_gmail="gmail" in intents and google_available,
        use_calendar="calendar" in intents and google_available,
        use_briefing="briefing" in intents,
        use_self="self" in intents,
        use_portfolio="portfolio" in intents,
        use_reasoning=use_reasoning,
        gmail_query=gmail_query,
        calendar_days=calendar_days,
        weather_location=weather_location,
    )

    # ── Guardrail 2: reasoning only allowed with complex intents ──
    if result.use_reasoning and not (
        result.use_rag or result.use_web or result.use_portfolio
    ):
        result.use_reasoning = False

    logger.info(
        "LLM router [%s]: intents=%s | reasoning=%s → final=%s",
        model,
        intents,
        parsed.get("use_reasoning"),
        result.use_reasoning,
    )
    _log_routing_sample(message, result, model)
    return result

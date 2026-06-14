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
from datetime import datetime, timezone

import httpx
from config import (
    MAX_TOKENS_SHORT,
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
    gmail_query: str
    calendar_days: int
    weather_location: str = field(default="")
    rag_query: str = field(default="")
    use_small_talk: bool = field(
        default=False
    )  # skip profile/memory injection entirely
    use_reasoning: bool = field(default=False)
    project_name: str = field(default="")


_ALLOWED_INTENTS = {
    "memory",
    "rag",
    "web",
    "weather",
    "gmail",
    "calendar",
    "briefing",
    "portfolio",
    "self",
}

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
        "ts": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
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
            "rag_query": result.rag_query,
            "use_reasoning": result.use_reasoning,
        },
        "model": model,
        "ok": None,  # null = not yet reviewed by human
    }
    path = os.path.join(ROUTER_DATA_DIR, "routing_samples.jsonl")
    try:
        if os.path.exists(path) and os.path.getsize(path) > 10 * 1024 * 1024:
            import shutil

            shutil.move(path, path + ".bak")  # rotation simple
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("Could not write routing sample: %s", exc)


# ── Core call ─────────────────────────────────────────────────────────────


async def llm_route(message: str, google_available: bool = True, last_jarvis: str | None = None) -> RouterResult | None:
    """
    Call the router model (OpenAI-compatible /v1/chat/completions).
    Returns RouterResult on success, None on any failure → caller falls back
    to the embedding router automatically.
    last_jarvis: last assistant response (truncated) — injected as <last_jarvis> for context-aware routing.
    """
    # Truncate to 400 chars — enough to classify intent without risking the router answering.
    routing_message = message[:400]
    last_jarvis_block = (
        f"<last_jarvis>{last_jarvis[:300]}</last_jarvis>\n" if last_jarvis else ""
    )
    prompt = get_prompt("ROUTER_USER").format(
        message=routing_message, last_jarvis_block=last_jarvis_block
    )

    try:
        # response_format is supported by OpenAI and mlx-lm ≥ 0.21.
        # Older mlx-lm versions ignore it — extract_llm_json() handles that.
        raw = await call_llm_async(
            [
                {"role": "system", "content": get_prompt("ROUTER_SYSTEM")},
                {"role": "user", "content": prompt},
            ],
            model=ROUTER_MODEL,
            api_url=ROUTER_API_URL,
            api_key=ROUTER_API_KEY,
            temperature=0.0,  # Hermes is designed for deterministic structured output
            max_tokens=MAX_TOKENS_SHORT,
            json_response=True,
            no_think=True,
            timeout=ROUTER_TIMEOUT,
        )
        parsed = extract_llm_json(raw)

    except httpx.TimeoutException:
        logger.warning(
            "LLM router timeout (%.1fs) — no routing info (all intents off)",
            ROUTER_TIMEOUT,
        )
        return None
    except Exception as exc:
        logger.warning(
            "LLM router error (%s) — no routing info (all intents off): %s",
            type(exc).__name__,
            exc,
        )
        return None

    # ── Guard: parsed must be a non-empty dict ──────────────────────────────
    if not isinstance(parsed, dict):
        logger.warning(
            "LLM router returned non-dict (%s): %r — falling back",
            type(parsed).__name__,
            str(parsed)[:200],
        )
        return None

    logger.debug("LLM router raw output: %r", raw[:300])

    # ── Extract and validate fields ──
    try:
        intents: list[str] = parsed.get("intents", [])
        if not isinstance(intents, list):
            intents = []
        intents = [i for i in intents if i in _ALLOWED_INTENTS]
        if not intents:
            intents = ["memory"]

        gmail_query: str = parsed.get("gmail_query") or ""
        _cal_raw = parsed.get("calendar_days")
        if _cal_raw is None:
            calendar_days = 7
        else:
            try:
                calendar_days = int(_cal_raw)
            except (ValueError, TypeError):
                calendar_days = 7

        weather_location: str = (parsed.get("weather_location") or "") if "weather" in intents else ""
        rag_query: str = (parsed.get("rag_query") or "") if "rag" in intents else ""
        project_name: str = parsed.get("project_name") or ""
        use_reasoning: bool = bool(parsed.get("use_reasoning", False))
    except Exception as exc:
        logger.warning(
            "LLM router field extraction failed (%s): %s — parsed=%r",
            type(exc).__name__,
            exc,
            str(parsed)[:300],
        )
        return None

    # ── Guardrail: prevent over-triggering reasoning on simple queries ──
    # SUPPRESSION on fait confiance aux deux etages de routage embed + 3B
    # _simple_query = (
    #    len(message) < 80
    #    and "?" not in message
    #    and not any(
    #        k in message.lower() for k in ["analyse", "explique", "compare", "pourquoi"]
    #    )
    # )

    # if _simple_query:
    #    use_reasoning = False

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
        rag_query=rag_query,
        project_name=project_name,
    )

    logger.info(
        "LLM router [%s]: intents=%s | reasoning=%s → final=%s",
        ROUTER_MODEL,
        intents,
        parsed.get("use_reasoning"),
        result.use_reasoning,
    )
    _log_routing_sample(message, result, ROUTER_MODEL)
    return result

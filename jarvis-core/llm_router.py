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


# ── Prompt ───────────────────────────────────────────────────────────────

_ROUTER_SYSTEM = """\
You are a routing and query-building assistant for a personal AI called Jarvis.
Your only job is to analyze a user message and return a JSON routing decision.
Never answer the user's question — only output the JSON."""

_ROUTER_USER = """\
Analyze this message and return a JSON routing decision.

intents: list of strings — which data sources are needed.
  Choose from: "memory", "rag", "web", "gmail", "calendar", "briefing", "self", "portfolio"

  - memory    : user refers to past conversations, personal facts, preferences, or anything
                Jarvis should already know about them. Also use for greetings, social messages,
                updates, thanks, or any conversational message with no specific data need.
  - rag       : user explicitly asks about their own documents, files, notes, or knowledge base.
  - web       : user asks about current events, news, live prices, weather, or anything requiring
                up-to-date information from the internet.
  - gmail     : user asks about emails, inbox, messages, or a specific sender / subject.
  - calendar  : user asks about their agenda, appointments, schedule, or upcoming events.
  - briefing  : user explicitly asks for their morning briefing or daily summary.
  - self      : user EXPLICITLY asks Jarvis about its own internal state — its goals, current
                focus, what it is thinking, or how it feels right now.
                ONLY fire for direct introspective questions such as:
                  "what are your goals", "quel est ton focus", "comment te sens-tu en ce moment",
                  "à quoi tu penses en ce moment", "what is your current objective",
                  "quel est ton état émotionnel", "qu'est-ce que tu ressens"
                Do NOT fire for casual social greetings like "comment tu vas", "ça va ?",
                "how are you" — these are chitchat → use ["memory"] instead.
                Do NOT fire when the user merely mentions Jarvis, addresses it, compliments it,
                or discusses its code / capabilities / improvements.
  - portfolio : user asks about their stock portfolio, share prices, positions, P&L, dividends,
                trading alerts, or any bourse / finance / portefeuille topic.

  Multiple intents allowed.
  Default rule: if the message is a greeting, update, chitchat, or nothing matches clearly
  → use ["memory"].

gmail_query: string or null
  Gmail search query (Gmail syntax). Use only when "gmail" is in intents.
  Examples:
    "from:amazon newer_than:7d"      (recent Amazon orders)
    "is:unread"                      (unread emails)
    "from:paul subject:invoice"      (invoice from Paul)
    "livraison colis newer_than:3d"  (delivery tracking)
    "newer_than:7d"                  (all recent emails)
  Set to null if gmail not needed.

calendar_days: integer or null
  Days ahead to fetch. Use only when "calendar" is in intents.
    7  → today / this week
    30 → this month
  Set to null if calendar not needed.

use_reasoning: boolean
  Sends the query to a powerful cloud model. Use VERY sparingly — default is false.
  true  ONLY for: medical/legal/regulatory analysis, hard multi-step logic puzzles,
                  complex code debugging across many files, deep scientific reasoning,
                  or tasks explicitly requiring expert-level nuanced judgment.
  false for everything else: chat, questions, summaries, portfolio, tasks, translations,
                  writing, coding assistance, explanations, data formatting, web lookup.
  When in doubt → false. The local model is strong enough for standard requests.

memory_scope: string
  Which memory layer to search. One of: "episodic", "autobiographical", "profile", "auto".
  - "episodic"        : search recent conversation summaries (past sessions, events, things user said).
                        Use when the question is about a specific past exchange or recent event.
                        Examples: "tu te souviens quand je t'ai parlé de X", "last week I told you"
  - "autobiographical": search long-term milestones and stable facts stored over months.
                        Use when the question is about major projects, life history, or identity.
                        Examples: "what are my current projects", "où en est mon projet Y"
  - "profile"         : user asks about preferences, settings, or static facts already in context.
                        No deep vector search needed.
                        Examples: "what language do I prefer", "what's my timezone"
  - "auto"            : search all memory layers. Use when unsure (default).

conversation_type: string
  Classify the nature of the message. One of: "conversational", "task", "question".
  - "conversational": greeting, thanks, social exchange, chitchat, sharing news, emotional message.
                      No document retrieval needed.
                      Examples: "bonjour", "merci", "super boulot aujourd'hui", "bonne nuit"
  - "task"          : user asks Jarvis to perform an action or create something.
                      Examples: "envoie un email à Paul", "crée une liste", "summarize this"
  - "question"      : user seeks information, an explanation, or a specific fact.
                      Examples: "qu'est-ce que X", "how does Y work", "quel temps fait-il"
  Default: "conversational" when in doubt.

User message: {message}

Respond with valid JSON only. No explanation, no markdown, no code fences.
Example: {{"intents": ["gmail", "web"], "gmail_query": "newer_than:7d", "calendar_days": null, "use_reasoning": false, "memory_scope": "auto", "conversation_type": "question"}}"""


# ── Result dataclass ──────────────────────────────────────────────────────

@dataclass
class RouterResult:
    use_memory:       bool
    use_rag:          bool
    use_web:          bool
    use_gmail:        bool
    use_calendar:     bool
    use_briefing:     bool
    use_self:         bool
    use_portfolio:    bool
    use_reasoning:    bool
    gmail_query:      str
    calendar_days:    int
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

    gmail_query:   str  = parsed.get("gmail_query") or ""
    calendar_days: int  = int(parsed.get("calendar_days") or 7)
    use_reasoning: bool = bool(parsed.get("use_reasoning", False))

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
        use_gmail         = "gmail"     in intents and google_available,
        use_calendar      = "calendar"  in intents and google_available,
        use_briefing      = "briefing"  in intents,
        use_self          = "self"      in intents,
        use_portfolio     = "portfolio" in intents,
        use_reasoning     = use_reasoning,
        gmail_query       = gmail_query,
        calendar_days     = calendar_days,
        memory_scope      = memory_scope,
        conversation_type = conversation_type,
    )

    logger.info(
        "LLM router [%s]: intents=%s gmail_query=%r calendar_days=%d use_reasoning=%s use_portfolio=%s memory_scope=%s conv_type=%s",
        ROUTER_MODEL, intents, gmail_query, calendar_days, use_reasoning, result.use_portfolio,
        memory_scope, conversation_type,
    )
    return result

import json
import logging
import os

from dotenv import load_dotenv

load_dotenv("/opt/jarvis/.env")

logger = logging.getLogger("jarvis-config")

# ── Shared OpenAI credentials (fallback for any tier not explicitly configured) ──
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", os.getenv("OPENAI_URL", "https://api.openai.com/v1"))

# ── Tier 1 — Router model (fast, cheap intent classifier) ─────────────────
# Now:    GPT-4.1-nano          → set nothing, uses OPENAI credentials
# Future: Qwen3-7B via mlx-lm  → set ROUTER_API_URL + ROUTER_API_KEY + ROUTER_MODEL
# Disable LLM router entirely:   set ROUTER_MODEL=""  (falls back to embedding router)
ROUTER_API_URL = os.getenv("ROUTER_API_URL") or OPENAI_API_URL
ROUTER_API_KEY = os.getenv("ROUTER_API_KEY") or OPENAI_API_KEY
ROUTER_MODEL   = os.getenv("ROUTER_MODEL", "gpt-4.1-nano")   # empty string intentionally disables LLM router
ROUTER_TIMEOUT = float(os.getenv("ROUTER_TIMEOUT") or "6")

# ── Tier 2 — Primary model (standard chat, trading, briefing, self-reflection) ──
# Now:    GPT-4o-mini              → set nothing, uses OPENAI credentials
# Future: Qwen3-30B-A3B via mlx-lm → set PRIMARY_API_URL + PRIMARY_API_KEY + PRIMARY_MODEL
PRIMARY_MODEL   = os.getenv("PRIMARY_MODEL", "gpt-4o-mini")
PRIMARY_API_URL = os.getenv("PRIMARY_API_URL") or OPENAI_API_URL
PRIMARY_API_KEY = os.getenv("PRIMARY_API_KEY") or OPENAI_API_KEY
PRIMARY_TIMEOUT = float(os.getenv("PRIMARY_TIMEOUT") or "60")

# ── Tier 2b — Analysis model (mood extraction, memory consolidation) ─────
# Now:    GPT-4o-mini              → defaults to PRIMARY credentials
# Future: Qwen3-30B-A3B via mlx-lm → set ANALYSIS_API_URL + ANALYSIS_API_KEY + ANALYSIS_MODEL
ANALYSIS_MODEL   = os.getenv("ANALYSIS_MODEL") or PRIMARY_MODEL
ANALYSIS_API_URL = os.getenv("ANALYSIS_API_URL") or PRIMARY_API_URL
ANALYSIS_API_KEY = os.getenv("ANALYSIS_API_KEY") or PRIMARY_API_KEY

# ── Tier 3 — Reasoning model (complex queries only, cloud-gated) ─────────
# Now:    GPT-5.1 (cloud)   → set REASONING_MODEL + optionally REASONING_API_*
# Only reached when the router sets use_reasoning=True.
REASONING_MODEL   = os.getenv("REASONING_MODEL") or PRIMARY_MODEL
REASONING_API_URL = os.getenv("REASONING_API_URL") or OPENAI_API_URL
REASONING_API_KEY = os.getenv("REASONING_API_KEY") or OPENAI_API_KEY
REASONING_TIMEOUT = float(os.getenv("REASONING_TIMEOUT") or "90")

# ── Vision model (image description — first stage of two-stage pipeline) ──
# Set to a vision-capable model (Qwen2.5-VL, gpt-4o, gpt-5.1, …).
# Leave empty to disable image support (images will be ignored with a warning).
VISION_MODEL   = os.getenv("VISION_MODEL") or PRIMARY_MODEL
VISION_API_URL = os.getenv("VISION_API_URL") or OPENAI_API_URL
VISION_API_KEY = os.getenv("VISION_API_KEY") or OPENAI_API_KEY
VISION_TIMEOUT = float(os.getenv("VISION_TIMEOUT") or "30")

# ── Model compatibility helpers ───────────────────────────────────────────

def is_qwen(model: str) -> bool:
    """True for Qwen models served locally via mlx-lm or Ollama."""
    return "qwen" in (model or "").lower()

def tokens_param(model: str) -> str:
    """
    Return the correct token-limit parameter name for the model.
    OpenAI o1/o3/gpt-5.x require 'max_completion_tokens'.
    All others (gpt-4o, Qwen via mlx-lm, Gemini-compat, etc.) use 'max_tokens'.

    Usage: json={..., **{tokens_param(REASONING_MODEL): 512}, ...}
    """
    _needs_completion = ("o1-", "o3-", "gpt-5", "gpt-4.5")
    return "max_completion_tokens" if any(x in (model or "") for x in _needs_completion) else "max_tokens"

def no_think_suffix(model: str) -> str:
    """
    Return '/no_think' suffix for Qwen3 models to disable chain-of-thought.
    Required for JSON-output tasks (router, analyzer) — prevents <think> blocks
    from breaking JSON parsing. Empty string for all other models.
    """
    return "\n/no_think" if is_qwen(model) else ""

# ── Infrastructure ────────────────────────────────────────────────────────
REDIS_URL              = os.getenv("REDIS_URL", "redis://redis:6379")
QDRANT_URL             = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION      = os.getenv("QDRANT_COLLECTION", "open-webui_knowledge")
QDRANT_MEMORY_COLLECTION = os.getenv("QDRANT_MEMORY_COLLECTION", "jarvis_memory")
RAG_TOP_K              = int(os.getenv("RAG_TOP_K", os.getenv("QDRANT_TOP_K", "5")))
RAG_SCORE_THRESHOLD    = float(os.getenv("RAG_SCORE_THRESHOLD", "0.4"))

# ── Features ──────────────────────────────────────────────────────────────
ENABLE_ANALYSIS  = os.getenv("ENABLE_ANALYSIS", "true").lower() == "true"
SELF_MEMORY_PATH = os.getenv("SELF_MEMORY_PATH", "/app/data/jarvis-self.json")
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ── Google Services (Gmail read+send, Calendar read) ──────────────────────
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REFRESH_TOKEN = os.getenv("GOOGLE_REFRESH_TOKEN", "")
GOOGLE_CALENDAR_ID   = os.getenv("GOOGLE_CALENDAR_ID", "primary")

# ── Morning Briefing ──────────────────────────────────────────────────────
BRIEFING_ENABLED  = os.getenv("BRIEFING_ENABLED", "true").lower() == "true"
BRIEFING_TIME     = os.getenv("BRIEFING_TIME", "07:30")        # HH:MM
BRIEFING_TIMEZONE = os.getenv("BRIEFING_TIMEZONE", "Europe/Paris")

# ── Proto-self reflection loop ─────────────────────────────────────────────
REFLECTION_INTERVAL_HOURS = int(os.getenv("REFLECTION_INTERVAL_HOURS", "6"))

# ── Autocoding — prompt self-modification ─────────────────────────────────
# Number of times a knowledge gap must be flagged before a prompt-refine is triggered
REFINE_PROMPT_THRESHOLD = int(os.getenv("REFINE_PROMPT_THRESHOLD", "3"))
# Stores prompt_proposals.json + prompt_overrides.json (inside the /app/data volume)
PROMPT_DATA_DIR = os.path.join(os.path.dirname(SELF_MEMORY_PATH), "prompts")

# ── Conversation storage limits ───────────────────────────────────────────
CHAT_MAX_MESSAGES = int(os.getenv("CHAT_MAX_MESSAGES", "100"))   # server-side Redis LTRIM cap
IOS_MAX_MESSAGES  = int(os.getenv("IOS_MAX_MESSAGES",  "50"))    # messages returned to iOS app
CHAT_LOG_TTL      = int(os.getenv("CHAT_LOG_TTL_DAYS", "90")) * 86400  # raw log expiry (seconds)

# ── Memory thresholds ─────────────────────────────────────────────────────
IMPORTANCE_THRESHOLD              = 0.35
RECALL_MEMORY_SIMILARITY_THRESHOLD = 0.7
AUTOBIO_IMPORTANCE_THRESHOLD      = 0.6
NOVELTY_THRESHOLD                 = 0.15
PROJECT_THRESHOLD                 = 0.6


# ══════════════════════════════════════════════════
#  USER MANAGEMENT — loaded from users_list.json
# ══════════════════════════════════════════════════

USERS_LIST_PATH = os.getenv("USERS_LIST", "/app/data/users_list.json")

# Full user objects keyed by access code
USERS: dict[str, dict] = {}

try:
    with open(USERS_LIST_PATH, encoding="utf-8") as _f:
        _raw: list[dict] = json.load(_f)
    for _u in _raw:
        code = _u.get("code", "").strip()
        if code:
            USERS[code] = _u
    logger.info("Loaded %d users from %s", len(USERS), USERS_LIST_PATH)
except FileNotFoundError:
    logger.error("users_list.json not found at %s — no users loaded", USERS_LIST_PATH)
except json.JSONDecodeError as _e:
    logger.error("users_list.json is invalid JSON: %s", _e)

# ── Derived dicts (backward-compatible with existing code) ────────────────
# code → firstname
USER_CODES: dict[str, str] = {code: u["firstname"] for code, u in USERS.items()}

# code → email (empty string = no email delivery)
USER_EMAILS: dict[str, str] = {code: u.get("mail", "") for code, u in USERS.items()}

# email → code (reverse index — used by the OpenWebUI proxy to identify users by email)
EMAIL_TO_CODE: dict[str, str] = {
    u.get("mail", "").lower(): code
    for code, u in USERS.items()
    if u.get("mail")
}

# code → city for weather
USER_CITIES: dict[str, str] = {code: u.get("city", "Paris") for code, u in USERS.items()}

# code → timezone name (IANA)
USER_TIMEZONES: dict[str, str] = {code: u.get("timezone", "Europe/Paris") for code, u in USERS.items()}

# codes with trading enabled
USER_TRADING: list[str] = [code for code, u in USERS.items() if u.get("trading", False)]

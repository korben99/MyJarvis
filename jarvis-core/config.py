import json
import logging
import os

from dotenv import load_dotenv

load_dotenv("/opt/jarvis/.env")

logger = logging.getLogger("jarvis-config")

# ── Core API (primary responder) ──────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", os.getenv("OPENAI_URL", "https://api.openai.com/v1"))
PRIMARY_MODEL  = os.getenv("PRIMARY_MODEL", "gpt-4o-mini")
ANALYSIS_MODEL = os.getenv("ANALYSIS_MODEL", "gpt-4o-mini")

# ── Tier 1 — Router model (fast, cheap intent classifier) ─────────────────
# Now:    GPT-4.1-nano  (uses OPENAI_API_KEY by default)
# Future: Qwen2.5-7B   via mlx-lm  →  set ROUTER_API_URL + ROUTER_MODEL
# Disable the LLM router entirely: set ROUTER_MODEL=""  (falls back to embedding router)
# Use `or fallback` instead of the default= parameter so that an empty string
# set by docker-compose (e.g. ROUTER_API_KEY=${ROUTER_API_KEY:-}) is treated
# the same as "not set" and falls back to the primary credentials.
ROUTER_API_URL = os.getenv("ROUTER_API_URL") or "https://api.openai.com/v1"
ROUTER_API_KEY = os.getenv("ROUTER_API_KEY") or OPENAI_API_KEY
ROUTER_MODEL   = os.getenv("ROUTER_MODEL", "gpt-4.1-nano")   # empty string intentionally disables LLM router
ROUTER_TIMEOUT = float(os.getenv("ROUTER_TIMEOUT") or "6")

# ── Tier 2 — Reasoning model (handles ALL user responses) ────────────────
# Now:    GPT-5.1       (uses OPENAI_API_KEY by default)
# Future: Qwen2.5-32B  via mlx-lm  →  set REASONING_API_URL + REASONING_MODEL
# Defaults to PRIMARY_MODEL so the system works out-of-the-box without configuration.
REASONING_MODEL   = os.getenv("REASONING_MODEL") or PRIMARY_MODEL
REASONING_API_URL = os.getenv("REASONING_API_URL") or OPENAI_API_URL
REASONING_API_KEY = os.getenv("REASONING_API_KEY") or OPENAI_API_KEY
REASONING_TIMEOUT = float(os.getenv("REASONING_TIMEOUT") or "90")

# ── Vision model (image description — first stage of two-stage pipeline) ──
# Set to a vision-capable model (Qwen2.5-VL, gpt-4o, gpt-5.1, …).
# Leave empty to disable image support (images will be ignored with a warning).
VISION_MODEL   = os.getenv("VISION_MODEL") or ""
VISION_API_URL = os.getenv("VISION_API_URL") or OPENAI_API_URL
VISION_API_KEY = os.getenv("VISION_API_KEY") or OPENAI_API_KEY
VISION_TIMEOUT = float(os.getenv("VISION_TIMEOUT") or "30")

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

# ── Conversation storage limits ───────────────────────────────────────────
CHAT_MAX_MESSAGES = int(os.getenv("CHAT_MAX_MESSAGES", "100"))   # server-side Redis LTRIM cap
IOS_MAX_MESSAGES  = int(os.getenv("IOS_MAX_MESSAGES",  "50"))    # messages returned to iOS app
CHAT_LOG_TTL      = int(os.getenv("CHAT_LOG_TTL_DAYS", "90")) * 86400  # raw log expiry (seconds)

# ── Memory thresholds ─────────────────────────────────────────────────────
IMPORTANCE_THRESHOLD              = 0.35
RECALL_MEMORY_SIMILARITY_THRESHOLD = 0.7
AUTOBIO_IMPORTANCE_THRESHOLD      = 0.6
NOVELTY_THRESHOLD                 = 0.15


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

# code → city for weather
USER_CITIES: dict[str, str] = {code: u.get("city", "Paris") for code, u in USERS.items()}

# codes with trading enabled
USER_TRADING: list[str] = [code for code, u in USERS.items() if u.get("trading", False)]

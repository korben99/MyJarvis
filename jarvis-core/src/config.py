import json
import logging
import os

from dotenv import load_dotenv

load_dotenv("/opt/jarvis/.env")

logger = logging.getLogger("jarvis-config")

# ── Shared OpenAI credentials (fallback for any tier not explicitly configured) ──
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_URL = os.getenv(
    "OPENAI_API_URL", os.getenv("OPENAI_URL", "https://api.openai.com/v1")
)

# ── Tier 1 — Router model (fast, cheap intent classifier) ─────────────────
# Now:    GPT-4.1-nano          → set nothing, uses OPENAI credentials
# Future: Qwen3-7B via mlx-lm  → set ROUTER_API_URL + ROUTER_API_KEY + ROUTER_MODEL
# Disable LLM router entirely:   set ROUTER_MODEL=""  (falls back to embedding router)
ROUTER_API_URL = os.getenv("ROUTER_API_URL") or OPENAI_API_URL
ROUTER_API_KEY = os.getenv("ROUTER_API_KEY") or OPENAI_API_KEY
ROUTER_MODEL = os.getenv(
    "ROUTER_MODEL", "gpt-4.1-nano"
)  # empty string intentionally disables LLM router
ROUTER_TIMEOUT = float(os.getenv("ROUTER_TIMEOUT") or "6")

# ── Tier 2 — Primary model (standard chat, trading, briefing, self-reflection) ──
# Now:    GPT-4o-mini              → set nothing, uses OPENAI credentials
# Future: Qwen3-30B-A3B via mlx-lm → set PRIMARY_API_URL + PRIMARY_API_KEY + PRIMARY_MODEL
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gpt-4o-mini")
PRIMARY_API_URL = os.getenv("PRIMARY_API_URL") or OPENAI_API_URL
PRIMARY_API_KEY = os.getenv("PRIMARY_API_KEY") or OPENAI_API_KEY
PRIMARY_TIMEOUT = float(os.getenv("PRIMARY_TIMEOUT") or "60")

# ── Tier 3 — Reasoning model (complex queries only, cloud-gated) ─────────
# Now:    GPT-5.1 (cloud)   → set REASONING_MODEL + optionally REASONING_API_*
# Only reached when the router sets use_reasoning=True.
REASONING_MODEL = os.getenv("REASONING_MODEL") or PRIMARY_MODEL
REASONING_API_URL = os.getenv("REASONING_API_URL") or OPENAI_API_URL
REASONING_API_KEY = os.getenv("REASONING_API_KEY") or OPENAI_API_KEY
REASONING_TIMEOUT = float(os.getenv("REASONING_TIMEOUT") or "90")

# ── Vision model (image description — first stage of two-stage pipeline) ──
# Set to a vision-capable model (Qwen2.5-VL, gpt-4o, gpt-5.1, …).
# Leave empty to disable image support (images will be ignored with a warning).
VISION_MODEL = os.getenv("VISION_MODEL") or PRIMARY_MODEL
VISION_API_URL = os.getenv("VISION_API_URL") or OPENAI_API_URL
VISION_API_KEY = os.getenv("VISION_API_KEY") or OPENAI_API_KEY
VISION_TIMEOUT = float(os.getenv("VISION_TIMEOUT") or "30")

# ── Local LLM mode — Apple Silicon / mlx-lm (M4 Pro) ─────────────────────
# Activé par LLM_LOCAL=yes dans .env.
# Écrase Router et Primary pour pointer vers les serveurs mlx-lm locaux.
# Le Reasoning reste sur cloud (cas rares, latence acceptable).
#
# Sur le Mac (deux terminaux) :
#   python -m mlx_lm.server --model $LOCAL_ROUTER_MODEL  --port $LOCAL_ROUTER_PORT
#   python -m mlx_lm.server --model $LOCAL_PRIMARY_MODEL --port $LOCAL_PRIMARY_PORT
#
# Mémoire unifiée conseillée :
#   24 GB → Qwen2.5-7B  (router) + Qwen2.5-14B  (primary)
#   48 GB → Qwen2.5-7B  (router) + Qwen2.5-32B  (primary)
LLM_LOCAL = os.getenv("LLM_LOCAL", "").lower() in ("yes", "true", "1")

if LLM_LOCAL:
    # Mode import direct MLX — pas de serveurs HTTP mlx-lm.
    # helpers.py route vers call_llm_local / call_llm_local_async directement.
    # Les API_URL / API_KEY ne sont pas utilisées pour l'inférence en mode local.
    ROUTER_MODEL = os.getenv(
        "ROUTER_MODEL_LOCAL", "mlx-community/Qwen2.5-3B-Instruct-8bit"
    )
    PRIMARY_MODEL = os.getenv("PRIMARY_MODEL_LOCAL", "Qwen/Qwen3-30B-A3B-MLX-4bit")
    VISION_MODEL = os.getenv(
        "VISION_MODEL_LOCAL", "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
    )

    logger.info(
        "Mode LLM local activé (import direct MLX) — router: %s  primary: %s  vision: %s",
        ROUTER_MODEL,
        PRIMARY_MODEL,
        VISION_MODEL,
    )

# ── Model compatibility helpers ───────────────────────────────────────────


def is_qwen(model: str) -> bool:
    """True for Qwen models served locally via mlx-lm or Ollama."""
    return "qwen" in (model or "").lower()


def is_qwen3(model: str) -> bool:
    """True specifically for Qwen3 models (support thinking_budget in chat template)."""
    return "qwen3" in (model or "").lower()


# Max tokens allowed for the <think> block in Qwen3 chat responses.
# Set via env THINKING_BUDGET_TOKENS. 0 = no budget (unlimited).
# Default 1024 = ~1-2s of thinking on 30B, enough for most questions.
THINKING_BUDGET_TOKENS = int(os.getenv("THINKING_BUDGET_TOKENS", "1024"))


def tokens_param(model: str) -> str:
    """
    Return the correct token-limit parameter name for the model.
    OpenAI o1/o3/gpt-5.x require 'max_completion_tokens'.
    All others (gpt-4o, Qwen via mlx-lm, Gemini-compat, etc.) use 'max_tokens'.

    Usage: json={..., **{tokens_param(REASONING_MODEL): 512}, ...}
    """
    _needs_completion = ("o1-", "o3-", "gpt-5", "gpt-4.5")
    return (
        "max_completion_tokens"
        if any(x in (model or "") for x in _needs_completion)
        else "max_tokens"
    )


def no_think_suffix(model: str) -> str:
    """
    Return '/no_think' suffix for Qwen3 models to disable chain-of-thought.
    Required for JSON-output tasks (router, analyzer) — prevents <think> blocks
    from breaking JSON parsing. Empty string for all other models.
    """
    return "\n/no_think" if is_qwen(model) else ""


# ── Infrastructure ────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "open-webui_knowledge")
QDRANT_MEMORY_COLLECTION = os.getenv("QDRANT_MEMORY_COLLECTION", "jarvis_memory")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", os.getenv("QDRANT_TOP_K", "5")))
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.4"))

# ── Features ──────────────────────────────────────────────────────────────
SELF_MEMORY_PATH = os.getenv("SELF_MEMORY_PATH", "/app/data/jarvis-self.json")
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ── Google Services (Gmail read+send, Calendar read) ──────────────────────
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID", "primary")

# ── Morning Briefing ──────────────────────────────────────────────────────
BRIEFING_ENABLED = os.getenv("BRIEFING_ENABLED", "true").lower() == "true"
BRIEFING_TIME = os.getenv("BRIEFING_TIME", "07:30")  # HH:MM
BRIEFING_TIMEZONE = os.getenv("BRIEFING_TIMEZONE", "Europe/Paris")

# ── Proto-self reflection loop ─────────────────────────────────────────────
REFLECTION_INTERVAL_HOURS = int(os.getenv("REFLECTION_INTERVAL_HOURS", "6"))
MAX_REFLECTION_TOKENS = 6000  # think block (~2000-4000) + JSON output (~500)
MAX_CHAIN_ITERATIONS = int(
    os.getenv("MAX_CHAIN_ITERATIONS", "3")
)  # max actions per reflection cycle
# ── Autocoding — prompt self-modification ─────────────────────────────────
# Number of times a knowledge gap must be flagged before a prompt-refine is triggered
REFINE_PROMPT_THRESHOLD = int(os.getenv("REFINE_PROMPT_THRESHOLD", "3"))
# Stores prompt_proposals.json + prompt_overrides.json (inside the /app/data volume)
PROMPT_DATA_DIR = os.path.join(os.path.dirname(SELF_MEMORY_PATH), "prompts")

# ── Router training data collector ────────────────────────────────────────
# JSONL file of (message, routing_json) pairs for future LoRA fine-tuning.
# Mount on host: ./RouterData:/app/router_data  (see docker-compose.yml)
ROUTER_DATA_DIR = os.getenv("ROUTER_DATA_DIR", "/app/router_data")

# ── Conversation storage limits ───────────────────────────────────────────
CHAT_MAX_MESSAGES = int(
    os.getenv("CHAT_MAX_MESSAGES", "100")
)  # server-side Redis LTRIM cap
IOS_MAX_MESSAGES = int(
    os.getenv("IOS_MAX_MESSAGES", "50")
)  # messages returned to iOS app
CHAT_LOG_TTL = (
    int(os.getenv("CHAT_LOG_TTL_DAYS", "90")) * 86400
)  # raw log expiry (seconds)

# ── Memory thresholds ─────────────────────────────────────────────────────
IMPORTANCE_THRESHOLD = 0.35
RECALL_MEMORY_SIMILARITY_THRESHOLD = 0.7
AUTOBIO_IMPORTANCE_THRESHOLD = 0.6
NOVELTY_THRESHOLD = 0.15
PROJECT_THRESHOLD = 0.6

# Fenêtre de récence pour le scoring des souvenirs autobiographiques dans search_memory().
# Les souvenirs épisodiques utilisent une fenêtre de 30 jours (hardcodée).
# Les autobiographiques (milestones durables) méritent une fenêtre plus longue pour rester
# pertinents dans le score de rappel même après plusieurs mois.
AUTOBIO_RECENCY_WINDOW_DAYS = int(os.getenv("AUTOBIO_RECENCY_WINDOW_DAYS", "365"))

# Fenêtre de rétention épisodique : les souvenirs épisodiques plus récents que cette valeur
# sont EXEMPTÉS de la consolidation mensuelle — ils restent accessibles en recall pendant
# cette période avant d'être compressés en autobio.
# 45 jours = fenêtre confortable (6 semaines de contexte moyen terme).
# Min raisonnable : 30j. Max : 60j.
EPISODIC_RETENTION_DAYS = int(os.getenv("EPISODIC_RETENTION_DAYS", "45"))

# Seuil de similarité cosine au-dessus duquel un nouvel événement autobiographique est
# considéré comme un doublon et ignoré (pas de stockage Qdrant).
# 0.85 = très similaire en sens mais formulation différente tolérée.
# Augmenter vers 0.95 pour être plus permissif (moins de dédup).
AUTOBIO_DEDUP_THRESHOLD = float(os.getenv("AUTOBIO_DEDUP_THRESHOLD", "0.85"))

# Décroissance mémorielle mensuelle des souvenirs autobiographiques.
# DECAY_FACTOR        : multiplicateur appliqué à importance à chaque passe (0.85 = -15 %/mois).
# DECAY_THRESHOLD     : en dessous, le souvenir est supprimé de Qdrant.
# DECAY_DURABLE_MIN   : importance initiale >= cette valeur → exempt de décroissance.
# CONSOLIDATION_IMPORTANCE : score assigné aux milestones issus de la consolidation mensuelle.
#                            Doit être == DECAY_DURABLE_MIN pour que ces souvenirs soient permanents.
MEMORY_DECAY_FACTOR = float(os.getenv("MEMORY_DECAY_FACTOR", "0.85"))
MEMORY_DECAY_THRESHOLD = float(os.getenv("MEMORY_DECAY_THRESHOLD", "0.15"))
MEMORY_DECAY_DURABLE_MIN = float(os.getenv("MEMORY_DECAY_DURABLE_MIN", "1.0"))
MEMORY_CONSOLIDATION_IMPORTANCE = float(
    os.getenv("MEMORY_CONSOLIDATION_IMPORTANCE", "1.0")
)

# Durée de rétention des projets "done" dans la liste Redis active (en jours).
# Passé ce délai, un projet terminé est retiré automatiquement au prochain update.
# Les projets importants sont de toute façon consolidés vers Qdrant (mémoire autobiographique)
# lors du nightly review, donc l'information n'est pas perdue.
DONE_PROJECT_TTL_DAYS = int(os.getenv("DONE_PROJECT_TTL_DAYS", "180"))

# Nombre maximum d'entrées conservées dans le journal de croissance (growth_log) de jarvis-self.json.
# Chaque entrée représente un résumé quotidien par utilisateur (1 entrée/utilisateur/jour).
# Avec 2 utilisateurs actifs, 180 = environ 3 mois de journal avant rotation.
GROWTH_LOG_MAX_ENTRIES = int(os.getenv("GROWTH_LOG_MAX_ENTRIES", "180"))


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

# codes with admin: true — allowed to approve/reject prompt proposals
USER_ADMINS: set[str] = {code for code, u in USERS.items() if u.get("admin") is True}

# code → email (empty string = no email delivery)
USER_EMAILS: dict[str, str] = {code: u.get("mail", "") for code, u in USERS.items()}

# email → code (reverse index — used by the OpenWebUI proxy to identify users by email)
EMAIL_TO_CODE: dict[str, str] = {
    u.get("mail", "").lower(): code for code, u in USERS.items() if u.get("mail")
}

# code → city for weather
USER_CITIES: dict[str, str] = {
    code: u.get("city", "Paris") for code, u in USERS.items()
}

# code → timezone name (IANA)
USER_TIMEZONES: dict[str, str] = {
    code: u.get("timezone", "Europe/Paris") for code, u in USERS.items()
}

# codes with trading enabled
USER_TRADING: list[str] = [code for code, u in USERS.items() if u.get("trading", False)]

# codes with a connected Google account (Gmail + Calendar access)
# Only these users receive calendar/Gmail data in briefings and chat.
USER_GOOGLE: set[str] = {code for code, u in USERS.items() if u.get("google", False)}

# Per-user Google OAuth refresh tokens.
# Loaded from GOOGLE_REFRESH_TOKEN_<CODE> env vars for each user with "google": true.
# No fallback — each user must have their own token (prevents cross-account calendar leaks).
# To enable a user: set "google": true in users_list.json AND add their token to .env.
GOOGLE_USER_TOKENS: dict[str, str] = {}
for _code in USER_GOOGLE:
    _tok = os.getenv(f"GOOGLE_REFRESH_TOKEN_{_code}", "")
    if _tok:
        GOOGLE_USER_TOKENS[_code] = _tok

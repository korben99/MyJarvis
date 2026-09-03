"""Singletons Redis + Qdrant partagés, et petits helpers JSON / session.

Une seule connexion Redis et une seule connexion Qdrant pour tout le process. Les
wrappers redis_get_json / redis_set_json encapsulent le motif get/set + json.
"""

import json
import os
from threading import Lock

import redis
from config import CHAT_LOG_TTL, QDRANT_URL, REDIS_URL
from qdrant_client import QdrantClient

from .logging_setup import get_logger

logger = get_logger("jarvis-helpers")

_redis_client: redis.Redis | None = None
_redis_lock = Lock()
_qdrant_client: QdrantClient | None = None
_qdrant_lock = Lock()


def get_redis() -> redis.Redis:
    """Return the shared Redis connection (singleton, thread-safe)."""
    global _redis_client
    if _redis_client is None:
        with _redis_lock:
            if _redis_client is None:
                _redis_client = redis.from_url(
                    REDIS_URL,
                    decode_responses=True,
                    socket_keepalive=True,
                    health_check_interval=30,
                    retry_on_timeout=True,
                )
    return _redis_client


_SESSION_SUMMARY_PREFIX = "session:summary:"


def get_session_summary_data(user_code: str, session_id: str) -> dict | None:
    """Return {text, last_ts} for the session conversation summary, or None."""
    raw = get_redis().get(f"{_SESSION_SUMMARY_PREFIX}{user_code}:{session_id}")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def set_session_summary_data(user_code: str, session_id: str, text: str, last_ts: float) -> None:
    """Persist conversation summary with the timestamp of the last covered message."""
    data = json.dumps({"text": text, "last_ts": last_ts}, ensure_ascii=False)
    get_redis().setex(
        f"{_SESSION_SUMMARY_PREFIX}{user_code}:{session_id}",
        CHAT_LOG_TTL,
        data,
    )


_STICKY_RAG_PREFIX = "jarvis:sticky_rag:"

# TTL PROPRE au contexte documentaire, et non celui des journaux de chat.
#
# Le collant sert à tenir un document pendant qu'on en PARLE, ce qui se compte en heures.
# Réutiliser `CHAT_LOG_TTL` (90 j) ferait se réinjecter un document ouvert une fois à chaque
# tour mémoire pendant trois mois. Passé ce délai, une vraie question relance un vrai RAG.
STICKY_RAG_TTL = int(os.getenv("STICKY_RAG_TTL_HOURS", "6")) * 3600

# Longueur minimale d'un extrait jugé digne d'être retenu. La 2e étape du RAG cherche DANS le
# document déjà identifié avec `score_threshold=0.0` (rag.py) — délibéré, mais elle laisse
# alors passer des fragments vides de sens qu'aucun seuil de score n'arrête. Filtrer ici plutôt
# que d'abaisser ce seuil : le fragment reste utilisable pour le tour en cours, il n'est
# simplement pas jugé assez substantiel pour être REJOUÉ sur les tours suivants.
_STICKY_MIN_CHARS = int(os.getenv("STICKY_RAG_MIN_CHARS", "200"))


def get_sticky_rag(user_code: str, session_id: str) -> list | None:
    """Return the last RAG chunks stored for this session, or None."""
    raw = get_redis().get(f"{_STICKY_RAG_PREFIX}{user_code}:{session_id}")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def set_sticky_rag(user_code: str, session_id: str, chunks: list) -> None:
    """Persist RAG chunks for automatic re-injection on subsequent memory turns.

    Les fragments trop courts sont écartés : ils ne portent pas de quoi nourrir un tour
    ultérieur, et ce sont eux qui polluaient le contexte le plus longtemps."""
    retenus = [c for c in chunks if len(c.get("text") or "") >= _STICKY_MIN_CHARS]
    if not retenus:
        logger.debug("set_sticky_rag: %d extrait(s) trop court(s) — rien de retenu", len(chunks))
        return
    try:
        get_redis().setex(
            f"{_STICKY_RAG_PREFIX}{user_code}:{session_id}",
            STICKY_RAG_TTL,
            json.dumps(retenus, ensure_ascii=False),
        )
    except Exception as exc:
        logger.warning("set_sticky_rag failed: %s", exc)


def get_qdrant() -> QdrantClient:
    """Return the shared Qdrant connection (singleton, thread-safe)."""
    global _qdrant_client
    if _qdrant_client is None:
        with _qdrant_lock:
            if _qdrant_client is None:
                _qdrant_client = QdrantClient(url=QDRANT_URL, timeout=5.0)
    return _qdrant_client


def redis_get_json(key: str, default=None):
    """
    Get a Redis key and deserialise it from JSON.
    Returns default if the key is missing or on any error.
    """
    try:
        data = get_redis().get(key)
        return json.loads(data) if data else default
    except Exception as exc:
        logger.warning("redis_get_json failed for key %s: %s", key, type(exc).__name__)
        return default


def redis_set_json(key: str, data, ttl: int | None = None) -> None:
    """
    Serialise data to JSON and store it in Redis.
    Pass ttl (seconds) to use SETEX instead of SET.
    """
    try:
        payload = json.dumps(data, ensure_ascii=False)
        r = get_redis()
        if ttl:
            r.setex(key, ttl, payload)
        else:
            r.set(key, payload)
    except Exception as exc:
        logger.warning("redis_set_json failed for key %s: %s", key, type(exc).__name__)

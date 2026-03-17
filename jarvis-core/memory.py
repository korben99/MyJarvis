"""
PROJECT JARVIS v7
Jarvis Memory System
====================
- Working memory: Redis (current session, mood, active context)
- Semantic memory: Redis hashes (user profile, preferences, projects)
- Episodic memory: Qdrant (timestamped conversation summaries)
- Self memory: JSON file (Jarvis identity and growth)
Redis
   short-term

Qdrant
   episodic memory
   autobiographical memory

compression
   knowledge abstraction

conversation → analyzer
             → memory_summary
             → episodic memory
             → autobiographical memory

Compression memory added, to be runned every month.


"""

import hashlib
import json
import logging
import os
import pickle
import tempfile
import time
import uuid
from datetime import datetime, timezone
from threading import Lock

import httpx
import redis
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList
from sentence_transformers import SentenceTransformer

from config import (
    ANALYSIS_API_KEY,
    ANALYSIS_API_URL,
    ANALYSIS_MODEL,
    AUTOBIO_IMPORTANCE_THRESHOLD,
    CHAT_LOG_TTL,
    CHAT_MAX_MESSAGES,
    EMBED_MODEL_NAME,
    ENABLE_ANALYSIS,
    IMPORTANCE_THRESHOLD,
    NOVELTY_THRESHOLD,
    QDRANT_COLLECTION,
    QDRANT_MEMORY_COLLECTION,
    QDRANT_URL,
    RAG_SCORE_THRESHOLD,
    RAG_TOP_K,
    RECALL_MEMORY_SIMILARITY_THRESHOLD,
    REDIS_URL,
    SELF_MEMORY_PATH,
    USER_CODES,
)

_qdrant = None
_embed_model = None
_embed_lock = Lock()
logger = logging.getLogger("jarvis-memory")

# ── qdran Connection ──


def get_qdrant():
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(
            url=QDRANT_URL,
            timeout=10
        )
    return _qdrant


# ── Embedding model — local-first, HF fallback ───────────────────────────

MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/app/data/model_cache")


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        with _embed_lock:
            if _embed_model is None:
                os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
                try:
                    # Fast path: model already on disk, no network call
                    _embed_model = SentenceTransformer(
                        EMBED_MODEL_NAME,
                        cache_folder=MODEL_CACHE_DIR,
                        local_files_only=True,
                    )
                    logger.info("Embedding model loaded from local cache (%s)", MODEL_CACHE_DIR)
                except Exception:
                    # First run or cache missing — download from HuggingFace
                    logger.info("Downloading embedding model from HuggingFace (one-time)...")
                    _embed_model = SentenceTransformer(
                        EMBED_MODEL_NAME,
                        cache_folder=MODEL_CACHE_DIR,
                    )
                    logger.info("Embedding model downloaded and cached at %s", MODEL_CACHE_DIR)
    return _embed_model


# ── Redis Connection ──

_redis = None


def get_redis():
    global _redis
    if _redis is None:
        _redis = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis


# ══════════════════════════════════════════════════
#  WORKING MEMORY — Current state, volatile
# ══════════════════════════════════════════════════


def get_working_memory(session_id: str) -> dict:
    """Get current session state."""
    r = get_redis()
    data = r.hgetall(f"working:{session_id}")
    return data or {}


def set_working_memory(session_id: str, key: str, value: str, ttl: int = 86400):
    """Set a working memory value. Expires after ttl seconds (default 24h)."""
    r = get_redis()
    r.hset(f"working:{session_id}", key, value)
    r.expire(f"working:{session_id}", ttl)


def get_emotional_state() -> dict:
    """Get Jarvis's current emotional state."""
    r = get_redis()
    data = r.get("jarvis:emotional_state")
    if data:
        return json.loads(data)
    return {
        "mood": "neutral",
        "energy": 0.7,
        "confidence": 0.8,
        "curiosity": 0.6,
        "concern": 0.0,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


def update_emotional_state(updates: dict):
    """Update emotional state with new values."""
    r = get_redis()
    state = get_emotional_state()
    state.update(updates)
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    # Clamp values to 0-1
    for key in ["energy", "confidence", "curiosity", "concern"]:
        if key in state and isinstance(state[key], (int, float)):
            state[key] = max(0.0, min(1.0, state[key]))
    r.set("jarvis:emotional_state", json.dumps(state))


# ══════════════════════════════════════════════════
#  SESSION MEMORY — Current conversation (200 message max)
# ══════════════════════════════════════════════════


_CHAT_LOG_TTL = CHAT_LOG_TTL  # configured in config.py (default 90 days)


def append_conversation_message(
    user_code: str, session_id: str, role: str, content: str
):
    r = get_redis()
    key = f"chat:{user_code}:{session_id}"

    entry = {"role": role, "content": content, "ts": time.time()}

    pipe = r.pipeline()
    pipe.rpush(key, json.dumps(entry))
    pipe.ltrim(key, -CHAT_MAX_MESSAGES, -1)
    pipe.expire(key, _CHAT_LOG_TTL)  # sliding TTL — resets on every message
    pipe.execute()


def get_conversation(user_code: str, session_id: str):
    r = get_redis()
    key = f"chat:{user_code}:{session_id}"

    entries = r.lrange(key, 0, -1)

    return [json.loads(e) for e in entries]


# ══════════════════════════════════════════════════
#  SEMANTIC MEMORY — Long-term knowledge about user
# ══════════════════════════════════════════════════

"""Get everything Jarvis knows about the user."""


def get_user_profile(user_code: str) -> dict:
    r = get_redis()
    data = r.hgetall(f"user:{user_code}:profile")
    return data or {}


def update_user_profile(user_code: str, key: str, value: str | None):
    """Add, update, or delete (value=None) a user profile fact."""
    r = get_redis()
    if value is None:
        r.hdel(f"user:{user_code}:profile", key)
        logger.info("User %s profile deleted: %s", user_code, key)
    else:
        r.hset(f"user:{user_code}:profile", key, value)
        logger.info("User %s profile updated: %s = %s", user_code, key, value)


def set_interest_weight(user_code: str, term: str, weight: float):
    """
    Set the importance weight for an interest term (0.0 = forgotten, 1.0 = normal, 2.0 = top).
    Weight=0 effectively removes the term from briefing and news queries.
    """
    r = get_redis()
    r.hset(f"user:{user_code}:interest_weights", term.lower(), str(weight))
    logger.info("User %s interest weight: %s = %.1f", user_code, term, weight)


def get_interest_weights(user_code: str) -> dict[str, float]:
    """Return {term: weight} dict. Missing terms default to 1.0."""
    r = get_redis()
    raw = r.hgetall(f"user:{user_code}:interest_weights")
    return {k: float(v) for k, v in raw.items()}


"""Get list of user's active projects."""


def get_user_projects(user_code: str) -> list:
    r = get_redis()
    data = r.get(f"user:{user_code}:projects")
    return json.loads(data) if data else []


"""Update active projects list."""


def update_user_projects(user_code: str, projects: list):
    r = get_redis()
    r.set(f"user:{user_code}:projects", json.dumps(projects))


"""Get user preferences (language, style, etc.)."""


def get_user_preferences(user_code: str) -> dict:
    r = get_redis()
    data = r.hgetall(f"user:{user_code}:preferences")
    return data or {}


"""Update a preference."""


def update_user_preference(user_code: str, key: str, value: str):
    r = get_redis()
    r.hset(f"user:{user_code}:preferences", key, value)


# ══════════════════════════════════════════════════
#  EPISODIC MEMORY — Conversation history + summaries
# ══════════════════════════════════════════════════


def log_conversation(
    user_code: str,
    session_id: str,
    user_msg: str,
    assistant_msg: str,
    mood: str = "neutral",
    topics: list = None,
    importance=0,
    memory_summary: str = None,
):
    """Log a conversation exchange to episodic memory."""
    r = get_redis()
    entry = {
        "timestamp": time.time(),
        "session_id": session_id,
        "user": user_msg[:500],  # Truncate for storage
        "assistant": assistant_msg[:500],
        "mood": mood,
        "topics": topics or [],
        "importance": importance,
        "memory_summary": memory_summary,
    }
    # Store in a sorted set by timestamp for easy retrieval
    score = time.time()
    r.zadd(f"episodic:{user_code}:conversations", {json.dumps(entry): score})

    # Keep only last 1000 exchanges (prevent unbounded growth)
    r.zremrangebyrank(f"episodic:{user_code}:conversations", 0, -1001)
    # Storage as embedded to Qdrant , is importance high enough (VARIABLE TO ADJUST)

    if entry.get("importance", 0) > IMPORTANCE_THRESHOLD:
        store_memory_vector(user_code, entry)

    summary = entry.get("memory_summary")

    if summary and entry.get("importance", 0) > AUTOBIO_IMPORTANCE_THRESHOLD:
        store_autobiographical_event(user_code, summary, entry["importance"])


def get_recent_conversations(user_code: str, hours: int = 24, limit: int = 20) -> list:
    """Get recent conversation exchanges."""
    r = get_redis()
    cutoff = time.time() - (hours * 3600)
    entries = r.zrangebyscore(
        f"episodic:{user_code}:conversations", cutoff, "+inf", start=0, num=limit
    )
    return [json.loads(e) for e in entries]


def get_conversation_summary(user_code: str, days: int = 7) -> str:
    """Get a text summary of recent conversations."""
    r = get_redis()
    cutoff = time.time() - (days * 86400)
    entries = r.zrangebyscore(f"episodic:{user_code}:conversations", cutoff, "+inf")
    parsed = [json.loads(e) for e in entries]

    if not parsed:
        return "No recent conversations."

    topics_seen = set()
    moods = []
    for e in parsed:
        topics_seen.update(e.get("topics", []))
        moods.append(e.get("mood", "neutral"))

    summary = f"Ces {days} derniers jours : {len(parsed)} échanges. "
    if topics_seen:
        summary += f"Sujets abordés : {', '.join(topics_seen)}. "
    if moods:
        from collections import Counter

        mood_counts = Counter(moods).most_common(3)
        summary += (
            f"Humeurs dominantes : {', '.join(f'{m}({c})' for m, c in mood_counts)}."
        )

    return summary


# ══════════════════════════════════════════════════
#  COMPLETE MEMORY TO QDRANT — Conversation history + summaries + AUTOBIOGRAPHIE
# ══════════════════════════════════════════════════


def compute_memory_novelty(user_code: str, text: str, limit: int = 5):
    """
    Estimate novelty of a memory by comparing it with recent vector memories.
    Returns a value between 0 and 1.
    """
    try:
        model = get_embed_model()
        qdrant = get_qdrant()

        vector = model.encode(text, normalize_embeddings=True).tolist()

        results = qdrant.query_points(
            collection_name=QDRANT_MEMORY_COLLECTION,
            query=vector,
            limit=limit,
            query_filter={
                "must": [
                    {"key": "user_code", "match": {"value": user_code}},
                ],
                "should": [
                    {"key": "memory_type", "match": {"value": "episodic"}},
                    {"key": "memory_type", "match": {"value": "autobiographical"}},
                ],
            },
        ).points

        if not results:
            return 1.0

        max_similarity = max(r.score for r in results)

        novelty = 1 - max_similarity

        return max(0, min(1, novelty))

    except Exception as e:
        logger.error("Novelty computation failed: %s", e)
        return 0.5


def store_memory_vector(user_code: str, entry: dict):
    """Store conversation exchange in vector memory (Qdrant).

    Requires a memory_summary — skips storage if absent.
    Raw exchange text is intentionally not used: embedding geometry between
    a natural-language query and a structured log string degrades recall quality.
    """
    try:
        text = (entry.get("memory_summary") or "").strip()
        if not text:
            logger.debug("store_memory_vector: skipped (no memory_summary)")
            return

        model = get_embed_model()
        qdrant = get_qdrant()
        logger.info("Vector memory candidate: %s", text[:80])
        novelty = compute_memory_novelty(user_code, text)
        if novelty < NOVELTY_THRESHOLD:
            return

        vector = model.encode(text, normalize_embeddings=True).tolist()
        qdrant.upsert(
            collection_name=QDRANT_MEMORY_COLLECTION,
            points=[
                {
                    "id": str(uuid.uuid4()),
                    "vector": vector,
                    "payload": {
                        "user_code": user_code,
                        "memory_type": "episodic",
                        "importance": entry.get("importance", 0),
                        "session_id": entry["session_id"],
                        "text": text,
                        "timestamp": entry["timestamp"],
                        "topics": entry.get("topics", []),
                        "mood": entry["mood"],
                        "novelty": novelty,
                    },
                }
            ],
        )

    except Exception as e:
        logger.error("Vector memory store failed: %s", e)


def store_autobiographical_event(user_code: str, summary: str, importance: float):
    """
    Store a major life / project milestone for the user.
    """
    try:
        model = get_embed_model()
        qdrant = get_qdrant()

        vector = model.encode(summary, normalize_embeddings=True).tolist()

        qdrant.upsert(
            collection_name=QDRANT_MEMORY_COLLECTION,
            points=[
                {
                    "id": str(uuid.uuid4()),
                    "vector": vector,
                    "payload": {
                        "user_code": user_code,
                        "memory_type": "autobiographical",
                        "text": summary,
                        "importance": importance,
                        "timestamp": time.time(),
                    },
                }
            ],
        )

        logger.info("Autobiographical memory stored: %s", summary)

    except Exception as e:
        logger.error("Autobiographical memory failed: %s", e)


def search_memory(user_code: str, query: str, limit: int = 5, memory_scope: str = "auto"):
    """Search vector memory. memory_scope filters to a specific layer or searches all ('auto')."""
    # Profile scope has no Qdrant data — Redis profile is already injected via build_memory_context()
    if memory_scope == "profile":
        return []

    try:
        model = get_embed_model()
        qdrant = get_qdrant()

        vector = model.encode(query, normalize_embeddings=True).tolist()

        if memory_scope == "episodic":
            type_filter = {"must": [{"key": "memory_type", "match": {"value": "episodic"}}]}
        elif memory_scope == "autobiographical":
            type_filter = {"must": [{"key": "memory_type", "match": {"value": "autobiographical"}}]}
        else:  # "auto" — search both layers
            type_filter = {
                "should": [
                    {"key": "memory_type", "match": {"value": "episodic"}},
                    {"key": "memory_type", "match": {"value": "autobiographical"}},
                ]
            }

        results = qdrant.query_points(
            collection_name=QDRANT_MEMORY_COLLECTION,
            query=vector,
            limit=limit * 3,
            query_filter={
                "must": [{"key": "user_code", "match": {"value": user_code}}],
                **type_filter,
            },
        ).points

        memories = []
        # memorie recall with filter of similarity
        now = time.time()

        for r in results:
            if r.score < RECALL_MEMORY_SIMILARITY_THRESHOLD:
                continue
            payload = r.payload

            # recall memory per recency 30 days window
            timestamp = payload.get("timestamp", now)
            recency = now - timestamp
            recency_bonus = max(0, min(1, 1 - recency / (60 * 60 * 24 * 30)))
            # Weighted blend: semantic similarity (primary) + importance + recency
            # All weights sum to 1.0 so the score stays in ~[0, 1]
            final_score = (
                r.score * 0.65
                + payload.get("importance", 0) * 0.25
                + recency_bonus * 0.1
            )

            memories.append(
                {
                    "text": payload["text"],
                    "timestamp": timestamp,
                    "score": final_score,
                }
            )
        # cognitive ranking
        memories.sort(key=lambda x: x["score"], reverse=True)

        return memories[:limit]

    except Exception as e:
        logger.error("Memory search failed: %s", e)
        return []


# ══════════════════════════════════════════════════
#  SELF MEMORY LOCK
# ══════════════════════════════════════════════════

# Shared lock for all read-modify-write cycles on jarvis-self.json.
# Use as: with self_memory_lock: data = get_self_memory(); ...; save_self_memory(data)
# threading.Lock works in both sync and async contexts (no await held while locked).
self_memory_lock = Lock()


# ══════════════════════════════════════════════════
#  ATOMIC FILE WRITE
# ══════════════════════════════════════════════════

def atomic_json_write(path: str, data, indent: int = 2) -> None:
    """
    Write *data* as JSON to *path* atomically.

    Uses a sibling temp file + os.replace() so that:
    - Concurrent readers always see either the previous complete file or the
      new complete file — never a truncated/partial state (race condition).
    - A crash mid-write leaves the original file intact (no corruption).

    os.replace() is a single syscall and is guaranteed atomic on POSIX.
    """
    dir_name = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ══════════════════════════════════════════════════
#  SELF MEMORY — Jarvis's identity and growth
# ══════════════════════════════════════════════════


def get_self_memory() -> dict:
    """Load jarvis-self.json; bootstrap with defaults if missing or corrupt."""
    try:
        with open(SELF_MEMORY_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        default = {
            "identity": {
                "name": "Jarvis",
                "version": "6.0",
                "created": datetime.now(timezone.utc).isoformat(),
                "personality": "Helpful, concise, direct. Dry humor when appropriate.",
            },
            "opinions": [],
            "learnings": [],
            "growth_log": [],
            "reflection_count": 0,
        }
        save_self_memory(default)
        return default
    except Exception as exc:
        logger.error("Could not load jarvis-self.json: %s", type(exc).__name__)
        return {}


def save_self_memory(data: dict) -> None:
    """Save jarvis-self.json atomically."""
    try:
        atomic_json_write(SELF_MEMORY_PATH, data)
    except Exception as exc:
        logger.error("Could not save jarvis-self.json: %s", type(exc).__name__)


def add_self_learning(learning: str):
    """Add a Jarvis self-improvement note (not a user fact)."""
    with self_memory_lock:
        data = get_self_memory()
        data.setdefault("learnings", []).append({
            "text": learning,
            "date": datetime.now(timezone.utc).isoformat(),
        })
        data["learnings"] = data["learnings"][-100:]
        save_self_memory(data)


def add_self_opinion(topic: str, opinion: str):
    """Add or update an opinion."""
    with self_memory_lock:
        data = get_self_memory()
        for o in data.get("opinions", []):
            if o["topic"] == topic:
                o["opinion"] = opinion
                o["updated"] = datetime.now(timezone.utc).isoformat()
                save_self_memory(data)
                return
        data.setdefault("opinions", []).append({
            "topic": topic,
            "opinion": opinion,
            "created": datetime.now(timezone.utc).isoformat(),
        })
        data["opinions"] = data["opinions"][-50:]
        save_self_memory(data)


# ══════════════════════════════════════════════════
#  CONTEXT BUILDER — Assemble memory for prompts
# ══════════════════════════════════════════════════


def get_user_timeline(user_code: str, limit: int = 20):
    """
    Retrieve the user's autobiographical timeline.
    """
    try:
        qdrant = get_qdrant()

        results = qdrant.scroll(
            collection_name=QDRANT_MEMORY_COLLECTION,
            scroll_filter={
                "must": [
                    {"key": "user_code", "match": {"value": user_code}},
                    {"key": "memory_type", "match": {"value": "autobiographical"}},
                ]
            },
            limit=limit,
        )[0]

        now_ts = time.time()
        timeline = []

        for r in results:
            ts = r.payload.get("timestamp", now_ts)
            imp = r.payload.get("importance", 0)
            # Recency bonus normalised over a 1-year window
            recency = max(0.0, 1.0 - (now_ts - ts) / (86400 * 365))
            rank = imp * 0.7 + recency * 0.3
            timeline.append(
                {
                    "text": r.payload["text"],
                    "timestamp": ts,
                    "importance": imp,
                    "_rank": rank,
                }
            )

        # Most important + recent events first
        timeline.sort(key=lambda x: x["_rank"], reverse=True)
        for e in timeline:
            e.pop("_rank", None)

        return timeline

    except Exception as e:
        logger.error("Timeline retrieval failed: %s", e)
        return []


def build_memory_context(session_id: str, user_code: str) -> str:
    """Build a memory context string to inject into the system prompt."""
    parts = []

    # User profile
    profile = get_user_profile(user_code)
    if profile:
        parts.append("=== PROFIL UTILISATEUR ===")
        for k, v in profile.items():
            parts.append(f"- {k}: {v}")

    # User preferences
    prefs = get_user_preferences(user_code)
    if prefs:
        parts.append("\n=== PRÉFÉRENCES ===")
        for k, v in prefs.items():
            parts.append(f"- {k}: {v}")

    # Active projects
    projects = get_user_projects(user_code)
    if projects:
        parts.append("\n=== PROJETS ACTIFS ===")
        for p in projects:
            if isinstance(p, dict):
                parts.append(f"- {p.get('name', 'sans nom')}: {p.get('status', '')}")
            else:
                parts.append(f"- {p}")

    # Recent conversation topics (last 24h)
    recent = get_recent_conversations(user_code, hours=24, limit=10)
    if recent:
        topics = set()
        for conv in recent:
            topics.update(conv.get("topics", []))
        if topics:
            parts.append("\n=== SUJETS RÉCENTS (24h) ===")
            parts.append(f"Sujets abordés : {', '.join(topics)}")

    # Emotional state
    emotion = get_emotional_state()
    if emotion.get("mood") != "neutral":
        parts.append("\n=== ÉTAT ÉMOTIONNEL ===")
        parts.append(f"Humeur actuelle : {emotion['mood']}")

    # Self identity
    self_mem = get_self_memory()
    if self_mem.get("learnings"):
        recent_learnings = self_mem["learnings"][-5:]
        parts.append("\n=== APPRENTISSAGES RÉCENTS ===")
        for ln in recent_learnings:
            parts.append(f"- {ln['text']}")

    # User Timeline
    timeline = get_user_timeline(user_code)

    if timeline:
        parts.append("\n=== FRISE CHRONOLOGIQUE ===")

        # timeline is already sorted by importance+recency desc — take the top 5
        for event in timeline[:5]:
            dt = datetime.fromtimestamp(event["timestamp"]).strftime("%Y-%m")
            parts.append(f"{dt}: {event['text']}")

    return "\n".join(parts) if parts else ""


# ══════════════════════════════════════════════════
#  MEMORY COMPRESSION / CLEANING. CALLED BY NIGHTLY SCRIPT
# ══════════════════════════════════════════════════
def _consolidate_user_memories(user_code: str, max_items: int = 20):

    try:
        qdrant = get_qdrant()

        results = qdrant.scroll(
            collection_name=QDRANT_MEMORY_COLLECTION,
            scroll_filter={
                "must": [
                    {"key": "user_code", "match": {"value": user_code}},
                    {"key": "memory_type", "match": {"value": "episodic"}},
                ]
            },
            order_by={
                "key": "timestamp",
                "direction": "asc"
            },
            limit=max_items,
        )[0]

        point_ids = [r.id for r in results]
        texts = [r.payload["text"] for r in results if r.payload.get("text")]

        if len(texts) < 5:
            return

        combined = "\n".join(texts)

        summary_prompt = f"""
Résume les souvenirs suivants concernant un utilisateur en un fait durable.

Souvenirs :
{combined}

Retourne une seule phrase en français décrivant un fait stable sur l'utilisateur.
"""

        resp = httpx.post(
            f"{ANALYSIS_API_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {ANALYSIS_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": ANALYSIS_MODEL,
                "messages": [{"role": "user", "content": summary_prompt}],
                "temperature": 0.1,
                "max_tokens": 120,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        summary = resp.json()["choices"][0]["message"]["content"]

        store_autobiographical_event(user_code, summary, 0.9)

        # Delete the episodic points that were just consolidated so they
        # are not re-processed on the next nightly run.
        qdrant.delete(
            collection_name=QDRANT_MEMORY_COLLECTION,
            points_selector=PointIdsList(points=point_ids),
        )

        logger.info("[%s] Memory consolidation: %s (%d episodic points deleted)", user_code, summary, len(point_ids))

    except Exception as e:
        logger.error("Memory consolidation failed for %s: %s", user_code, e)


def consolidate_memories(user_code: str = None, max_items: int = 20):
    """
    Nightly memory consolidation.

    If user_code is provided -> consolidate only this user.
    If user_code is None -> iterate over all users.
    """
    try:
        if user_code:
            _consolidate_user_memories(user_code, max_items)
            return

        for uc in USER_CODES.keys():
            logger.info(f"Memory consolidation for user {uc}")
            _consolidate_user_memories(uc, max_items)

    except Exception as e:
        logger.error("Global memory consolidation failed: %s", e)

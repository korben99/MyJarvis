"""
PROJECT JARVIS v8
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
import os
import pickle
from collections import Counter
import tempfile
import time
import uuid
from datetime import datetime, timezone
from threading import Lock

from qdrant_client.models import PointIdsList
from sentence_transformers import SentenceTransformer

from config import (
    AUTOBIO_DEDUP_THRESHOLD,
    AUTOBIO_IMPORTANCE_THRESHOLD,
    AUTOBIO_RECENCY_WINDOW_DAYS,
    CHAT_LOG_TTL,
    DONE_PROJECT_TTL_DAYS,
    CHAT_MAX_MESSAGES,
    EMBED_MODEL_NAME,
    IMPORTANCE_THRESHOLD,
    MEMORY_DECAY_FACTOR,
    MEMORY_DECAY_THRESHOLD,
    MEMORY_DECAY_DURABLE_MIN,
    MEMORY_CONSOLIDATION_IMPORTANCE,
    NOVELTY_THRESHOLD,
    PRIMARY_API_KEY,
    PRIMARY_API_URL,
    PRIMARY_MODEL,
    PRIMARY_TIMEOUT,
    QDRANT_COLLECTION,
    QDRANT_MEMORY_COLLECTION,
    RAG_SCORE_THRESHOLD,
    RAG_TOP_K,
    RECALL_MEMORY_SIMILARITY_THRESHOLD,
    ROUTER_API_KEY,
    ROUTER_API_URL,
    ROUTER_MODEL,
    ROUTER_TIMEOUT,
    SELF_MEMORY_PATH,
    USER_CODES,
)

from helpers import call_llm, extract_llm_json, get_logger, get_qdrant, get_redis, redis_get_json, redis_set_json


logger = get_logger("jarvis-memory")

# ── Embedding model — local-first, HF fallback ───────────────────────────
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/app/data/model_cache")
_embed_model = None
_embed_lock = Lock()
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
    return redis_get_json("jarvis:emotional_state") or {
        "mood": "neutral",
        "energy": 0.7,
        "confidence": 0.8,
        "curiosity": 0.6,
        "concern": 0.0,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


def update_emotional_state(updates: dict):
    """Update emotional state with new values."""
    state = get_emotional_state()
    state.update(updates)
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    # Clamp values to 0-1
    for key in ["energy", "confidence", "curiosity", "concern"]:
        if key in state and isinstance(state[key], (int, float)):
            state[key] = max(0.0, min(1.0, state[key]))
    redis_set_json("jarvis:emotional_state", state)


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

# ── Profile key dedup helpers ─────────────────────────────────────────────

# Scalar aliases: O(1) resolution before any LLM call
_SCALAR_CANONICAL: dict[str, str] = {
    "ville": "location",    "city": "location",
    "metier": "profession", "emploi": "profession",
    "employeur": "current_employer", "entreprise": "current_employer",
    "societe": "current_employer",   "company": "current_employer",
    "prenom": "name",       "prénom": "name",
    "revenu": "capital",    "patrimoine": "capital",
    "inquietude": "concerns",
    "voyages_prevus": "travel_plans",
}

# Namespace families: keys in the same family are compared together
_NS_FAMILY: dict[str, frozenset] = {
    "hobby":         frozenset({"hobby", "interest", "loisir", "passion", "activite"}),
    "skill":         frozenset({"skill", "competence", "technologie", "outil"}),
    "placement":     frozenset({"placement", "investissement", "epargne"}),
    "projet":        frozenset({"projet", "project"}),
    "preoccupation": frozenset({"preoccupation", "concerns", "inquietude"}),
}


def _key_prefix(key: str) -> str | None:
    return key.split(":")[0] if ":" in key else None


def _candidate_keys(new_key: str, existing_keys: list[str]) -> list[str]:
    """Narrow the dedup candidate set to the same namespace family."""
    new_prefix = _key_prefix(new_key)
    if new_prefix is None:
        return [k for k in existing_keys if _key_prefix(k) is None]
    family = next(
        (members for members in _NS_FAMILY.values() if new_prefix in members),
        frozenset({new_prefix}),
    )
    return [k for k in existing_keys if _key_prefix(k) in family]


def get_user_profile(user_code: str) -> dict:
    r = get_redis()
    data = r.hgetall(f"user:{user_code}:profile")
    return data or {}


def _normalize_profile_key(user_code: str, new_key: str, existing_keys: list[str]) -> str | None:
    """
    Find whether new_key is semantically equivalent to an existing profile key.
    Returns the existing key to evict, or None if new_key is genuinely new.

    Three-stage pipeline (cheapest first):
      1. Verbatim match          — O(1), no LLM
      2. Scalar canonical alias  — O(1), no LLM
      3. Category-aware LLM      — only within the same namespace family
    """
    if not existing_keys or new_key in existing_keys:
        return None

    # Stage 1: scalar canonical alias
    stripped = new_key.lower().replace("-", "_").replace(" ", "_")
    if stripped in _SCALAR_CANONICAL:
        canonical = _SCALAR_CANONICAL[stripped]
        if canonical in existing_keys:
            logger.info("User %s profile key '%s' → canonical '%s' (no LLM)", user_code, new_key, canonical)
            return canonical

    if not ROUTER_MODEL:
        return None

    # Stage 2: category-aware LLM on reduced candidate set
    candidates = _candidate_keys(new_key, existing_keys)
    if not candidates:
        return None

    try:
        keys_list = ", ".join(f'"{k}"' for k in candidates)
        prompt = (
            f"Clés existantes (même catégorie) : [{keys_list}]\n"
            f"Nouvelle clé : \"{new_key}\"\n\n"
            f"DOUBLON = même concept sous un nom ou préfixe différent :\n"
            f"  • \"hobby:ia\" == \"interest:ia\" → OUI\n"
            f"  • \"hobby:kart\" == \"loisir:kart\" → OUI\n"
            f"  • \"hobby:kart\" == \"hobby:tennis\" → NON\n"
            f'JSON uniquement : {{"match": "clé_existante"}} ou {{"match": null}}'
        )
        raw = call_llm(
            [{"role": "user", "content": prompt}],
            model=ROUTER_MODEL,
            api_url=ROUTER_API_URL,
            api_key=ROUTER_API_KEY,
            temperature=0.0,
            max_tokens=40,
            no_think=True,
            timeout=ROUTER_TIMEOUT,
        )
        parsed = extract_llm_json(raw)
        match = parsed.get("match")
        if match and match in existing_keys:
            logger.info("User %s profile key '%s' deduped → '%s'", user_code, new_key, match)
            return match
        return None

    except Exception as exc:
        logger.warning("Profile key normalization skipped (%s): %s", new_key, exc)
        return None


def update_user_profile(user_code: str, key: str, value: str | None):
    """Add, update, or delete (value=None) a user profile fact.

    Preventive duplicate guard: before writing a new key, the router LLM checks
    whether it is semantically equivalent to an existing key.  If a match is found,
    the old key is deleted before the new one is written, preventing profile bloat.
    """
    r = get_redis()
    profile_redis_key = f"user:{user_code}:profile"

    if value is None:
        r.hdel(profile_redis_key, key)
        logger.info("User %s profile deleted: %s", user_code, key)
    else:
        existing_keys = r.hkeys(profile_redis_key)

        # Key normalization: if new_key is semantically equivalent to an existing key
        # (same concept or same category:item under a synonym category), evict the old
        # key and write under the new name — no value merging, each key is atomic.
        duplicate = _normalize_profile_key(user_code, key, existing_keys)
        if duplicate:
            r.hdel(profile_redis_key, duplicate)

        r.hset(profile_redis_key, key, value)
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


def get_user_projects(user_code: str) -> list:
    """Return the user's project list from Redis."""
    return redis_get_json(f"user:{user_code}:projects", default=[])


def update_user_projects(user_code: str, projects: list):
    """Persist the project list.
    Done projects older than DONE_PROJECT_TTL_DAYS are dropped.
    Schema: {name, status, first_mentioned, last_update, description?}
    """
    cutoff = datetime.now(timezone.utc).timestamp() - DONE_PROJECT_TTL_DAYS * 86400
    result = []
    for p in projects:
        if p.get("status") == "done" and p.get("last_update"):
            try:
                if datetime.fromisoformat(p["last_update"]).timestamp() < cutoff:
                    continue
            except (ValueError, TypeError):
                pass
        entry = {
            "name":            p["name"],
            "status":          p.get("status", "in_progress"),
            "first_mentioned": p.get("first_mentioned"),
            "last_update":     p.get("last_update"),
        }
        if p.get("description"):
            entry["description"] = p["description"]
        result.append(entry)
    redis_set_json(f"user:{user_code}:projects", result)


def get_user_preferences(user_code: str) -> dict:
    """Get user preferences (language, style, etc.)."""
    r = get_redis()
    data = r.hgetall(f"user:{user_code}:preferences")
    return data or {}


def update_user_preference(user_code: str, key: str, value: str):
    """Update a preference."""
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
        mood_counts = Counter(moods).most_common(3)
        summary += (
            f"Humeurs dominantes : {', '.join(f'{m}({c})' for m, c in mood_counts)}."
        )

    return summary


def _fuzzy_project_name(name: str, project_map: dict) -> str | None:
    """
    Find the best matching project name by word overlap (≥60% threshold).
    Handles minor variations: "Jarvis" matches "Jarvis v7", case-insensitive.
    """
    words_new = set(name.lower().split())
    best_match: str | None = None
    best_score = 0.0
    for existing_name in project_map:
        words_ex = set(existing_name.lower().split())
        overlap = len(words_new & words_ex)
        if overlap == 0:
            continue
        score = overlap / max(len(words_new), len(words_ex))
        if score > best_score and score >= 0.6:
            best_score = score
            best_match = existing_name
    return best_match


def apply_project_updates(user_code: str, project_events: list[str]):
    projects = get_user_projects(user_code)
    now = datetime.now(timezone.utc).isoformat()

    project_map: dict[str, dict] = {p["name"]: p for p in projects}

    for event in project_events:
        try:
            action, name = event.split(":", 1)
            name = name.strip()
        except ValueError:
            continue

        # Exact match first, then fuzzy — prevents name drift duplicates
        resolved = name if name in project_map else (_fuzzy_project_name(name, project_map) or name)

        if action == "create":
            if resolved not in project_map:
                project_map[resolved] = {
                    "name": resolved,
                    "status": "in_progress",
                    "first_mentioned": now,
                    "last_update": now,
                }
            else:
                project_map[resolved]["last_update"] = now  # already exists → update

        elif action == "update":
            if resolved in project_map:
                project_map[resolved]["last_update"] = now
            else:
                project_map[resolved] = {
                    "name": resolved,
                    "status": "in_progress",
                    "first_mentioned": now,
                    "last_update": now,
                }

        elif action == "done":
            if resolved in project_map:
                project_map[resolved]["status"] = "done"
                project_map[resolved]["last_update"] = now

    update_user_projects(user_code, list(project_map.values()))

# ══════════════════════════════════════════════════
#  COMPLETE MEMORY TO QDRANT — Conversation history + summaries + AUTOBIOGRAPHIE
# ══════════════════════════════════════════════════


def compute_memory_novelty(user_code: str, text: str, vector: list | None = None, limit: int = 5):
    """
    Estimate novelty of a memory by comparing it with recent vector memories.
    Returns a value between 0 and 1.

    Pass a pre-computed *vector* to avoid re-encoding the text when the caller
    already has the embedding (e.g. store_memory_vector).
    """
    try:
        model = get_embed_model()
        qdrant = get_qdrant()

        if vector is None:
            vector = model.encode(text, normalize_embeddings=True).tolist()

        results = qdrant.query_points(
            collection_name=QDRANT_MEMORY_COLLECTION,
            query=vector,
            limit=limit,
            query_filter={
                "must": [
                    {"key": "user_code", "match": {"value": user_code}},
                    {
                        "should": [
                            {"key": "memory_type", "match": {"value": "episodic"}},
                            {"key": "memory_type", "match": {"value": "autobiographical"}},
                        ]
                    },
                ],
            },
        ).points

        if not results:
            return 1.0

        max_similarity = max(r.score for r in results)
        return max(0, min(1, 1 - max_similarity))

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
        vector = model.encode(text, normalize_embeddings=True).tolist()
        novelty = compute_memory_novelty(user_code, text, vector=vector)
        if novelty < NOVELTY_THRESHOLD:
            return

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

    Skips storage if a semantically identical autobiographical memory already exists
    (cosine similarity ≥ AUTOBIO_DEDUP_THRESHOLD) to prevent the collection from
    accumulating redundant variants of the same fact over time.
    """
    try:
        model = get_embed_model()
        qdrant = get_qdrant()

        vector = model.encode(summary, normalize_embeddings=True).tolist()

        # Dedup check: skip if a very similar autobio already exists
        existing = qdrant.query_points(
            collection_name=QDRANT_MEMORY_COLLECTION,
            query=vector,
            limit=1,
            query_filter={
                "must": [
                    {"key": "user_code",    "match": {"value": user_code}},
                    {"key": "memory_type",  "match": {"value": "autobiographical"}},
                ]
            },
        ).points
        if existing and existing[0].score >= AUTOBIO_DEDUP_THRESHOLD:
            logger.debug(
                "Autobio dedup: skipping '%s' (similar=%.2f)", summary[:60], existing[0].score
            )
            return

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

        user_filter = {"key": "user_code", "match": {"value": user_code}}
        if memory_scope in ("episodic", "autobiographical"):
            query_filter = {"must": [user_filter, type_filter["must"][0]]}
        else:
            # Nested should inside must: user_code MUST match AND memory_type must be
            # episodic OR autobiographical (explicit — points without memory_type excluded).
            query_filter = {
                "must": [
                    user_filter,
                    {"should": type_filter["should"]},
                ]
            }

        results = qdrant.query_points(
            collection_name=QDRANT_MEMORY_COLLECTION,
            query=vector,
            limit=limit * 3,
            query_filter=query_filter,
        ).points

        memories = []
        # memorie recall with filter of similarity
        now = time.time()

        for r in results:
            if r.score < RECALL_MEMORY_SIMILARITY_THRESHOLD:
                continue
            payload = r.payload

            # Recency window: 30 days for episodic, AUTOBIO_RECENCY_WINDOW_DAYS for autobiographical.
            # Autobiographical memories are durable milestones — they stay relevant for months.
            timestamp = payload.get("timestamp", now)
            recency = now - timestamp
            mem_type = payload.get("memory_type", "episodic")
            recency_window = (
                AUTOBIO_RECENCY_WINDOW_DAYS * 86400
                if mem_type == "autobiographical"
                else 30 * 86400
            )
            recency_bonus = max(0, min(1, 1 - recency / recency_window))
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


def build_memory_context(session_id: str, user_code: str, self_mem: dict | None = None) -> str:
    """Build a memory context string to inject into the system prompt.

    Pass an already-loaded *self_mem* dict to avoid a redundant JSON read when
    the caller (build_system_prompt) has already called get_self_memory().
    """
    if self_mem is None:
        self_mem = get_self_memory()
    parts = []

    # User profile — namespaced keys (hobby:kart) are grouped by category for readability
    profile = get_user_profile(user_code)
    if profile:
        parts.append("=== PROFIL UTILISATEUR ===")
        grouped: dict[str, list[str]] = {}
        scalars: list[tuple[str, str]] = []
        for k, v in profile.items():
            if ":" in k:
                category, _ = k.split(":", 1)
                grouped.setdefault(category, []).append(v)
            else:
                scalars.append((k, v))
        for k, v in scalars:
            parts.append(f"- {k}: {v}")
        for category, values in grouped.items():
            parts.append(f"- {category}: {', '.join(values)}")

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

    # Tomorrow suggestions — written by nightly review, consumed today
    try:
        raw = get_redis().get(f"jarvis:{user_code}:tomorrow_suggestions")
        suggestions = json.loads(raw) if raw else []
    except Exception:
        suggestions = []
    if suggestions:
        parts.append("\n=== SUJETS À ABORDER AUJOURD'HUI ===")
        for s in suggestions:
            parts.append(f"- {s}")

    return "\n".join(parts) if parts else ""


# ══════════════════════════════════════════════════
#  MEMORY COMPRESSION / CLEANING. CALLED BY NIGHTLY SCRIPT
# ══════════════════════════════════════════════════
def _consolidate_user_memories(user_code: str, batch_size: int = 50):
    """
    Consolidate all episodic memories for a user into autobiographical milestones.

    Runs in a loop, processing batches of `batch_size` oldest episodic points until
    fewer than 5 remain (not enough to form a meaningful summary). This ensures a full
    catch-up on the first run after a period of inactivity, not just a fixed-size pass.
    """
    qdrant = get_qdrant()
    total_deleted = 0

    while True:
        try:
            results = qdrant.scroll(
                collection_name=QDRANT_MEMORY_COLLECTION,
                scroll_filter={
                    "must": [
                        {"key": "user_code", "match": {"value": user_code}},
                        {"key": "memory_type", "match": {"value": "episodic"}},
                    ]
                },
                order_by={"key": "timestamp", "direction": "asc"},
                limit=batch_size,
            )[0]

            point_ids = [r.id for r in results]
            texts = [r.payload["text"] for r in results if r.payload.get("text")]

            if len(texts) < 5:
                break  # Nothing left worth consolidating

            combined = "\n".join(texts)

            summary_prompt = f"""Voici des souvenirs de conversations avec un utilisateur :

{combined}

Identifie les faits durables et distincts sur cet utilisateur (habitudes, préférences, projets, traits de caractère…).
Retourne une liste JSON de phrases en français, chacune décrivant un fait stable.
S'il n'y a aucun fait durable à extraire, retourne une liste vide.
Réponds uniquement avec le JSON, rien d'autre : ["fait 1", "fait 2"]"""

            raw = call_llm(
                [{"role": "user", "content": summary_prompt}],
                model=PRIMARY_MODEL,
                api_url=PRIMARY_API_URL,
                api_key=PRIMARY_API_KEY,
                temperature=0.1,
                max_tokens=400,
                json_response=False,
                no_think=True,
                timeout=30.0,
            )

            parsed = extract_llm_json(raw)
            facts = parsed if isinstance(parsed, list) else []
            facts = [f.strip().strip('"\'')[:300] for f in facts if isinstance(f, str) and f.strip()]

            for fact in facts:
                store_autobiographical_event(user_code, fact, MEMORY_CONSOLIDATION_IMPORTANCE)

            qdrant.delete(
                collection_name=QDRANT_MEMORY_COLLECTION,
                points_selector=PointIdsList(points=point_ids),
            )

            total_deleted += len(point_ids)
            logger.info(
                "[%s] Consolidation batch: %d facts stored, %d points deleted",
                user_code, len(facts), len(point_ids),
            )

        except Exception as e:
            logger.error("Memory consolidation failed for %s: %s", user_code, e)
            break

    if total_deleted:
        logger.info("[%s] Memory consolidation complete: %d episodic points total", user_code, total_deleted)


def _curative_profile_cleanup(user_code: str):
    """
    Nightly curative cleanup of the Redis user profile hash.

    Sends the full profile to the analysis LLM and asks it to identify:
    - Semantic duplicates (same fact under two different key names)
    - Obsolete/contradictory keys superseded by a more recent entry

    The LLM returns {"keys_to_delete": ["key1", "key2"]}  — each key is then
    deleted via HDEL.  The surviving key keeps the most up-to-date value.
    No write is performed other than deletions (values are never rewritten here).

    Skip condition: profile has fewer than 5 keys (not worth the LLM call).
    """
    r = get_redis()
    profile = r.hgetall(f"user:{user_code}:profile")
    if len(profile) < 5:
        return

    try:
        profile_str = "\n".join(f'- "{k}": {v}' for k, v in profile.items())
        prompt = (
            f"Voici le profil Redis d'un utilisateur ({len(profile)} clés) :\n"
            f"{profile_str}\n\n"
            f"Identifie les clés à supprimer parmi :\n"
            f"1. Doublons sémantiques — même information sous deux noms différents "
            f"(garde la clé la plus descriptive/récente, supprime l'autre)\n"
            f"2. Entrées obsolètes ou contredites par une autre clé plus récente\n\n"
            f"Réponds uniquement en JSON, rien d'autre : "
            f'{{\"keys_to_delete\": [\"clé1\", \"clé2\"]}} '
            f"ou {{\"keys_to_delete\": []}} si le profil est propre."
        )

        parsed = extract_llm_json(call_llm(
            [{"role": "user", "content": prompt}],
            model=PRIMARY_MODEL,
            api_url=PRIMARY_API_URL,
            api_key=PRIMARY_API_KEY,
            temperature=0.0,
            max_tokens=120,
            no_think=True,
            timeout=30.0,
        ))
        # Only delete keys that actually exist in the profile (safety guard)
        keys_to_delete = [k for k in parsed.get("keys_to_delete", []) if k in profile]

        if keys_to_delete:
            r.hdel(f"user:{user_code}:profile", *keys_to_delete)
            logger.info("[%s] Curative profile cleanup: deleted %s", user_code, keys_to_delete)
        else:
            logger.info("[%s] Curative profile cleanup: profile is clean", user_code)

    except Exception as exc:
        logger.error("Curative profile cleanup failed for %s: %s", user_code, exc)


def _decay_autobiographical_memories(user_code: str) -> int:
    """
    Monthly decay pass on autobiographical memories.

    For each autobiographical memory:
    - If importance >= MEMORY_DECAY_DURABLE_MIN → exempt (milestone permanent).
    - Otherwise: decayed = importance * (MEMORY_DECAY_FACTOR ^ age_months).
      - If decayed < MEMORY_DECAY_THRESHOLD → deleted from Qdrant.
      - Else → payload updated with the new (lower) importance.

    Returns the number of memories deleted.
    """
    qdrant    = get_qdrant()
    now_ts    = time.time()
    deleted   = 0
    updated   = 0
    offset    = None

    while True:
        results, next_offset = qdrant.scroll(
            collection_name=QDRANT_MEMORY_COLLECTION,
            scroll_filter={
                "must": [
                    {"key": "user_code",   "match": {"value": user_code}},
                    {"key": "memory_type", "match": {"value": "autobiographical"}},
                ]
            },
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not results:
            break

        to_delete = []
        for point in results:
            importance = float(point.payload.get("importance", 0.7))

            # Durable milestones are never decayed
            if importance >= MEMORY_DECAY_DURABLE_MIN:
                continue

            stored_at  = float(point.payload.get("timestamp", now_ts))
            age_months = (now_ts - stored_at) / (30 * 86400)
            decayed    = importance * (MEMORY_DECAY_FACTOR ** age_months)

            if decayed < MEMORY_DECAY_THRESHOLD:
                to_delete.append(point.id)
            elif decayed < importance - 0.01:
                qdrant.set_payload(
                    collection_name=QDRANT_MEMORY_COLLECTION,
                    payload={"importance": round(decayed, 4)},
                    points=[point.id],
                )
                updated += 1

        if to_delete:
            qdrant.delete(
                collection_name=QDRANT_MEMORY_COLLECTION,
                points_selector=PointIdsList(points=to_delete),
            )
            deleted += len(to_delete)
            logger.info("[%s] Autobio decay: %d stale memories deleted", user_code, len(to_delete))

        offset = next_offset
        if offset is None:
            break

    if deleted or updated:
        logger.info(
            "[%s] Autobio decay complete: %d deleted, %d importance updated",
            user_code, deleted, updated,
        )
    return deleted


def consolidate_memories(user_code: str = None, max_items: int = 20):
    """
    Monthly memory consolidation. Single public entry point — called on day 1 of each month
    by the nightly review scheduler, and on demand by the LLM self-action 'consolidate_memory'.

    If user_code is provided -> consolidate only this user.
    If user_code is None -> iterate over all users.

    Steps per user:
    1. Episodic consolidation  → compress episodic memories into autobiographical milestones
    2. Autobiographical decay  → reduce importance over time, delete stale memories
    3. Curative profile cleanup → remove duplicate / stale Redis profile keys
    """
    users = [user_code] if user_code else list(USER_CODES.keys())
    for uc in users:
        logger.info("Memory consolidation starting for user %s", uc)
        try:
            _consolidate_user_memories(uc, max_items)
            _decay_autobiographical_memories(uc)
            _curative_profile_cleanup(uc)
            logger.info("Memory consolidation done for user %s", uc)
        except Exception as e:
            logger.error("Memory consolidation failed for user %s: %s", uc, e)

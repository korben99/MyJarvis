"""Mémoire épisodique côté Redis : journal des échanges (sorted set) et lecture
des conversations récentes."""

import json
import time

from helpers import get_logger, get_redis

logger = get_logger("jarvis-memory")


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
    _now = time.time()  # single call — score and timestamp stay consistent
    entry = {
        "timestamp": _now,
        "session_id": session_id,
        "user": user_msg[:500],  # Truncate for storage
        "assistant": assistant_msg[:500],
        "mood": mood,
        "topics": topics or [],
        "importance": importance,
        "memory_summary": memory_summary,
        "satisfaction": "unknown",  # back-filled by analyzer (LLM) after each batch
    }
    # Store in a sorted set by timestamp for easy retrieval
    r.zadd(f"convlog:{user_code}", {json.dumps(entry): _now})

    # Keep only last 1000 exchanges (prevent unbounded growth)
    r.zremrangebyrank(f"convlog:{user_code}", 0, -1001)


def get_recent_conversations(user_code: str, hours: int = 24, limit: int = 20) -> list:
    """Get recent conversation exchanges (the most recent `limit`, oldest first).

    zrevrangebyscore takes the NEWEST entries of the window — zrangebyscore with
    num=limit returned the oldest ones, silently dropping recent exchanges on
    busy days. The list is then reversed to keep chronological order for callers."""
    r = get_redis()
    cutoff = time.time() - (hours * 3600)
    entries = r.zrevrangebyscore(
        f"convlog:{user_code}", "+inf", cutoff, start=0, num=limit
    )
    result = []
    for e in entries:
        try:
            result.append(json.loads(e))
        except (json.JSONDecodeError, ValueError):
            logger.warning("Skipping corrupted convlog entry for %s", user_code)
    result.reverse()
    return result

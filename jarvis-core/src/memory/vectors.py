"""Mémoire vectorielle (Qdrant) : nouveauté, stockage épisodique + autobiographique,
rétractation/archivage, recherche cognitive avec reconsolidation.

Dépend de `embed` (encodage) et de `profile` (pondération d'intérêts au ranking).
`_invalidate_timeline_cache` vit ici car ses seuls appelants sont les stockages autobio.
"""

import asyncio
import re
import time
import uuid
from datetime import date
from threading import Thread

import numpy as np
from config import (
    AUTOBIO_DEDUP_THRESHOLD,
    AUTOBIO_RECENCY_WINDOW_DAYS,
    MEMORY_DECAY_DURABLE_MIN,
    NOVELTY_THRESHOLD,
    QDRANT_MEMORY_COLLECTION,
    RECALL_MEMORY_SIMILARITY_THRESHOLD,
)
from helpers import get_logger, get_qdrant, get_redis
from qdrant_client.models import PointIdsList

from .embed import get_embed_model
from .profile import get_interest_weights

logger = get_logger("jarvis-memory")


def _invalidate_timeline_cache(user_code: str) -> None:
    """Called after a new autobiographical memory is stored."""
    try:
        get_redis().delete(f"cache:timeline:{user_code}")
    except Exception:
        pass


# ══════════════════════════════════════════════════
#  COMPLETE MEMORY TO QDRANT — Conversation history + summaries + AUTOBIOGRAPHIE
# ══════════════════════════════════════════════════


def compute_memory_novelty(
    user_code: str, text: str, vector: list | None = None, limit: int = 5
):
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
                            {
                                "key": "memory_type",
                                "match": {"value": "autobiographical"},
                            },
                        ]
                    },
                ],
            },
        ).points

        if not results:
            return 1.0

        # Clamp to [0, 1]: collection uses Distance.DOT, scores can exceed 1.0
        max_similarity = max(min(r.score, 1.0) for r in results)
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

        # #1 — Invariant: vector must be unit-norm before storage
        _norm = float(np.linalg.norm(vector))
        if abs(_norm - 1.0) > 0.01:
            logger.error(
                "[memory_invariant] NON-NORMALIZED vector for user=%s norm=%.4f — re-normalizing",
                user_code,
                _norm,
            )
            _arr = np.array(vector, dtype=np.float32)
            vector = (_arr / _norm).tolist()

        novelty = compute_memory_novelty(user_code, text, vector=vector)

        # #1 — Invariant: novelty must be in [0, 1]
        if not (0.0 <= novelty <= 1.0):
            logger.error(
                "[memory_invariant] novelty out of range for user=%s novelty=%.4f — clamping",
                user_code,
                novelty,
            )
            novelty = max(0.0, min(1.0, novelty))

        if novelty < NOVELTY_THRESHOLD:
            # #5 — Structured decision log
            logger.info(
                "[memory_decision] user=%s stored=False type=episodic reason=duplicate novelty=%.3f summary=%r",
                user_code,
                novelty,
                text[:80],
            )
            return

        # Deterministic ID: same (user_code, text) always produces the same UUID.
        # Qdrant upsert with an existing ID silently overwrites the point,
        # preventing duplicate entries when the same memory is stored twice.
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{user_code}:{text}"))
        importance = entry.get("importance", 0)

        qdrant.upsert(
            collection_name=QDRANT_MEMORY_COLLECTION,
            points=[
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "user_code": user_code,
                        "memory_type": "episodic",
                        "importance": importance,
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
        # #5 — Structured decision log
        logger.info(
            "[memory_decision] user=%s stored=True type=episodic novelty=%.3f importance=%.2f summary=%r",
            user_code,
            novelty,
            importance,
            text[:80],
        )

    except Exception as e:
        logger.error("Vector memory store failed: %s", e)


def store_autobiographical_event(
    user_code: str, summary: str, importance: float
) -> bool:
    """
    Store a major life / project milestone for the user.

    Skips storage if a semantically identical autobiographical memory already exists
    (cosine similarity ≥ AUTOBIO_DEDUP_THRESHOLD) to prevent the collection from
    accumulating redundant variants of the same fact over time.

    Returns True when the call left a mark — a new point, or an existing one reinforced
    with a higher importance. Returns False when the fact was dropped as a duplicate, or
    when the write failed.

    Ce retour existe pour `_consolidate_user_memories`, qui SUPPRIME les points épisodiques
    après avoir tenté d'écrire leurs résumés. Sans lui (avant le 21/08/2026), l'appelant ne
    pouvait pas distinguer « écrit » de « écarté par la dédup » : des faits tous
    silencieusement dédupliqués faisaient quand même détruire le lot d'origine. Les
    appelants qui ignorent la valeur de retour restent valides.
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
                    {"key": "user_code", "match": {"value": user_code}},
                    {"key": "memory_type", "match": {"value": "autobiographical"}},
                ]
            },
        ).points
        # The collection uses Distance.DOT — raw dot product score can exceed 1.0 when
        # stored vectors were uploaded without normalization. Clamp to [0, 1] before
        # comparing against the threshold to avoid spurious dedup skips or false hits.
        dedup_score = min(existing[0].score, 1.0) if existing else 0.0
        if existing and dedup_score >= AUTOBIO_DEDUP_THRESHOLD:
            # Reinforce the existing memory if the new submission carries higher importance
            existing_importance = float(existing[0].payload.get("importance", 0))
            if importance > existing_importance:
                qdrant.set_payload(
                    collection_name=QDRANT_MEMORY_COLLECTION,
                    payload={"importance": round(importance, 4)},
                    points=[existing[0].id],
                )
                logger.debug(
                    "Autobio dedup: reinforced '%s' %.2f → %.2f",
                    summary[:60],
                    existing_importance,
                    importance,
                )
                return True
            logger.debug(
                "Autobio dedup: skipping '%s' (similar=%.2f)",
                summary[:60],
                dedup_score,
            )
            return False

        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{user_code}:autobio:{summary}"))

        qdrant.upsert(
            collection_name=QDRANT_MEMORY_COLLECTION,
            points=[
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "user_code": user_code,
                        "memory_type": "autobiographical",
                        "status": "current",  # explicit — archive sets this to "past"
                        "text": summary,
                        "importance": importance,
                        "timestamp": time.time(),
                    },
                }
            ],
        )

        logger.info("Autobiographical memory stored: %s", summary)
        _invalidate_timeline_cache(user_code)
        return True

    except Exception as e:
        logger.error("Autobiographical memory failed: %s", e)
        return False


def _autobio_op(user_code: str, query: str, threshold: float, action: str) -> int:
    """Shared implementation for retract/archive operations on autobiographical memories.

    action="retract" → hard delete (reserved for errors/duplicates).
    action="archive"  → payload update status="past" (outdated facts, keeps history).
    """
    try:
        model = get_embed_model()
        qdrant = get_qdrant()
        vector = model.encode(query, normalize_embeddings=True).tolist()

        filt: dict = {
            "must": [
                {"key": "user_code", "match": {"value": user_code}},
                {"key": "memory_type", "match": {"value": "autobiographical"}},
            ]
        }
        if action == "archive":
            # Only archive current facts — already-archived ones are skipped
            filt["must_not"] = [{"key": "status", "match": {"value": "past"}}]

        results = qdrant.query_points(
            collection_name=QDRANT_MEMORY_COLLECTION,
            query=vector,
            limit=5,
            query_filter=filt,
        ).points

        to_act = [r.id for r in results if min(r.score, 1.0) >= threshold]
        if not to_act:
            return 0

        if action == "retract":
            qdrant.delete(
                collection_name=QDRANT_MEMORY_COLLECTION,
                points_selector=PointIdsList(points=to_act),
            )
            logger.info(
                "Autobio retracted %d point(s) for '%s'", len(to_act), query[:60]
            )
        else:
            qdrant.set_payload(
                collection_name=QDRANT_MEMORY_COLLECTION,
                payload={"status": "past", "archived_date": date.today().isoformat()},
                points=to_act,
            )
            logger.info(
                "Autobio archived %d point(s) for '%s'", len(to_act), query[:60]
            )

        _invalidate_timeline_cache(user_code)
        return len(to_act)
    except Exception as e:
        logger.error("_autobio_op(%s) failed: %s", action, e)
        return 0


def retract_autobiographical_event(
    user_code: str, query: str, threshold: float = 0.88
) -> int:
    """Delete autobiographical memories semantically matching the query.
    Reserved for genuine errors and strict duplicates — not for outdated facts.
    Higher threshold than archive (0.88 vs 0.78) — hard delete requires stricter match."""
    return _autobio_op(user_code, query, threshold, "retract")


def archive_autobiographical_event(
    user_code: str, query: str, threshold: float = 0.78
) -> int:
    """Mark autobiographical memories as past (status='past') without deleting them.
    Used when a fact is no longer current but retains historical value
    (e.g. changed jobs, stopped a hobby). Deprioritised in recall via status_factor."""
    return _autobio_op(user_code, query, threshold, "archive")


def get_autobiographical_facts(
    user_code: str, limit: int = 40, newest_first: bool = False
) -> list[str]:
    """Return current (non-archived) autobiographical memory summaries.

    Default: sorted oldest-first (temporal progression visible for cleaning).
    newest_first=True: most recent facts first (used to seed NIGHTLY_FACTS context).

    All facts fetched up to max(limit*2, 100) are returned without truncation
    so the nightly cleaning LLM sees recently-added duplicates, not just old facts.
    """
    try:
        qdrant = get_qdrant()
        results = qdrant.scroll(
            collection_name=QDRANT_MEMORY_COLLECTION,
            scroll_filter={
                "must": [
                    {"key": "user_code", "match": {"value": user_code}},
                    {"key": "memory_type", "match": {"value": "autobiographical"}},
                ],
                "must_not": [{"key": "status", "match": {"value": "past"}}],
            },
            limit=max(limit * 2, 100),
            with_payload=True,
        )[0]

        results.sort(key=lambda r: r.payload.get("timestamp", 0), reverse=newest_first)
        return [r.payload["text"] for r in results]
    except Exception as e:
        logger.error("get_autobiographical_facts failed: %s", e)
        return []


def _build_memory_filter(user_code: str, scope: str) -> dict:
    """Build Qdrant query_filter for a memory search by scope."""
    user_clause = {"key": "user_code", "match": {"value": user_code}}
    if scope in ("episodic", "autobiographical"):
        return {
            "must": [user_clause, {"key": "memory_type", "match": {"value": scope}}]
        }
    # "auto" — both layers; must_not absent types from slipping in
    return {
        "must": [
            user_clause,
            {
                "should": [
                    {"key": "memory_type", "match": {"value": "episodic"}},
                    {"key": "memory_type", "match": {"value": "autobiographical"}},
                ]
            },
        ]
    }


def search_memory(
    user_code: str, query: str, limit: int = 5, memory_scope: str = "auto"
):
    """Search vector memory. memory_scope filters to a specific layer or searches all ('auto')."""
    # Profile scope has no Qdrant data — Redis profile is already injected via build_memory_context()
    if memory_scope == "profile":
        return []

    try:
        model = get_embed_model()
        qdrant = get_qdrant()

        vector = model.encode(query, normalize_embeddings=True).tolist()

        results = qdrant.query_points(
            collection_name=QDRANT_MEMORY_COLLECTION,
            query=vector,
            limit=limit * 3,
            query_filter=_build_memory_filter(user_code, memory_scope),
        ).points

        memories = []
        # memorie recall with filter of similarity
        now = time.time()
        # Fetch once — avoids one Redis round-trip per result
        interest_weights = get_interest_weights(user_code)

        for r in results:
            # Clamp to [0, 1]: the collection uses Distance.DOT so scores can exceed 1.0
            # for old vectors that were stored before normalize_embeddings was enforced.
            sim = min(r.score, 1.0)
            if sim < RECALL_MEMORY_SIMILARITY_THRESHOLD:
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
            # Archived (past) facts are still findable but ranked lower so current
            # facts take priority; a 0.4 factor ensures past facts appear in
            # positions ~4-5 when a semantically close current fact scores higher.
            status_factor = 0.4 if payload.get("status") == "past" else 1.0
            # Weighted blend: similarity + importance + recency
            # All weights sum to 1.0 so the score stays in ~[0, 1]
            final_score = (
                sim * 0.50 + payload.get("importance", 0) * 0.30 + recency_bonus * 0.20
            ) * status_factor

            # Interest-weight boost: user-declared topics nudge ranking gently.
            # Cap at 0.08 so a strong semantic match (Δ≥0.08) is never overridden.
            # weight 1.0 = neutral (no boost); weight 3.0 → +0.08 (max).
            if interest_weights:
                text_lower = payload.get("text", "").lower()
                best_weight = max(
                    (
                        w
                        for term, w in interest_weights.items()
                        if re.search(
                            r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)",
                            text_lower,
                        )
                    ),
                    default=1.0,
                )
                interest_boost = min(0.08, max(0.0, (best_weight - 1.0) * 0.04))
                final_score = min(1.0, final_score + interest_boost)

            memories.append(
                {
                    "text": payload["text"],
                    "timestamp": timestamp,
                    "score": final_score,
                    "_id": r.id,
                    "_sim": sim,  # clamped similarity — used for reconsolidation gate
                    "_mem_type": mem_type,
                    "_importance": payload.get("importance", 0),
                    "_status": payload.get("status", "current"),
                }
            )
        # cognitive ranking
        memories.sort(key=lambda x: x["score"], reverse=True)
        top = memories[:limit]

        # Reconsolidation: recalling a memory reinforces it (neuroscience analogy).
        # Conditions (both must hold):
        #   1. autobiographical only — episodic memories are transient by design; boosting them
        #      would delay consolidation and bloat the episodic collection.
        #   2. raw semantic similarity > 0.82 — only strongly relevant recalls count;
        #      vaguely related memories (0.70–0.82) are not reinforced.
        # Cap at MEMORY_DECAY_DURABLE_MIN - 0.05 = 0.95 so they remain subject to monthly decay.
        _REINFORCE_SIM_THRESHOLD = 0.82
        _reinforce_cap = MEMORY_DECAY_DURABLE_MIN - 0.05
        try:
            for m in top:
                if m["_mem_type"] != "autobiographical":
                    continue
                if m["_status"] == "past":  # don't reinforce archived memories
                    continue
                if m["_sim"] < _REINFORCE_SIM_THRESHOLD:
                    continue
                old_imp = m["_importance"]
                new_imp = min(round(old_imp + 0.05, 4), _reinforce_cap)
                if new_imp > old_imp:
                    Thread(
                        target=qdrant.set_payload,
                        kwargs={
                            "collection_name": QDRANT_MEMORY_COLLECTION,
                            "payload": {"importance": new_imp},
                            "points": [m["_id"]],
                        },
                        daemon=True,
                    ).start()
        except Exception as _e:
            logger.warning("Memory reinforcement failed (non-blocking): %s", _e)

        # Strip internal fields before returning
        for m in top:
            m.pop("_id", None)
            m.pop("_sim", None)
            m.pop("_mem_type", None)
            m.pop("_importance", None)
            m.pop("_status", None)

        return top

    except Exception as e:
        logger.error("Memory search failed: %s", e)
        return []


async def async_search_memory(
    user_code: str, query: str, limit: int = 5, memory_scope: str = "auto"
) -> list:
    """Async-safe wrapper for search_memory.

    search_memory calls model.encode() which is CPU/GPU-bound and synchronous.
    Calling it directly from an async route would block the event loop.
    This wrapper always delegates to a thread pool — callers never need to
    remember to wrap it themselves.
    """
    return await asyncio.to_thread(search_memory, user_code, query, limit, memory_scope)

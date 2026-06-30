"""
rag.py — Qdrant document retrieval (RAG)
==========================================
Two-stage retrieval:

  Stage 1 — Document identification
    a) Title match : keyword substring search on known filenames.
       If query words (≥3 chars, stopwords excluded) match a doc name → candidate set restricted to
       those docs.  A quick semantic query (limit=3) confirms the best doc among
       candidates.
    b) Fallback : global semantic search (limit=5, score ≥ RAG_SCORE_THRESHOLD).
       Top-scoring chunk identifies the target document.

  Stage 2 — Focused semantic retrieval
    Vector search filtered to the identified document, score ≥ RAG_DOC_THRESHOLD
    (lower than global threshold since we're already in the right document).
    Returns top_k best chunks, each up to CHUNK_MAX_CHARS characters.

Doc names are cached in memory after first load (lazy, cleared on restart).
"""

import asyncio
import json
import re

import deps
from config import QDRANT_COLLECTION, RAG_SCORE_THRESHOLD, RAG_TOP_K
from helpers import _FR_STOPWORDS, get_logger
from qdrant_client.models import FieldCondition, Filter, MatchAny

logger = get_logger("jarvis-rag")

# Within-document threshold: more permissive than global since we're in the
# right document — avoids returning completely off-topic chunks.
RAG_DOC_THRESHOLD = max(0.25, RAG_SCORE_THRESHOLD - 0.15)

# Max characters per chunk (≈ 600 tokens).  Larger than before since we
# focus on one document and want coherent passages.
CHUNK_MAX_CHARS = 2500

# ── Doc-name cache ─────────────────────────────────────────────────────────────

_doc_names: set[str] | None = None


def _load_doc_names() -> set[str]:
    global _doc_names
    if _doc_names is not None:
        return _doc_names
    names: set[str] = set()
    offset = None
    while True:
        results, next_offset = deps.QDRANT_CLIENT.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=250,
            offset=offset,
            with_payload=["metadata", "file_name"],
            with_vectors=False,
        )
        for r in results:
            meta = r.payload.get("metadata") or {}
            name = meta.get("name") or r.payload.get("file_name") or ""
            if name:
                names.add(name)
        if next_offset is None:
            break
        offset = next_offset
    _doc_names = names
    logger.debug("RAG doc-name cache: %d documents", len(_doc_names))
    return _doc_names


def _title_candidates(query: str) -> list[str]:
    """Return doc names whose filename contains at least one significant query word."""
    words = {
        w for w in re.sub(r"[^\w]", " ", query.lower()).split()
        if len(w) >= 3 and w not in _FR_STOPWORDS
    }
    if not words:
        return []
    try:
        return [
            n for n in _load_doc_names()
            if any(
                re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", n.lower())
                for w in words
            )
        ]
    except Exception:
        return []


def _extract_chunk(hit) -> dict | None:
    payload = hit.payload
    text = ""
    if "_node_content" in payload:
        nc = payload["_node_content"]
        if isinstance(nc, dict):
            text = nc.get("text", "")
        elif isinstance(nc, str):
            try:
                text = json.loads(nc).get("text", "") or nc
            except Exception:
                text = nc
    else:
        text = payload.get("text", payload.get("content", ""))
    if not isinstance(text, str):
        text = str(text)
    # Normalize whitespace: collapse runs of spaces/tabs, cap consecutive newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return None
    # Reject encoded/binary content (e.g. 2D barcode raw data extracted from PDFs).
    # Valid text always has ~15-20% spaces; < 2% space ratio on a long string → garbage.
    if len(text) > 80 and text.count(" ") / len(text) < 0.02:
        return None
    meta = payload.get("metadata") or {}
    source = (
        meta.get("name") or meta.get("source") or payload.get("file_name") or "unknown"
    )
    if len(text) > CHUNK_MAX_CHARS:
        truncated = text[:CHUNK_MAX_CHARS]
        last_space = truncated.rfind(" ")
        text = (truncated[:last_space] if last_space > CHUNK_MAX_CHARS // 2 else truncated) + "…"
    return {"text": text, "source": source, "score": hit.score}


def _doc_name_from_hit(hit) -> str | None:
    chunk = _extract_chunk(hit)
    return chunk["source"] if chunk else None


# ── Main search ────────────────────────────────────────────────────────────────


async def search_documents(query: str, top_k: int = RAG_TOP_K) -> list[dict]:
    """Two-stage RAG: identify best document, then semantic search within it."""
    try:
        loop = asyncio.get_running_loop()
        if deps.EMBED_MODEL is None:
            raise RuntimeError("Embedding model not initialized")

        vector = await loop.run_in_executor(
            None,
            lambda: deps.EMBED_MODEL.encode(query, normalize_embeddings=True).tolist(),
        )

        # ── Stage 1a: title-based document identification ──────────────────────
        target_doc: str | None = None
        candidates = await loop.run_in_executor(None, _title_candidates, query)

        if candidates:
            # Semantic confirmation within title-matched docs
            title_hits = await loop.run_in_executor(
                None,
                lambda: deps.QDRANT_CLIENT.query_points(
                    collection_name=QDRANT_COLLECTION,
                    query=vector,
                    limit=3,
                    score_threshold=0.0,
                    query_filter=Filter(must=[
                        FieldCondition(key="metadata.name", match=MatchAny(any=candidates))
                    ]),
                ).points,
            )
            if title_hits:
                target_doc = _doc_name_from_hit(title_hits[0])
                logger.info("RAG stage-1 title match: '%s'", target_doc)

        # ── Stage 1b: global semantic fallback ─────────────────────────────────
        if not target_doc:
            global_hits = await loop.run_in_executor(
                None,
                lambda: deps.QDRANT_CLIENT.query_points(
                    collection_name=QDRANT_COLLECTION,
                    query=vector,
                    limit=5,
                    score_threshold=RAG_SCORE_THRESHOLD,
                ).points,
            )
            if not global_hits:
                logger.info("RAG: no results for: %s", query[:50])
                return []
            target_doc = _doc_name_from_hit(global_hits[0])
            logger.info("RAG stage-1 semantic fallback: '%s'", target_doc)

        if not target_doc:
            return []

        # ── Stage 2: focused semantic retrieval within target document ──────────
        doc_filter = Filter(must=[
            FieldCondition(key="metadata.name", match=MatchAny(any=[target_doc]))
        ])
        doc_hits = await loop.run_in_executor(
            None,
            lambda: deps.QDRANT_CLIENT.query_points(
                collection_name=QDRANT_COLLECTION,
                query=vector,
                limit=top_k,
                score_threshold=RAG_DOC_THRESHOLD,
                query_filter=doc_filter,
            ).points,
        )

        chunks = [c for hit in doc_hits if (c := _extract_chunk(hit))]
        logger.info(
            "RAG stage-2: %d/%d chunks from '%s' (threshold=%.2f) for: %s",
            len(chunks), top_k, target_doc, RAG_DOC_THRESHOLD, query[:50],
        )

        # Fallback: if threshold filtered everything, return best-scored chunks anyway
        if not chunks:
            doc_hits = await loop.run_in_executor(
                None,
                lambda: deps.QDRANT_CLIENT.query_points(
                    collection_name=QDRANT_COLLECTION,
                    query=vector,
                    limit=top_k,
                    score_threshold=0.0,
                    query_filter=doc_filter,
                ).points,
            )
            chunks = [c for hit in doc_hits if (c := _extract_chunk(hit))]
            if chunks:
                logger.info(
                    "RAG stage-2 fallback (no threshold): %d chunks from '%s'",
                    len(chunks), target_doc,
                )

        return chunks

    except Exception as e:
        logger.warning("RAG search failed: %s", e)
        return []

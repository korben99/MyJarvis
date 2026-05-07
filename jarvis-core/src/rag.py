"""
rag.py — Qdrant document retrieval (RAG)
==========================================
Embeds a query with the shared EMBED_MODEL and searches the Qdrant knowledge base.
"""

import asyncio
import json

import deps
from config import QDRANT_COLLECTION, RAG_SCORE_THRESHOLD, RAG_TOP_K
from helpers import get_logger

logger = get_logger("jarvis-rag")


async def search_documents(query: str, top_k: int = RAG_TOP_K) -> list[dict]:
    """Embed query and search Qdrant for relevant chunks."""
    try:
        loop = asyncio.get_running_loop()
        if deps.EMBED_MODEL is None:
            raise RuntimeError("Embedding model not initialized")
        vector = await loop.run_in_executor(
            None,
            lambda: deps.EMBED_MODEL.encode(query, normalize_embeddings=True).tolist(),
        )
        results = await loop.run_in_executor(
            None,
            lambda: deps.QDRANT_CLIENT.query_points(
                collection_name=QDRANT_COLLECTION,
                query=vector,
                limit=top_k,
                score_threshold=RAG_SCORE_THRESHOLD,
            ).points,
        )

        chunks = []
        for hit in results:
            payload = hit.payload
            text = ""
            if "_node_content" in payload:
                nc = payload["_node_content"]
                if isinstance(nc, dict):
                    text = nc.get("text", "")
                else:
                    try:
                        node = json.loads(nc)
                        text = node.get("text", "")
                    except Exception:
                        text = str(nc)
            else:
                text = payload.get("text", payload.get("content", ""))

            # OpenWebUI stores metadata in payload["metadata"] dict
            meta = payload.get("metadata") or {}
            source = (
                meta.get("name")
                or meta.get("source")
                or payload.get("file_name")
                or "unknown"
            )

            if text:
                chunks.append({"text": text[:1500], "source": source, "score": hit.score})

        logger.info("RAG: %d relevant chunks for: %s", len(chunks), query[:50])
        return chunks
    except Exception as e:
        logger.warning("RAG search failed: %s", e)
        return []

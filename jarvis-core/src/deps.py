"""
deps.py — Shared runtime singletons
=====================================
Single source of truth for all process-wide singleton objects and budget constants.
All other modules import from here instead of creating their own instances.

Mutable state (EMBED_MODEL) must be accessed as deps.EMBED_MODEL — never imported
with `from deps import EMBED_MODEL` because the lifespan rebinds the name at startup.
"""

import os

import httpx

from helpers import get_logger, get_qdrant, get_redis

logger = get_logger("jarvis-deps")

HAS_MEMORY = True

# ── Context budgets (max chars per source injected into the system prompt) ──
MEMORY_CHAR_BUDGET   = int(os.getenv("MEMORY_CHAR_BUDGET",   "2500"))
RAG_CHAR_BUDGET      = int(os.getenv("RAG_CHAR_BUDGET",      "4000"))
WEB_CHAR_BUDGET      = int(os.getenv("WEB_CHAR_BUDGET",      "2000"))
GOOGLE_CHAR_BUDGET   = int(os.getenv("GOOGLE_CHAR_BUDGET",   "3000"))
TOTAL_CONTEXT_BUDGET = int(os.getenv("TOTAL_CONTEXT_BUDGET", "10000"))

# ── HTTP clients ──────────────────────────────────────────────────────────────
HTTP_CLIENT = httpx.AsyncClient(timeout=30.0)

# Per-timeout streaming clients — reuse connection pools across requests.
_STREAM_CLIENTS: dict[float, httpx.AsyncClient] = {}


def get_stream_client(timeout: float) -> httpx.AsyncClient:
    if timeout not in _STREAM_CLIENTS:
        _STREAM_CLIENTS[timeout] = httpx.AsyncClient(timeout=timeout)
    return _STREAM_CLIENTS[timeout]


# ── Data stores ───────────────────────────────────────────────────────────────
REDIS_CLIENT  = get_redis()
QDRANT_CLIENT = get_qdrant()

# ── Embedding model (set at startup by lifespan in main.py) ──────────────────
# Access as deps.EMBED_MODEL — do NOT import this name directly.
EMBED_MODEL = None

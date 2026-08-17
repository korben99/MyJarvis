"""Modèle d'embedding — local d'abord, fallback HuggingFace. Singleton thread-safe."""

import os
from threading import Lock

from config import EMBED_MODEL_NAME
from helpers import get_logger
from sentence_transformers import SentenceTransformer

logger = get_logger("jarvis-memory")

# ── Embedding model — local-first, HF fallback ───────────────────────────
MODEL_CACHE_DIR = os.getenv(
    "MODEL_CACHE_DIR", "/opt/jarvis/jarvis-core/JarvisData/model_cache"
)
_embed_model = None
_embed_lock = Lock()


def _best_device() -> str:
    """Return 'mps' on Apple Silicon, 'cuda' if available, else 'cpu'."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        with _embed_lock:
            if _embed_model is None:
                os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
                device = _best_device()
                try:
                    # Fast path: model already on disk, no network call
                    _embed_model = SentenceTransformer(
                        EMBED_MODEL_NAME,
                        cache_folder=MODEL_CACHE_DIR,
                        local_files_only=True,
                        device=device,
                        model_kwargs={"torch_dtype": "float16"},
                    )
                    logger.info(
                        "Embedding model loaded from local cache (%s) on %s",
                        MODEL_CACHE_DIR,
                        device,
                    )
                except Exception:
                    # First run or cache missing — download from HuggingFace
                    logger.info(
                        "Downloading embedding model from HuggingFace (one-time)..."
                    )
                    _embed_model = SentenceTransformer(
                        EMBED_MODEL_NAME,
                        cache_folder=MODEL_CACHE_DIR,
                        device=device,
                        model_kwargs={"torch_dtype": "float16"},
                    )
                    logger.info(
                        "Embedding model downloaded and cached at %s on %s",
                        MODEL_CACHE_DIR,
                        device,
                    )
    return _embed_model

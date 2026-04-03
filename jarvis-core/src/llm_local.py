"""
llm_local.py — Inférence MLX directe (Apple Silicon, sans serveur HTTP)
========================================================================
Actif uniquement quand LLM_LOCAL=yes dans .env.

Les modèles sont chargés une seule fois au démarrage (preload_models) ou
au premier appel (lazy). Router et Primary partagent le même objet si leur
chemin est identique (optimisation un-seul-modèle).

TurboQuant KV cache (TURBO_QUANT=yes) :
  Applique apply_turboquant_cache après le chargement pour étendre la
  fenêtre de contexte sans surcoût mémoire proportionnel.
  Variables d'environnement :
    TURBO_QUANT=yes          activer (défaut : non)
    TURBO_QUANT_BITS=3       bits de quantification KV (défaut : 3)
    TURBO_QUANT_SINK=128     fp16_sink_size en tokens (défaut : 128)

Exports publics :
  preload_models()            → charge les modèles au démarrage (main.py lifespan)
  call_llm_local(...)         → str   (sync — utiliser via asyncio.to_thread)
  call_llm_local_async(...)   → str   (async non-streaming)
  stream_local(...)           → AsyncGenerator[str, None]  (streaming chat)
"""

import asyncio
import logging
import os
import threading
from collections import OrderedDict
from typing import AsyncGenerator

from config import (
    LLM_LOCAL,
    PRIMARY_MODEL,
    ROUTER_MODEL,
    THINKING_BUDGET_TOKENS,
    no_think_suffix,
)
from mlx_lm import generate, load, stream_generate
from mlx_lm.sample_utils import make_sampler

try:
    from mlx_lm.models.cache import make_prompt_cache as _make_prompt_cache
    _KV_CACHE_AVAILABLE = True
except ImportError:
    _KV_CACHE_AVAILABLE = False

HF_HOME = os.getenv("HF_HOME", "/opt/jarvis/models")
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HUB_CACHE"]

logger = logging.getLogger("jarvis-llm-local")


# ── TurboQuant ────────────────────────────────────────────────────────────

TURBO_QUANT = os.getenv("TURBO_QUANT", "").lower() in ("yes", "true", "1")
TURBO_QUANT_BITS = int(os.getenv("TURBO_QUANT_BITS", "3"))
TURBO_QUANT_SINK = int(os.getenv("TURBO_QUANT_SINK", "128"))


# ── Registre de modèles ───────────────────────────────────────────────────

_model_cache: dict[str, tuple] = {}  # model_path → (model, tokenizer)
_load_lock = threading.Lock()  # protège le chargement concurrent
_infer_lock = (
    asyncio.Lock()
)  # sérialise toutes les inférences MLX (contrainte Metal GPU)

# ── Session KV cache (prefix caching) ─────────────────────────────────────
# Stores one mlx_lm prompt_cache object per session_id.
# The cache accumulates KV pairs turn-by-turn; only new tokens are processed
# each turn once the common prefix (system + history) is cached.
# LRU eviction keeps memory bounded.

_session_kv: OrderedDict = OrderedDict()   # session_id → prompt_cache object
_MAX_SESSIONS = 5                          # max concurrent session caches


def _get_session_cache(session_id: str, model):
    """
    Return the existing KV cache for this session (LRU refresh) or create a new one.
    Thread-safe: called from _worker thread while _infer_lock is held.
    """
    if not _KV_CACHE_AVAILABLE or not session_id:
        return None
    if session_id in _session_kv:
        _session_kv.move_to_end(session_id)
        return _session_kv[session_id]
    # Evict oldest if at capacity
    if len(_session_kv) >= _MAX_SESSIONS:
        evicted, _ = _session_kv.popitem(last=False)
        logger.debug("KV cache: evicted session %s", evicted)
    cache = _make_prompt_cache(model)
    _session_kv[session_id] = cache
    logger.debug("KV cache: new session %s (total=%d)", session_id, len(_session_kv))
    return cache


def _load_model(model_path: str) -> tuple:
    """
    Charge un modèle mlx-lm et applique TurboQuant si activé.
    Thread-safe via double-checked locking.
    """
    if model_path in _model_cache:
        return _model_cache[model_path]

    with _load_lock:
        if model_path in _model_cache:
            return _model_cache[model_path]

        logger.info("Chargement modèle MLX : %s", model_path)

        logger.info("Loading model from: %s", model_path)
        logger.info("Exists: %s", os.path.exists(model_path))

        model, tokenizer = load(model_path)

        if TURBO_QUANT:
            try:
                import mlx_lm.models.cache as _cache_mod
                from mlx_core.cache import TurboQuantKVCache, apply_turboquant_cache

                apply_turboquant_cache(
                    model,
                    bits=TURBO_QUANT_BITS,
                    fp16_sink_size=TURBO_QUANT_SINK,
                )
                # Re-patch make_prompt_cache pour les architectures Qwen3 MoE
                # où head_dim est sur l.self_attn et non directement sur la couche.
                _bits, _sink = TURBO_QUANT_BITS, TURBO_QUANT_SINK

                class _PatchedCache(TurboQuantKVCache):
                    def __init__(self, head_dim, n_kv_heads, **kwargs):
                        super().__init__(
                            head_dim=head_dim,
                            n_kv_heads=n_kv_heads,
                            pq_bits=_bits,
                            fp16_sink_size=_sink,
                        )

                def _make_prompt_cache(m, max_kv_size=None):
                    caches = []
                    for layer in m.layers:
                        attn = getattr(layer, "self_attn", layer)
                        hd = getattr(attn, "head_dim", None) or getattr(
                            layer, "head_dim", 64
                        )
                        nkv = getattr(attn, "n_kv_heads", None) or getattr(
                            layer, "n_kv_heads", 8
                        )
                        caches.append(_PatchedCache(head_dim=hd, n_kv_heads=nkv))
                    return caches

                _cache_mod.make_prompt_cache = _make_prompt_cache
                logger.info(
                    "TurboQuant activé : bits=%d sink=%d",
                    TURBO_QUANT_BITS,
                    TURBO_QUANT_SINK,
                )
            except ImportError:
                logger.warning(
                    "TURBO_QUANT=yes mais mlx_core.cache introuvable — "
                    "installer mlx-core ou désactiver TURBO_QUANT"
                )

        _model_cache[model_path] = (model, tokenizer)
        logger.info("Modèle prêt : %s", model_path)
        return model, tokenizer


def preload_models() -> None:
    """
    Préchargement au démarrage de jarvis-api (appeler depuis main.py lifespan).
    Évite la latence du premier appel utilisateur.
    Si ROUTER_MODEL == PRIMARY_MODEL, un seul modèle est chargé.
    """
    if not LLM_LOCAL:
        return
    model_paths = {ROUTER_MODEL, PRIMARY_MODEL}  # set → déduplique
    for path in model_paths:
        _load_model(path)
    logger.info("MLX : %d modèle(s) préchargé(s)", len(model_paths))

    # Warmup JIT : déclenche la compilation MLX au démarrage, pas au 1er appel utilisateur.
    # Sans ça, le 1er generate() peut prendre 3-5s supplémentaires (JIT + graph build).
    warmup_msgs = [{"role": "user", "content": "ok"}]
    for path in model_paths:
        try:
            _generate_sync(
                path, warmup_msgs, temperature=0.0, max_tokens=1, no_think=False
            )
            logger.info("MLX warmup OK : %s", path)
        except Exception as exc:
            logger.warning("MLX warmup failed for %s : %s", path, exc)


# ── Prompt ────────────────────────────────────────────────────────────────


def _build_prompt(
    messages: list[dict],
    tokenizer,
    model_path: str,
    no_think: bool,
) -> str:
    """
    Convertit les messages OpenAI en prompt via le chat template du tokenizer.

    KV-cache compatibility: message content is NEVER modified here.
    Instead of appending /no_think to the last message (which would change
    the cached token sequence), we use thinking_budget=0 which the template
    inserts near the generation prompt — after the cacheable prefix.

    no_think=True  → thinking_budget=0   (disables <think> block entirely)
    no_think=False → thinking_budget=THINKING_BUDGET_TOKENS (limits think length)
    """
    budget = 0 if no_think else THINKING_BUDGET_TOKENS
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            thinking_budget=budget,
        )
    except TypeError:
        # Tokenizer does not support thinking_budget — fall back to suffix for no_think
        if no_think:
            suffix = no_think_suffix(model_path)
            if suffix:
                last = messages[-1]
                messages = [*messages[:-1], {**last, "content": last["content"] + suffix}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


# ── Inférence synchrone (cœur) ────────────────────────────────────────────


def _generate_sync(
    model_path: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    no_think: bool,
    session_id: str = "",
) -> str:
    """Génération complète (non-streaming). Bloquant — wrapper pour asyncio.to_thread."""
    model, tokenizer = _load_model(model_path)
    prompt = _build_prompt(messages, tokenizer, model_path, no_think)
    kv_cache = _get_session_cache(session_id, model)
    return generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=temperature),
        verbose=False,
        **({"prompt_cache": kv_cache} if kv_cache is not None else {}),
    )


# ── API publique ──────────────────────────────────────────────────────────


def call_llm_local(
    messages: list[dict],
    *,
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 500,
    no_think: bool = False,
    session_id: str = "",
    **_kwargs,  # absorbe api_url, api_key, json_response, timeout (non utilisés)
) -> str:
    """
    Inférence synchrone directe.
    Depuis du code async, toujours appeler via asyncio.to_thread pour ne pas
    bloquer la boucle événementielle.
    """
    return _generate_sync(model, messages, temperature, max_tokens, no_think, session_id)


async def call_llm_local_async(
    messages: list[dict],
    *,
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 500,
    no_think: bool = False,
    session_id: str = "",
    **_kwargs,
) -> str:
    """
    Inférence async non-streaming.
    Sérialisée par lock par modèle — router et primary indépendants.
    """
    # ====DEBUG====
    import time as _time

    _t_lock_wait = _time.time()
    logger.debug(
        "[TTFT] call_llm_local_async: waiting for _infer_lock (model=%s)",
        model.split("/")[-1],
    )
    # ====DEBUG====
    async with _infer_lock:
        # ====DEBUG====
        _t_infer_start = _time.time()
        logger.debug(
            "[TTFT] call_llm_local_async: lock acquired — waited %.3fs",
            _t_infer_start - _t_lock_wait,
        )
        # ====DEBUG====
        result = await asyncio.to_thread(
            _generate_sync, model, messages, temperature, max_tokens, no_think, session_id
        )
        # ====DEBUG====
        logger.debug(
            "[TTFT] call_llm_local_async: inference done — %.3fs (total %.3fs)",
            _time.time() - _t_infer_start,
            _time.time() - _t_lock_wait,
        )
        # ====DEBUG====
        return result


async def stream_local(
    messages: list[dict],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 8192,
    no_think: bool = False,
    session_id: str = "",
    **_kwargs,
) -> AsyncGenerator[str, None]:
    """
    Streaming token-par-token via mlx_lm.stream_generate.
    Remplace stream_openai() de llm_client.py en mode local.

    Implémentation :
    - stream_generate tourne dans un thread dédié (synchrone, bloquant)
    - Les chunks transitent par une asyncio.Queue vers l'event loop
    - _infer_lock garantit qu'un seul flux est actif à la fois
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    # ====DEBUG====
    import time as _time

    _t_lock_wait = _time.time()
    logger.debug(
        "[TTFT] stream_local: waiting for _infer_lock (model=%s)", model.split("/")[-1]
    )
    # ====DEBUG====

    def _worker():
        # ====DEBUG====
        _t_infer = _time.time()
        logger.debug(
            "[TTFT] stream_local: inference started (lock held %.3fs)",
            _t_infer - _t_lock_wait,
        )
        # ====DEBUG====
        mlx_model, tokenizer = _load_model(model)
        prompt = _build_prompt(messages, tokenizer, model, no_think)
        kv_cache = _get_session_cache(session_id, mlx_model)
        if kv_cache is not None:
            logger.debug(
                "[KV] session=%s cache_offset=%d",
                session_id, getattr(kv_cache[0] if isinstance(kv_cache, list) else kv_cache, "offset", "?"),
            )
        first = True
        try:
            for chunk in stream_generate(
                mlx_model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=make_sampler(temp=temperature),
                **({"prompt_cache": kv_cache} if kv_cache is not None else {}),
            ):
                if chunk.text:
                    # ====DEBUG====
                    if first:
                        logger.debug(
                            "[TTFT] stream_local: first token generated — %.3fs since inference start",
                            _time.time() - _t_infer,
                        )
                        first = False
                    # ====DEBUG====
                    loop.call_soon_threadsafe(queue.put_nowait, chunk.text)
        except Exception as exc:
            logger.error("stream_local erreur : %s", exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinelle fin de flux

    async with _infer_lock:
        # ====DEBUG====
        logger.debug(
            "[TTFT] stream_local: lock acquired — waited %.3fs",
            _time.time() - _t_lock_wait,
        )
        # ====DEBUG====
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

"""
llm_local.py — Inférence MLX directe (Apple Silicon, sans serveur HTTP)
========================================================================
Actif uniquement quand LLM_LOCAL=yes dans .env.

Les modèles sont chargés une seule fois au démarrage (preload_models) ou
au premier appel (lazy). Router et Primary partagent le même objet si leur
chemin est identique (optimisation un-seul-modèle).

KV cache quantifié (QUANT_KV=yes) :
  Utilise QuantizedKVCache de mlx_lm pour réduire la bande passante mémoire
  pendant le décodage (4-bit = 4× moins de données lues par token généré).
  S'applique uniquement au modèle primaire (le routeur garde KVCache standard).
  Variables d'environnement :
    QUANT_KV=yes       activer (défaut : non)
    QUANT_KV_BITS=4    bits de quantification (4 ou 8, défaut : 4)

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
from typing import Any, AsyncGenerator

from config import (
    LLM_LOCAL,
    PRIMARY_MODEL,
    ROUTER_MODEL,
    THINKING_BUDGET_TOKENS,
    is_qwen3,
)
from mlx_lm import generate, load, stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler

try:
    import mlx_lm.models.cache as _cache_mod

    _KV_CACHE_AVAILABLE = True
except ImportError:
    _cache_mod = None
    _KV_CACHE_AVAILABLE = False

HF_HOME = os.getenv("HF_HOME", "/opt/jarvis/models")
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HUB_CACHE"]

logger = logging.getLogger("jarvis-llm-local")


# ── KV cache quantifié ────────────────────────────────────────────────────

QUANT_KV = os.getenv("QUANT_KV", "").lower() in ("yes", "true", "1")
QUANT_KV_BITS = int(os.getenv("QUANT_KV_BITS", "4"))


# ── Registre de modèles ───────────────────────────────────────────────────

_model_cache: dict[str, tuple] = {}  # model_path → (model, tokenizer)
_load_lock = threading.Lock()  # protège le chargement concurrent
_infer_lock = (
    asyncio.Lock()
)  # sérialise toutes les inférences MLX (contrainte Metal GPU)

# ── Session KV cache (prefix caching) ─────────────────────────────────────
# Key: "{session_id}:{'nt'|'think'}" — thinking mode included to prevent
# prefix-token mismatch (enable_thinking changes the generation prompt suffix).
# Both modes cache independently; LRU eviction keeps memory bounded.

_session_kv: OrderedDict = OrderedDict()        # cache_key → prompt_cache object
_session_kv_first_hash: dict[str, int] = {}     # cache_key → hash(messages[1].content)
_MAX_SESSIONS = 5  # max concurrent session caches


def _first_complete_json(text: str) -> str | None:
    """
    Return the first complete top-level JSON object or array found in *text*, or None.

    Scans character-by-character tracking brace/bracket depth and string escapes.
    Used by _generate_sync to stop streaming as soon as JSON output is complete.
    """
    start = -1
    open_char = close_char = ""
    depth = 0
    in_string = False
    escape_next = False

    for i, c in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if c == "\\" and in_string:
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if start == -1:
            if c == "{":
                start, open_char, close_char, depth = i, "{", "}", 1
            elif c == "[":
                start, open_char, close_char, depth = i, "[", "]", 1
        else:
            if c == open_char:
                depth += 1
            elif c == close_char:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def _strip_thinking(text: str) -> str:
    if not text:
        return text

    # Cas Qwen / instruct
    if "Final Answer:" in text:
        return text.split("Final Answer:")[-1].strip()

    # Cas classique reasoning leak
    if "Thinking Process:" in text:
        return text.split("Thinking Process:")[-1].strip()

    return text


def _get_session_cache(session_id: str, model, no_think: bool = False):
    """
    Return the existing KV cache for this session/mode (LRU refresh) or create a new one.
    Thread-safe: called from inference thread while _infer_lock is held.

    Cache key = session_id + thinking mode.
    Rationale: enable_thinking=False appends '<think>\\n\\n</think>\\n\\n' to the prompt,
    enable_thinking=True appends '<think>\\n'. These are different generation-prompt suffixes,
    so KV state built under one mode is invalid for the other. Separating the keys allows
    both modes to benefit from prefix caching independently.
    """
    if not _KV_CACHE_AVAILABLE or not session_id:
        return None
    # Include thinking mode in key to prevent prefix-token mismatch
    cache_key = f"{session_id}:{'nt' if no_think else 'think'}"
    if cache_key in _session_kv:
        _session_kv.move_to_end(cache_key)
        cache = _session_kv[cache_key]
        offset = _kv_offset(cache)
        logger.debug(
            "KV cache: HIT  key=%s offset=%s slots=%d/%d",
            cache_key,
            offset,
            len(_session_kv),
            _MAX_SESSIONS,
        )
        return cache
    # Evict oldest if at capacity
    if len(_session_kv) >= _MAX_SESSIONS:
        evicted, _ = _session_kv.popitem(last=False)
        _session_kv_first_hash.pop(evicted, None)
        logger.debug(
            "KV cache: EVICT key=%s slots=%d/%d",
            evicted,
            len(_session_kv),
            _MAX_SESSIONS,
        )

    cache = _cache_mod.make_prompt_cache(model)
    logger.debug(
        "KV cache: MISS  key=%s slots=%d/%d", cache_key, len(_session_kv), _MAX_SESSIONS
    )
    _session_kv[cache_key] = cache
    return cache


def _kv_offset(cache) -> str:
    """Extract KV cache offset for logging (list-of-layers or single object)."""
    obj = cache[0] if isinstance(cache, list) and cache else cache
    return str(getattr(obj, "offset", "?"))


def _load_model(model_path: str) -> tuple:
    """
    Charge un modèle mlx-lm. Thread-safe via double-checked locking.
    """
    if model_path in _model_cache:
        return _model_cache[model_path]

    with _load_lock:
        if model_path in _model_cache:
            return _model_cache[model_path]

        logger.info("Chargement modèle MLX : %s", model_path)
        model, tokenizer = load(model_path)
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
    warmup_msgs = [
        {"role": "system", "content": "Tu es un assistant."},
        {"role": "user", "content": "Salut"},
    ]
    for path in model_paths:
        try:
            _generate_sync(
                path,
                warmup_msgs,
                temperature=0.0,
                max_tokens=32,  # assez pour le JIT sans déclencher le warning de troncature
                no_think=True,
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
    Construit le prompt final via apply_chat_template.

    Qwen3.x uniquement : passe enable_thinking pour contrôler le bloc <think>.
      enable_thinking=False → <think>\\n\\n</think>\\n\\n inséré (no-think)
      enable_thinking=True  → <think>\\n inséré (thinking libre)

    Fallback : si enable_thinking n'est pas accepté (TypeError), réessaie sans.
    """
    qwen3 = is_qwen3(model_path)
    base_kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}

    if qwen3:
        # Build thinking kwargs:
        # - no_think=True  → enable_thinking=False + thinking_budget=0 (belt+suspenders:
        #   mlx-lm issue #1625 — enable_thinking=False alone may not suppress thinking in
        #   some library versions; thinking_budget=0 enforces the budget at template level)
        # - no_think=False → enable_thinking=True + THINKING_BUDGET_TOKENS cap
        #   (thinking + output share the same max_tokens budget in mlx-lm; Qwen3 open-source
        #   correctly honours thinking_budget in apply_chat_template)
        if no_think:
            think_kwargs: dict[str, Any] = {"enable_thinking": False, "thinking_budget": 0}
        else:
            think_kwargs = {"enable_thinking": True}
            if THINKING_BUDGET_TOKENS > 0:
                think_kwargs["thinking_budget"] = THINKING_BUDGET_TOKENS

        # Try full kwargs; fall back if tokenizer version is too old
        try:
            return tokenizer.apply_chat_template(messages, **base_kwargs, **think_kwargs)
        except TypeError:
            # thinking_budget not supported → retry with only enable_thinking
            try:
                return tokenizer.apply_chat_template(
                    messages, **base_kwargs, enable_thinking=not no_think
                )
            except TypeError:
                logger.debug(
                    "enable_thinking not supported for %s — falling back to default template",
                    model_path.split("/")[-1],
                )

    return tokenizer.apply_chat_template(messages, **base_kwargs)



# ── Inférence synchrone (cœur) ────────────────────────────────────────────
def _generate_sync(
    model_path: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    no_think: bool,
    session_id: str = "",
    json_response: bool = False,
) -> str:
    """Génération complète (non-streaming). Bloquant — wrapper pour asyncio.to_thread.

    json_response=True + no_think=True : utilise stream_generate avec early-stop
    dès que le premier objet JSON complet est détecté. Évite de générer des centaines
    de tokens superflus après la } finale (explications, markdown, …).
    """
    model, tokenizer = _load_model(model_path)
    prompt = _build_prompt(messages, tokenizer, model_path, no_think)

    prompt_tokens = len(tokenizer.encode(prompt))
    kv_cache = _get_session_cache(session_id, model, no_think)
    quant_kwargs = {"kv_bits": QUANT_KV_BITS, "kv_group_size": 64} if QUANT_KV else {}
    # Global 10k ceiling. Thinking + output tokens are counted together in mlx-lm,
    # so thinking_budget (set in _build_prompt) reserves the thinking portion.
    effective_max = min(max_tokens, 10000)
    model_short = model_path.split("/")[-1]

    # Qwen3 recommended: top_k=20 always; top_p=0.95 (thinking) / 0.8 (no-think); min_p=0.0
    sampler = make_sampler(
        temp=temperature,
        top_p=0.95 if not no_think else 0.8,
        top_k=20,
        min_p=0.0,
    )
    # Repetition penalty — prevents the model from looping on the same tokens,
    # especially critical at low temperatures (near-greedy) where the model can
    # get stuck in "eneeneeneene..." style degenerate output.
    logits_procs = make_logits_processors(repetition_penalty=1.1, repetition_context_size=20)
    cache_kwargs: dict = {"prompt_cache": kv_cache} if kv_cache is not None else {}

    early_stopped = False

    if json_response and no_think:
        # Stream token-by-token; break as soon as a complete JSON object/array
        # is found. GPU stops exactly there — no wasted tokens after the closing }.
        raw_chunks: list[str] = []
        for chunk in stream_generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=effective_max,
            sampler=sampler,
            logits_processors=logits_procs,
            **cache_kwargs,
            **quant_kwargs,
        ):
            if chunk.text:
                raw_chunks.append(chunk.text)
                if _first_complete_json("".join(raw_chunks)) is not None:
                    early_stopped = True
                    break
        raw = "".join(raw_chunks)
        result = _first_complete_json(raw) or raw
    else:
        result = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=effective_max,
            sampler=sampler,
            logits_processors=logits_procs,
            verbose=False,
            **cache_kwargs,
            **quant_kwargs,
        )

    # ── Stats réponse ──────────────────────────────────────────────
    resp_tokens = len(tokenizer.encode(result))
    thinking_active = "</think>" in result
    call_type = "json" if json_response else "text"
    stop_label = "early-stop" if early_stopped else "eos/limit"
    pct = resp_tokens * 100 // effective_max if effective_max else 0

    logger.info(
        "[LLM-STATS] %s | %s no_think=%s | %s | prompt=%d tok | resp=%d/%d tok (%d%%)",
        model_short, call_type, no_think, stop_label,
        prompt_tokens, resp_tokens, effective_max, pct,
    )
    if not early_stopped and resp_tokens >= int(effective_max * 0.9):
        logger.warning(
            "[LLM-STATS] POSSIBLE TRUNCATION — resp=%d tok near limit=%d (model=%s)",
            resp_tokens, effective_max, model_short,
        )

    # ── Strip thinking block, keep actual answer ──────────────────
    # split("</think>", 1)[-1] → keeps everything AFTER </think>
    # Truncated case: model hit token budget mid-reasoning → no </think>.
    # Discard from <think> onwards to avoid leaking raw reasoning.
    if "</think>" in result:
        result = result.split("</think>", 1)[-1].strip()
    elif "<think>" in result:
        result = result.split("<think>", 1)[0].strip()
    elif not no_think:
        # enable_thinking=True → the chat template appends <think>\n to the prompt.
        # The model's raw output starts INSIDE the think block, so there is no
        # <think> tag in `result`. If </think> was never generated (truncated
        # mid-reasoning), there is no actual answer — return empty so callers
        # can detect the failure cleanly instead of receiving raw markdown.
        logger.warning(
            "_generate_sync: thinking truncated before </think> (model=%s) — returning empty",
            model_short,
        )
        result = ""

    return _strip_thinking(result)


# ── API publique ──────────────────────────────────────────────────────────


def call_llm_local(
    messages: list[dict],
    *,
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 500,
    no_think: bool = False,
    session_id: str = "",
    json_response: bool = False,
    **_kwargs,  # absorbe api_url, api_key, timeout (non utilisés)
) -> str:
    """
    Inférence synchrone directe.
    Depuis du code async, toujours appeler via asyncio.to_thread pour ne pas
    bloquer la boucle événementielle.
    """
    return _generate_sync(
        model, messages, temperature, max_tokens, no_think, session_id, json_response
    )


async def call_llm_local_async(
    messages: list[dict],
    *,
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 500,
    no_think: bool = False,
    session_id: str = "",
    json_response: bool = False,
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
            _generate_sync,
            model,
            messages,
            temperature,
            max_tokens,
            no_think,
            session_id,
            json_response,
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
    max_tokens: int = 10000,
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

        # ── Encode prompt une seule fois (réutilisé pour cache validation ET stats) ─
        prompt_tokens = len(tokenizer.encode(prompt))
        model_short = model.split("/")[-1]

        kv_cache = _get_session_cache(session_id, mlx_model, no_think)

        # ── Validation préfixe KV cache par hash du premier message d'historique ──
        # mlx_lm suppose que les cache.offset premiers tokens du nouveau prompt sont
        # IDENTIQUES à ceux du dernier appel. La fenêtre glissante hist[-8:] peut
        # supprimer le message le plus ancien → préfixe différent → output corrompu.
        #
        # Détection fiable : on compare le hash du contenu du premier message
        # d'historique (messages[1]). Si ce message change entre deux tours,
        # la fenêtre a glissé → on recrée le cache proprement.
        # La comparaison par token count seul (ancienne approche) ratait les
        # glissements qui font croître le prompt net (drop 400 tok, ajout 600 tok).
        if kv_cache is not None and session_id:
            _cache_key = f"{session_id}:{'nt' if no_think else 'think'}"
            # messages = [system, ...hist..., current_user] — first hist msg is [1]
            _cur_hash = hash(messages[1]["content"]) if len(messages) > 2 else 0
            _stored_hash = _session_kv_first_hash.get(_cache_key)
            if _stored_hash is not None and _stored_hash != _cur_hash:
                # First history message changed → window shifted → stale KV
                _session_kv.pop(_cache_key, None)
                _session_kv_first_hash.pop(_cache_key, None)
                kv_cache = _get_session_cache(session_id, mlx_model, no_think)
                logger.debug(
                    "KV cache: REBUILT (first hist msg changed — window shifted) key=%s",
                    _cache_key,
                )
            # Record current first-hist-message hash for next-turn validation
            if kv_cache is not None:
                _session_kv_first_hash[_cache_key] = _cur_hash

        first = True
        raw_chunks: list[str] = []  # accumule la réponse brute pour stats
        # Global 10k ceiling — thinking + output are counted together in mlx-lm.
        # thinking_budget in the chat template reserves the thinking portion.
        budget = min(max_tokens, 10000)
        quant_kwargs = {"kv_bits": QUANT_KV_BITS, "kv_group_size": 64} if QUANT_KV else {}
        try:
            for chunk in stream_generate(
                mlx_model,
                tokenizer,
                prompt=prompt,
                max_tokens=budget,
                # Qwen3 recommended: top_k=20; top_p=0.95 (thinking) / 0.8 (no-think); min_p=0.0
                sampler=make_sampler(
                    temp=temperature,
                    top_p=0.95 if not no_think else 0.8,
                    top_k=20,
                    min_p=0.0,
                ),
                logits_processors=make_logits_processors(
                    repetition_penalty=1.1, repetition_context_size=20
                ),
                **({"prompt_cache": kv_cache} if kv_cache is not None else {}),
                **quant_kwargs,
            ):
                if chunk.text:
                    text = chunk.text
                    raw_chunks.append(text)

                    if first:
                        logger.debug(
                            "[TTFT] stream_local: first token generated — %.3fs since inference start",
                            _time.time() - _t_infer,
                        )
                        first = False

                    # Forward all tokens to sse() which handles <think>/<think>
                    # filtering. Do NOT break at </think> — the actual answer
                    # is generated AFTER </think> and must be streamed.
                    loop.call_soon_threadsafe(queue.put_nowait, text)

        except Exception as exc:
            logger.error("stream_local erreur : %s", exc)
        finally:
            # ── Stats réponse ──────────────────────────────────────
            raw_resp = "".join(raw_chunks)
            resp_tokens = len(tokenizer.encode(raw_resp))
            thinking_active = "</think>" in raw_resp
            pct = resp_tokens * 100 // budget if budget else 0
            logger.info(
                "[LLM-STATS] %s | stream no_think=%s thinking=%s | eos/limit"
                " | prompt=%d tok | resp=%d/%d tok (%d%%)",
                model_short, no_think, thinking_active,
                prompt_tokens, resp_tokens, budget, pct,
            )
            if resp_tokens >= int(budget * 0.9):
                logger.warning(
                    "[LLM-STATS] POSSIBLE TRUNCATION — resp=%d tok near limit=%d (model=%s)",
                    resp_tokens, budget, model_short,
                )
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

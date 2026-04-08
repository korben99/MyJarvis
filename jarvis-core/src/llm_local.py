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
import copy
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, AsyncGenerator

from config import (
    LLM_LOCAL,
    PRIMARY_MODEL,
    REASONING_MODEL,
    ROUTER_MODEL,
    THINKING_BUDGET_TOKENS,
    is_hermes,
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

# ── Debug prompt/response logging ─────────────────────────────────────────
# Activé via LLM_DEBUG_PROMPTS=yes dans .env.
# Écrit le prompt brut et la réponse brute (avant stripping) dans logs/prompts.log.

LLM_DEBUG_PROMPTS = os.getenv("LLM_DEBUG_PROMPTS", "").lower() in ("yes", "true", "1")
_PROMPTS_LOG_PATH = "/opt/jarvis/logs/prompts.log"
_PROMPTS_LOG_SEP = "=" * 80


def _debug_log(model_short: str, no_think: bool, prompt: str, raw_output: str) -> None:
    """Écrit prompt + réponse brute dans prompts.log si LLM_DEBUG_PROMPTS=yes."""
    if not LLM_DEBUG_PROMPTS:
        return
    import datetime

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = (
        f"\n{_PROMPTS_LOG_SEP}\n"
        f"[{ts}] model={model_short}  no_think={no_think}\n"
        f"--- PROMPT ---\n{prompt}\n"
        f"--- RESPONSE (raw) ---\n{raw_output}\n"
        f"{_PROMPTS_LOG_SEP}\n"
    )
    try:
        with open(_PROMPTS_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(entry)
    except Exception as exc:
        logger.warning("_debug_log: impossible d'écrire prompts.log : %s", exc)


# ── KV cache quantifié ────────────────────────────────────────────────────

QUANT_KV = os.getenv("QUANT_KV", "").lower() in ("yes", "true", "1")
QUANT_KV_BITS = int(os.getenv("QUANT_KV_BITS", "4"))


# ── Profils d'inférence par modèle ────────────────────────────────────────


@dataclass(frozen=True)
class _ModelProfile:
    """Sampling + generation parameters for a given model family.

    Centralises all per-model tuning so call sites are free of if/else.
    Resolved once per call via _model_profile().
    """

    temp_think: float  # temperature when enable_thinking=True
    temp_nothink: float  # temperature when enable_thinking=False / no-think
    top_p_think: float  # top_p when enable_thinking=True
    top_p_nothink: float  # top_p when enable_thinking=False / no-think
    top_k: int  # 0 = disabled
    min_p: float
    repetition_penalty: float  # > 1.0 penalises repeated tokens
    repetition_context_size: int  # how many past tokens to look at
    frequency_penalty: float  # additive penalty proportional to frequency
    use_quant_kv: bool  # whether to enable KV cache quantisation
    stop_tokens: tuple[str, ...]  # extra stop strings beyond the tokenizer's EOS


def _model_profile(model_path: str) -> _ModelProfile:
    """Return the inference profile for *model_path*.

    Two profiles:
    - Qwen3 (primary): thinking-aware sampler + KV quantisation allowed.
    - Generic (router / non-Qwen3): relaxed sampler, no KV quantisation.
      Applying top_k=20 or 4-bit KV to small instruct models (e.g. Qwen2.5-3B-8bit)
      causes degenerate looping — these must stay at their default settings.
    """
    if is_qwen3(model_path):
        # Official Qwen3 recommendations: temp 0.6 (think) / 0.7 (no-think),
        # top_p 0.95 / 0.80, top_k 20, min_p 0.
        return _ModelProfile(
            temp_think=0.6,
            temp_nothink=0.7,
            top_p_think=0.95,
            top_p_nothink=0.80,
            top_k=20,
            min_p=0.0,
            repetition_penalty=1.1,
            repetition_context_size=64,
            frequency_penalty=0.05,
            use_quant_kv=QUANT_KV,
            stop_tokens=(),
        )
    if is_hermes(model_path):
        # Hermes-3 (Llama 3.2 base) — purpose-built for structured/JSON output.
        # Temperature=0 is safe (no greedy loops), no penalties needed (trained format).
        # stop_tokens: Hermes occasionally generates continuation text after <|im_end|>
        # (tokenizer EOS not always sufficient); explicit stop string closes the leak.
        return _ModelProfile(
            temp_think=0.0,
            temp_nothink=0.0,
            top_p_think=1.0,
            top_p_nothink=1.0,
            top_k=0,
            min_p=0.0,
            repetition_penalty=1.0,  # already-quantised Q4; penalties degrade JSON quality
            repetition_context_size=64,
            frequency_penalty=0.0,
            use_quant_kv=False,  # already Q4 affine quantised
            stop_tokens=("<|im_end|>",),
        )
    # Generic profile — fallback for any other small model
    return _ModelProfile(
        temp_think=0.7,
        temp_nothink=0.7,
        top_p_think=0.90,
        top_p_nothink=0.90,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.3,
        repetition_context_size=64,
        frequency_penalty=0.10,
        use_quant_kv=False,
        stop_tokens=(),
    )


# ── Registre de modèles ───────────────────────────────────────────────────

_model_cache: dict[str, tuple] = {}  # model_path → (model, tokenizer)
_load_lock = threading.Lock()  # protège le chargement concurrent
_infer_lock = (
    threading.Lock()
)  # sérialise toutes les inférences MLX (contrainte Metal GPU)
# threading.Lock (pas asyncio.Lock) : garantit la mutual exclusion entre les appelants
# sync (call_llm_local via asyncio.to_thread) ET les appelants async (call_llm_local_async,
# stream_local). asyncio.Lock est invisible depuis les threads → race GPU sans threading.Lock.

# ── System prompt KV cache ────────────────────────────────────────────────
# Caches the prefilled KV states for the system prompt (which is token-identical
# across every turn). Each streaming inference gets a deepcopy of this cache so
# only the conversation history + current user message need prefill processing.
#
# Multi-turn session caching was removed: Qwen3's enable_thinking template adds
# <think>…</think> tokens to the generation prompt that are absent from
# historical message reconstruction, causing permanent token-sequence mismatch.

_sys_kv: dict[str, tuple[int, Any]] = {}  # model_path → (sys_hash, base_cache)
_sys_kv_lock = threading.Lock()  # protects lazy build of the base cache


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


def _make_system_kv(model_path: str, model, tokenizer, system_content: str) -> Any:
    """
    Pre-fill the system prompt into a fresh KV cache.
    Called once per unique (model_path, system_content) pair — never call directly.

    Strategy: run stream_generate for 1 token to drive the prefill, then trim
    the cache offset back to the exact system token count so the 1 garbage
    decode token is overwritten by real tokens on the first actual inference.
    """
    sys_messages = [{"role": "system", "content": system_content}]
    sys_prompt_text = tokenizer.apply_chat_template(
        sys_messages, tokenize=False, add_generation_prompt=False
    )
    sys_token_count = len(tokenizer.encode(sys_prompt_text))

    profile = _model_profile(model_path)
    quant_kwargs = (
        {"kv_bits": QUANT_KV_BITS, "kv_group_size": 64} if profile.use_quant_kv else {}
    )
    cache = _cache_mod.make_prompt_cache(model)
    sampler = make_sampler(temp=0.0, top_p=1.0, top_k=0, min_p=0.0)

    try:
        for _ in stream_generate(
            model,
            tokenizer,
            prompt=sys_prompt_text,
            max_tokens=1,
            sampler=sampler,
            prompt_cache=cache,
            **quant_kwargs,
        ):
            break  # stop after prefill + 1 decode step

        # Trim offset to system token count — the 1 garbage token will be
        # overwritten by the first real token on the next call.
        for layer_cache in cache:
            if hasattr(layer_cache, "offset"):
                layer_cache.offset = sys_token_count

    except Exception as exc:
        logger.warning("System KV cache build failed: %s", exc)
        return None

    logger.info(
        "KV cache: system prompt prefilled (%d tok, model=%s)",
        sys_token_count,
        model_path.split("/")[-1],
    )
    return cache


def _get_system_cache(model_path: str, model, tokenizer, system_content: str) -> Any:
    """
    Return a fresh deepcopy of the system prompt KV cache (lazy init, thread-safe).
    Returns None on any failure → caller proceeds without cache.
    Must be called from inside _infer_lock.
    """
    if not _KV_CACHE_AVAILABLE or not system_content:
        return None
    h = hash(system_content)
    with _sys_kv_lock:
        if model_path not in _sys_kv or _sys_kv[model_path][0] != h:
            base = _make_system_kv(model_path, model, tokenizer, system_content)
            if base is None:
                return None
            _sys_kv[model_path] = (h, base)
        try:
            return copy.deepcopy(_sys_kv[model_path][1])
        except Exception as exc:
            logger.warning(
                "KV cache: deepcopy failed (%s) — running without cache", exc
            )
            return None


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

    # Limit Metal allocator cache to 4 GB so unused buffers between inferences
    # are released promptly rather than retained indefinitely by the allocator.
    # This keeps headroom available for KV caches during long conversations.
    try:
        import mlx.core as _mx

        _mx.set_cache_limit(4 * 1024**3)
        logger.info("MLX Metal cache limit set to 4 GB")
    except Exception as exc:
        logger.warning("MLX set_cache_limit failed (non-fatal): %s", exc)

    model_paths = {ROUTER_MODEL, PRIMARY_MODEL, REASONING_MODEL}  # set → déduplique
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
                max_tokens=50,  # assez pour le JIT sans déclencher le warning de troncature
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
            think_kwargs: dict[str, Any] = {
                "enable_thinking": False,
                "thinking_budget": 0,
            }
        else:
            think_kwargs = {"enable_thinking": True}
            if THINKING_BUDGET_TOKENS > 0:
                think_kwargs["thinking_budget"] = THINKING_BUDGET_TOKENS

        # Try full kwargs; fall back if tokenizer version is too old
        try:
            return tokenizer.apply_chat_template(
                messages, **base_kwargs, **think_kwargs
            )
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
    profile = _model_profile(model_path)
    quant_kwargs = (
        {"kv_bits": QUANT_KV_BITS, "kv_group_size": 64} if profile.use_quant_kv else {}
    )
    effective_max = min(max_tokens, 10000)
    model_short = model_path.split("/")[-1]

    # System prompt KV cache — pre-filled once, deepcopied per call.
    _sys_content = (
        messages[0]["content"]
        if messages and messages[0].get("role") == "system"
        else ""
    )
    kv_cache = _get_system_cache(model_path, model, tokenizer, _sys_content)
    _cache_kwarg = {"prompt_cache": kv_cache} if kv_cache is not None else {}

    sampler = make_sampler(
        temp=temperature,
        top_p=profile.top_p_nothink if no_think else profile.top_p_think,
        top_k=profile.top_k,
        min_p=profile.min_p,
    )
    logits_procs = make_logits_processors(
        repetition_penalty=profile.repetition_penalty,
        repetition_context_size=profile.repetition_context_size,
        frequency_penalty=profile.frequency_penalty,
    )

    early_stopped = False
    _think_already_stripped = False

    if json_response:
        # Stream token-by-token and stop as soon as a complete JSON object is found.
        #
        # no_think=True  → model outputs JSON directly; scan from the start.
        # no_think=False → model outputs <think>…</think> THEN JSON; wait for
        #                  </think> before scanning so we don't stop inside the
        #                  think block on any JSON-like structure.
        #
        # This prevents wasted GPU cycles after the closing } AND avoids
        # returning a truncated JSON when the model stops mid-object.
        #
        # O(n) optimisation: accumulate via += instead of "".join(list) per token,
        # and only call _first_complete_json when the current chunk contains "}".
        raw_so_far = ""
        seen_end_think = no_think  # True immediately if no thinking expected

        for chunk in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=effective_max,
            sampler=sampler,
            logits_processors=logits_procs,
            **quant_kwargs,
            **_cache_kwarg,
        ):
            if chunk.text:
                raw_so_far += chunk.text

                if not seen_end_think and "</think>" in raw_so_far:
                    seen_end_think = True

                # Manual stop-token check (mlx_lm.generate/stream_generate do not
                # accept a `stop` kwarg — handle it in the loop instead).
                if profile.stop_tokens and any(t in raw_so_far for t in profile.stop_tokens):
                    early_stopped = True
                    break

                if seen_end_think and "}" in chunk.text:
                    # Only scan when a closing brace arrived — avoids O(n²) scan per token.
                    after_think = (
                        raw_so_far.split("</think>", 1)[-1]
                        if "</think>" in raw_so_far
                        else raw_so_far
                    )
                    if _first_complete_json(after_think) is not None:
                        early_stopped = True
                        break

        # Truncate at stop token if hit
        raw = raw_so_far
        if profile.stop_tokens:
            for st in profile.stop_tokens:
                if st in raw:
                    raw = raw.split(st, 1)[0]
        # For thinking mode: extract the JSON portion after </think>
        if "</think>" in raw:
            json_portion = raw.split("</think>", 1)[-1]
        else:
            json_portion = raw
        extracted = _first_complete_json(json_portion)
        if extracted is not None:
            # Complete JSON found — result is already clean, skip the stripping section.
            result = extracted
            early_stopped = True  # marks that we stopped at a clean boundary
            _think_already_stripped = True
        else:
            # Incomplete JSON — fall through to the stripping section with the raw text.
            result = raw
            _think_already_stripped = False
    else:
        result = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=effective_max,
            sampler=sampler,
            logits_processors=logits_procs,
            verbose=False,
            **quant_kwargs,
            **_cache_kwarg,
        )
        # Truncate at stop token (mlx_lm.generate does not accept a `stop` kwarg)
        if profile.stop_tokens:
            for st in profile.stop_tokens:
                if st in result:
                    result = result.split(st, 1)[0]

    # ── Debug log brut (avant stripping) ──────────────────────────
    _debug_log(model_short, no_think, prompt, result)

    # ── Stats réponse ──────────────────────────────────────────────
    resp_tokens = len(tokenizer.encode(result))
    thinking_active = "</think>" in result
    call_type = "json" if json_response else "text"
    stop_label = "early-stop" if early_stopped else "eos/limit"
    pct = resp_tokens * 100 // effective_max if effective_max else 0

    logger.info(
        "[LLM-STATS] %s | %s no_think=%s | %s | prompt=%d tok | resp=%d/%d tok (%d%%)",
        model_short,
        call_type,
        no_think,
        stop_label,
        prompt_tokens,
        resp_tokens,
        effective_max,
        pct,
    )
    if not early_stopped and resp_tokens >= int(effective_max * 0.9):
        logger.warning(
            "[LLM-STATS] POSSIBLE TRUNCATION — resp=%d tok near limit=%d (model=%s)",
            resp_tokens,
            effective_max,
            model_short,
        )

    # ── Strip thinking block, keep actual answer ──────────────────
    # split("</think>", 1)[-1] → keeps everything AFTER </think>
    # Truncated case: model hit token budget mid-reasoning → no </think>.
    # Discard from <think> onwards to avoid leaking raw reasoning.
    #
    # Skip when json_response=True already extracted a clean JSON object
    # (result contains no <think> tags and stripping would erroneously empty it).
    if _think_already_stripped:
        pass  # think block was handled inline; result is already clean
    elif "</think>" in result:
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
    Acquiert _infer_lock (threading.Lock) pour sérialiser avec les appelants async.
    """
    with _infer_lock:
        return _generate_sync(
            model,
            messages,
            temperature,
            max_tokens,
            no_think,
            session_id,
            json_response,
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
    # Acquiert threading.Lock depuis un thread pool pour ne pas bloquer l'event loop.
    await asyncio.to_thread(_infer_lock.acquire)
    # ====DEBUG====
    _t_infer_start = _time.time()
    logger.debug(
        "[TTFT] call_llm_local_async: lock acquired — waited %.3fs",
        _t_infer_start - _t_lock_wait,
    )
    # ====DEBUG====
    try:
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
    finally:
        _infer_lock.release()


async def stream_local(
    messages: list[dict],
    model: str,
    temperature: float | None = None,  # None → utilise temp_think/nothink du profil
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

        # System prompt KV cache — pre-filled once, deepcopied per call.
        # Only the tokens after the system prompt need prefill on each turn.
        _sys_content = (
            messages[0]["content"]
            if messages and messages[0].get("role") == "system"
            else ""
        )
        kv_cache = _get_system_cache(model, mlx_model, tokenizer, _sys_content)

        first = True
        raw_chunks: list[str] = []  # accumule la réponse brute pour stats
        # Budget vient du pipeline : 1500 (no_think) ou 4000 (reasoning).
        # Hard cap à 10000 pour éviter les runaway en cas de mauvais passage.
        budget = min(max_tokens, 10000)
        profile = _model_profile(model)
        # Température : utilise la valeur du profil par défaut (think vs no-think),
        # ou la valeur explicite si le caller en a passé une.
        effective_temp = (
            temperature
            if temperature is not None
            else (profile.temp_nothink if no_think else profile.temp_think)
        )
        quant_kwargs = (
            {"kv_bits": QUANT_KV_BITS, "kv_group_size": 64}
            if profile.use_quant_kv
            else {}
        )
        try:
            for chunk in stream_generate(
                mlx_model,
                tokenizer,
                prompt=prompt,
                max_tokens=budget,
                sampler=make_sampler(
                    temp=effective_temp,
                    top_p=profile.top_p_nothink if no_think else profile.top_p_think,
                    top_k=profile.top_k,
                    min_p=profile.min_p,
                ),
                logits_processors=make_logits_processors(
                    repetition_penalty=profile.repetition_penalty,
                    repetition_context_size=profile.repetition_context_size,
                    frequency_penalty=profile.frequency_penalty,
                ),
                **({"prompt_cache": kv_cache} if kv_cache is not None else {}),
                **quant_kwargs,
            ):
                if chunk.text:
                    text = chunk.text
                    stop_hit = False

                    # Manual stop-token check for models that need it (e.g. Hermes).
                    if profile.stop_tokens:
                        acc = "".join(raw_chunks) + text
                        for st in profile.stop_tokens:
                            if st in acc:
                                text = acc.split(st, 1)[0][len("".join(raw_chunks)):]
                                stop_hit = True
                                break

                    raw_chunks.append(text)

                    if first and text:
                        logger.debug(
                            "[TTFT] stream_local: first token generated — %.3fs since inference start",
                            _time.time() - _t_infer,
                        )
                        first = False

                    if text:
                        loop.call_soon_threadsafe(queue.put_nowait, text)

                    if stop_hit:
                        break

        except Exception as exc:
            logger.error("stream_local erreur : %s", exc)
        finally:
            # ── Stats réponse ──────────────────────────────────────
            raw_resp = "".join(raw_chunks)
            _debug_log(model_short, no_think, prompt, raw_resp)
            resp_tokens = len(tokenizer.encode(raw_resp))
            thinking_active = "</think>" in raw_resp
            pct = resp_tokens * 100 // budget if budget else 0
            logger.info(
                "[LLM-STATS] %s | stream no_think=%s thinking=%s | eos/limit"
                " | prompt=%d tok | resp=%d/%d tok (%d%%)",
                model_short,
                no_think,
                thinking_active,
                prompt_tokens,
                resp_tokens,
                budget,
                pct,
            )
            if resp_tokens >= int(budget * 0.9):
                logger.warning(
                    "[LLM-STATS] POSSIBLE TRUNCATION — resp=%d tok near limit=%d (model=%s)",
                    resp_tokens,
                    budget,
                    model_short,
                )
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinelle fin de flux

    # Acquiert threading.Lock depuis un thread pool pour ne pas bloquer l'event loop.
    await asyncio.to_thread(_infer_lock.acquire)
    # ====DEBUG====
    logger.debug(
        "[TTFT] stream_local: lock acquired — waited %.3fs",
        _time.time() - _t_lock_wait,
    )
    # ====DEBUG====
    try:
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
    finally:
        _infer_lock.release()

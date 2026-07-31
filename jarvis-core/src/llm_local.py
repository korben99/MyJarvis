"""
llm_localLRU.py — MLX inference with LRUPromptCache (drop-in for llm_local.py).

Rollback: mv llm_local.py llm_local.py.old && mv llm_localLRU.py llm_local.py

Changes vs llm_local.py
-----------------------
• _sys_kv / _sys_kv_lock  →  _lru_caches (one LRUPromptCache per model-path)
• Each call: fetch_nearest_cache() returns best cached prefix → only *remaining*
  tokens are sent to stream_generate (system prefix is never re-processed).
• After generation: insert(prompt_tokens + output_tokens) so future turns share
  the accumulated context prefix, not just the system prompt.
• Session benefit: consecutive turns reuse everything up to where their token
  sequences diverge (system + prior user/assistant turns).

Config env vars (new)
---------------------
LRU_KV_SIZE=4    max cached sequences per model (default 4)
LRU_KV_GB=4.0    total RAM budget for all LRU caches in GB (default 4.0)
"""

import asyncio
import concurrent.futures
import copy
import datetime
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, List, Optional, Union

from config import (
    LLM_LOCAL,
    MAX_TOKENS_HARD_CAP,
    PRIMARY_MODEL,
    QWEN36_NINJA_TEMPLATE,
    REASONING_MODEL,
    ROUTER_MODEL,
    USE_THINKING_BUDGET_PROCESSOR,
    VISION_MODEL,
    is_hermes,
    is_qwen,
    is_qwen25,
    is_qwen3,
    is_qwen36,
)
from prompts import VISION_USER_PROMPT, get_prompt
import mlx.core as mx
from mlx_lm import generate, load, stream_generate
from mlx_lm.models.cache import (
    ArraysCache,
    LRUPromptCache,
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)
from mlx_lm.sample_utils import make_logits_processors, make_sampler

# ── ArraysCache trim patch — REMOVED ─────────────────────────────────────
# Qwen3.6 uses a hybrid architecture: full_attention → KVCache (trimmable),
# linear_attention → ArraysCache (recurrent state, NOT trimmable).
#
# A previous patch made ArraysCache.trim() a no-op to enable multi-turn LRU hits.
# Root-cause bug: on a partial LRU hit covering a previous full turn, KVCache was
# correctly trimmed to the matching prefix but ArraysCache kept the full previous
# generation state (think block + response + <|im_end|>).  This inconsistency caused
# the model to generate EOS immediately after a forced </think>, producing empty
# responses.
#
# Without the patch, LRUPromptCache.fetch_nearest_cache skips partial-hit entries
# that require trimming (since ArraysCache is not trimmable) and falls back to the
# system-only prefix.
#
# In thinking mode that fallback still applies, and it is correct: the raw-output LRU
# key (including <think>) never matches the next turn's prompt (which uses the clean
# response), so a multi-turn hit would always need a large trim.
#
# In no_think mode, _build_prompt passes preserve_thinking=True so the template renders
# past assistant turns exactly as they were generated (empty think block included).  The
# stored key is then a true *prefix* of the next turn's prompt, which fetch_nearest_cache
# serves without any trim — so multi-turn reuse works despite the hybrid cache.
# Measured on the 6-turn scripts/test_lru_cache.py conversation (Qwen3.6, --no-think):
# tokens reused per turn 46/46/46/46/46/46 → 46/154/265/444/623/833 (turn 6: 5% → 96%).

HF_HOME = os.getenv("HF_HOME", "/opt/jarvis/models")
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HUB_CACHE"]

logger = logging.getLogger("jarvis-llm-local")

LLM_DEBUG_PROMPTS = os.getenv("LLM_DEBUG_PROMPTS", "").lower() in ("yes", "true", "1")
_PROMPTS_LOG_PATH = "/opt/jarvis/logs/prompts.log"

QUANT_KV = os.getenv("QUANT_KV", "").lower() in ("yes", "true", "1")
QUANT_KV_BITS = int(os.getenv("QUANT_KV_BITS", "4"))

LRU_KV_MAX_SIZE = int(os.getenv("LRU_KV_SIZE", "4"))
LRU_KV_MAX_BYTES = int(float(os.getenv("LRU_KV_GB", "4.0")) * 1024**3)


# ── Debug logging ─────────────────────────────────────────────────────────

def _debug_log(model_short: str, no_think: bool, prompt: str, raw_output: str, skip: bool = False) -> None:
    if not LLM_DEBUG_PROMPTS or skip:
        return
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "=" * 80
    entry = (
        f"\n{sep}\n[{ts}] model={model_short}  no_think={no_think}\n"
        f"--- PROMPT ---\n{prompt}\n--- RESPONSE (raw) ---\n{raw_output}\n{sep}\n"
    )
    try:
        with open(_PROMPTS_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(entry)
    except Exception as exc:
        logger.warning("_debug_log: cannot write prompts.log: %s", exc)


# ── Model profiles ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _ModelProfile:
    temp_think: float
    temp_nothink: float
    top_p_think: float
    top_p_nothink: float
    top_k: int
    min_p: float
    repetition_penalty: float
    repetition_context_size: int
    frequency_penalty: float
    presence_penalty: float
    use_quant_kv: bool
    stop_tokens: tuple[str, ...]


def _model_profile(model_path: str) -> _ModelProfile:
    """Return sampling profile. Qwen3.6 checked before Qwen3 (is_qwen36 ⊂ is_qwen3)."""
    if is_qwen36(model_path):
        return _ModelProfile(
            temp_think=1.0, temp_nothink=0.7,
            top_p_think=0.95, top_p_nothink=0.80,
            top_k=20, min_p=0.0,
            repetition_penalty=1.0, repetition_context_size=256,
            frequency_penalty=0.15, presence_penalty=1.5,
            use_quant_kv=QUANT_KV, stop_tokens=(),
        )
    if is_qwen25(model_path):
        return _ModelProfile(
            temp_think=0.0, temp_nothink=0.0,
            top_p_think=1.0, top_p_nothink=1.0,
            top_k=0, min_p=0.0,
            repetition_penalty=1.0, repetition_context_size=64,
            frequency_penalty=0.0, presence_penalty=0.0,
            use_quant_kv=False, stop_tokens=(),
        )
    if is_qwen3(model_path):
        return _ModelProfile(
            temp_think=0.6, temp_nothink=0.7,
            top_p_think=0.95, top_p_nothink=0.80,
            top_k=20, min_p=0.0,
            repetition_penalty=1.1, repetition_context_size=64,
            frequency_penalty=0.05, presence_penalty=1.5,
            use_quant_kv=QUANT_KV, stop_tokens=(),
        )
    if is_hermes(model_path):
        return _ModelProfile(
            temp_think=0.0, temp_nothink=0.0,
            top_p_think=1.0, top_p_nothink=1.0,
            top_k=0, min_p=0.0,
            repetition_penalty=1.0, repetition_context_size=64,
            frequency_penalty=0.0, presence_penalty=0.0,
            use_quant_kv=False, stop_tokens=("<|im_end|>",),
        )
    return _ModelProfile(
        temp_think=0.7, temp_nothink=0.7,
        top_p_think=0.90, top_p_nothink=0.90,
        top_k=0, min_p=0.0,
        repetition_penalty=1.3, repetition_context_size=64,
        frequency_penalty=0.10, presence_penalty=0.0,
        use_quant_kv=False, stop_tokens=(),
    )


# ── Model registry ────────────────────────────────────────────────────────

_model_cache: dict[str, tuple] = {}
_load_lock = threading.Lock()
_infer_lock = threading.Lock()

_chat_waiters: int = 0
_chat_waiters_lock = threading.Lock()

_bg_wakeup: asyncio.Event | None = None
_bg_loop: asyncio.AbstractEventLoop | None = None  # loop that owns _bg_wakeup


def _wake_bg_waiters() -> None:
    """Wake background-priority waiters from any thread.

    asyncio.Event is not thread-safe: setting it from a worker thread must go
    through call_soon_threadsafe on the loop that owns the event."""
    if _bg_wakeup is None or _bg_loop is None:
        return
    try:
        _bg_loop.call_soon_threadsafe(_bg_wakeup.set)
    except RuntimeError:
        pass  # loop closed (shutdown)

# LRU prompt cache: one LRUPromptCache per model-path.
# Replaces the old _sys_kv dict (single system-prompt entry per model).
# All fetch/insert operations happen under _infer_lock (serialized by GPU lock).
# _lru_lock only protects creation of new LRUPromptCache instances.
_lru_caches: dict[str, LRUPromptCache] = {}
_lru_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────

def _norm_think_close(s: str) -> str:
    """Normalise Qwen3.6 hallucination: </think > → </think>."""
    return s.replace("</think >", "</think>")


def _first_complete_json(text: str) -> Optional[str]:
    """Return the first valid complete JSON object in text, or None."""
    start = -1
    stack: list[str] = []
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
                start = i
                stack.append(c)
            continue

        if c in "{[":
            stack.append(c)
        elif c in "}]":
            if not stack:
                continue
            last = stack.pop()
            if (last == "{" and c != "}") or (last == "[" and c != "]"):
                start = -1
                stack.clear()
                continue
            if not stack:
                candidate = text[start : i + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    start = -1
    return None


def _strip_thinking(text: str) -> str:
    """Strip legacy reasoning markers that occasionally leak through."""
    if "Final Answer:" in text:
        return text.split("Final Answer:")[-1].strip()
    if "Thinking Process:" in text:
        return text.split("Thinking Process:")[-1].strip()
    return text


def _log_stats(
    model_short: str, call_type: str, no_think: bool, stop_label: str,
    prompt_tokens: int, resp_tokens: int, effective_max: int,
) -> None:
    pct = resp_tokens * 100 // effective_max if effective_max else 0
    logger.info(
        "[LLM-STATS] %s | %s no_think=%s | %s | prompt=%d tok | resp=%d/%d tok (%d%%)",
        model_short, call_type, no_think, stop_label, prompt_tokens, resp_tokens, effective_max, pct,
    )
    if stop_label != "early-stop" and resp_tokens >= int(effective_max * 0.9):
        logger.warning(
            "[LLM-STATS] POSSIBLE TRUNCATION — resp=%d tok near limit=%d (model=%s)",
            resp_tokens, effective_max, model_short,
        )


def _prompt_token_ids(prompt_text: str, tokenizer) -> list[int]:
    """Tokenize a prompt string the same way stream_generate does internally."""
    bos = getattr(tokenizer, "bos_token", None)
    add_special = bos is None or not prompt_text.startswith(bos)
    return tokenizer.encode(prompt_text, add_special_tokens=add_special)


# ── LRU cache helpers ─────────────────────────────────────────────────────

def _get_lru(model_path: str) -> LRUPromptCache:
    """Get or create the LRU cache instance for a model path."""
    with _lru_lock:
        if model_path not in _lru_caches:
            _lru_caches[model_path] = LRUPromptCache(
                max_size=LRU_KV_MAX_SIZE, max_bytes=LRU_KV_MAX_BYTES
            )
            logger.info(
                "LRU: created cache for %s (max_size=%d, max_bytes=%.1f GB)",
                model_path.split("/")[-1], LRU_KV_MAX_SIZE, LRU_KV_MAX_BYTES / 1024**3,
            )
        return _lru_caches[model_path]


def _eval_kv_cache(cache) -> None:
    """Materialize all MLX arrays in a KV cache into device-wide Metal buffers.
    Must be called before storing a cache built in one thread for use in another —
    unevaluated (lazy) arrays hold a reference to the creating thread's Metal stream,
    which is destroyed when that thread exits, causing 'There is no Stream(gpu, N)'."""
    to_eval = []
    for layer in cache:
        for attr in ("keys", "values"):
            v = getattr(layer, attr, None)
            if v is not None:
                if isinstance(v, list):
                    to_eval.extend(x for x in v if x is not None)
                else:
                    to_eval.append(v)
        state = getattr(layer, "state", None)
        if state:
            to_eval.extend(x for x in state if x is not None)
    if to_eval:
        mx.eval(*to_eval)


def _system_prefix_text(model_path: str, tokenizer, system_content: str) -> str | None:
    """Render the system-only head of the prompt, or None if it can't be done safely.

    The result MUST be a textual prefix of what _build_prompt produces for the same
    system message: _lru_get_cache slices the real prompt at len(sys_token_ids) and
    feeds only the tail to the model, so a prefix mismatch silently misaligns the KV
    cache with the prompt (garbled answers, no exception raised).

    Recent Qwen ChatML templates (3.5, 3.6, …) raise 'No user query found in messages.'
    on a system-only message list — they scan the messages backwards for a user turn.
    For those, the ChatML system block is emitted literally (byte-identical to what the
    template emits for that block, `|trim` included).  Padding the message list with an
    empty user turn instead would satisfy the template but append a full
    `<|im_start|>user\\n<|im_end|>\\n` turn that the real prompt does not contain,
    breaking the prefix invariant — hence the explicit render-and-verify below.

    Returning None disables system-KV caching for the call: slower, still correct.
    """
    model_short = model_path.split("/")[-1]
    candidate: str | None = None
    try:
        candidate = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_content}],
            tokenize=False, add_generation_prompt=False,
        )
    except Exception as exc:
        if not is_qwen(model_path):
            logger.warning(
                "System prefix: template rejects a system-only message list (%s: %s) "
                "— system KV cache disabled", model_short, exc,
            )
            return None
        candidate = f"<|im_start|>system\n{system_content.strip()}<|im_end|>\n"
        logger.debug(
            "System prefix: template requires a user turn (%s) — ChatML block used", model_short
        )

    # Verify the invariant against a real two-message render. Cheap (one jinja pass,
    # only on an LRU miss) and it also catches whitespace-level template quirks.
    try:
        probe = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_content}, {"role": "user", "content": "?"}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception as exc:
        logger.warning(
            "System prefix: probe render failed (%s: %s) — system KV cache disabled",
            model_short, exc,
        )
        return None
    if not probe.startswith(candidate):
        logger.warning(
            "System prefix is not a prefix of the rendered prompt (%s) — system KV cache "
            "disabled. prefix tail=%r prompt head=%r",
            model_short, candidate[-40:], probe[:len(candidate) + 20],
        )
        return None
    return candidate


def _make_system_kv(
    model_path: str, model, tokenizer, system_content: str
) -> tuple[Any, list[int]] | None:
    """Pre-fill system prompt into a KV cache.
    Returns (cache, sys_token_ids) or None on failure."""
    sys_prompt_text = _system_prefix_text(model_path, tokenizer, system_content)
    if sys_prompt_text is None:
        return None
    sys_token_ids = _prompt_token_ids(sys_prompt_text, tokenizer)
    sys_token_count = len(sys_token_ids)
    profile = _model_profile(model_path)
    quant_kwargs = (
        {"kv_bits": QUANT_KV_BITS, "kv_group_size": 64, "quantized_kv_start": 256}
        if profile.use_quant_kv else {}
    )
    cache = make_prompt_cache(model)
    try:
        for _ in stream_generate(
            model, tokenizer, prompt=sys_prompt_text, max_tokens=1,
            sampler=make_sampler(temp=0.0, top_p=1.0, top_k=0, min_p=0.0),
            prompt_cache=cache, **quant_kwargs,
        ):
            break
        for layer_cache in cache:
            if hasattr(layer_cache, "offset"):
                layer_cache.offset = sys_token_count
    except Exception as exc:
        logger.warning("System KV cache build failed: %s", exc)
        return None
    _eval_kv_cache(cache)
    logger.info(
        "LRU: system prompt prefilled (%d tok, model=%s)",
        sys_token_count, model_path.split("/")[-1],
    )
    return cache, sys_token_ids


def _lru_get_cache(
    model_path: str,
    model,
    tokenizer,
    sys_content: str,
    prompt_token_ids: list[int],
) -> tuple[Any | None, list[int]]:
    """
    Fetch the best cached prefix for this prompt from the LRU.
    Returns (cache, remaining_tokens).  Caller passes remaining_tokens to stream_generate.
    Must be called while holding _infer_lock.

    LRU hit  → deepcopy of cached KV, remaining = tokens not yet in cache.
    LRU miss → fresh system KV, system entry stored in LRU, remaining = prompt[K_sys:].
    No sys   → (None, prompt_token_ids).
    """
    if not sys_content:
        return None, prompt_token_ids

    lru = _get_lru(model_path)
    # Use model_path (str) as the trie key — MLX Model objects are not hashable.
    # One LRU per model_path, so model_path is a correct unique identifier.
    cache, remaining = lru.fetch_nearest_cache(model_path, prompt_token_ids)

    if cache is not None:
        cached_len = len(prompt_token_ids) - len(remaining)
        logger.debug(
            "LRU hit: %d/%d prompt tok cached (remaining=%d) model=%s",
            cached_len, len(prompt_token_ids), len(remaining),
            model_path.split("/")[-1],
        )
        if not remaining:
            # Exact match: cache covers all prompt tokens.
            # stream_generate needs ≥1 token — trim one and replay the last prompt token.
            if can_trim_prompt_cache(cache):
                trim_prompt_cache(cache, 1)
                remaining = [prompt_token_ids[-1]]
                logger.debug("LRU exact hit — trimmed 1 tok to allow decode")
            else:
                logger.warning(
                    "LRU exact hit but cache not trimmable (model=%s) — falling back",
                    model_path.split("/")[-1],
                )
                return None, prompt_token_ids
        _eval_kv_cache(cache)
        return cache, remaining

    # LRU miss: build fresh system KV cache
    result = _make_system_kv(model_path, model, tokenizer, sys_content)
    if result is None:
        return None, prompt_token_ids
    sys_cache, sys_token_ids = result

    remaining = prompt_token_ids[len(sys_token_ids):]
    if not remaining:
        # Prompt is exactly the system prompt — the cache already covers every
        # prompt token; reusing it while re-feeding the full prompt would process
        # the system tokens twice (misaligned positions). Store the entry for
        # future calls and bypass the cache for this one.
        lru.insert_cache(model_path, sys_token_ids, sys_cache, cache_type="system")
        logger.warning("LRU: prompt has no tokens beyond system prompt — cache bypassed")
        return None, prompt_token_ids

    # Store system entry in LRU so it can seed the next call's prefix lookup.
    # This entry is superseded by the fuller assistant entry inserted post-generation.
    lru.insert_cache(model_path, sys_token_ids, copy.deepcopy(sys_cache), cache_type="system")

    logger.debug(
        "LRU miss: system KV built (%d tok), remaining=%d tok, model=%s",
        len(sys_token_ids), len(remaining), model_path.split("/")[-1],
    )
    return sys_cache, remaining


def _metal_mem_str() -> str:
    """Return a short Metal memory status string, or empty string if unavailable."""
    try:
        active_mb = mx.get_active_memory() / 1024**2
        cache_mb = mx.get_cache_memory() / 1024**2
        return f"Metal active={active_mb:.0f} MB cache={cache_mb:.0f} MB"
    except Exception:
        return ""


def _lru_insert(
    model_path: str,
    model,
    prompt_token_ids: list[int],
    raw_output: str,
    cache,
    tokenizer,
) -> None:
    """Insert the completed (prompt + output) sequence into the LRU. Non-fatal on error."""
    try:
        out_ids = tokenizer.encode(raw_output, add_special_tokens=False)
        if not out_ids:
            return
        lru = _get_lru(model_path)
        _eval_kv_cache(cache)
        lru.insert_cache(model_path, prompt_token_ids + out_ids, cache, cache_type="assistant")
        used_gb = lru.nbytes / 1024**3
        metal = _metal_mem_str()
        logger.debug(
            "LRU insert: key=%d tok (prompt=%d + output=%d) model=%s — %d/%d slots | %.2f/%.1f GB (lru) | %s",
            len(prompt_token_ids) + len(out_ids),
            len(prompt_token_ids), len(out_ids),
            model_path.split("/")[-1],
            len(lru), lru.max_size,
            used_gb, lru.max_bytes / 1024**3,
            metal,
        )
    except Exception as exc:
        logger.warning("LRU insert failed (%s) — non-fatal", exc)


def get_lru_stats() -> dict:
    """Return current LRU memory usage for all loaded models. Safe to call at any time."""
    result = {}
    with _lru_lock:
        for model_path, lru in _lru_caches.items():
            by_type = lru.stats_by_type()
            result[model_path.split("/")[-1]] = {
                "slots_used": len(lru),
                "slots_max": lru.max_size,
                "bytes_used": lru.nbytes,
                "bytes_max": lru.max_bytes,
                "gb_used": round(lru.nbytes / 1024**3, 3),
                "gb_max": round(lru.max_bytes / 1024**3, 1),
                "pct_bytes": round(lru.nbytes * 100 / lru.max_bytes, 1) if lru.max_bytes else 0,
                "by_type": {
                    t: {
                        "slots": v["n_sequences"],
                        "gb": round(v["n_bytes"] / 1024**3, 3),
                    }
                    for t, v in by_type.items()
                },
            }
    return result


# ── Model loading ─────────────────────────────────────────────────────────

def _load_model(model_path: str) -> tuple:
    """Load and cache a model (double-checked locking)."""
    if model_path in _model_cache:
        return _model_cache[model_path]
    with _load_lock:
        if model_path in _model_cache:
            return _model_cache[model_path]
        logger.info("Loading MLX model: %s", model_path)
        model, tokenizer = load(model_path)
        if is_qwen36(model_path):
            if os.path.isfile(QWEN36_NINJA_TEMPLATE):
                with open(QWEN36_NINJA_TEMPLATE, encoding="utf-8") as f:
                    tokenizer.chat_template = f.read()
                logger.info("Ninja-patch template applied: %s", model_path.split("/")[-1])
            else:
                logger.warning(
                    "Ninja-patch template not found (%s) — using default. "
                    "Run scripts/download_models.py to fetch it.",
                    QWEN36_NINJA_TEMPLATE,
                )
        _model_cache[model_path] = (model, tokenizer)
        logger.info("Model ready: %s", model_path)
        return model, tokenizer


def preload_models(primary_system_content: str = "") -> None:
    """Load and warm-up models at startup. Call from main.py lifespan."""
    if not LLM_LOCAL:
        return
    try:
        import mlx.core as _mx
        _mx.set_cache_limit(4 * 1024**3)
        try:
            info = _mx.device_info()
            max_ws = info.get("max_recommended_working_set_size", 0)
            if max_ws > 0:
                wired = int(max_ws * 0.70)
                _mx.set_wired_limit(wired)
                logger.info("MLX: cache_limit=4 GB, wired_limit=%.0f GB", wired / 1024**3)
            else:
                logger.info("MLX: cache_limit=4 GB (wired_limit skipped — device_info unavailable)")
        except Exception as exc:
            logger.info("MLX: cache_limit=4 GB (wired_limit skipped: %s)", exc)
    except Exception as exc:
        logger.warning("MLX memory limits: %s (non-fatal)", exc)

    model_paths = {ROUTER_MODEL, PRIMARY_MODEL, REASONING_MODEL}
    for path in model_paths:
        _load_model(path)
    logger.info("MLX: %d model(s) preloaded", len(model_paths))

    if VISION_MODEL:
        try:
            _vlm_executor.submit(_load_vlm).result(timeout=300)
            logger.info("MLX VLM preloaded: %s", VISION_MODEL)
            _vlm_executor.submit(_warmup_vlm).result(timeout=120)
        except Exception as exc:
            logger.warning("MLX VLM preload/warmup failed (non-fatal): %s", exc)

    for path in model_paths:
        if path == PRIMARY_MODEL and primary_system_content:
            sys_content = primary_system_content
        elif path == ROUTER_MODEL:
            sys_content = get_prompt("ROUTER_SYSTEM")
        else:
            sys_content = "Tu es un assistant."
        warmup_msgs = [{"role": "system", "content": sys_content}, {"role": "user", "content": "Salut"}]
        try:
            _generate_sync(path, warmup_msgs, temperature=None, max_tokens=100, no_think=True)
            logger.info("MLX warmup OK: %s", path)
        except Exception as exc:
            logger.warning("MLX warmup failed for %s: %s", path, exc)


# ── Prompt building ───────────────────────────────────────────────────────

def _build_prompt(
    messages: list[dict],
    tokenizer,
    model_path: str,
    no_think: bool,
    thinking_budget: int = 0,
) -> str:
    """
    Build the final prompt via apply_chat_template.

    For Qwen3.x, passes enable_thinking to control the <think> block, with two
    fallback retries if the tokenizer doesn't support newer kwargs.

    A string-level guard is applied after templating to enforce the correct think
    state regardless of template variant:
      no_think=True  → must end with </think>\\n\\n (closed empty block)
      no_think=False → must end with open <think>\\n

    Note: Qwen3.6 does not honour <budget_remaining> tags — it was not trained with
    Qwen3's budget-forcing mechanism. thinking_budget is therefore ignored for Qwen3.6
    at the template level; ThinkingBudgetProcessor handles the cap at logit level.
    """
    base_kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}

    if not is_qwen3(model_path):
        return tokenizer.apply_chat_template(messages, **base_kwargs)

    if no_think:
        # preserve_thinking makes the template render *past* assistant turns with their
        # (empty) think block, matching the `<think>\n\n</think>\n\n` scaffolding appended
        # below at generation time. Without it the two renderings differ by 4 tokens right
        # after `<|im_start|>assistant\n`, so every stored LRU entry stops being a prefix
        # of the next turn's prompt and needs a trim the hybrid Qwen3.6 cache cannot do
        # (ArraysCache is not trimmable) — multi-turn reuse collapses to the system prefix.
        # Only correct under no_think: in thinking mode the KV holds the real reasoning
        # text while history stores the stripped response, so alignment is impossible.
        # No-op for templates without the flag (Qwen2.5, Hermes, Qwen3-VL: verified).
        think_kwargs: dict[str, Any] = {
            "enable_thinking": False,
            "thinking_budget": 0,
            "preserve_thinking": True,
        }
    elif thinking_budget > 0 and not is_qwen36(model_path):
        think_kwargs = {"enable_thinking": True, "thinking_budget": thinking_budget}
    else:
        think_kwargs = {"enable_thinking": True}

    try:
        prompt = tokenizer.apply_chat_template(messages, **base_kwargs, **think_kwargs)
    except TypeError:
        try:
            prompt = tokenizer.apply_chat_template(messages, **base_kwargs, enable_thinking=not no_think)
        except TypeError:
            logger.debug("enable_thinking not supported for %s — default template", model_path.split("/")[-1])
            prompt = tokenizer.apply_chat_template(messages, **base_kwargs)

    if no_think:
        if prompt.endswith("<think>\n") or prompt.rstrip().endswith("<think>"):
            prompt = prompt.rstrip("\n") + "\n</think>\n\n"
        elif "</think>" not in _norm_think_close(prompt[-30:]):
            prompt = prompt.rstrip("\n") + "\n<think>\n\n</think>\n\n"
    else:
        _tail = _norm_think_close(prompt[-100:])
        if not ("<think>" in _tail and "</think>" not in _tail[_tail.rfind("<think>"):]):
            prompt = prompt.rstrip("\n") + "\n<think>\n"

    if thinking_budget > 0 and not no_think and not is_qwen36(model_path):
        if "<budget_remaining>" not in prompt[-120:]:
            logger.warning(
                "_build_prompt thinking_budget=%d → <budget_remaining> NOT injected "
                "(ninja-patch or DWQ checkpoint); budget hint lost. prompt tail: %r",
                thinking_budget, prompt[-120:],
            )
    return prompt


# ── ThinkingBudgetProcessor ───────────────────────────────────────────────

class ThinkingBudgetProcessor:
    """
    MLX logits processor: forces </think> after `budget_tokens` thinking tokens.

    Soft phase (last 10% of budget): progressively boost </think> logit.
    Hard phase (at budget): all logits → -1e9, </think> → +100.
    The .reshape(logits.shape) on returned tensors preserves the batch dim
    expected by downstream processors (e.g. repetition_penalty uses logits[:, tokens]).
    """

    def __init__(self, tokenizer, budget_tokens: int):
        import mlx.core as mx
        self.budget = budget_tokens
        self._mx = mx
        self._initialized = False
        self._thinking_count = 0
        self._thinking_done = False
        self._end_think_id: int | None = None
        self._newline_id: int | None = None
        self._force_close = False  # True after \n forced — next step forces </think>

        if budget_tokens <= 0:
            return
        try:
            vocab = tokenizer.get_vocab()
            end_id = vocab.get("</think>")
            if end_id is not None:
                self._end_think_id = int(end_id)
            else:
                ids = tokenizer.encode("</think>", add_special_tokens=False)
                if len(ids) == 1:
                    self._end_think_id = ids[0]
                else:
                    logger.warning("ThinkingBudgetProcessor: </think> is %d tokens — disabled", len(ids))
            nl_ids = tokenizer.encode("\n", add_special_tokens=False)
            if len(nl_ids) == 1:
                self._newline_id = nl_ids[0]
        except Exception as exc:
            logger.warning("ThinkingBudgetProcessor init: %s", exc)

    def _force_token(self, token_id: int, logits):
        mx = self._mx
        n = logits.shape[-1]
        left = mx.full([token_id], -1e9, dtype=logits.dtype)
        mid = mx.array([100.0], dtype=logits.dtype)
        right = mx.full([n - token_id - 1], -1e9, dtype=logits.dtype)
        return mx.concatenate([left, mid, right]).reshape(logits.shape)

    def __call__(self, tokens, logits):
        mx = self._mx
        if self._thinking_done or self.budget <= 0 or self._end_think_id is None:
            return logits
        if not self._initialized:
            self._initialized = True
            return logits

        last_token = int(tokens[-1])
        if last_token == self._end_think_id:
            self._thinking_done = True
            return logits

        # Step 2: \n was just forced — now force </think>
        if self._force_close:
            self._force_close = False
            logger.debug("ThinkingBudgetProcessor: forcing </think>")
            return self._force_token(self._end_think_id, logits)

        self._thinking_count += 1
        n = logits.shape[-1]
        idx = self._end_think_id

        if self._thinking_count >= self.budget:
            # Step 1: force \n so </think> lands on its own line (matches training distribution).
            if self._newline_id is not None:
                self._force_close = True
                logger.debug("ThinkingBudgetProcessor: budget=%d — forcing \\n</think>", self._thinking_count)
                return self._force_token(self._newline_id, logits)
            logger.debug("ThinkingBudgetProcessor: budget=%d reached — forcing </think>", self._thinking_count)
            return self._force_token(self._end_think_id, logits)

        soft_start = max(0, self.budget - max(1, self.budget // 10))
        if self._thinking_count >= soft_start:
            progress = (self._thinking_count - soft_start) / max(self.budget - soft_start, 1)
            boost = progress * 15.0
            left = mx.zeros([idx], dtype=logits.dtype)
            mid = mx.array([boost], dtype=logits.dtype)
            right = mx.zeros([n - idx - 1], dtype=logits.dtype)
            return logits + mx.concatenate([left, mid, right]).reshape(logits.shape)

        return logits


def _make_thinking_budget_proc(tokenizer, no_think: bool, thinking_budget: int):
    if USE_THINKING_BUDGET_PROCESSOR and not no_think and thinking_budget > 0:
        return ThinkingBudgetProcessor(tokenizer, thinking_budget)
    return None


# ── Inference setup (shared between _generate_sync and stream_local) ──────

def _setup_gen(
    profile: _ModelProfile,
    tokenizer,
    no_think: bool,
    thinking_budget: int,
    temperature: float | None,
    max_tokens: int,
) -> tuple:
    """Build (sampler, logits_procs, quant_kwargs, effective_max)."""
    effective_max = min(max_tokens, MAX_TOKENS_HARD_CAP)
    quant_kwargs = (
        {"kv_bits": QUANT_KV_BITS, "kv_group_size": 64, "quantized_kv_start": 256}
        if profile.use_quant_kv else {}
    )
    effective_temp = (
        temperature if temperature is not None
        else (profile.temp_nothink if no_think else profile.temp_think)
    )
    sampler = make_sampler(
        temp=effective_temp,
        top_p=profile.top_p_nothink if no_think else profile.top_p_think,
        top_k=profile.top_k,
        min_p=profile.min_p,
    )
    procs = list(make_logits_processors(
        repetition_penalty=profile.repetition_penalty,
        repetition_context_size=profile.repetition_context_size,
        frequency_penalty=profile.frequency_penalty,
        frequency_context_size=profile.repetition_context_size,
        presence_penalty=profile.presence_penalty,
        presence_context_size=profile.repetition_context_size,
    ))
    budget_proc = _make_thinking_budget_proc(tokenizer, no_think, thinking_budget)
    if budget_proc is not None:
        procs.insert(0, budget_proc)
    return sampler, procs, quant_kwargs, effective_max


def _stream_to_json(
    model,
    tokenizer,
    prompt: Union[str, List[int]],
    effective_max: int,
    sampler,
    logits_procs: list,
    quant_kwargs: dict,
    cache_kwarg: dict,
    no_think: bool,
    stop_tokens: tuple[str, ...],
) -> tuple[str, bool, bool]:
    """
    Stream-generate with early-stop on the first complete JSON object.
    Returns (raw_normalized, seen_end_think, early_stopped).
    prompt accepts str or List[int] (token ids for LRU remaining-token path).
    """
    raw_so_far = ""
    seen_end_think = no_think
    early_stopped = False
    max_stop_len = max((len(t) for t in stop_tokens), default=0)

    for chunk in stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=effective_max,
        sampler=sampler, logits_processors=logits_procs,
        **quant_kwargs, **cache_kwarg,
    ):
        if not chunk.text:
            continue
        raw_so_far += chunk.text

        if not seen_end_think and "</think>" in _norm_think_close(raw_so_far):
            seen_end_think = True

        if max_stop_len and any(t in raw_so_far[-(max_stop_len * 2):] for t in stop_tokens):
            early_stopped = True
            break

        if seen_end_think and "}" in chunk.text:
            raw_norm = _norm_think_close(raw_so_far)
            after_think = raw_norm.split("</think>", 1)[-1] if "</think>" in raw_norm else raw_norm
            if _first_complete_json(after_think) is not None:
                early_stopped = True
                break

    raw = _norm_think_close(raw_so_far)
    for st in stop_tokens:
        if st in raw:
            raw = raw.split(st, 1)[0]
    return raw, seen_end_think, early_stopped


# ── Core sync inference ───────────────────────────────────────────────────

def _generate_sync(
    model_path: str,
    messages: list[dict],
    temperature: float | None,
    max_tokens: int,
    no_think: bool,
    session_id: str = "",
    json_response: bool = False,
    thinking_budget: int = 0,
    skip_debug_log: bool = False,
) -> str:
    """Blocking generation. Always call via asyncio.to_thread from async code."""
    model, tokenizer = _load_model(model_path)
    prompt_text = _build_prompt(messages, tokenizer, model_path, no_think, thinking_budget)
    # Force JSON start for Hermes: append "{" after the generation prompt so the model
    # cannot open with a chat response. The "{" is restored before extraction below.
    if json_response and is_hermes(model_path):
        prompt_text = prompt_text.rstrip("\n") + "{"
    tok_ids = _prompt_token_ids(prompt_text, tokenizer)
    prompt_tokens = len(tok_ids)
    profile = _model_profile(model_path)
    model_short = model_path.split("/")[-1]

    sys_content = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""
    lru_cache, remaining = _lru_get_cache(model_path, model, tokenizer, sys_content, tok_ids)
    cache_kwarg = {"prompt_cache": lru_cache} if lru_cache is not None else {}

    sampler, logits_procs, quant_kwargs, effective_max = _setup_gen(
        profile, tokenizer, no_think, thinking_budget, temperature, max_tokens
    )

    early_stopped = False
    truncated_by_stop = False

    if json_response:
        result, seen_end_think, early_stopped = _stream_to_json(
            model, tokenizer, remaining, effective_max, sampler, logits_procs,
            quant_kwargs, cache_kwarg, no_think, profile.stop_tokens,
        )
    else:
        result = generate(
            model, tokenizer, prompt=remaining, max_tokens=effective_max,
            sampler=sampler, logits_processors=logits_procs, verbose=False,
            **quant_kwargs, **cache_kwarg,
        )
        for st in profile.stop_tokens:
            if st in result:
                result = result.split(st, 1)[0]
                truncated_by_stop = True
        seen_end_think = not no_think

    _debug_log(model_short, no_think, prompt_text, result, skip=skip_debug_log)
    resp_tokens = len(result) // 4
    _log_stats(model_short, "json" if json_response else "text", no_think,
               "early-stop" if early_stopped else "eos/limit",
               prompt_tokens, resp_tokens, effective_max)

    # Insert full sequence into LRU for future prefix reuse.
    # result is the raw LLM output (including think block if any) — inserted before stripping.
    # Skipped after early-stop or stop-token truncation: the KV cache then contains
    # generated tokens that the re-encoded key would not declare, and a future
    # partial hit on that entry would misalign positions. The system-prefix entry
    # stored by _lru_get_cache keeps system-level caching working in those cases.
    if lru_cache is not None and result and not early_stopped and not truncated_by_stop:
        _lru_insert(model_path, model, tok_ids, result, lru_cache, tokenizer)
    mx.clear_cache()
    metal = _metal_mem_str()
    if metal:
        logger.debug("Metal after clear_cache (sync): %s", metal)

    # For json_response: extract the clean JSON object after </think>.
    if json_response:
        if "</think>" in result:
            json_portion = result.rsplit("</think>", 1)[-1]
        elif seen_end_think:
            json_portion = result
        else:
            logger.warning("_generate_sync: </think> never generated (raw=%r…)", result[:80])
            json_portion = None
        # Restore the "{" that was injected into the prompt for Hermes.
        if json_portion is not None and is_hermes(model_path):
            json_portion = "{" + json_portion
        extracted = _first_complete_json(json_portion) if json_portion is not None else None
        if extracted is not None:
            return extracted

    # Strip thinking block from result.
    if "</think>" in result:
        return _strip_thinking(result.rsplit("</think>", 1)[-1].strip())
    if "<think>" in result:
        # Unclosed explicit <think>: only the text BEFORE the tag is usable.
        before_think = _strip_thinking(result.split("<think>", 1)[0].strip())
        if before_think or no_think:
            return before_think
        # Think mode with nothing before the tag → same fallback as truncation below
        # (previously returned "" silently).
        logger.warning("_generate_sync: unclosed <think>, no visible text (model=%s)", model_short)
        return "⚠️ Réponse incomplète (budget de réflexion dépassé). Reformule ou augmente max_tokens."
    if not no_think:
        logger.warning("_generate_sync: thinking truncated before </think> (model=%s)", model_short)
        return "⚠️ Réponse incomplète (budget de réflexion dépassé). Reformule ou augmente max_tokens."
    return _strip_thinking(result)


# ── Public API ────────────────────────────────────────────────────────────

def call_llm_local(
    messages: list[dict],
    *,
    model: str,
    temperature: float | None = None,
    max_tokens: int = 500,
    no_think: bool = False,
    session_id: str = "",
    json_response: bool = False,
    thinking_budget: int = 0,
    **_kwargs,
) -> str:
    """Sync inference. From async code always call via asyncio.to_thread."""
    with _infer_lock:
        result = _generate_sync(
            model, messages, temperature, max_tokens, no_think,
            session_id, json_response, thinking_budget,
        )
    _wake_bg_waiters()
    return result


def call_llm_local_bg(
    messages: list[dict],
    *,
    model: str,
    temperature: float | None = None,
    max_tokens: int = 3000,
    no_think: bool = False,
    session_id: str = "",
    json_response: bool = False,
    thinking_budget: int = 0,
    **_kwargs,
) -> str:
    """Sync background-priority inference: yields the GPU to chat callers.

    Thread-safe counterpart of call_llm_local_async_bg for sync background tasks
    (analyzer profile dedup, nightly jobs). Polls instead of waiting on the
    asyncio event — a plain thread cannot await _bg_wakeup. Falls back to a
    blocking acquire after 300 s so a busy chat session cannot starve it forever."""
    _t0 = time.time()
    while True:
        with _chat_waiters_lock:
            if _chat_waiters == 0 and _infer_lock.acquire(blocking=False):
                break
        if time.time() - _t0 > 300.0:
            logger.warning("[BG-INFER-SYNC] 300 s of chat priority — acquiring anyway")
            _infer_lock.acquire()
            break
        time.sleep(0.25)
    try:
        return _generate_sync(
            model, messages, temperature, max_tokens, no_think,
            session_id, json_response, thinking_budget,
        )
    finally:
        _infer_lock.release()
        _wake_bg_waiters()


async def call_llm_local_async(
    messages: list[dict],
    *,
    model: str,
    temperature: float | None = None,
    max_tokens: int = 500,
    no_think: bool = False,
    session_id: str = "",
    json_response: bool = False,
    thinking_budget: int = 0,
    **_kwargs,
) -> str:
    """Async non-streaming inference, high priority (chat calls)."""
    _t0 = time.time()
    with _chat_waiters_lock:
        global _chat_waiters
        _chat_waiters += 1
    try:
        await asyncio.to_thread(_infer_lock.acquire)
    finally:
        with _chat_waiters_lock:
            _chat_waiters -= 1
    logger.debug("[TTFT] call_llm_local_async: lock acquired — waited %.3fs", time.time() - _t0)
    try:
        return await asyncio.to_thread(
            _generate_sync, model, messages, temperature, max_tokens,
            no_think, session_id, json_response, thinking_budget,
        )
    finally:
        _infer_lock.release()
        if _bg_wakeup is not None:
            _bg_wakeup.set()


async def call_llm_local_async_bg(
    messages: list[dict],
    *,
    model: str,
    temperature: float | None = None,
    max_tokens: int = 3000,
    no_think: bool = False,
    session_id: str = "",
    json_response: bool = False,
    thinking_budget: int = 0,
    **_kwargs,
) -> str:
    """Async non-streaming, background priority: yields GPU when a chat caller is waiting."""
    global _bg_wakeup, _bg_loop
    if _bg_wakeup is None:
        _bg_wakeup = asyncio.Event()
        _bg_loop = asyncio.get_running_loop()
        _bg_wakeup.set()
    _t0 = time.time()
    while True:
        with _chat_waiters_lock:
            if _chat_waiters == 0:
                acquired = _infer_lock.acquire(blocking=False)
                if acquired:
                    _bg_wakeup.clear()
                    break
        _bg_wakeup.clear()
        with _chat_waiters_lock:
            if _chat_waiters == 0:
                acquired = _infer_lock.acquire(blocking=False)
                if acquired:
                    break
        _waited = time.time() - _t0
        if _waited > 5.0 and int(_waited) % 10 == 0:
            logger.debug("[BG-INFER] waiting for GPU (chat priority) — %.0fs", _waited)
        await _bg_wakeup.wait()

    logger.debug("[BG-INFER] lock acquired after %.3fs", time.time() - _t0)
    try:
        return await asyncio.to_thread(
            _generate_sync, model, messages, temperature, max_tokens,
            no_think, session_id, json_response, thinking_budget,
        )
    finally:
        _infer_lock.release()
        if _bg_wakeup is not None:
            _bg_wakeup.set()


async def stream_local(
    messages: list[dict],
    model: str,
    temperature: float | None = None,
    max_tokens: int = MAX_TOKENS_HARD_CAP,
    no_think: bool = False,
    session_id: str = "",
    thinking_budget: int = 0,
    skip_debug_log: bool = False,
    **_kwargs,
) -> AsyncGenerator[str, None]:
    """Token-by-token streaming via mlx_lm.stream_generate."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    stop_flag = threading.Event()

    _t_lock_wait = time.time()

    def _worker():
        _t_infer = time.time()
        logger.debug("[TTFT] stream_local: inference started (lock held %.3fs)",
                     _t_infer - _t_lock_wait)
        # Sentinels first: the finally below must be safe to run even if setup
        # (_load_model, _build_prompt, _lru_get_cache) fails — it releases the
        # GPU lock and unblocks the consumer, so it must ALWAYS execute.
        raw_chunks: list[str] = []
        first = True
        _generation_ok = True
        _truncated_by_stop = False
        model_short = model.split("/")[-1]
        prompt_text = ""
        prompt_tokens = 0
        budget = 0
        tok_ids: list[int] = []
        lru_cache = None
        mlx_model = tokenizer = None
        try:
            mlx_model, tokenizer = _load_model(model)
            prompt_text = _build_prompt(messages, tokenizer, model, no_think, thinking_budget)

            # Tokenize for LRU lookup
            tok_ids = _prompt_token_ids(prompt_text, tokenizer)
            prompt_tokens = len(tok_ids)

            sys_content = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""
            lru_cache, remaining = _lru_get_cache(model, mlx_model, tokenizer, sys_content, tok_ids)
            cache_kwarg = {"prompt_cache": lru_cache} if lru_cache is not None else {}

            profile = _model_profile(model)
            sampler, stream_procs, quant_kwargs, budget = _setup_gen(
                profile, tokenizer, no_think, thinking_budget, temperature, max_tokens
            )

            for chunk in stream_generate(
                mlx_model, tokenizer, prompt=remaining, max_tokens=budget,
                sampler=sampler, logits_processors=stream_procs,
                **cache_kwarg, **quant_kwargs,
            ):
                if stop_flag.is_set():
                    break
                if not chunk.text:
                    continue

                text = _norm_think_close(chunk.text)
                if profile.stop_tokens:
                    joined = "".join(raw_chunks)
                    acc = joined + text
                    for st in profile.stop_tokens:
                        if st in acc:
                            _truncated_by_stop = True
                            text = acc.split(st, 1)[0][len(joined):]
                            if text:
                                raw_chunks.append(text)
                                loop.call_soon_threadsafe(queue.put_nowait, text)
                            return

                raw_chunks.append(text)
                if first and text:
                    logger.debug("[TTFT] stream_local: first token — %.3fs since inference start",
                                 time.time() - _t_infer)
                    first = False
                loop.call_soon_threadsafe(queue.put_nowait, text)

        except Exception as exc:
            logger.error("stream_local error: %s", exc)
            _generation_ok = False
        finally:
            raw_resp = "".join(raw_chunks)
            _debug_log(model_short, no_think, prompt_text, raw_resp, skip=skip_debug_log)
            resp_tokens = len(raw_resp) // 4
            thinking_active = "</think>" in raw_resp or "</think >" in raw_resp
            _log_stats(model_short, "stream", no_think, "eos/limit",
                       prompt_tokens, resp_tokens, budget)
            if thinking_active:
                logger.debug("[LLM-STATS] thinking active in stream response")

            # Insert full sequence into LRU for future prefix reuse.
            # Only insert on clean completion (not on error, stop_flag abort, or
            # stop-token truncation — truncation leaves tokens in the KV cache that
            # the re-encoded key would not declare → misaligned future partial hits).
            if (
                lru_cache is not None and raw_resp and _generation_ok
                and not stop_flag.is_set() and not _truncated_by_stop
            ):
                _lru_insert(model, mlx_model, tok_ids, raw_resp, lru_cache, tokenizer)

            # Free MLX compute buffers accumulated during inference.
            # Without this, the 4 GB compute cache fills up across multiple large requests
            # and pushes total Metal usage past the available unified memory limit.
            mx.clear_cache()
            metal = _metal_mem_str()
            if metal:
                logger.debug("Metal after clear_cache (stream): %s", metal)

            # Release the GPU lock HERE, once inference and clear_cache are truly
            # finished. Releasing from the generator's finally (previous behaviour)
            # opened a race on client disconnect: a new request could start inference
            # while this worker was still mid-step. It also required the event loop
            # to run for the lock to be freed — a sync LLM call blocking the loop
            # could then deadlock against a stream holding the lock.
            _infer_lock.release()
            _wake_bg_waiters()
            loop.call_soon_threadsafe(queue.put_nowait, None)

    with _chat_waiters_lock:
        global _chat_waiters
        _chat_waiters += 1
    try:
        await asyncio.to_thread(_infer_lock.acquire)
    finally:
        with _chat_waiters_lock:
            _chat_waiters -= 1
    logger.debug("[TTFT] stream_local: lock acquired — waited %.3fs", time.time() - _t_lock_wait)

    # From here the worker owns the lock release (in its finally). Only guard the
    # window where the thread could fail to start.
    try:
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
    except BaseException:
        _infer_lock.release()
        _wake_bg_waiters()
        raise

    try:
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
    finally:
        stop_flag.set()  # worker exits at next chunk and releases the lock itself


# ── Vision model (mlx_vlm) ────────────────────────────────────────────────
# Uses _vlm_lock, NOT _infer_lock: VLM and text inference are sequential in the
# pipeline (describe_images runs before Qwen3.6), so they never compete for the GPU.

_vlm_model = None
_vlm_processor = None
_vlm_config = None
_vlm_lock = threading.Lock()
_vlm_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="jarvis-vlm")


def _load_vlm() -> None:
    global _vlm_model, _vlm_processor, _vlm_config
    if _vlm_model is not None:
        return
    from config import VISION_MODEL
    from mlx_vlm import load as vlm_load
    from mlx_vlm.utils import load_config as vlm_load_config
    logger.info("VLM: loading %s…", VISION_MODEL)
    t0 = time.time()
    _vlm_model, _vlm_processor = vlm_load(VISION_MODEL)
    _vlm_config = vlm_load_config(VISION_MODEL)
    logger.info("VLM: loaded in %.1fs", time.time() - t0)


def _warmup_vlm() -> None:
    """Run a tiny dummy inference to JIT-compile MLX VLM graphs. Must run in _vlm_executor."""
    import io
    from PIL import Image as _PIL
    from mlx_vlm import generate as vlm_generate
    from mlx_vlm.prompt_utils import apply_chat_template as vlm_apply_chat_template
    from mlx_vlm.utils import load_image as vlm_load_image

    _load_vlm()
    buf = io.BytesIO()
    _PIL.new("RGB", (32, 32), color=(128, 128, 128)).save(buf, format="JPEG")
    buf.seek(0)
    image = vlm_load_image(buf)
    prompt_text = VISION_USER_PROMPT.format(text_prompt="Décris brièvement.")
    formatted = vlm_apply_chat_template(_vlm_processor, _vlm_config, prompt_text, num_images=1)
    vlm_generate(_vlm_model, _vlm_processor, formatted, image=image, max_tokens=10, temperature=0.0, verbose=False)
    logger.info("VLM warmup OK: %s", VISION_MODEL.split("/")[-1] if VISION_MODEL else "?")


def _describe_images_sync(image_parts: list, text_prompt: str) -> str:
    import base64 as _b64
    import io
    from mlx_vlm import generate as vlm_generate
    from mlx_vlm.prompt_utils import apply_chat_template as vlm_apply_chat_template
    from mlx_vlm.utils import load_image as vlm_load_image

    from PIL import Image as _PIL

    _load_vlm()
    mx.clear_cache()  # libère les buffers de calcul MLX avant l'inférence VLM
    images = []
    for part in image_parts:
        url = (part.get("image_url") or {}).get("url", "")
        if url.startswith("data:"):
            _, b64data = url.split(",", 1)
            pil_img = _PIL.open(io.BytesIO(_b64.b64decode(b64data))).convert("RGB")
            # Resize to max 1024px — reduces visual tiles and speeds up inference significantly.
            # iPhone photos (~12 MP) generate dozens of tiles at full resolution.
            max_dim = 1024
            if max(pil_img.size) > max_dim:
                ratio = max_dim / max(pil_img.size)
                pil_img = pil_img.resize(
                    (int(pil_img.size[0] * ratio), int(pil_img.size[1] * ratio)),
                    _PIL.LANCZOS,
                )
                logger.debug("vision: resized to %s", pil_img.size)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            buf.seek(0)
            images.append(vlm_load_image(buf))
        elif url.startswith("http"):
            images.append(url)
        else:
            logger.warning(
                "_describe_images_sync: URL dropped (not data: or http) — url=%r",
                url[:80],
            )
    if not images:
        logger.warning("_describe_images_sync: no image decoded from %d part(s)", len(image_parts))
        return ""

    prompt_text = VISION_USER_PROMPT.format(
        text_prompt=text_prompt or "Décris cette image dans son ensemble."
    )
    formatted = vlm_apply_chat_template(_vlm_processor, _vlm_config, prompt_text, num_images=len(images))
    result = vlm_generate(
        _vlm_model, _vlm_processor, formatted,
        image=images[0] if len(images) == 1 else images,
        max_tokens=700, temperature=0.7,
        repetition_penalty=1.5, repetition_context_size=256, verbose=False,
    )
    return result.text if hasattr(result, "text") else str(result)


async def describe_images_local(image_parts: list, text_prompt: str) -> str:
    """Async VLM inference. Same executor as preload → same thread → same MLX Metal stream."""
    def _run():
        with _vlm_lock:
            return _describe_images_sync(image_parts, text_prompt)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_vlm_executor, _run)

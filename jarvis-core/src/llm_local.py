"""
llm_local.py — MLX inference with system-prompt KV cache.
Active when LLM_LOCAL=yes. Models loaded once at startup (preload_models) or lazily.
Router/Primary share the same object when their paths are identical.

Public API
----------
preload_models()             → warm-up at startup (main.py lifespan)
call_llm_local(...)          → str   (sync)
call_llm_local_async(...)    → str   (async non-streaming, high priority)
call_llm_local_async_bg(...) → str   (async non-streaming, yields GPU to chat)
stream_local(...)            → AsyncGenerator[str, None]
describe_images_local(...)   → str   (async, mlx_vlm)
"""

import asyncio
import copy
import datetime
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Optional

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
    is_qwen3,
    is_qwen36,
)
from prompts import VISION_USER_PROMPT
from mlx_lm import generate, load, stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_logits_processors, make_sampler

HF_HOME = os.getenv("HF_HOME", "/opt/jarvis/models")
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HUB_CACHE"]

logger = logging.getLogger("jarvis-llm-local")

LLM_DEBUG_PROMPTS = os.getenv("LLM_DEBUG_PROMPTS", "").lower() in ("yes", "true", "1")
_PROMPTS_LOG_PATH = "/opt/jarvis/logs/prompts.log"

QUANT_KV = os.getenv("QUANT_KV", "").lower() in ("yes", "true", "1")
QUANT_KV_BITS = int(os.getenv("QUANT_KV_BITS", "4"))


# ── Debug logging ─────────────────────────────────────────────────────────

def _debug_log(model_short: str, no_think: bool, prompt: str, raw_output: str) -> None:
    if not LLM_DEBUG_PROMPTS:
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
        # Hermes occasionally continues after <|im_end|> — explicit stop string closes the leak.
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
_infer_lock = threading.Lock()  # serialises all MLX inference (GPU Metal not preemptable)

# Background tasks yield the GPU when a chat caller is waiting.
_chat_waiters: int = 0
_chat_waiters_lock = threading.Lock()

# System-prompt KV cache: prefilled once, deepcopied per call.
# Multi-turn session caching was removed: Qwen3's enable_thinking template adds
# <think>…</think> to the generation prompt, causing token mismatch on rebuild.
_sys_kv: dict[str, tuple[int, Any]] = {}
_sys_kv_lock = threading.Lock()


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


# ── System-prompt KV cache ────────────────────────────────────────────────

def _make_system_kv(model_path: str, model, tokenizer, system_content: str) -> Any:
    """Pre-fill system prompt into a KV cache. Called once per unique (path, content)."""
    if is_qwen36(model_path):
        # Qwen3.6 ninja template requires ≥1 user message; build system block directly.
        sys_prompt_text = f"<|im_start|>system\n{system_content}<|im_end|>\n"
    else:
        sys_prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_content}],
            tokenize=False, add_generation_prompt=False,
        )
    sys_token_count = len(tokenizer.encode(sys_prompt_text))
    profile = _model_profile(model_path)
    quant_kwargs = {"kv_bits": QUANT_KV_BITS, "kv_group_size": 64} if profile.use_quant_kv else {}
    cache = make_prompt_cache(model)
    try:
        for _ in stream_generate(
            model, tokenizer, prompt=sys_prompt_text, max_tokens=1,
            sampler=make_sampler(temp=0.0, top_p=1.0, top_k=0, min_p=0.0),
            prompt_cache=cache, **quant_kwargs,
        ):
            break
        # Trim offset to system token count — the 1 garbage decode token is overwritten next call.
        for layer_cache in cache:
            if hasattr(layer_cache, "offset"):
                layer_cache.offset = sys_token_count
    except Exception as exc:
        logger.warning("System KV cache build failed: %s", exc)
        return None
    logger.info("KV cache: system prompt prefilled (%d tok, model=%s)",
                sys_token_count, model_path.split("/")[-1])
    return cache


def _get_system_cache(model_path: str, model, tokenizer, system_content: str) -> Any:
    """Return a deepcopy of the system KV cache (lazy, thread-safe). Must hold _infer_lock."""
    if not system_content:
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
            logger.warning("KV cache deepcopy failed (%s) — running without cache", exc)
            return None


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


def preload_models() -> None:
    """Load and warm-up models at startup. Call from main.py lifespan."""
    if not LLM_LOCAL:
        return
    try:
        import mlx.core as _mx
        # 4 GB allocator cache: release unused buffers promptly to keep headroom for KV caches.
        _mx.set_cache_limit(12 * 1024**3)
        # Wired limit: 85% of Metal's max working set size (macOS-enforced cap).
        try:
            info = _mx.device_info()
            max_ws = info.get("max_working_set_size", 0)
            if max_ws > 0:
                wired = int(max_ws * 0.85)
                _mx.metal.set_wired_limit(wired)
                logger.info("MLX: cache_limit=12 GB, wired_limit=%.0f GB", wired / 1024**3)
            else:
                logger.info("MLX: cache_limit=12 GB (wired_limit skipped — device_info unavailable)")
        except Exception as exc:
            logger.info("MLX: cache_limit=12 GB (wired_limit skipped: %s)", exc)
    except Exception as exc:
        logger.warning("MLX memory limits: %s (non-fatal)", exc)

    model_paths = {ROUTER_MODEL, PRIMARY_MODEL, REASONING_MODEL}
    for path in model_paths:
        _load_model(path)
    logger.info("MLX: %d model(s) preloaded", len(model_paths))

    if VISION_MODEL:
        try:
            _load_vlm()
            logger.info("MLX VLM preloaded: %s", VISION_MODEL)
        except Exception as exc:
            logger.warning("MLX VLM preload failed (non-fatal): %s", exc)

    # JIT warmup: compile MLX graphs at startup, not on the first user request.
    warmup_msgs = [
        {"role": "system", "content": "Tu es un assistant."},
        {"role": "user", "content": "Salut"},
    ]
    for path in model_paths:
        try:
            _generate_sync(path, warmup_msgs, temperature=0.0, max_tokens=100, no_think=True)
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
        think_kwargs: dict[str, Any] = {"enable_thinking": False, "thinking_budget": 0}
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

    # String-level guard: enforce think state after template (handles all variants).
    if no_think:
        if prompt.endswith("<think>\n") or prompt.rstrip().endswith("<think>"):
            # Template opened think block but didn't close it → force-close.
            prompt = prompt.rstrip("\n") + "\n</think>\n\n"
        elif "</think>" not in _norm_think_close(prompt[-30:]):
            # Ninja-patch (Qwen3.6): bare assistant prefix → inject empty block.
            # Without it, Qwen3.6 generates <think> spontaneously on its first token.
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
        self._initialized = False  # first call = last prompt token, skip it
        self._thinking_count = 0
        self._thinking_done = False
        self._end_think_id: int | None = None

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
        except Exception as exc:
            logger.warning("ThinkingBudgetProcessor init: %s", exc)

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

        self._thinking_count += 1
        n = logits.shape[-1]
        idx = self._end_think_id

        if self._thinking_count >= self.budget:
            logger.debug("ThinkingBudgetProcessor: budget=%d reached — forcing </think>", self._thinking_count)
            left = mx.full([idx], -1e9, dtype=logits.dtype)
            mid = mx.array([100.0], dtype=logits.dtype)
            right = mx.full([n - idx - 1], -1e9, dtype=logits.dtype)
            return mx.concatenate([left, mid, right]).reshape(logits.shape)

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
    quant_kwargs = {"kv_bits": QUANT_KV_BITS, "kv_group_size": 64} if profile.use_quant_kv else {}
    effective_temp = (
        temperature if (temperature is not None and temperature > 0)
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
    prompt: str,
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

    Waits for </think> before scanning for JSON to avoid false positives
    inside the thinking block.
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
    temperature: float,
    max_tokens: int,
    no_think: bool,
    session_id: str = "",
    json_response: bool = False,
    thinking_budget: int = 0,
) -> str:
    """Blocking generation. Always call via asyncio.to_thread from async code."""
    model, tokenizer = _load_model(model_path)
    prompt = _build_prompt(messages, tokenizer, model_path, no_think, thinking_budget)
    prompt_tokens = len(tokenizer.encode(prompt))
    profile = _model_profile(model_path)
    model_short = model_path.split("/")[-1]

    sys_content = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""
    kv_cache = _get_system_cache(model_path, model, tokenizer, sys_content)
    cache_kwarg = {"prompt_cache": kv_cache} if kv_cache is not None else {}

    sampler, logits_procs, quant_kwargs, effective_max = _setup_gen(
        profile, tokenizer, no_think, thinking_budget, temperature, max_tokens
    )

    early_stopped = False

    if json_response:
        result, seen_end_think, early_stopped = _stream_to_json(
            model, tokenizer, prompt, effective_max, sampler, logits_procs,
            quant_kwargs, cache_kwarg, no_think, profile.stop_tokens,
        )
    else:
        result = generate(
            model, tokenizer, prompt=prompt, max_tokens=effective_max,
            sampler=sampler, logits_processors=logits_procs, verbose=False,
            **quant_kwargs, **cache_kwarg,
        )
        for st in profile.stop_tokens:
            if st in result:
                result = result.split(st, 1)[0]
        seen_end_think = not no_think  # not used in non-json path

    _debug_log(model_short, no_think, prompt, result)
    resp_tokens = len(tokenizer.encode(result))
    _log_stats(model_short, "json" if json_response else "text", no_think,
               "early-stop" if early_stopped else "eos/limit",
               prompt_tokens, resp_tokens, effective_max)

    # For json_response: extract the clean JSON object after </think>.
    if json_response:
        if "</think>" in result:
            json_portion = result.split("</think>", 1)[-1]
        elif seen_end_think:
            json_portion = result
        else:
            logger.warning("_generate_sync: </think> never generated (raw=%r…)", result[:80])
            json_portion = None
        extracted = _first_complete_json(json_portion) if json_portion is not None else None
        if extracted is not None:
            return extracted
        # Fall through: incomplete JSON — strip thinking and return raw for caller to parse.

    # Strip thinking block from result.
    if "</think>" in result:
        return _strip_thinking(result.split("</think>", 1)[-1].strip())
    if "<think>" in result:
        return _strip_thinking(result.split("<think>", 1)[0].strip())
    if not no_think:
        # Thinking was expected but </think> never came — truncated mid-reasoning.
        logger.warning("_generate_sync: thinking truncated before </think> (model=%s)", model_short)
        partial = result.split("<think>", 1)[0].strip() if "<think>" in result else ""
        return partial or "⚠️ Réponse incomplète (budget de réflexion dépassé). Reformule ou augmente max_tokens."
    return _strip_thinking(result)


# ── Public API ────────────────────────────────────────────────────────────

def call_llm_local(
    messages: list[dict],
    *,
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 500,
    no_think: bool = False,
    session_id: str = "",
    json_response: bool = False,
    thinking_budget: int = 0,
    **_kwargs,
) -> str:
    """Sync inference. From async code always call via asyncio.to_thread."""
    with _infer_lock:
        return _generate_sync(
            model, messages, temperature, max_tokens, no_think,
            session_id, json_response, thinking_budget,
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


async def call_llm_local_async_bg(
    messages: list[dict],
    *,
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 3000,
    no_think: bool = False,
    session_id: str = "",
    json_response: bool = False,
    thinking_budget: int = 0,
    **_kwargs,
) -> str:
    """Async non-streaming, background priority: yields GPU when a chat caller is waiting."""
    _t0 = time.time()
    while True:
        with _chat_waiters_lock:
            if _chat_waiters == 0:
                acquired = _infer_lock.acquire(blocking=False)
                if acquired:
                    break
        _waited = time.time() - _t0
        if _waited > 5.0 and int(_waited) % 10 == 0:
            logger.debug("[BG-INFER] waiting for GPU (chat priority) — %.0fs", _waited)
        await asyncio.sleep(2.0)

    logger.debug("[BG-INFER] lock acquired after %.3fs", time.time() - _t0)
    try:
        return await asyncio.to_thread(
            _generate_sync, model, messages, temperature, max_tokens,
            no_think, session_id, json_response, thinking_budget,
        )
    finally:
        _infer_lock.release()


async def stream_local(
    messages: list[dict],
    model: str,
    temperature: float | None = None,
    max_tokens: int = MAX_TOKENS_HARD_CAP,
    no_think: bool = False,
    session_id: str = "",
    thinking_budget: int = 0,
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
        mlx_model, tokenizer = _load_model(model)
        prompt = _build_prompt(messages, tokenizer, model, no_think, thinking_budget)
        prompt_tokens = len(tokenizer.encode(prompt))
        model_short = model.split("/")[-1]

        sys_content = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""
        kv_cache = _get_system_cache(model, mlx_model, tokenizer, sys_content)
        cache_kwarg = {"prompt_cache": kv_cache} if kv_cache is not None else {}

        profile = _model_profile(model)
        sampler, stream_procs, quant_kwargs, budget = _setup_gen(
            profile, tokenizer, no_think, thinking_budget, temperature, max_tokens
        )

        raw_chunks: list[str] = []
        first = True
        try:
            for chunk in stream_generate(
                mlx_model, tokenizer, prompt=prompt, max_tokens=budget,
                sampler=sampler, logits_processors=stream_procs,
                **cache_kwarg, **quant_kwargs,
            ):
                if stop_flag.is_set():
                    break
                if not chunk.text:
                    continue

                text = chunk.text
                if profile.stop_tokens:
                    joined = "".join(raw_chunks)
                    acc = joined + text
                    for st in profile.stop_tokens:
                        if st in acc:
                            text = acc.split(st, 1)[0][len(joined):]
                            if text:
                                raw_chunks.append(text)
                                loop.call_soon_threadsafe(queue.put_nowait, text)
                            return  # finally block sends the None sentinel

                raw_chunks.append(text)
                if first and text:
                    logger.debug("[TTFT] stream_local: first token — %.3fs since inference start",
                                 time.time() - _t_infer)
                    first = False
                loop.call_soon_threadsafe(queue.put_nowait, text)

        except Exception as exc:
            logger.error("stream_local error: %s", exc)
        finally:
            raw_resp = "".join(raw_chunks)
            _debug_log(model_short, no_think, prompt, raw_resp)
            resp_tokens = len(tokenizer.encode(raw_resp))
            thinking_active = "</think>" in raw_resp or "</think >" in raw_resp
            _log_stats(model_short, "stream", no_think, "eos/limit",
                       prompt_tokens, resp_tokens, budget)
            if thinking_active:
                logger.debug("[LLM-STATS] thinking active in stream response")
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

    try:
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
    finally:
        stop_flag.set()
        _infer_lock.release()


# ── Vision model (mlx_vlm) ────────────────────────────────────────────────
# Uses _vlm_lock, NOT _infer_lock: VLM and text inference are sequential in the
# pipeline (describe_images runs before Qwen3.6), so they never compete for the GPU.
# Holding _infer_lock during VLM load + Metal compilation would block background
# tasks for minutes on first run.

_vlm_model = None
_vlm_processor = None
_vlm_config = None
_vlm_lock = threading.Lock()


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


def _describe_images_sync(image_parts: list, text_prompt: str) -> str:
    import base64 as _b64
    import io
    from mlx_vlm import generate as vlm_generate
    from mlx_vlm.prompt_utils import apply_chat_template as vlm_apply_chat_template
    from mlx_vlm.utils import load_image as vlm_load_image

    _load_vlm()
    images = []
    for part in image_parts:
        url = (part.get("image_url") or {}).get("url", "")
        if url.startswith("data:"):
            _, b64data = url.split(",", 1)
            images.append(vlm_load_image(io.BytesIO(_b64.b64decode(b64data))))
        elif url.startswith("http"):
            images.append(url)
    if not images:
        return ""

    prompt_text = VISION_USER_PROMPT.format(
        text_prompt=text_prompt or "Décris cette image dans son ensemble."
    )
    formatted = vlm_apply_chat_template(_vlm_processor, _vlm_config, prompt_text, num_images=len(images))
    result = vlm_generate(
        _vlm_model, _vlm_processor, formatted,
        image=images[0] if len(images) == 1 else images,
        max_tokens=1200, temperature=0.7,
        repetition_penalty=1.3, repetition_context_size=64, verbose=False,
    )
    return result.text if hasattr(result, "text") else str(result)


async def describe_images_local(image_parts: list, text_prompt: str) -> str:
    """Async VLM inference. Uses _vlm_lock so VLM load never blocks text inference."""
    await asyncio.to_thread(_vlm_lock.acquire)
    try:
        return await asyncio.to_thread(_describe_images_sync, image_parts, text_prompt)
    finally:
        _vlm_lock.release()

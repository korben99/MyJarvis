"""
llm_local_KVCache.py — Inférence MLX avec cache système-prompt (KVCache classique)
====================================================================================
Variante stable de llm_local.py. Pour activer : cp llm_local_KVCache.py llm_local.py

Différence vs llm_local_LRU.py :
  Cache uniquement le prompt système (prefill une fois, deepcopy par appel).
  Plus simple, aucune dépendance aux nouvelles API mlx_lm.

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
  describe_images_local(...)  → str   (async, mlx_vlm — image description)
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
    PRIMARY_MODEL,
    QWEN36_NINJA_TEMPLATE,
    REASONING_MODEL,
    ROUTER_MODEL,
    THINKING_BUDGET_TOKENS,
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
    presence_penalty: float  # fixed additive penalty if token has appeared at all (OpenAI-style)
    use_quant_kv: bool  # whether to enable KV cache quantisation
    stop_tokens: tuple[str, ...]  # extra stop strings beyond the tokenizer's EOS


def _model_profile(model_path: str) -> _ModelProfile:
    """Return the inference profile for *model_path*.

    Three profiles (checked in priority order):
    - Qwen3.6 (is_qwen36): temp_think=0.7, repetition_penalty=1.05.
    - Qwen3   (is_qwen3) : temp_think=0.6, repetition_penalty=1.1.
      is_qwen36() checked FIRST — it's a subset of is_qwen3().
    - Generic (router / non-Qwen3): relaxed sampler, no KV quantisation.
      Applying top_k=20 or 4-bit KV to small instruct models (e.g. Qwen2.5-3B-8bit)
      causes degenerate looping — these must stay at their default settings.
    """
    if is_qwen36(model_path):
        # Qwen3.6 official recommendations: temp_think=0.7 (conservative; official says 1.0
        # for open-ended but 0.7 avoids hallucination on structured tasks like Jarvis uses).
        # top_p / top_k identical to Qwen3. Slightly reduced repetition_penalty (1.05 vs 1.1)
        # — Qwen3.6 has better native diversity from larger expert pool (256 vs 64).
        return _ModelProfile(
            temp_think=1.0,
            temp_nothink=0.7,
            top_p_think=0.95,
            top_p_nothink=0.80,
            top_k=20,
            min_p=0.0,
            repetition_penalty=1.0,
            repetition_context_size=256,  # fenêtre élargie — couvre les copies depuis le prompt injecté
            frequency_penalty=0.15,  # accumulative — brise les boucles que presence_penalty seul ne stoppe pas
            presence_penalty=1.5,  # official Qwen3.6 recommendation — pénalité fixe dès la 1ère occurrence
            use_quant_kv=QUANT_KV,
            stop_tokens=(),
        )
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
            presence_penalty=1.5,
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
            presence_penalty=0.0,  # structured JSON output — penalties degrade quality
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
        presence_penalty=0.0,
        use_quant_kv=False,
        stop_tokens=(),
    )


# ── Registre de modèles ───────────────────────────────────────────────────

_model_cache: dict[str, tuple] = {}  # model_path → (model, tokenizer)
_load_lock = threading.Lock()  # protège le chargement concurrent
_infer_lock = threading.Lock()
# _infer_lock sérialise TOUTES les inférences MLX (router + primary confondus).
# Intentionnel : le GPU Metal est partagé et ne supporte pas deux kernels simultanés.
# Si router et primary doivent être indépendants, remplacer par dict[str, Lock].

# ── Chat priority tracker ─────────────────────────────────────────────────
# Compteur thread-safe des appels chat en attente du lock.
# Les tâches background (analyzer) cèdent le GPU si ce compteur > 0,
# garantissant que le chat n'attend jamais derrière l'analyzer.
_chat_waiters: int = 0
_chat_waiters_lock = threading.Lock()

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


def _first_complete_json(text: str) -> Optional[str]:
    """
    Return the first valid complete top-level JSON object or array found in *text*, or None.
    """

    start = -1
    stack = []
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
            if c in "{[":
                start = i
                stack.append(c)
            continue

        # Gestion des ouvertures imbriquées
        if c in "{[":
            stack.append(c)

        elif c in "}]":
            if not stack:
                continue

            last = stack.pop()

            # Vérification cohérence {} vs []
            if (last == "{" and c != "}") or (last == "[" and c != "]"):
                # reset complet si incohérent
                start = -1
                stack.clear()
                continue

            if not stack:
                candidate = text[start : i + 1]

                # Validation JSON réelle
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    # si invalide → on continue à chercher
                    start = -1

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
    if is_qwen36(model_path):
        # Qwen3.6 ninja-patch template requires ≥1 user message — system-only input
        # raises jinja2.TemplateError("No user query found in messages.").
        # Construct the system block directly: format is stable across Qwen3.x (ChatML).
        sys_prompt_text = f"<|im_start|>system\n{system_content}<|im_end|>\n"
    else:
        sys_prompt_text = tokenizer.apply_chat_template(
            sys_messages, tokenize=False, add_generation_prompt=False
        )
    sys_token_count = len(tokenizer.encode(sys_prompt_text))

    profile = _model_profile(model_path)
    quant_kwargs = (
        {"kv_bits": QUANT_KV_BITS, "kv_group_size": 64} if profile.use_quant_kv else {}
    )
    cache = make_prompt_cache(model)
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

        # ── Ninja patch template (Qwen3.6 only) ─────────────────────────
        # Chemin local défini par QWEN36_NINJA_TEMPLATE (config.py).
        # Téléchargé via scripts/download_models.py — indépendant du cache HF.
        # enable_thinking=False → aucun tag <think> dans le prompt (0 token overhead).
        if is_qwen36(model_path):
            if os.path.isfile(QWEN36_NINJA_TEMPLATE):
                with open(QWEN36_NINJA_TEMPLATE, encoding="utf-8") as _f:
                    tokenizer.chat_template = _f.read()
                logger.info(
                    "Ninja-patch template applied: %s", model_path.split("/")[-1]
                )
            else:
                logger.warning(
                    "Ninja-patch template introuvable (%s) — template par défaut utilisé. "
                    "Lancer scripts/download_models.py pour le télécharger.",
                    QWEN36_NINJA_TEMPLATE,
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

    # Limit Metal allocator cache to 4 GB so unused buffers between inferences
    # are released promptly rather than retained indefinitely by the allocator.
    # This keeps headroom available for KV caches during long conversations.
    try:
        import mlx.core as _mx

        _mx.set_cache_limit(12 * 1024**3)
        logger.info("MLX Metal cache limit set to 12 GB")
    except Exception as exc:
        logger.warning("MLX set_cache_limit failed (non-fatal): %s", exc)

    model_paths = {ROUTER_MODEL, PRIMARY_MODEL, REASONING_MODEL}  # set → déduplique
    for path in model_paths:
        _load_model(path)
    logger.info("MLX : %d modèle(s) préchargé(s)", len(model_paths))

    if VISION_MODEL:
        try:
            _load_vlm()
            logger.info("MLX VLM préchargé : %s", VISION_MODEL)
        except Exception as exc:
            logger.warning("MLX VLM preload failed (non-fatal): %s", exc)

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
                max_tokens=100,  # assez pour le JIT sans déclencher le warning de troncature
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
    thinking_budget: int = 0,
) -> str:
    """
    Construit le prompt final via apply_chat_template.

    Qwen3.x : passe enable_thinking pour contrôler le bloc <think>.
      Qwen3 standard   enable_thinking=False → <think>\\n\\n</think>\\n\\n (no-think)
      Qwen3.6 + ninja  enable_thinking=False → préfixe nu (pas de tag <think>)
      Tout Qwen3.x     enable_thinking=True  → <think>\\n (thinking libre)
                       + thinking_budget=N   → <think>\\n<budget_remaining>N</budget_remaining>\\n

    La ninja patch guard (en fin de fonction) normalise le suffixe brut :
      - préfixe nu (ninja patch no_think) → injecte <think>\\n\\n</think>\\n\\n
        car sans ce bloc le modèle génère <think> spontanément.
      - bloc ouvert non fermé             → force-close </think>\\n\\n
      - no_think=False sans <think>       → force-open <think>\\n

    Fallback : si enable_thinking n'est pas accepté (TypeError), réessaie sans.
    thinking_budget > 0 : cap de tokens de thinking (format <budget_remaining>N</budget_remaining>).
      Ignoré si no_think=True.
      Ignoré si le tokenizer ne supporte pas le paramètre (TypeError → fallback sans budget).
    """
    qwen3 = is_qwen3(model_path)
    base_kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}

    if qwen3:
        # Build thinking kwargs:
        # - no_think=True  → enable_thinking=False + thinking_budget=0 (belt+suspenders:
        #   mlx-lm issue #1625 — enable_thinking=False alone may not suppress thinking in
        #   some library versions; thinking_budget=0 enforces the budget at template level)
        # - no_think=False, thinking_budget=0  → enable_thinking=True, thinking libre
        #   Sur une tâche complexe, Qwen3.6 peut penser ~1900 tokens naturellement.
        # - no_think=False, thinking_budget>0  → enable_thinking=True + tag <budget_remaining>N>
        #   Injecté par le ninja template (qwen36_ninja.jinja).
        #
        #   COMPORTEMENT MESURÉ sur Qwen3.6-35B-A3B (test scripts/test_thinking_budget.py) :
        #   Le tag <budget_remaining> n'agit PAS comme un cap précis sur Qwen3.6 —
        #   le modèle n'a pas été entraîné avec le mécanisme budget-forcing de Qwen3.
        #   Sur certains prompts, le modèle émet '<budget>N</budget>' comme réponse
        #   visible (garbled echo du tag) au lieu de générer du contenu — comportement
        #   non-déterministe et lié à la sensibilité au prompt. Désactivé pour Qwen3.6.
        if no_think:
            think_kwargs: dict[str, Any] = {
                "enable_thinking": False,
                "thinking_budget": 0,
            }
        elif thinking_budget > 0 and not is_qwen36(model_path):
            think_kwargs = {"enable_thinking": True, "thinking_budget": thinking_budget}
        else:
            think_kwargs = {"enable_thinking": True}

        # Try full kwargs; fall back if tokenizer version is too old
        try:
            prompt = tokenizer.apply_chat_template(
                messages, **base_kwargs, **think_kwargs
            )
        except TypeError:
            # thinking_budget not supported → retry with only enable_thinking
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, **base_kwargs, enable_thinking=not no_think
                )
            except TypeError:
                logger.debug(
                    "enable_thinking not supported for %s — falling back to default template",
                    model_path.split("/")[-1],
                )
                prompt = tokenizer.apply_chat_template(messages, **base_kwargs)

        # ── Ninja patch guard — enforce think state at raw string level ──
        # Belt+suspenders on top of enable_thinking kwargs. Works with all template
        # variants: standard Qwen3, Qwen3.6 ninja patch, and any broken fallback.
        #
        # no_think=True:
        #   standard Qwen3  → ends with </think>\n\n            → no-op (already closed)
        #   ninja patch     → ends with <|im_start|>assistant\n → inject empty block (*)
        #   broken template → ends with <think>\n               → force-close
        #
        # (*) Without seeding, Qwen3.6 spontaneously generates <think> as its first token
        # regardless of enable_thinking=False — the model defaults to thinking mode when it
        # sees a bare assistant prefix. An explicit empty block prevents this.
        #
        # no_think=False:
        #   standard/ninja  → ends with <think>\n               → no-op (already open)
        #   thinking_budget → ends with <budget_remaining>N</budget_remaining>\n → no-op
        #   broken fallback → ends without open <think>         → force-open
        #   DWQ + budget    → ends with <think>\n</think>\n\n   → force-open (budget lost
        #                     but generation proceeds; ninja patch re-opens the block)
        if no_think:
            if prompt.endswith("<think>\n") or prompt.rstrip().endswith("<think>"):
                # Template opened think block but didn't close it → force-close
                prompt = prompt.rstrip("\n") + "\n</think>\n\n"
            elif "</think>" not in prompt[-30:]:
                # No think block at all (ninja patch) → inject empty block
                prompt = prompt.rstrip("\n") + "\n<think>\n\n</think>\n\n"
            # else: standard template already closed the block → no-op
        else:
            # Check for an unclosed <think> block in the last 100 chars.
            # Handles both the plain "<think>\n" case and the thinking_budget
            # "<think>\n<budget_remaining>N</budget_remaining>\n" case.
            _tail = prompt[-100:]
            _think_open = "<think>" in _tail and "</think>" not in _tail[_tail.rfind("<think>"):]
            if not _think_open:
                prompt = prompt.rstrip("\n") + "\n<think>\n"

        # Diagnostic — vérifier que thinking_budget est bien injecté par le tokenizer.
        # Ignoré pour Qwen3.6 : l'absence de <budget_remaining> est intentionnelle
        # (le modèle n'a pas été entraîné avec ce mécanisme — voir commentaire ci-dessus).
        if thinking_budget > 0 and not no_think and not is_qwen36(model_path):
            tail = prompt[-120:]
            if "<budget_remaining>" in tail:
                logger.info(
                    "_build_prompt thinking_budget=%d → <budget_remaining> injected (tokenizer OK)",
                    thinking_budget,
                )
            else:
                logger.warning(
                    "_build_prompt thinking_budget=%d → <budget_remaining> NOT injected "
                    "(ninja-patch template or DWQ checkpoint); budget hint lost, "
                    "ninja guard keeps <think> open. prompt tail: %r",
                    thinking_budget,
                    tail,
                )
        return prompt

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
    thinking_budget: int = 0,
) -> str:
    """Génération complète (non-streaming). Bloquant — wrapper pour asyncio.to_thread.

    json_response=True + no_think=True : utilise stream_generate avec early-stop
    dès que le premier objet JSON complet est détecté. Évite de générer des centaines
    de tokens superflus après la } finale (explications, markdown, …).
    """
    model, tokenizer = _load_model(model_path)
    prompt = _build_prompt(messages, tokenizer, model_path, no_think, thinking_budget)

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

    # Use call-site temperature if explicitly set (> 0); otherwise fall back to
    # the model profile's recommended value (Qwen3.6: 1.0 think / 0.7 no-think).
    # Passing temperature=0.0 or temperature≤0 means "use profile default".
    effective_temp = (
        temperature
        if temperature > 0
        else (profile.temp_nothink if no_think else profile.temp_think)
    )
    sampler = make_sampler(
        temp=effective_temp,
        top_p=profile.top_p_nothink if no_think else profile.top_p_think,
        top_k=profile.top_k,
        min_p=profile.min_p,
    )
    logits_procs = make_logits_processors(
        repetition_penalty=profile.repetition_penalty,
        repetition_context_size=profile.repetition_context_size,
        frequency_penalty=profile.frequency_penalty,
        frequency_context_size=profile.repetition_context_size,
        presence_penalty=profile.presence_penalty,
        presence_context_size=profile.repetition_context_size,
    )

    early_stopped = False
    _think_already_stripped = False
    _max_stop_len = max((len(t) for t in profile.stop_tokens), default=0)

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
                # Check only the tail (2× max stop-token length) — O(1) per token
                # instead of O(n) on the full accumulated string.
                if _max_stop_len and any(
                    t in raw_so_far[-(_max_stop_len * 2):] for t in profile.stop_tokens
                ):
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
            "_generate_sync: thinking truncated before </think> (model=%s) — returning raw",
            model_short,
        )
        # Retourner le début du raisonnement partiel plutôt que rien :
        result = result.split("<think>", 1)[0].strip() if "<think>" in result else ""
        if not result:
            result = "⚠️ Réponse incomplète (budget de réflexion dépassé). Reformule ou augmente max_tokens."

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
    thinking_budget: int = 0,
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
            thinking_budget,
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
    """
    Inférence async non-streaming.
    Sérialisée par lock par modèle — router et primary indépendants.
    """
    _t0 = time.time()
    logger.debug(
        "[TTFT] call_llm_local_async: waiting for _infer_lock (model=%s)",
        model.split("/")[-1],
    )
    # Signale qu'un appel chat haute priorité est en attente du lock.
    # L'analyzer vérifie ce compteur via call_llm_local_async_bg et cède si > 0.
    with _chat_waiters_lock:
        global _chat_waiters
        _chat_waiters += 1
    try:
        await asyncio.to_thread(_infer_lock.acquire)
    finally:
        with _chat_waiters_lock:
            _chat_waiters -= 1
    logger.debug(
        "[TTFT] call_llm_local_async: lock acquired — waited %.3fs", time.time() - _t0
    )
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
            thinking_budget,
        )
        logger.debug(
            "[TTFT] call_llm_local_async: done — total %.3fs", time.time() - _t0
        )
        return result
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
    """
    Inférence async non-streaming — priorité basse (tâches background).

    Cède le GPU quand un appel chat est en attente (_chat_waiters > 0) :
      - Tente d'acquérir _infer_lock sans bloquer (acquire(blocking=False))
      - Si le lock est libre ET aucun chat en attente → acquiert immédiatement
      - Sinon → attend 2s et réessaie (yield vers l'event loop entre les tentatives)

    Garantie : un appel chat ne sera jamais bloqué par l'analyzer.
    Limite : si l'analyzer est déjà en génération, le chat doit attendre la fin
    du token courant (inévitable — le GPU Metal ne supporte pas la préemption).
    """
    _t0 = time.time()
    while True:
        with _chat_waiters_lock:
            if _chat_waiters == 0:
                acquired = _infer_lock.acquire(blocking=False)
                if acquired:
                    break
        # Chat en attente ou lock pris — céder et réessayer
        _waited = time.time() - _t0
        if _waited > 5.0 and int(_waited) % 10 == 0:
            logger.debug("[BG-INFER] waiting for GPU (chat priority) — %.0fs", _waited)
        await asyncio.sleep(2.0)

    logger.debug("[BG-INFER] lock acquired after %.3fs", time.time() - _t0)
    try:
        return await asyncio.to_thread(
            _generate_sync,
            model,
            messages,
            temperature,
            max_tokens,
            no_think,
            session_id,
            json_response,
            thinking_budget,
        )
    finally:
        _infer_lock.release()


async def stream_local(
    messages: list[dict],
    model: str,
    temperature: float | None = None,  # None → utilise temp_think/nothink du profil
    max_tokens: int = 10000,
    no_think: bool = False,
    session_id: str = "",
    thinking_budget: int = 0,
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
    # Set by the finally block when the caller (SSE generator) is cancelled so the
    # worker thread can stop generating tokens instead of running to completion.
    stop_flag = threading.Event()

    _t_lock_wait = time.time()
    logger.debug(
        "[TTFT] stream_local: waiting for _infer_lock (model=%s)", model.split("/")[-1]
    )

    def _worker():
        _t_infer = time.time()
        logger.debug(
            "[TTFT] stream_local: inference started (lock held %.3fs)",
            _t_infer - _t_lock_wait,
        )
        mlx_model, tokenizer = _load_model(model)
        prompt = _build_prompt(messages, tokenizer, model, no_think, thinking_budget)

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
        # Budget vient du pipeline : 1500 (no_think) / 3000 (synthesis) / 4000 (reasoning).
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
                    frequency_context_size=profile.repetition_context_size,
                    presence_penalty=profile.presence_penalty,
                    presence_context_size=profile.repetition_context_size,
                ),
                **({"prompt_cache": kv_cache} if kv_cache is not None else {}),
                **quant_kwargs,
            ):
                if stop_flag.is_set():
                    break

                if chunk.text:
                    text = chunk.text
                    stop_hit = False

                    # Manual stop-token check for models that need it (e.g. Hermes).
                    if profile.stop_tokens:
                        acc = "".join(raw_chunks) + text
                        for st in profile.stop_tokens:
                            if st in acc:
                                text = acc.split(st, 1)[0][len("".join(raw_chunks)) :]
                                stop_hit = True
                                break

                    raw_chunks.append(text)

                    if first and text:
                        logger.debug(
                            "[TTFT] stream_local: first token generated — %.3fs since inference start",
                            time.time() - _t_infer,
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

    # Signale qu'un appel chat haute priorité est en attente du lock.
    with _chat_waiters_lock:
        global _chat_waiters
        _chat_waiters += 1
    try:
        await asyncio.to_thread(_infer_lock.acquire)
    finally:
        with _chat_waiters_lock:
            _chat_waiters -= 1
    logger.debug(
        "[TTFT] stream_local: lock acquired — waited %.3fs", time.time() - _t_lock_wait
    )
    try:
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
    finally:
        # Signal the worker to stop (no-op if it already finished normally).
        # This prevents the MLX generation loop from continuing to run after a
        # client disconnect (CancelledError), which would otherwise hold the
        # GPU busy and make a second concurrent inference possible.
        stop_flag.set()
        _infer_lock.release()


# ── Vision model (mlx_vlm) ────────────────────────────────────────────────────
# Lazy-loaded on first image request.
# Uses its own _vlm_lock — NOT _infer_lock — because:
#   1. VLM and text inference are sequential in the pipeline (describe_images
#      runs before Qwen3.6), so they never compete for the GPU.
#   2. Holding _infer_lock during VLM load + Metal kernel compilation (can take
#      several minutes on first run) blocks background tasks indefinitely.
#   3. _vlm_lock still serialises concurrent image requests correctly.

_vlm_model = None
_vlm_processor = None
_vlm_config = None
_vlm_lock = threading.Lock()


def _load_vlm() -> None:
    """Load the VLM model once (must be called while _infer_lock is held)."""
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
    """
    Synchronous VLM inference — run via asyncio.to_thread() only.
    image_parts: list of resolved OpenAI image_url dicts
      {"type": "image_url", "image_url": {"url": "data:…;base64,…" | "https://…"}}
    """
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
    formatted = vlm_apply_chat_template(
        _vlm_processor, _vlm_config, prompt_text, num_images=len(images)
    )
    result = vlm_generate(
        _vlm_model,
        _vlm_processor,
        formatted,
        image=images[0] if len(images) == 1 else images,
        max_tokens=1200,
        temperature=0.7,
        repetition_penalty=1.3,
        repetition_context_size=64,
        verbose=False,
    )
    return result.text if hasattr(result, "text") else str(result)


async def describe_images_local(image_parts: list, text_prompt: str) -> str:
    """
    Async entry point for local VLM inference.
    Uses _vlm_lock (not _infer_lock) so VLM load/compilation never blocks
    background text inference tasks.
    """
    await asyncio.to_thread(_vlm_lock.acquire)
    try:
        return await asyncio.to_thread(_describe_images_sync, image_parts, text_prompt)
    finally:
        _vlm_lock.release()

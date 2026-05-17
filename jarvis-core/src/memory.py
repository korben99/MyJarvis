"""
PROJECT JARVIS v8
Jarvis Memory System
====================
- Working memory: Redis (current session, mood, active context)
- Semantic memory: Redis hashes (user profile, preferences, projects)
- Episodic memory: Qdrant (timestamped conversation summaries)
- Self memory: JSON file (Jarvis identity and growth)
Redis
   short-term

Qdrant
   episodic memory
   autobiographical memory

compression
   knowledge abstraction

conversation → analyzer
             → memory_summary
             → episodic memory
             → autobiographical memory

Compression memory added, to be runned every month.


"""

import asyncio
import json
import os
import re
import tempfile
import time
import uuid
from datetime import date, datetime, timezone
from threading import Lock, Thread

import numpy as np

from config import (
    AUTOBIO_DEDUP_THRESHOLD,
    AUTOBIO_RECENCY_WINDOW_DAYS,
    CHAT_LOG_TTL,
    CHAT_MAX_MESSAGES,
    DONE_PROJECT_TTL_DAYS,
    EMBED_MODEL_NAME,
    EPISODIC_RETENTION_DAYS,
    IMPORTANCE_THRESHOLD,
    MEMORY_CONSOLIDATION_IMPORTANCE,
    MEMORY_DECAY_DURABLE_MIN,
    MEMORY_DECAY_FACTOR,
    MEMORY_DECAY_THRESHOLD,
    NOVELTY_THRESHOLD,
    MAX_TOKENS_COMPACT,
    MAX_TOKENS_SHORT,
    PRIMARY_API_KEY,
    PRIMARY_API_URL,
    PRIMARY_MODEL,
    PRIMARY_TIMEOUT,
    QDRANT_COLLECTION,
    QDRANT_MEMORY_COLLECTION,
    RAG_SCORE_THRESHOLD,
    RAG_TOP_K,
    RECALL_MEMORY_SIMILARITY_THRESHOLD,
    ROUTER_API_KEY,
    ROUTER_API_URL,
    ROUTER_MODEL,
    ROUTER_TIMEOUT,
    SELF_MEMORY_PATH,
    USER_CODES,
    llm_timeout,
)
from helpers import (
    call_llm,
    extract_llm_json,
    get_logger,
    get_qdrant,
    get_redis,
    normalize_key,
    redis_get_json,
    redis_set_json,
    rel_time_fr,
)
from prompts import get_prompt
from qdrant_client.models import PointIdsList
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


# ══════════════════════════════════════════════════
#  WORKING MEMORY — Current state, volatile
# ══════════════════════════════════════════════════


def get_working_memory(session_id: str) -> dict:
    """Get current session state."""
    r = get_redis()
    data = r.hgetall(f"working:{session_id}")
    return data or {}


def set_working_memory(session_id: str, key: str, value: str, ttl: int = 86400):
    """Set a working memory value. Expires after ttl seconds (default 24h)."""
    r = get_redis()
    r.hset(f"working:{session_id}", key, value)
    r.expire(f"working:{session_id}", ttl)


_CONCERN_DECAY_PER_HOUR = 0.05  # concern loses 0.05/h → fully decayed in 20h
_ENERGY_DECAY_PER_HOUR = 0.02  # energy drifts back to 0.7 baseline at 0.02/h
_emotional_state_lock = Lock()  # guards the read-modify-write decay cycle


def _apply_decay_unlocked(state: dict, elapsed_h: float) -> bool:
    """Apply concern/energy time-decay to *state* in-place. Returns True if anything changed."""
    changed = False
    concern = state.get("concern", 0.0)
    if concern > 0:
        new_concern = max(0.0, concern - elapsed_h * _CONCERN_DECAY_PER_HOUR)
        if abs(new_concern - concern) > 0.001:
            state["concern"] = round(new_concern, 3)
            changed = True
    energy = state.get("energy", 0.7)
    if abs(energy - 0.7) > 0.001:
        direction = 1 if energy < 0.7 else -1
        new_energy = energy + direction * elapsed_h * _ENERGY_DECAY_PER_HOUR
        new_energy = min(0.7, new_energy) if direction == 1 else max(0.7, new_energy)
        if abs(new_energy - energy) > 0.001:
            state["energy"] = round(new_energy, 3)
            changed = True
    return changed


def get_emotional_state() -> dict:
    """Get Jarvis's current emotional state, with time-based decay applied.

    concern decays toward 0.0 at 0.05/h (fully cleared in ~20h without new stress).
    energy drifts toward 0.7 baseline at 0.02/h.
    Decay is applied lazily on read — no scheduler needed.
    """
    _default = {
        "mood": "neutral",
        "energy": 0.7,
        "confidence": 0.8,
        "curiosity": 0.6,
        "concern": 0.0,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    with _emotional_state_lock:
        state = redis_get_json("jarvis:emotional_state")
        if not state:
            return _default

        last_updated = state.get("last_updated")
        if last_updated:
            try:
                elapsed_h = max(
                    0.0,
                    (
                        datetime.now(timezone.utc)
                        - datetime.fromisoformat(last_updated)
                    ).total_seconds()
                    / 3600,
                )
                if _apply_decay_unlocked(state, elapsed_h):
                    state["last_updated"] = datetime.now(timezone.utc).isoformat()
                    redis_set_json("jarvis:emotional_state", state)

            except (ValueError, TypeError):
                pass

    return state


def update_emotional_state(updates: dict):
    """Update emotional state atomically — decay applied then updates merged, all under lock."""
    with _emotional_state_lock:
        state = redis_get_json("jarvis:emotional_state") or {
            "mood": "neutral",
            "energy": 0.7,
            "confidence": 0.8,
            "curiosity": 0.6,
            "concern": 0.0,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        last_updated = state.get("last_updated")
        if last_updated:
            try:
                elapsed_h = max(
                    0.0,
                    (datetime.now(timezone.utc) - datetime.fromisoformat(last_updated)).total_seconds() / 3600,
                )
                _apply_decay_unlocked(state, elapsed_h)
            except (ValueError, TypeError):
                pass
        state.update(updates)
        state["last_updated"] = datetime.now(timezone.utc).isoformat()
        for key in ["energy", "confidence", "curiosity", "concern"]:
            if key in state and isinstance(state[key], (int, float)):
                state[key] = max(0.0, min(1.0, state[key]))
        redis_set_json("jarvis:emotional_state", state)


# ══════════════════════════════════════════════════
#  SESSION MEMORY — Current conversation (200 message max)
# ══════════════════════════════════════════════════


_CHAT_LOG_TTL = CHAT_LOG_TTL  # configured in config.py (default 90 days)


def append_conversation_message(
    user_code: str, session_id: str, role: str, content: str
):
    r = get_redis()
    key = f"chat:{user_code}:{session_id}"

    entry = {"role": role, "content": content, "ts": time.time()}

    pipe = r.pipeline()
    pipe.rpush(key, json.dumps(entry))
    pipe.ltrim(key, -CHAT_MAX_MESSAGES, -1)
    pipe.expire(key, _CHAT_LOG_TTL)  # sliding TTL — resets on every message
    pipe.execute()


def get_conversation(user_code: str, session_id: str, limit: int | None = None):
    r = get_redis()
    key = f"chat:{user_code}:{session_id}"

    start = -limit if limit else 0
    entries = r.lrange(key, start, -1)

    return [json.loads(e) for e in entries]


# ══════════════════════════════════════════════════
#  SEMANTIC MEMORY — Long-term knowledge about user
# ══════════════════════════════════════════════════

"""Get everything Jarvis knows about the user."""

# ── Profile key dedup helpers ─────────────────────────────────────────────

# Scalar aliases: O(1) resolution before any LLM call
_SCALAR_CANONICAL: dict[str, str] = {
    "ville": "location",
    "city": "location",
    "metier": "profession",
    "emploi": "profession",
    "employeur": "current_employer",
    "entreprise": "current_employer",
    "societe": "current_employer",
    "company": "current_employer",
    "prenom": "name",
    "prénom": "name",
    "revenu": "capital",
    "patrimoine": "capital",
    "inquietude": "concerns",
    "voyages_prevus": "travel_plans",
}

# Namespace families: keys in the same family are compared together for dedup.
# Each family is isolated — namespaces from different families are NEVER compared.
_NS_FAMILY: dict[str, frozenset] = {
    "hobby":         frozenset({"hobby", "interest", "loisir", "passion", "activite", "sport"}),
    "skill":         frozenset({"skill", "competence"}),
    "tech":          frozenset({"technologie", "outil"}),
    "placement":     frozenset({"placement", "investissement", "epargne"}),
    "projet":        frozenset({"projet", "project"}),
    "preoccupation": frozenset({"preoccupation", "concerns", "inquietude"}),
    "etude":         frozenset({"etude", "matiere", "specialite", "option", "note"}),
    # situation and famille are intentionally NOT grouped together:
    # situation = facts about the user, famille = facts about third parties.
}


def _key_prefix(key: str) -> str | None:
    return key.split(":")[0] if ":" in key else None


def _profile_key_fast_match(new_key: str, existing_keys: list[str]) -> str | None:
    """Stages 0–1 of profile key dedup — no LLM call.

    Stage 0: case/accent-insensitive exact match via normalize_key().
    Stage 1: scalar canonical alias lookup (_SCALAR_CANONICAL dict).

    Returns the matching existing key to evict, or None.
    Called by both _normalize_profile_key (single) and _normalize_profile_keys_batch.
    """
    new_key_norm = normalize_key(new_key)
    # Stage 0: case/accent-insensitive exact match
    for k in existing_keys:
        if normalize_key(k) == new_key_norm and k != new_key:
            return k
    # Stage 1: scalar canonical alias (new_key_norm already lowercased + normalised)
    canonical = _SCALAR_CANONICAL.get(new_key_norm)
    if canonical and canonical in existing_keys:
        return canonical
    return None


def _candidate_keys(new_key: str, existing_keys: list[str]) -> list[str]:
    """Narrow the dedup candidate set to the same namespace family."""
    new_prefix = _key_prefix(new_key)
    if new_prefix is None:
        return [k for k in existing_keys if _key_prefix(k) is None]
    family = next(
        (members for members in _NS_FAMILY.values() if new_prefix in members),
        frozenset({new_prefix}),
    )
    return [k for k in existing_keys if _key_prefix(k) in family]


def _normalize_profile_keys_batch(
    user_code: str, new_keys: list[str], existing_keys: list[str]
) -> dict[str, str | None]:
    """
    Batch version of _normalize_profile_key.
    Returns {new_key: existing_key_to_evict_or_None} for all new_keys.

    Fast paths (stages 0-1) are applied per key with no LLM.
    Remaining unresolved keys are grouped by prefix family and sent in a
    single LLM call per group — O(families) instead of O(keys).
    """
    result: dict[str, str | None] = {k: None for k in new_keys}
    unresolved: list[str] = []

    for new_key in new_keys:
        if new_key in existing_keys:
            continue  # exact match already present — just overwrite, no eviction
        fast = _profile_key_fast_match(new_key, existing_keys)
        if fast:
            logger.info(
                "User %s profile key '%s' → fast match '%s' (no LLM)",
                user_code,
                new_key,
                fast,
            )
            result[new_key] = fast
            continue
        unresolved.append(new_key)

    if not unresolved:
        return result

    # Stage 2: group unresolved by prefix family, one LLM call per group
    groups: dict[str, list[str]] = {}
    for new_key in unresolved:
        prefix = _key_prefix(new_key)
        family_key = next(
            (fk for fk, members in _NS_FAMILY.items() if prefix in members),
            prefix or "__none__",
        )
        groups.setdefault(family_key, []).append(new_key)

    for _family, group_keys in groups.items():
        seen: set[str] = set()
        candidates: list[str] = []
        for new_key in group_keys:
            for c in _candidate_keys(new_key, existing_keys):
                if c not in seen:
                    seen.add(c)
                    candidates.append(c)
        if not candidates:
            continue

        try:
            keys_list = ", ".join(f'"{k}"' for k in candidates)
            new_list = ", ".join(f'"{k}"' for k in group_keys)
            # Response wrapped in {"matches": [...]} so extract_llm_json (object-only)
            # can parse it without modification.
            prompt = (
                f"Clés existantes : [{keys_list}]\n"
                f"Nouvelles clés  : [{new_list}]\n\n"
                "Pour chaque nouvelle clé, indique le doublon exact parmi les existantes.\n"
                "RÈGLE STRICTE : doublon = même sujet ET même concept. Même préfixe namespace ≠ doublon.\n"
                "Si le moindre doute → null.\n"
                'Réponds : {"matches": [{"new": "clé", "match": "existante_ou_null"}, ...]}'
            )
            raw = call_llm(
                [
                    {
                        "role": "system",
                        "content": (
                            "Tu es un détecteur de doublons de clés de profil. "
                            "Réponds UNIQUEMENT avec du JSON valide.\n"
                            "RÈGLE : deux clés sont des doublons UNIQUEMENT si elles décrivent "
                            "le MÊME sujet ET le MÊME concept. Même préfixe ≠ doublon.\n"
                            "Exemples VRAIS doublons (namespace synonyme, même sujet) :\n"
                            '  existantes: ["hobby:kart"] nouvelles: ["loisir:kart"] '
                            '→ {"matches": [{"new": "loisir:kart", "match": "hobby:kart"}]}\n'
                            "Exemples NON doublons (même préfixe, sujets ou concepts différents) :\n"
                            '  existantes: ["situation:parents_location"] nouvelles: ["situation:lieu_residence"] '
                            '→ {"matches": [{"new": "situation:lieu_residence", "match": null}]}\n'
                            '  existantes: ["competence:ia"] nouvelles: ["competence:bricolage"] '
                            '→ {"matches": [{"new": "competence:bricolage", "match": null}]}\n'
                            '  existantes: ["hobby:kart"] nouvelles: ["hobby:tennis"] '
                            '→ {"matches": [{"new": "hobby:tennis", "match": null}]}'
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=PRIMARY_MODEL,
                api_url=PRIMARY_API_URL,
                api_key=PRIMARY_API_KEY,
                temperature=0.0,
                max_tokens=MAX_TOKENS_SHORT,
                json_response=True,
                no_think=True,
                timeout=ROUTER_TIMEOUT,
            )
            parsed = extract_llm_json(raw)
            matches = parsed.get("matches") if isinstance(parsed, dict) else None
            if not isinstance(matches, list):
                logger.warning(
                    "Batch profile key normalization: unexpected response format for group (%s)",
                    ", ".join(group_keys),
                )
            else:
                for item in matches:
                    nk = item.get("new")
                    match = item.get("match")
                    # Skip if already resolved (model may return duplicate `new` entries).
                    # result[nk] is None for unresolved keys; non-None means fast path or
                    # a prior item in this loop already set it — don't overwrite.
                    if (
                        nk in group_keys
                        and result.get(nk) is None
                        and match
                        and match in existing_keys
                    ):
                        logger.info(
                            "User %s profile key batch '%s' deduped → '%s'",
                            user_code,
                            nk,
                            match,
                        )
                        result[nk] = match
        except Exception as exc:
            logger.warning(
                "Batch profile key normalization failed for group (%s): %s",
                ", ".join(group_keys),
                exc,
            )

    return result


def get_user_profile(user_code: str) -> dict:
    r = get_redis()
    data = r.hgetall(f"user:{user_code}:profile")
    return data or {}


def _normalize_profile_key(
    user_code: str, new_key: str, existing_keys: list[str]
) -> str | None:
    """
    Find whether new_key is semantically equivalent to an existing profile key.
    Returns the existing key to evict, or None if new_key is genuinely new.

    Three-stage pipeline (cheapest first):
      1. Verbatim match          — O(1), no LLM
      2. Scalar canonical alias  — O(1), no LLM
      3. Category-aware LLM      — only within the same namespace family
    """
    if not existing_keys or new_key in existing_keys:
        return None

    # Stages 0–1: fast path (no LLM)
    fast = _profile_key_fast_match(new_key, existing_keys)
    if fast:
        logger.info(
            "User %s profile key '%s' → fast match '%s' (no LLM)",
            user_code,
            new_key,
            fast,
        )
        return fast

    # Stage 2: category-aware LLM on reduced candidate set
    candidates = _candidate_keys(new_key, existing_keys)
    if not candidates:
        return None

    try:
        keys_list = ", ".join(f'"{k}"' for k in candidates)
        prompt = (
            f"Clés existantes (même catégorie) : [{keys_list}]\n"
            f'Nouvelle clé : "{new_key}"\n\n'
            f'Est-ce un doublon ? Réponds : {{"match": "clé_existante"}} ou {{"match": null}}'
        )

        raw = call_llm(
            [
                {
                    "role": "system",
                    "content": (
                        "Tu es un détecteur de doublons de clés de profil. "
                        "Réponds UNIQUEMENT avec du JSON valide.\n"
                        "RÈGLE : deux clés sont des doublons UNIQUEMENT si elles décrivent "
                        "le MÊME sujet ET le MÊME concept. Même préfixe ≠ doublon.\n"
                        "Exemples VRAIS doublons (namespace synonyme, même sujet) :\n"
                        '  "hobby:kart" vs "loisir:kart" → {"match": "hobby:kart"}\n'
                        '  "hobby:ia" vs "interest:ia" → {"match": "hobby:ia"}\n'
                        "Exemples NON doublons (même préfixe, sujets ou concepts différents) :\n"
                        '  "situation:parents_location" vs "situation:lieu_residence" → {"match": null}\n'
                        '  "competence:ia" vs "competence:bricolage" → {"match": null}\n'
                        '  "hobby:kart" vs "hobby:tennis" → {"match": null}'
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=PRIMARY_MODEL,
            api_url=PRIMARY_API_URL,
            api_key=PRIMARY_API_KEY,
            temperature=0.0,
            max_tokens=MAX_TOKENS_SHORT,
            json_response=True,
            no_think=True,
            timeout=ROUTER_TIMEOUT,
        )

        parsed = extract_llm_json(raw)
        match = parsed.get("match")

        if match and match in existing_keys:
            logger.info(
                "User %s profile key '%s' deduped → '%s'", user_code, new_key, match
            )
            return match

        return None

    except Exception as exc:
        logger.warning("Profile key normalization skipped (%s): %s", new_key, exc)
        return None


def _write_profile_fact(
    r,
    profile_key: str,
    ts_key: str,
    user_code: str,
    key: str,
    value: str,
    duplicate: str | None,
    now_ts: int,
) -> None:
    """Apply a single profile key write (with optional duplicate eviction)."""
    if duplicate:
        old_dup_val = r.hget(profile_key, duplicate)
        logger.info(
            "User %s profile key normalized: '%s' (was: %s) → replaced by '%s'",
            user_code,
            duplicate,
            old_dup_val or "(empty)",
            key,
        )
        r.hdel(profile_key, duplicate)
        r.hdel(ts_key, duplicate)
    r.hset(profile_key, key, value)
    r.hset(ts_key, key, now_ts)
    logger.info("User %s profile updated: %s = %s", user_code, key, value)


def update_user_profile(user_code: str, key: str, value: str | None):
    """Add, update, or delete (value=None or "") a user profile fact.

    Preventive duplicate guard: before writing a new key, the router LLM checks
    whether it is semantically equivalent to an existing key.  If a match is found,
    the old key is deleted before the new one is written, preventing profile bloat.

    Every write/delete is mirrored to the shadow timestamp hash
    user:{user_code}:profile:ts so that curative cleanup can reason about recency.
    """
    r = get_redis()
    profile_redis_key = f"user:{user_code}:profile"
    profile_ts_key = f"user:{user_code}:profile:ts"

    if not value:  # None or empty string → delete
        old_val = r.hget(profile_redis_key, key)
        r.hdel(profile_redis_key, key)
        r.hdel(profile_ts_key, key)
        logger.info(
            "User %s profile deleted: %s (was: %s)",
            user_code,
            key,
            old_val or "(empty)",
        )
    else:
        if ":" in key:
            ns = key.split(":")[0]
            if ns not in _ALLOWED_PROFILE_NAMESPACES:
                logger.warning(
                    "update_user_profile: rejected unauthorized namespace key %r for %s",
                    key, user_code,
                )
                return

        existing_keys = r.hkeys(profile_redis_key)

        # Key normalization: if new_key is semantically equivalent to an existing key
        # (same concept or same category:item under a synonym category), evict the old
        # key and write under the new name — no value merging, each key is atomic.
        duplicate = _normalize_profile_key(user_code, key, existing_keys)
        _write_profile_fact(
            r,
            profile_redis_key,
            profile_ts_key,
            user_code,
            key,
            value,
            duplicate,
            int(time.time()),
        )


# Whitelist of authorized profile namespaces (prefix before ":").
# Scalar keys with no prefix (name, travel_preference…) are always allowed.
# Any key with a prefix NOT in this set is silently rejected.
_ALLOWED_PROFILE_NAMESPACES = frozenset({
    "situation", "famille", "profession", "competence", "loisir", "sport",
    "technologie", "sante", "objectif", "etude", "placement", "preference",
    "interet", "apprecie", "aversion", "langue",
})

# Hard cap: never persist more than this many new facts per analyzer call.
_MAX_FACTS_PER_BATCH = 6


def update_user_profile_batch(user_code: str, facts: list[dict]) -> None:
    """
    Apply a list of profile facts in one batch:
    - single Redis hkeys read shared across all facts
    - one LLM dedup call per prefix family (instead of one per key)
    - all writes applied sequentially after dedup resolution
    """
    if not facts:
        return

    # Guard: only allow explicitly authorized namespaces.
    # Scalar keys (no ":") are always allowed.
    filtered = []
    for f in facts:
        k = f.get("key", "")
        ns = k.split(":")[0] if ":" in k else ""
        if ns and ns not in _ALLOWED_PROFILE_NAMESPACES:
            logger.warning(
                "update_user_profile_batch: rejected unauthorized namespace key '%s' for %s",
                k, user_code,
            )
            continue
        filtered.append(f)
    facts = filtered

    # Hard cap: guard against analyzer returning an implausibly large fact list.
    if len(facts) > _MAX_FACTS_PER_BATCH:
        logger.warning(
            "update_user_profile_batch: %d facts exceeds cap %d for %s — truncating",
            len(facts), _MAX_FACTS_PER_BATCH, user_code,
        )
        facts = facts[:_MAX_FACTS_PER_BATCH]

    r = get_redis()
    profile_redis_key = f"user:{user_code}:profile"
    profile_ts_key = f"user:{user_code}:profile:ts"

    existing_keys = r.hkeys(profile_redis_key)

    new_facts = [f for f in facts if "key" in f and f.get("value")]
    delete_facts = [f for f in facts if "key" in f and not f.get("value")]

    dedup_map = (
        _normalize_profile_keys_batch(
            user_code, [f["key"] for f in new_facts], existing_keys
        )
        if new_facts
        else {}
    )

    now_ts = int(time.time())

    for fact in delete_facts:
        key = fact["key"]
        old_val = r.hget(profile_redis_key, key)
        r.hdel(profile_redis_key, key)
        r.hdel(profile_ts_key, key)
        logger.info(
            "User %s profile deleted: %s (was: %s)",
            user_code,
            key,
            old_val or "(empty)",
        )

    # Track evicted keys so that if two new facts resolve to the same existing key,
    # only the first write evicts it — subsequent writes skip the already-gone eviction.
    already_evicted: set[str] = set()
    for fact in new_facts:
        dup = dedup_map.get(fact["key"])
        if dup in already_evicted:
            dup = None  # eviction already done by a prior fact in this batch
        _write_profile_fact(
            r,
            profile_redis_key,
            profile_ts_key,
            user_code,
            fact["key"],
            fact["value"],
            dup,
            now_ts,
        )
        if dup:
            already_evicted.add(dup)


def set_interest_weight(user_code: str, term: str, weight: float):
    """
    Set the importance weight for an interest term (0.0 = forgotten, 1.0 = normal, 2.0 = top).
    Weight=0 effectively removes the term from briefing and news queries.
    """
    r = get_redis()
    r.hset(f"user:{user_code}:interest_weights", term.lower(), str(weight))
    logger.info("User %s interest weight: %s = %.1f", user_code, term, weight)


def get_interest_weights(user_code: str) -> dict[str, float]:
    """Return {term: weight} dict. Missing terms default to 1.0."""
    r = get_redis()
    raw = r.hgetall(f"user:{user_code}:interest_weights")
    return {k: float(v) for k, v in raw.items()}


def get_user_projects(user_code: str) -> list:
    """Return the user's project list from Redis."""
    return redis_get_json(f"user:{user_code}:projects", default=[])


def update_user_projects(user_code: str, projects: list):
    """Persist the project list.
    Done projects older than DONE_PROJECT_TTL_DAYS are dropped.
    Schema: {name, status, first_mentioned, last_update, description?, updates?}
    updates: [{date: "YYYY-MM-DD", summary: "..."}], capped at 20 FIFO.
    """
    cutoff = datetime.now(timezone.utc).timestamp() - DONE_PROJECT_TTL_DAYS * 86400
    result = []
    for p in projects:
        if p.get("status") == "done" and p.get("last_update"):
            try:
                if datetime.fromisoformat(p["last_update"]).timestamp() < cutoff:
                    continue
            except (ValueError, TypeError):
                pass
        entry = {
            "name": p["name"],
            "status": p.get("status", "in_progress"),
            "first_mentioned": p.get("first_mentioned"),
            "last_update": p.get("last_update"),
        }
        if p.get("description"):
            entry["description"] = p["description"]
        if p.get("updates"):
            entry["updates"] = p["updates"][-20:]
        result.append(entry)
    redis_set_json(f"user:{user_code}:projects", result)


def get_project_detail(user_code: str, project_name: str) -> dict | None:
    """Return the full project dict (with updates timeline) by fuzzy name match."""
    projects = get_user_projects(user_code)
    project_map = {p["name"]: p for p in projects}
    if project_name in project_map:
        return project_map[project_name]
    resolved = _fuzzy_project_name(project_name, project_map, threshold=0.4)
    return project_map.get(resolved) if resolved else None


def get_project_timeline_text(project: dict) -> str:
    """Format a project's timeline for prompt injection."""
    status_fr = {"done": "terminé", "in_progress": "en cours", "active": "en cours"}.get(
        project.get("status", ""), project.get("status", "en cours")
    )
    lines = [f"Projet : {project['name']} ({status_fr})"]
    updates = project.get("updates") or []
    if updates:
        lines.append("Historique :")
        for u in updates:
            d = u.get("date", "?")
            s = u.get("summary", "")
            if s:
                lines.append(f"  {d} : {s}")
    elif project.get("description"):
        lines.append(f"Description : {project['description']}")
    return "\n".join(lines)


def get_user_preferences(user_code: str) -> dict:
    """Get user preferences (language, style, etc.)."""
    r = get_redis()
    data = r.hgetall(f"user:{user_code}:preferences")
    return data or {}


def update_user_preference(user_code: str, key: str, value: str):
    """Update a preference."""
    r = get_redis()
    r.hset(f"user:{user_code}:preferences", key, value)


# ══════════════════════════════════════════════════
#  EPISODIC MEMORY — Conversation history + summaries
# ══════════════════════════════════════════════════


def log_conversation(
    user_code: str,
    session_id: str,
    user_msg: str,
    assistant_msg: str,
    mood: str = "neutral",
    topics: list = None,
    importance=0,
    memory_summary: str = None,
):
    """Log a conversation exchange to episodic memory."""
    r = get_redis()
    _now = time.time()  # single call — score and timestamp stay consistent
    entry = {
        "timestamp": _now,
        "session_id": session_id,
        "user": user_msg[:500],  # Truncate for storage
        "assistant": assistant_msg[:500],
        "mood": mood,
        "topics": topics or [],
        "importance": importance,
        "memory_summary": memory_summary,
        "satisfaction": "unknown",  # back-filled by analyzer (LLM) after each batch
    }
    # Store in a sorted set by timestamp for easy retrieval
    r.zadd(f"convlog:{user_code}", {json.dumps(entry): _now})

    # Keep only last 1000 exchanges (prevent unbounded growth)
    r.zremrangebyrank(f"convlog:{user_code}", 0, -1001)


def get_recent_conversations(user_code: str, hours: int = 24, limit: int = 20) -> list:
    """Get recent conversation exchanges."""
    r = get_redis()
    cutoff = time.time() - (hours * 3600)
    entries = r.zrangebyscore(
        f"convlog:{user_code}", cutoff, "+inf", start=0, num=limit
    )
    result = []
    for e in entries:
        try:
            result.append(json.loads(e))
        except (json.JSONDecodeError, ValueError):
            logger.warning("Skipping corrupted convlog entry for %s", user_code)
    return result


def _normalize_project_name(name: str) -> str:
    """Normalize a project name to a consistent space-separated form.

    Replaces slug-style hyphens (between word chars) with spaces so that
    LLM-generated kebab-case names ("installation-attelage-bmw") and natural
    language names ("installation attelage bmw") are stored identically.
    Em-dashes (—) used as title separators are preserved.
    """
    normalized = re.sub(r"(?<=\w)-(?=\w)", " ", name)
    return re.sub(r" {2,}", " ", normalized).strip()


def _fuzzy_project_name(
    name: str, project_map: dict, threshold: float = 0.6
) -> str | None:
    """Return the best-matching project name from project_map, or None.

    Scoring uses word overlap with two metrics (max is taken):
    - General: overlap / max(|A|, |B|)  — Jaccard-like
    - Subset : overlap / min(|A|, |B|)  — catches subset names ("Jarvis" → "Jarvis v9")

    Tokens are split on spaces, hyphens, and em-dashes so that slugified names
    and natural-language names score identically.

    When two projects tie, in_progress beats done — avoids re-opening a closed
    project when an active one with similar words exists.
    """
    tokenize = lambda s: set(re.split(r"[\s\-—]+", s.lower()))
    words_new = tokenize(name)
    best_match: str | None = None
    best_score = 0.0
    for existing_name, existing_proj in project_map.items():
        words_ex = tokenize(existing_name)
        overlap = len(words_new & words_ex)
        if overlap == 0:
            continue
        general = overlap / max(len(words_new), len(words_ex))
        subset = overlap / min(len(words_new), len(words_ex))
        score = max(general, subset)
        if score < threshold:
            continue
        is_active = existing_proj.get("status") != "done"
        best_is_active = project_map.get(best_match, {}).get("status") != "done" if best_match else False
        if score > best_score or (score == best_score and is_active and not best_is_active):
            best_score = score
            best_match = existing_name
    return best_match


def apply_project_updates(user_code: str, project_events: list[dict]):
    """Apply structured project events from the analyzer.

    Each event is a dict: {name, action, summary, [rename_to]}
      action  : "create" | "update" | "done" | "rename"
      summary : 1-sentence description of what happened (appended to updates timeline)
      rename_to : new name (rename only)
    """
    projects = get_user_projects(user_code)
    now = datetime.now(timezone.utc).isoformat()
    today = now[:10]  # YYYY-MM-DD

    project_map: dict[str, dict] = {p["name"]: p for p in projects}

    for event in project_events:
        if not isinstance(event, dict):
            continue
        action = event.get("action", "").strip()
        name = _normalize_project_name(event.get("name", "").strip())
        summary = (event.get("summary") or "").strip()
        if not name or action not in ("create", "update", "done", "rename"):
            continue

        # Exact match first, then fuzzy — prevents name drift duplicates
        resolved = (
            name
            if name in project_map
            else (_fuzzy_project_name(name, project_map) or name)
        )

        if action == "create":
            if resolved not in project_map:
                soft_match = _fuzzy_project_name(name, project_map, threshold=0.4)
                if soft_match:
                    resolved = soft_match
                    project_map[resolved]["last_update"] = now
                    logger.debug(
                        "Project create: '%s' soft-matched to '%s' — skipping create",
                        name,
                        soft_match,
                    )
                else:
                    project_map[resolved] = {
                        "name": resolved,
                        "status": "in_progress",
                        "first_mentioned": now,
                        "last_update": now,
                        "updates": [],
                    }
            else:
                project_map[resolved]["last_update"] = now

        elif action == "update":
            if resolved not in project_map:
                project_map[resolved] = {
                    "name": resolved,
                    "status": "in_progress",
                    "first_mentioned": now,
                    "last_update": now,
                    "updates": [],
                }
            else:
                project_map[resolved]["last_update"] = now

        elif action == "done":
            if resolved in project_map:
                project_map[resolved]["status"] = "done"
                project_map[resolved]["last_update"] = now
            else:
                logger.warning(
                    "Project done: '%s' not found — possible LLM confabulation (raw='%s')",
                    resolved, name,
                )

        elif action == "rename":
            old_raw = _normalize_project_name(event.get("name", "").strip())
            new_name = _normalize_project_name(event.get("rename_to", "").strip())
            if not new_name:
                continue
            old_resolved = (
                old_raw
                if old_raw in project_map
                else (_fuzzy_project_name(old_raw, project_map) or old_raw)
            )
            if old_resolved in project_map:
                entry = project_map.pop(old_resolved)
                entry["name"] = new_name
                entry["last_update"] = now
                project_map[new_name] = entry
            else:
                logger.warning(
                    "Project rename: '%s' not found — possible LLM confabulation (raw='%s', rename_to='%s')",
                    old_resolved, old_raw, new_name,
                )
            continue  # rename has no summary entry

        # Append summary to updates timeline (all actions except rename)
        if summary and resolved in project_map:
            proj = project_map[resolved]
            proj.setdefault("updates", [])
            proj["updates"].append({"date": today, "summary": summary})
            proj["description"] = summary  # compact context always shows latest

    update_user_projects(user_code, list(project_map.values()))


# ══════════════════════════════════════════════════
#  COMPLETE MEMORY TO QDRANT — Conversation history + summaries + AUTOBIOGRAPHIE
# ══════════════════════════════════════════════════


def compute_memory_novelty(
    user_code: str, text: str, vector: list | None = None, limit: int = 5
):
    """
    Estimate novelty of a memory by comparing it with recent vector memories.
    Returns a value between 0 and 1.

    Pass a pre-computed *vector* to avoid re-encoding the text when the caller
    already has the embedding (e.g. store_memory_vector).
    """
    try:
        model = get_embed_model()
        qdrant = get_qdrant()

        if vector is None:
            vector = model.encode(text, normalize_embeddings=True).tolist()

        results = qdrant.query_points(
            collection_name=QDRANT_MEMORY_COLLECTION,
            query=vector,
            limit=limit,
            query_filter={
                "must": [
                    {"key": "user_code", "match": {"value": user_code}},
                    {
                        "should": [
                            {"key": "memory_type", "match": {"value": "episodic"}},
                            {
                                "key": "memory_type",
                                "match": {"value": "autobiographical"},
                            },
                        ]
                    },
                ],
            },
        ).points

        if not results:
            return 1.0

        # Clamp to [0, 1]: collection uses Distance.DOT, scores can exceed 1.0
        max_similarity = max(min(r.score, 1.0) for r in results)
        return max(0, min(1, 1 - max_similarity))

    except Exception as e:
        logger.error("Novelty computation failed: %s", e)
        return 0.5


def store_memory_vector(user_code: str, entry: dict):
    """Store conversation exchange in vector memory (Qdrant).

    Requires a memory_summary — skips storage if absent.
    Raw exchange text is intentionally not used: embedding geometry between
    a natural-language query and a structured log string degrades recall quality.
    """
    try:
        text = (entry.get("memory_summary") or "").strip()
        if not text:
            logger.debug("store_memory_vector: skipped (no memory_summary)")
            return

        model = get_embed_model()
        qdrant = get_qdrant()
        logger.info("Vector memory candidate: %s", text[:80])
        vector = model.encode(text, normalize_embeddings=True).tolist()

        # #1 — Invariant: vector must be unit-norm before storage
        _norm = float(np.linalg.norm(vector))
        if abs(_norm - 1.0) > 0.01:
            logger.error(
                "[memory_invariant] NON-NORMALIZED vector for user=%s norm=%.4f — re-normalizing",
                user_code, _norm,
            )
            _arr = np.array(vector, dtype=np.float32)
            vector = (_arr / _norm).tolist()

        novelty = compute_memory_novelty(user_code, text, vector=vector)

        # #1 — Invariant: novelty must be in [0, 1]
        if not (0.0 <= novelty <= 1.0):
            logger.error(
                "[memory_invariant] novelty out of range for user=%s novelty=%.4f — clamping",
                user_code, novelty,
            )
            novelty = max(0.0, min(1.0, novelty))

        if novelty < NOVELTY_THRESHOLD:
            # #5 — Structured decision log
            logger.info(
                "[memory_decision] user=%s stored=False type=episodic reason=duplicate novelty=%.3f summary=%r",
                user_code, novelty, text[:80],
            )
            return

        # Deterministic ID: same (user_code, text) always produces the same UUID.
        # Qdrant upsert with an existing ID silently overwrites the point,
        # preventing duplicate entries when the same memory is stored twice.
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{user_code}:{text}"))
        importance = entry.get("importance", 0)

        qdrant.upsert(
            collection_name=QDRANT_MEMORY_COLLECTION,
            points=[
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "user_code": user_code,
                        "memory_type": "episodic",
                        "importance": importance,
                        "session_id": entry["session_id"],
                        "text": text,
                        "timestamp": entry["timestamp"],
                        "topics": entry.get("topics", []),
                        "mood": entry["mood"],
                        "novelty": novelty,
                    },
                }
            ],
        )
        # #5 — Structured decision log
        logger.info(
            "[memory_decision] user=%s stored=True type=episodic novelty=%.3f importance=%.2f summary=%r",
            user_code, novelty, importance, text[:80],
        )

    except Exception as e:
        logger.error("Vector memory store failed: %s", e)


def store_autobiographical_event(user_code: str, summary: str, importance: float):
    """
    Store a major life / project milestone for the user.

    Skips storage if a semantically identical autobiographical memory already exists
    (cosine similarity ≥ AUTOBIO_DEDUP_THRESHOLD) to prevent the collection from
    accumulating redundant variants of the same fact over time.
    """
    try:
        model = get_embed_model()
        qdrant = get_qdrant()

        vector = model.encode(summary, normalize_embeddings=True).tolist()

        # Dedup check: skip if a very similar autobio already exists
        existing = qdrant.query_points(
            collection_name=QDRANT_MEMORY_COLLECTION,
            query=vector,
            limit=1,
            query_filter={
                "must": [
                    {"key": "user_code", "match": {"value": user_code}},
                    {"key": "memory_type", "match": {"value": "autobiographical"}},
                ]
            },
        ).points
        # The collection uses Distance.DOT — raw dot product score can exceed 1.0 when
        # stored vectors were uploaded without normalization. Clamp to [0, 1] before
        # comparing against the threshold to avoid spurious dedup skips or false hits.
        dedup_score = min(existing[0].score, 1.0) if existing else 0.0
        if existing and dedup_score >= AUTOBIO_DEDUP_THRESHOLD:
            # Reinforce the existing memory if the new submission carries higher importance
            existing_importance = float(existing[0].payload.get("importance", 0))
            if importance > existing_importance:
                qdrant.set_payload(
                    collection_name=QDRANT_MEMORY_COLLECTION,
                    payload={"importance": round(importance, 4)},
                    points=[existing[0].id],
                )
                logger.debug(
                    "Autobio dedup: reinforced '%s' %.2f → %.2f",
                    summary[:60],
                    existing_importance,
                    importance,
                )
            else:
                logger.debug(
                    "Autobio dedup: skipping '%s' (similar=%.2f)",
                    summary[:60],
                    dedup_score,
                )
            return

        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{user_code}:autobio:{summary}"))

        qdrant.upsert(
            collection_name=QDRANT_MEMORY_COLLECTION,
            points=[
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "user_code": user_code,
                        "memory_type": "autobiographical",
                        "status": "current",  # explicit — archive sets this to "past"
                        "text": summary,
                        "importance": importance,
                        "timestamp": time.time(),
                    },
                }
            ],
        )

        logger.info("Autobiographical memory stored: %s", summary)
        _invalidate_timeline_cache(user_code)

    except Exception as e:
        logger.error("Autobiographical memory failed: %s", e)


def _autobio_op(user_code: str, query: str, threshold: float, action: str) -> int:
    """Shared implementation for retract/archive operations on autobiographical memories.

    action="retract" → hard delete (reserved for errors/duplicates).
    action="archive"  → payload update status="past" (outdated facts, keeps history).
    """
    try:
        model = get_embed_model()
        qdrant = get_qdrant()
        vector = model.encode(query, normalize_embeddings=True).tolist()

        filt: dict = {
            "must": [
                {"key": "user_code", "match": {"value": user_code}},
                {"key": "memory_type", "match": {"value": "autobiographical"}},
            ]
        }
        if action == "archive":
            # Only archive current facts — already-archived ones are skipped
            filt["must_not"] = [{"key": "status", "match": {"value": "past"}}]

        results = qdrant.query_points(
            collection_name=QDRANT_MEMORY_COLLECTION,
            query=vector,
            limit=5,
            query_filter=filt,
        ).points

        to_act = [r.id for r in results if min(r.score, 1.0) >= threshold]
        if not to_act:
            return 0

        if action == "retract":
            qdrant.delete(
                collection_name=QDRANT_MEMORY_COLLECTION,
                points_selector=PointIdsList(points=to_act),
            )
            logger.info(
                "Autobio retracted %d point(s) for '%s'", len(to_act), query[:60]
            )
        else:
            qdrant.set_payload(
                collection_name=QDRANT_MEMORY_COLLECTION,
                payload={"status": "past", "archived_date": date.today().isoformat()},
                points=to_act,
            )
            logger.info(
                "Autobio archived %d point(s) for '%s'", len(to_act), query[:60]
            )

        _invalidate_timeline_cache(user_code)
        return len(to_act)
    except Exception as e:
        logger.error("_autobio_op(%s) failed: %s", action, e)
        return 0


def retract_autobiographical_event(
    user_code: str, query: str, threshold: float = 0.88
) -> int:
    """Delete autobiographical memories semantically matching the query.
    Reserved for genuine errors and strict duplicates — not for outdated facts.
    Higher threshold than archive (0.88 vs 0.78) — hard delete requires stricter match."""
    return _autobio_op(user_code, query, threshold, "retract")


def archive_autobiographical_event(
    user_code: str, query: str, threshold: float = 0.78
) -> int:
    """Mark autobiographical memories as past (status='past') without deleting them.
    Used when a fact is no longer current but retains historical value
    (e.g. changed jobs, stopped a hobby). Deprioritised in recall via status_factor."""
    return _autobio_op(user_code, query, threshold, "archive")


def get_autobiographical_facts(
    user_code: str, limit: int = 40, newest_first: bool = False
) -> list[str]:
    """Return current (non-archived) autobiographical memory summaries.

    Default: sorted oldest-first (temporal progression visible for cleaning).
    newest_first=True: most recent facts first (used to seed NIGHTLY_FACTS context).

    All facts fetched up to max(limit*2, 100) are returned without truncation
    so the nightly cleaning LLM sees recently-added duplicates, not just old facts.
    """
    try:
        qdrant = get_qdrant()
        results = qdrant.scroll(
            collection_name=QDRANT_MEMORY_COLLECTION,
            scroll_filter={
                "must": [
                    {"key": "user_code", "match": {"value": user_code}},
                    {"key": "memory_type", "match": {"value": "autobiographical"}},
                ],
                "must_not": [{"key": "status", "match": {"value": "past"}}],
            },
            limit=max(limit * 2, 100),
            with_payload=True,
        )[0]

        results.sort(
            key=lambda r: r.payload.get("timestamp", 0), reverse=newest_first
        )
        return [r.payload["text"] for r in results]
    except Exception as e:
        logger.error("get_autobiographical_facts failed: %s", e)
        return []


def _build_memory_filter(user_code: str, scope: str) -> dict:
    """Build Qdrant query_filter for a memory search by scope."""
    user_clause = {"key": "user_code", "match": {"value": user_code}}
    if scope in ("episodic", "autobiographical"):
        return {
            "must": [user_clause, {"key": "memory_type", "match": {"value": scope}}]
        }
    # "auto" — both layers; must_not absent types from slipping in
    return {
        "must": [
            user_clause,
            {
                "should": [
                    {"key": "memory_type", "match": {"value": "episodic"}},
                    {"key": "memory_type", "match": {"value": "autobiographical"}},
                ]
            },
        ]
    }


def search_memory(
    user_code: str, query: str, limit: int = 5, memory_scope: str = "auto"
):
    """Search vector memory. memory_scope filters to a specific layer or searches all ('auto')."""
    # Profile scope has no Qdrant data — Redis profile is already injected via build_memory_context()
    if memory_scope == "profile":
        return []

    try:
        model = get_embed_model()
        qdrant = get_qdrant()

        vector = model.encode(query, normalize_embeddings=True).tolist()

        results = qdrant.query_points(
            collection_name=QDRANT_MEMORY_COLLECTION,
            query=vector,
            limit=limit * 3,
            query_filter=_build_memory_filter(user_code, memory_scope),
        ).points

        memories = []
        # memorie recall with filter of similarity
        now = time.time()
        # Fetch once — avoids one Redis round-trip per result
        interest_weights = get_interest_weights(user_code)

        for r in results:
            # Clamp to [0, 1]: the collection uses Distance.DOT so scores can exceed 1.0
            # for old vectors that were stored before normalize_embeddings was enforced.
            sim = min(r.score, 1.0)
            if sim < RECALL_MEMORY_SIMILARITY_THRESHOLD:
                continue
            payload = r.payload

            # Recency window: 30 days for episodic, AUTOBIO_RECENCY_WINDOW_DAYS for autobiographical.
            # Autobiographical memories are durable milestones — they stay relevant for months.
            timestamp = payload.get("timestamp", now)
            recency = now - timestamp
            mem_type = payload.get("memory_type", "episodic")
            recency_window = (
                AUTOBIO_RECENCY_WINDOW_DAYS * 86400
                if mem_type == "autobiographical"
                else 30 * 86400
            )
            recency_bonus = max(0, min(1, 1 - recency / recency_window))
            # Archived (past) facts are still findable but ranked lower so current
            # facts take priority; a 0.4 factor ensures past facts appear in
            # positions ~4-5 when a semantically close current fact scores higher.
            status_factor = 0.4 if payload.get("status") == "past" else 1.0
            # Weighted blend: similarity + importance + recency
            # All weights sum to 1.0 so the score stays in ~[0, 1]
            final_score = (
                sim * 0.50 + payload.get("importance", 0) * 0.30 + recency_bonus * 0.20
            ) * status_factor

            # Interest-weight boost: user-declared topics nudge ranking gently.
            # Cap at 0.08 so a strong semantic match (Δ≥0.08) is never overridden.
            # weight 1.0 = neutral (no boost); weight 3.0 → +0.08 (max).
            if interest_weights:
                text_lower = payload.get("text", "").lower()
                best_weight = max(
                    (
                        w for term, w in interest_weights.items()
                        if re.search(
                            r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)",
                            text_lower,
                        )
                    ),
                    default=1.0,
                )
                interest_boost = min(0.08, max(0.0, (best_weight - 1.0) * 0.04))
                final_score = min(1.0, final_score + interest_boost)

            memories.append(
                {
                    "text": payload["text"],
                    "timestamp": timestamp,
                    "score": final_score,
                    "_id": r.id,
                    "_sim": sim,  # clamped similarity — used for reconsolidation gate
                    "_mem_type": mem_type,
                    "_importance": payload.get("importance", 0),
                    "_status": payload.get("status", "current"),
                }
            )
        # cognitive ranking
        memories.sort(key=lambda x: x["score"], reverse=True)
        top = memories[:limit]

        # Reconsolidation: recalling a memory reinforces it (neuroscience analogy).
        # Conditions (both must hold):
        #   1. autobiographical only — episodic memories are transient by design; boosting them
        #      would delay consolidation and bloat the episodic collection.
        #   2. raw semantic similarity > 0.82 — only strongly relevant recalls count;
        #      vaguely related memories (0.70–0.82) are not reinforced.
        # Cap at MEMORY_DECAY_DURABLE_MIN - 0.05 = 0.95 so they remain subject to monthly decay.
        _REINFORCE_SIM_THRESHOLD = 0.82
        _reinforce_cap = MEMORY_DECAY_DURABLE_MIN - 0.05
        try:
            for m in top:
                if m["_mem_type"] != "autobiographical":
                    continue
                if m["_status"] == "past":  # don't reinforce archived memories
                    continue
                if m["_sim"] < _REINFORCE_SIM_THRESHOLD:
                    continue
                old_imp = m["_importance"]
                new_imp = min(round(old_imp + 0.05, 4), _reinforce_cap)
                if new_imp > old_imp:
                    Thread(
                        target=qdrant.set_payload,
                        kwargs={
                            "collection_name": QDRANT_MEMORY_COLLECTION,
                            "payload": {"importance": new_imp},
                            "points": [m["_id"]],
                        },
                        daemon=True,
                    ).start()
        except Exception as _e:
            logger.warning("Memory reinforcement failed (non-blocking): %s", _e)

        # Strip internal fields before returning
        for m in top:
            m.pop("_id", None)
            m.pop("_sim", None)
            m.pop("_mem_type", None)
            m.pop("_importance", None)
            m.pop("_status", None)

        return top

    except Exception as e:
        logger.error("Memory search failed: %s", e)
        return []


async def async_search_memory(
    user_code: str, query: str, limit: int = 5, memory_scope: str = "auto"
) -> list:
    """Async-safe wrapper for search_memory.

    search_memory calls model.encode() which is CPU/GPU-bound and synchronous.
    Calling it directly from an async route would block the event loop.
    This wrapper always delegates to a thread pool — callers never need to
    remember to wrap it themselves.
    """
    return await asyncio.to_thread(search_memory, user_code, query, limit, memory_scope)


# ══════════════════════════════════════════════════
#  SELF MEMORY LOCK
# ══════════════════════════════════════════════════

# Shared lock for all read-modify-write cycles on jarvis-self.json.
# Use as: with self_memory_lock: data = get_self_memory(); ...; save_self_memory(data)
# threading.Lock works in both sync and async contexts (no await held while locked).
self_memory_lock = Lock()


# ══════════════════════════════════════════════════
#  ATOMIC FILE WRITE
# ══════════════════════════════════════════════════


def atomic_json_write(path: str, data, indent: int = 2) -> None:
    """
    Write *data* as JSON to *path* atomically.

    Uses a sibling temp file + os.replace() so that:
    - Concurrent readers always see either the previous complete file or the
      new complete file — never a truncated/partial state (race condition).
    - A crash mid-write leaves the original file intact (no corruption).

    os.replace() is a single syscall and is guaranteed atomic on POSIX.
    """
    dir_name = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ══════════════════════════════════════════════════
#  SELF MEMORY — Jarvis's identity and growth
# ══════════════════════════════════════════════════


def get_self_memory() -> dict:
    """Load jarvis-self.json; bootstrap with defaults if missing or corrupt."""
    try:
        with open(SELF_MEMORY_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        default = {
            "identity": {
                "name": "Jarvis",
                "version": "6.0",
                "created": datetime.now(timezone.utc).isoformat(),
                "personality": "Helpful, concise, direct. Dry humor when appropriate.",
            },
            "opinions": [],
            "learnings": [],
            "growth_log": [],
            "reflection_count": 0,
        }
        save_self_memory(default)
        return default
    except Exception as exc:
        logger.error("Could not load jarvis-self.json: %s", type(exc).__name__)
        return {}


def save_self_memory(data: dict) -> None:
    """Save jarvis-self.json atomically."""
    try:
        atomic_json_write(SELF_MEMORY_PATH, data)
    except Exception as exc:
        logger.error("Could not save jarvis-self.json: %s", type(exc).__name__)


def add_self_learning(learning: str):
    """Add a Jarvis self-improvement note (not a user fact)."""
    with self_memory_lock:
        data = get_self_memory()
        data.setdefault("learnings", []).append(
            {
                "text": learning,
                "date": datetime.now(timezone.utc).isoformat(),
            }
        )
        data["learnings"] = data["learnings"][-100:]
        save_self_memory(data)


# ══════════════════════════════════════════════════
#  CONTEXT BUILDER — Assemble memory for prompts
# ══════════════════════════════════════════════════


_TIMELINE_CACHE_TTL = 300  # 5 minutes — invalidated on new autobio store


def get_user_timeline(user_code: str, limit: int = 20):
    """
    Retrieve the user's autobiographical timeline.

    Scrolls up to 200 points to ensure the top-N by importance+recency are
    correctly identified — scroll() returns in arbitrary Qdrant order.

    Result is cached in Redis for 5 minutes to avoid a 200-point Qdrant scroll
    on every chat request. Invalidated whenever a new autobiographical memory
    is stored (store_autobiographical_event calls _invalidate_timeline_cache).
    """
    cache_key = f"cache:timeline:{user_code}"
    r = get_redis()
    try:
        cached = r.get(cache_key)
        if cached:
            return json.loads(cached)[:limit]
    except Exception:
        pass

    try:
        qdrant = get_qdrant()

        results = qdrant.scroll(
            collection_name=QDRANT_MEMORY_COLLECTION,
            scroll_filter={
                "must": [
                    {"key": "user_code", "match": {"value": user_code}},
                    {"key": "memory_type", "match": {"value": "autobiographical"}},
                ],
                # Archived (past) facts are excluded from the chat timeline —
                # they remain searchable via search_memory but are not injected
                # into every conversation as if they were still current.
                "must_not": [{"key": "status", "match": {"value": "past"}}],
            },
            limit=200,  # over-fetch then rank — scroll order is arbitrary
        )[0]

        now_ts = time.time()
        timeline = []

        for r_ in results:
            ts = r_.payload.get("timestamp", now_ts)
            imp = r_.payload.get("importance", 0)
            # Recency bonus normalised over a 1-year window
            recency = max(0.0, 1.0 - (now_ts - ts) / (86400 * 365))
            rank = imp * 0.7 + recency * 0.3
            timeline.append(
                {
                    "text": r_.payload["text"],
                    "timestamp": ts,
                    "importance": imp,
                    "_rank": rank,
                }
            )

        # Most important + recent events first
        timeline.sort(key=lambda x: x["_rank"], reverse=True)
        for e in timeline:
            e.pop("_rank", None)

        # Cache top-50 (more than enough for any prompt injection)
        try:
            r.setex(
                cache_key,
                _TIMELINE_CACHE_TTL,
                json.dumps(timeline[:50], ensure_ascii=False),
            )
        except Exception:
            pass

        return timeline[:limit]

    except Exception as e:
        logger.error("Timeline retrieval failed: %s", e)
        return []


def _invalidate_timeline_cache(user_code: str) -> None:
    """Called after a new autobiographical memory is stored."""
    try:
        get_redis().delete(f"cache:timeline:{user_code}")
    except Exception:
        pass


def build_memory_context(
    session_id: str,
    user_code: str,
    self_mem: dict | None = None,
    include_suggestions: bool = True,
) -> str:
    """Build a memory context string to inject into the system prompt.

    Pass an already-loaded *self_mem* dict to avoid a redundant JSON read when
    the caller (build_system_prompt) has already called get_self_memory().

    include_suggestions — set False for pure utility intents (weather/calendar/gmail)
                          to skip the SUJETS À ABORDER section (~50 tokens saved).

    All Redis reads are batched into a single pipeline round-trip.
    """
    if self_mem is None:
        self_mem = get_self_memory()
    parts = []

    # ── Single Redis pipeline round-trip for all scalar/hash reads ──────────
    r = get_redis()
    pipe = r.pipeline(transaction=False)
    pipe.hgetall(f"user:{user_code}:profile")  # 0
    pipe.hgetall(f"user:{user_code}:preferences")  # 1
    pipe.get(f"user:{user_code}:projects")  # 2
    pipe.get("jarvis:emotional_state")  # 3
    pipe.get(f"jarvis:{user_code}:tomorrow_suggestions")  # 4
    pipe.get(f"cache:timeline:{user_code}")  # 5 — avoids 2nd Redis RTT
    _pipe_results = pipe.execute()

    profile = _pipe_results[0] or {}
    prefs = _pipe_results[1] or {}
    _proj_raw = _pipe_results[2]
    _emotion_raw = _pipe_results[3]
    _sugg_raw = _pipe_results[4]
    _timeline_cached = _pipe_results[5]

    # User profile — namespaced keys (hobby:kart) are grouped by category for readability
    if profile:
        grouped: dict[str, list[str]] = {}
        scalars: list[tuple[str, str]] = []
        for k, v in profile.items():
            if ":" in k:
                category, subkey = k.split(":", 1)
                grouped.setdefault(category, []).append(f"{subkey}={v}")
            else:
                scalars.append((k, v))
        plines = [f"- {k}: {v}" for k, v in scalars]
        plines += [f"- {cat}: {', '.join(vals)}" for cat, vals in grouped.items()]
        parts.append(
            "<profil_utilisateur>\n" + "\n".join(plines) + "\n</profil_utilisateur>"
        )

    # User preferences
    if prefs:
        plines = [f"- {k}: {v}" for k, v in prefs.items()]
        parts.append("<preferences>\n" + "\n".join(plines) + "\n</preferences>")

    # Active projects only — done projects are not useful context for chat
    try:
        projects = json.loads(_proj_raw) if _proj_raw else []
    except Exception:
        projects = []
    active_projects = [
        p for p in projects if isinstance(p, dict) and p.get("status") != "done"
    ]
    if active_projects:
        plines = [f"- {p.get('name', 'sans nom')}" for p in active_projects]
        parts.append(
            "<projets_actifs>\n"
            "[Exhaustif — absent = clôturé]\n"
            + "\n".join(plines)
            + "\n</projets_actifs>"
        )

    # Emotional state — read-only decay applied inline (no Redis write, no lock)
    try:
        emotion = json.loads(_emotion_raw) if _emotion_raw else {}
    except Exception:
        emotion = {}
    if not emotion:
        emotion = {
            "mood": "neutral",
            "energy": 0.7,
            "confidence": 0.8,
            "curiosity": 0.6,
            "concern": 0.0,
        }
    else:
        _last = emotion.get("last_updated")
        if _last:
            try:
                _eh = max(
                    0.0,
                    (datetime.now(timezone.utc) - datetime.fromisoformat(_last)).total_seconds() / 3600,
                )
                _apply_decay_unlocked(emotion, _eh)
            except (ValueError, TypeError):
                pass
    # Injection etat emotionnel
    _MOOD_FR = {
        "happy": "joyeux",
        "attentive": "attentif",
        "supportive": "bienveillant",
        "engaged": "curieux",
        "gentle": "calme",
        "focused": "concentré",
        "neutral": "neutre",
    }
    _mood = emotion.get("mood", "neutral")
    _concern = emotion.get("concern", 0.0)
    _energy = emotion.get("energy", 0.7)
    _emotion_lines = []
    if _mood != "neutral":
        _emotion_lines.append(f"Humeur : {_MOOD_FR.get(_mood, _mood)}")
    if _concern > 0.3:
        _emotion_lines.append(f"Préoccupation : {_concern:.1f}")
    if abs(_energy - 0.7) > 0.15:
        _energy_label = "élevée" if _energy > 0.7 else "basse"
        _emotion_lines.append(f"Énergie : {_energy_label} ({_energy:.1f})")
    if _emotion_lines:
        parts.append(
            "<etat_emotionnel>\n"
            + "\n".join(f"- {l}" for l in _emotion_lines)
            + "\n</etat_emotionnel>"
        )

    # Self identity — apprentissages de Jarvis (guide interne, pas des faits sur l'utilisateur)
    if self_mem.get("learnings"):
        recent_learnings = self_mem["learnings"][-5:]
        plines = [f"- {ln['text']}" for ln in recent_learnings]
        parts.append(
            "<apprentissages_jarvis>\n"
            + "\n".join(plines)
            + "\n</apprentissages_jarvis>"
        )

    if self_mem.get("self_notes"):
        recent_notes = self_mem["self_notes"][-5:]
        _notes_lines = [f"- {n['text']}" for n in recent_notes if n.get("text")]
        if _notes_lines:
            parts.append(
                "<notes_jarvis>\n" + "\n".join(_notes_lines) + "\n</notes_jarvis>"
            )

    # User Timeline — served from pipeline cache hit [5]; fallback to Qdrant on miss
    if _timeline_cached:
        try:
            timeline = json.loads(_timeline_cached)[:7]
        except Exception:
            timeline = get_user_timeline(user_code, limit=7)
    else:
        timeline = get_user_timeline(user_code, limit=7)
    if timeline:
        plines = [
            f"({rel_time_fr(event['timestamp'])}) {event['text']}" for event in timeline
        ]
        parts.append(
            "<frise_chronologique>\n" + "\n".join(plines) + "\n</frise_chronologique>"
        )

    # Tomorrow suggestions — written by nightly review, consumed today.
    # Skipped for pure utility intents (weather/calendar/gmail) — irrelevant noise.
    if include_suggestions:
        try:
            suggestions = json.loads(_sugg_raw) if _sugg_raw else []
        except Exception:
            suggestions = []
        if suggestions:
            plines = [f"- {s}" for s in suggestions]
            parts.append(
                "<sujets_a_aborder>\n" + "\n".join(plines) + "\n</sujets_a_aborder>"
            )

    # User relation — always injected so every conversation has a tonal directive.
    # self_mem is already loaded at the top of this function (no extra I/O).
    _default_rel = {
        "affinity": 0.5,
        "interaction_style": "direct",
        "average_interaction_mood": "measured",
    }
    rel = {**_default_rel, **self_mem.get("user_relations", {}).get(user_code, {})}
    _aff = rel["affinity"]
    _aff_label = (
        "forte"
        if _aff >= 0.8
        else "bonne"
        if _aff >= 0.6
        else "modérée"
        if _aff >= 0.4
        else "faible"
    )
    parts.append(
        f"<relation_avec_utilisateur>\n"
        f"- Affinité : {_aff_label}\n"
        f"- Style de communication préféré : {rel['interaction_style']}\n"
        f"- Humeur moyenne des échanges : {rel['average_interaction_mood']}\n"
        f"</relation_avec_utilisateur>"
    )

    return "\n\n".join(parts) if parts else ""


# ══════════════════════════════════════════════════
#  MEMORY COMPRESSION / CLEANING. CALLED BY NIGHTLY SCRIPT
# ══════════════════════════════════════════════════
def _consolidate_user_memories(user_code: str, batch_size: int = 50):
    """
    Consolidate all episodic memories for a user into autobiographical milestones.

    Only memories older than EPISODIC_RETENTION_DAYS are eligible — recent episodic
    memories are preserved as a short-term context window (default: 45 days).

    Runs in a loop, processing batches of `batch_size` oldest eligible points until
    fewer than 5 remain (not enough to form a meaningful summary).
    """
    qdrant = get_qdrant()
    total_deleted = 0
    cutoff_ts = time.time() - EPISODIC_RETENTION_DAYS * 86400

    while True:
        try:
            results = qdrant.scroll(
                collection_name=QDRANT_MEMORY_COLLECTION,
                scroll_filter={
                    "must": [
                        {"key": "user_code", "match": {"value": user_code}},
                        {"key": "memory_type", "match": {"value": "episodic"}},
                        {"key": "timestamp", "range": {"lt": cutoff_ts}},
                    ]
                },
                order_by={"key": "timestamp", "direction": "asc"},
                limit=batch_size,
            )[0]

            point_ids = [r.id for r in results]
            texts = [r.payload["text"] for r in results if r.payload.get("text")]

            if len(texts) < 5:
                break  # Nothing left worth consolidating

            combined = "\n".join(texts)

            raw = call_llm(
                [
                    {
                        "role": "user",
                        "content": get_prompt("CONSOLIDATION_PROMPT").format(
                            combined=combined
                        ),
                    }
                ],
                model=PRIMARY_MODEL,
                api_url=PRIMARY_API_URL,
                api_key=PRIMARY_API_KEY,
                temperature=0.0,
                max_tokens=MAX_TOKENS_COMPACT,
                json_response=True,
                no_think=True,
                timeout=llm_timeout(MAX_TOKENS_COMPACT),
            )

            parsed = extract_llm_json(raw)
            facts = parsed.get("facts", []) if isinstance(parsed, dict) else []
            facts = [
                f.strip().strip("\"'")[:300]
                for f in facts
                if isinstance(f, str) and f.strip()
            ]

            if not facts:
                logger.warning(
                    "[%s] Consolidation: LLM returned 0 facts for %d episodic points — skipping deletion",
                    user_code,
                    len(point_ids),
                )
                break

            for fact in facts:
                store_autobiographical_event(
                    user_code, fact, MEMORY_CONSOLIDATION_IMPORTANCE
                )

            qdrant.delete(
                collection_name=QDRANT_MEMORY_COLLECTION,
                points_selector=PointIdsList(points=point_ids),
            )

            total_deleted += len(point_ids)
            logger.info(
                "[%s] Consolidation batch: %d facts stored, %d points deleted",
                user_code,
                len(facts),
                len(point_ids),
            )

        except Exception as e:
            logger.error("Memory consolidation failed for %s: %s", user_code, e)
            break

    if total_deleted:
        logger.info(
            "[%s] Memory consolidation complete: %d episodic points total",
            user_code,
            total_deleted,
        )


def curative_profile_cleanup(user_code: str):
    """
    Curative cleanup of the Redis user profile hash. Called nightly by the nightly
    review scheduler (run_nightly_interaction_review). Separated from monthly
    consolidation so duplicate keys are caught within 24 h instead of 30 days.

    Sends the full profile to the analysis LLM and asks it to identify:
    - Semantic duplicates (same fact under two different key names)
    - Obsolete/contradictory keys superseded by a more recent entry

    The LLM returns {"updates": {...}, "keys_to_delete": ["key1"]} — updates are
    applied first (merge-before-delete), then duplicates are deleted via HDEL.

    Skip condition: profile has fewer than 5 keys (not worth the LLM call).
    """
    r = get_redis()
    profile_redis_key = f"user:{user_code}:profile"
    profile_ts_key = f"user:{user_code}:profile:ts"
    profile = r.hgetall(profile_redis_key)
    if len(profile) < 5:
        return

    try:
        timestamps = r.hgetall(
            profile_ts_key
        )  # key → unix timestamp string (may be empty for old keys)

        def _fmt_ts(k: str) -> str:
            raw = timestamps.get(k)
            if not raw:
                return "date inconnue"
            try:
                return rel_time_fr(int(raw))
            except Exception:
                return "date inconnue"

        profile_str = "\n".join(
            f'- "{k}" (mis à jour : {_fmt_ts(k)}): {v}' for k, v in profile.items()
        )
        prompt = get_prompt("CURATIVE_CLEANUP_PROMPT").format(
            profile_count=len(profile),
            profile_str=profile_str,
        )

        parsed = extract_llm_json(
            call_llm(
                [{"role": "user", "content": prompt}],
                model=PRIMARY_MODEL,
                api_url=PRIMARY_API_URL,
                api_key=PRIMARY_API_KEY,
                temperature=0.0,
                max_tokens=MAX_TOKENS_COMPACT,
                json_response=True,
                no_think=True,
                timeout=llm_timeout(MAX_TOKENS_COMPACT),
            )
        )

        # Apply consolidation updates BEFORE any deletion (merge-before-delete)
        updates = parsed.get("updates", {}) if isinstance(parsed, dict) else {}
        now_ts = int(time.time())
        for key, value in updates.items():
            if key in profile and isinstance(value, str) and value.strip():
                r.hset(profile_redis_key, key, value.strip())
                r.hset(profile_ts_key, key, now_ts)
                logger.info(
                    "[%s] curative_profile_cleanup: UPDATE '%s' → '%s'",
                    user_code,
                    key,
                    value.strip(),
                )

        # Only delete keys that actually exist in the profile (safety guard)
        keys_to_delete = [k for k in parsed.get("keys_to_delete", []) if k in profile]

        # Safety: never delete a key that was just updated (merge target)
        keys_to_delete = [k for k in keys_to_delete if k not in updates]

        if keys_to_delete:
            for key in keys_to_delete:
                old_val = r.hget(profile_redis_key, key)
                logger.warning(
                    "[%s] curative_profile_cleanup: DELETE '%s' (was: %s, ts: %s)",
                    user_code,
                    key,
                    old_val or "(empty)",
                    _fmt_ts(key),
                )
            r.hdel(profile_redis_key, *keys_to_delete)
            r.hdel(profile_ts_key, *keys_to_delete)
            logger.info(
                "[%s] curative_profile_cleanup: deleted %s", user_code, keys_to_delete
            )
        elif not updates:
            logger.info("[%s] curative_profile_cleanup: profile is clean", user_code)

    except Exception as exc:
        logger.error("curative_profile_cleanup failed for %s: %s", user_code, exc)


def _decay_autobiographical_memories(user_code: str) -> int:
    """
    Monthly decay pass on autobiographical memories.

    For each autobiographical memory:
    - If importance >= MEMORY_DECAY_DURABLE_MIN → exempt (milestone permanent).
    - Otherwise: decayed = importance * (MEMORY_DECAY_FACTOR ^ age_months).
      - If decayed < MEMORY_DECAY_THRESHOLD → deleted from Qdrant.
      - Else → payload updated with the new (lower) importance.

    Returns the number of memories deleted.
    """
    qdrant = get_qdrant()
    deleted = 0
    updated = 0
    offset = None

    while True:
        results, next_offset = qdrant.scroll(
            collection_name=QDRANT_MEMORY_COLLECTION,
            scroll_filter={
                "must": [
                    {"key": "user_code", "match": {"value": user_code}},
                    {"key": "memory_type", "match": {"value": "autobiographical"}},
                ]
            },
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not results:
            break

        to_delete = []
        for point in results:
            importance = float(point.payload.get("importance", 0.7))

            # Durable milestones are never decayed
            if importance >= MEMORY_DECAY_DURABLE_MIN:
                continue

            # One multiplicative step per monthly run — avoids double-counting decay
            # since importance is already the post-previous-run value.
            # Human analogy: each month, memory fades by a fixed % of its current strength.
            decayed = importance * MEMORY_DECAY_FACTOR

            if decayed < MEMORY_DECAY_THRESHOLD:
                to_delete.append(point.id)
            else:
                qdrant.set_payload(
                    collection_name=QDRANT_MEMORY_COLLECTION,
                    payload={"importance": round(decayed, 4)},
                    points=[point.id],
                )
                updated += 1

        if to_delete:
            qdrant.delete(
                collection_name=QDRANT_MEMORY_COLLECTION,
                points_selector=PointIdsList(points=to_delete),
            )
            deleted += len(to_delete)
            logger.info(
                "[%s] Autobio decay: %d stale memories deleted",
                user_code,
                len(to_delete),
            )

        offset = next_offset
        if offset is None:
            break

    if deleted or updated:
        logger.info(
            "[%s] Autobio decay complete: %d deleted, %d importance updated",
            user_code,
            deleted,
            updated,
        )
    return deleted


def consolidate_memories(user_code: str = None, max_items: int = 20):
    """
    Monthly memory consolidation. Single public entry point — called on day 1 of each month
    by the nightly review scheduler, and on demand by the LLM self-action 'consolidate_memory'.

    If user_code is provided -> consolidate only this user.
    If user_code is None -> iterate over all users.

    Steps per user:
    1. Episodic consolidation  → compress episodic memories into autobiographical milestones
    2. Autobiographical decay  → reduce importance over time, delete stale memories

    Note: curative_profile_cleanup() is NOT called here — it runs nightly so duplicates
    are caught within 24 h rather than waiting up to 30 days.
    """
    users = [user_code] if user_code else list(USER_CODES.keys())
    for uc in users:
        logger.info("Memory consolidation starting for user %s", uc)
        try:
            _consolidate_user_memories(uc, max_items)
            _decay_autobiographical_memories(uc)
            logger.info("Memory consolidation done for user %s", uc)
        except Exception as e:
            logger.error("Memory consolidation failed for user %s: %s", uc, e)

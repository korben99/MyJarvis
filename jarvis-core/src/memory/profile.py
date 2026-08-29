"""Mémoire sémantique — profil utilisateur : déduplication de clés (fast paths +
LLM par famille de namespace), écriture de faits, pondération d'intérêts, préférences.
"""

import time

from config import (
    DEFAULT_TEMP,
    MAX_TOKENS_SHORT,
    PRIMARY_API_KEY,
    PRIMARY_API_URL,
    PRIMARY_MODEL,
    ROUTER_TIMEOUT,
)
from helpers import (
    call_llm_bg,
    extract_llm_json,
    get_logger,
    get_redis,
    normalize_key,
)

logger = get_logger("jarvis-memory")


# ══════════════════════════════════════════════════
#  SEMANTIC MEMORY — Long-term knowledge about user
# ══════════════════════════════════════════════════

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
#
# Chaque famille = UN namespace canonique inscriptible, plus ses synonymes — y compris
# ceux qui ne sont PLUS inscriptibles mais subsistent dans des profils écrits avant
# _ALLOWED_PROFILE_NAMESPACES. C'est la vraie fonction de cette table, et elle n'était dite
# nulle part : elle sert de PASSERELLE DE MIGRATION, en permettant à une nouvelle clé
# `etude:*` de se rapprocher d'une ancienne `specialite:*`. Le profil de ZSXEDC en porte
# encore trois : `specialite:`, `option:`, `matière:`.
#
# Ce qui était faux, en revanche, et corrigé ici : les exemples des deux prompts de dédup
# enseignaient au modèle de rapprocher VERS le legacy (« loisir:kart → hobby:kart »),
# c'est-à-dire vers une clé que plus rien ne peut écrire. La convergence doit aller dans
# l'autre sens — vers le namespace canonique, seul inscriptible.
#
# Les familles sont disjointes : `_candidate_keys` retient la première qui contient le
# préfixe, donc un préfixe présent dans deux familles rendrait le regroupement dépendant
# de l'ordre d'itération. Un préfixe seul dans sa catégorie n'a pas besoin d'entrée — le
# repli de `_candidate_keys` le traite déjà comme sa propre famille.
#
# situation et famille ne sont volontairement PAS regroupés :
# situation = faits sur l'utilisateur, famille = faits sur des tiers.
#
# `preoccupation / concerns / inquietude` a été retirée : `concerns` est la cible SCALAIRE
# de `_SCALAR_CANONICAL["inquietude"]`, donc le même jeton était à la fois un scalaire et
# un préfixe de namespace. Une clé est l'un ou l'autre.
_NS_FAMILY: dict[str, frozenset] = {
    "loisir": frozenset(
        {"loisir", "sport", "interet", "apprecie", "hobby", "interest", "passion", "activite"}
    ),
    "competence": frozenset({"competence", "technologie", "skill", "outil"}),
    "placement": frozenset({"placement", "investissement", "epargne"}),
    "etude": frozenset({"etude", "matiere", "matière", "specialite", "option", "note"}),
    "preference": frozenset({"preference", "aversion"}),
    "objectif": frozenset({"objectif", "projet", "project"}),
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
    # Stage 0.5: namespaced key whose subkey resolves to an existing scalar
    # e.g. situation:name → name, situation:prenom → name
    if ":" in new_key:
        subkey_norm = normalize_key(new_key.split(":", 1)[1])
        if subkey_norm in existing_keys:
            return subkey_norm
        sub_canonical = _SCALAR_CANONICAL.get(subkey_norm)
        if sub_canonical and sub_canonical in existing_keys:
            return sub_canonical
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
            # Priorité background : ces dedups tournent depuis l'analyzer planifié —
            # call_llm prendrait le lock GPU en priorité chat et retarderait l'utilisateur.
            raw = call_llm_bg(
                [
                    {
                        "role": "system",
                        "content": (
                            "Tu es un détecteur de doublons de clés de profil. "
                            "Réponds UNIQUEMENT avec du JSON valide.\n"
                            "RÈGLE : deux clés sont des doublons UNIQUEMENT si elles décrivent "
                            "le MÊME sujet ET le MÊME concept. Même préfixe ≠ doublon.\n"
                            "Exemples VRAIS doublons (namespace synonyme, même sujet) :\n"
                            '  existantes: ["sport:kart"] nouvelles: ["loisir:kart"] '
                            '→ {"matches": [{"new": "loisir:kart", "match": "sport:kart"}]}\n'
                            "Exemples NON doublons (même préfixe, sujets ou concepts différents) :\n"
                            '  existantes: ["situation:parents_location"] nouvelles: ["situation:lieu_residence"] '
                            '→ {"matches": [{"new": "situation:lieu_residence", "match": null}]}\n'
                            '  existantes: ["competence:ia"] nouvelles: ["competence:bricolage"] '
                            '→ {"matches": [{"new": "competence:bricolage", "match": null}]}\n'
                            '  existantes: ["loisir:kart"] nouvelles: ["loisir:tennis"] '
                            '→ {"matches": [{"new": "loisir:tennis", "match": null}]}'
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=PRIMARY_MODEL,
                api_url=PRIMARY_API_URL,
                api_key=PRIMARY_API_KEY,
                temperature=DEFAULT_TEMP,
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

        # Priorité background — voir _normalize_profile_keys_batch.
        raw = call_llm_bg(
            [
                {
                    "role": "system",
                    "content": (
                        "Tu es un détecteur de doublons de clés de profil. "
                        "Réponds UNIQUEMENT avec du JSON valide.\n"
                        "RÈGLE : deux clés sont des doublons UNIQUEMENT si elles décrivent "
                        "le MÊME sujet ET le MÊME concept. Même préfixe ≠ doublon.\n"
                        "Exemples VRAIS doublons (namespace synonyme, même sujet) :\n"
                        '  "sport:kart" vs "loisir:kart" → {"match": "sport:kart"}\n'
                        '  "technologie:ia" vs "competence:ia" → {"match": "technologie:ia"}\n'
                        "Exemples NON doublons (même préfixe, sujets ou concepts différents) :\n"
                        '  "situation:parents_location" vs "situation:lieu_residence" → {"match": null}\n'
                        '  "competence:ia" vs "competence:bricolage" → {"match": null}\n'
                        '  "loisir:kart" vs "loisir:tennis" → {"match": null}'
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=PRIMARY_MODEL,
            api_url=PRIMARY_API_URL,
            api_key=PRIMARY_API_KEY,
            temperature=DEFAULT_TEMP,
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

    # La liste blanche de namespaces ne s'applique QU'À L'ÉCRITURE, délibérément. Des
    # profils portent encore des clés écrites avant elle — `specialite:`, `option:`,
    # `matière:` chez ZSXEDC. Les soumettre au même filtre les rendrait
    # indélébiles : le nettoyage curatif ne pourrait plus jamais s'en défaire. On garde
    # donc la suppression ouverte à toute clé.
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
                    key,
                    user_code,
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
_ALLOWED_PROFILE_NAMESPACES = frozenset(
    {
        "situation",
        "famille",
        "profession",
        "competence",
        "loisir",
        "sport",
        "technologie",
        "sante",
        "objectif",
        "etude",
        "placement",
        "preference",
        "interet",
        "apprecie",
        "aversion",
        "langue",
    }
)

# Chaque famille doit contenir AU MOINS un namespace inscriptible : c'est vers lui que la
# déduplication fait converger, et une famille qui n'en contient aucun ne pourrait
# rapprocher que des clés mortes entre elles. On ne vérifie pas l'inverse — une famille
# contient par construction des synonymes legacy, non inscriptibles, et c'est sa raison
# d'être.
for _nom, _membres in _NS_FAMILY.items():  # pragma: no cover - garde de démarrage
    if not (_membres & _ALLOWED_PROFILE_NAMESPACES):
        raise RuntimeError(
            f"memory.profile: la famille {_nom!r} ne contient aucun namespace "
            f"inscriptible — la déduplication n'aurait nulle part où converger."
        )

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
                k,
                user_code,
            )
            continue
        filtered.append(f)
    facts = filtered

    # Hard cap: guard against analyzer returning an implausibly large fact list.
    if len(facts) > _MAX_FACTS_PER_BATCH:
        logger.warning(
            "update_user_profile_batch: %d facts exceeds cap %d for %s — truncating",
            len(facts),
            _MAX_FACTS_PER_BATCH,
            user_code,
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


def get_user_preferences(user_code: str) -> dict:
    """Get user preferences (language, style, etc.)."""
    r = get_redis()
    data = r.hgetall(f"user:{user_code}:preferences")
    return data or {}


def update_user_preference(user_code: str, key: str, value: str):
    """Update a preference."""
    r = get_redis()
    r.hset(f"user:{user_code}:preferences", key, value)

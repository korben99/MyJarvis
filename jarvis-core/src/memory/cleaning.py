"""Compression / nettoyage de la mémoire, appelé par la revue nocturne et la
consolidation mensuelle : consolidation épisodique→autobiographique, nettoyage
curatif du profil, narratif de profil, décroissance des souvenirs autobiographiques.

Couche haute du paquet : dépend de `vectors` (stockage/lecture autobio) et `profile`.
"""

import time

from config import (
    DEFAULT_TEMP,
    EPISODIC_RETENTION_DAYS,
    MAX_TOKENS_COMPACT,
    MEMORY_CONSOLIDATION_IMPORTANCE,
    MEMORY_DECAY_DURABLE_MIN,
    MEMORY_DECAY_FACTOR,
    MEMORY_DECAY_THRESHOLD,
    MAX_TOKENS_THINK_MEDIUM,
    PRIMARY_API_KEY,
    PRIMARY_API_URL,
    PRIMARY_MODEL,
    PROFILE_NARRATIVE_TOKENS,
    QDRANT_MEMORY_COLLECTION,
    THINKING_BUDGET_MEDIUM,
    USER_CODES,
    USERS,
    llm_timeout,
)
from helpers import (
    call_llm_bg,
    extract_llm_json,
    get_logger,
    get_qdrant,
    get_redis,
    rel_time_fr,
)
from prompts import get_prompt
from qdrant_client.models import PointIdsList

from .profile import get_interest_weights
from .vectors import get_autobiographical_facts, store_autobiographical_event

logger = get_logger("jarvis-memory")

# Nombre de lots consécutifs entièrement dédupliqués au-delà duquel on arrête la passe.
# Borne le cas où toute la fenêtre ancienne se résume en faits déjà connus : on avance
# plutôt que de bloquer, mais on ne paie pas un appel LLM par lot indéfiniment.
_MAX_LOTS_ECARTES = 5


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

    def _filtre(borne_basse: float | None) -> dict:
        plage: dict = {"lt": cutoff_ts}
        if borne_basse is not None:
            plage["gt"] = borne_basse
        return {
            "must": [
                {"key": "user_code", "match": {"value": user_code}},
                {"key": "memory_type", "match": {"value": "episodic"}},
                {"key": "timestamp", "range": plage},
            ]
        }

    # Sanité de l'index. `timestamp` doit être indexé en `float` — c'est ce que le payload
    # y écrit (time.time()). Un index `integer` n'indexe aucune valeur flottante, et
    # `scroll(order_by="timestamp")` rend alors une liste VIDE sans lever la moindre
    # erreur : la consolidation sortirait à son premier tour (`len(texts) < 5`) sans que
    # rien ne le signale. D'où ce comptage — on compare ce que le scroll rend à ce que le
    # compteur annonce, la seule façon de distinguer « plus rien à faire » de « index muet ».
    try:
        eligibles = qdrant.count(
            collection_name=QDRANT_MEMORY_COLLECTION,
            count_filter=_filtre(None),
            exact=True,
        ).count
    except Exception as exc:
        logger.warning("[%s] Consolidation: comptage impossible (%s)", user_code, exc)
        eligibles = -1

    borne_basse: float | None = None
    lots_ecartes = 0

    while True:
        try:
            results = qdrant.scroll(
                collection_name=QDRANT_MEMORY_COLLECTION,
                scroll_filter=_filtre(borne_basse),
                order_by={"key": "timestamp", "direction": "asc"},
                limit=batch_size,
            )[0]

            point_ids = [r.id for r in results]
            texts = [r.payload["text"] for r in results if r.payload.get("text")]

            # Garde limitée au PREMIER tour : `eligibles` est un instantané pris avant la
            # boucle, et ne décrit plus l'état réel dès qu'un lot a été supprimé. Il faut
            # les DEUX conditions — `borne_basse is None` ne couvre que la branche « lot
            # entièrement dédupliqué », qui est la seule à poser cette borne ; sur un
            # parcours nominal elle reste None de bout en bout et laisserait la sortie
            # normale de boucle, stock épuisé, franchir la garde.
            if not results and eligibles >= 5 and borne_basse is None and not total_deleted:
                logger.error(
                    "[%s] Consolidation: %d point(s) épisodique(s) éligibles mais le "
                    "scroll ordonné n'en rend AUCUN — l'index de payload `timestamp` est "
                    "probablement absent ou du mauvais type (il doit être `float`). "
                    "Consolidation impossible, aucune donnée touchée.",
                    user_code, eligibles,
                )
                break

            if len(texts) < 5:
                break  # Nothing left worth consolidating

            combined = "\n".join(texts)

            raw = call_llm_bg(
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
                temperature=DEFAULT_TEMP,
                max_tokens=MAX_TOKENS_THINK_MEDIUM,
                thinking_budget=THINKING_BUDGET_MEDIUM,
                json_response=True,
                no_think=False,
                timeout=llm_timeout(MAX_TOKENS_THINK_MEDIUM),
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

            # On compte ce qui est RÉELLEMENT entré en mémoire avant de détruire la source.
            # store_autobiographical_event écarte silencieusement un fait trop proche d'un
            # souvenir existant : sans ce comptage (auparavant), un lot dont tous
            # les faits étaient dédupliqués voyait quand même ses 50 points épisodiques
            # supprimés — la source détruite, rien de gagné. Le garde `if not facts`
            # ci-dessus ne voit pas ce cas, puisque la génération, elle, a réussi.
            ecrits = sum(
                store_autobiographical_event(
                    user_code, fact, MEMORY_CONSOLIDATION_IMPORTANCE
                )
                for fact in facts
            )

            if not ecrits:
                # On CONSERVE la source — c'était déjà le cas — mais on AVANCE au lieu de
                # sortir. Sortir figeait la consolidation de cet utilisateur pour de bon :
                # le lot suivant est sélectionné par « les plus anciens d'abord », donc le
                # passage suivant resélectionnait exactement le même lot, le redéduplicait,
                # et ressortait. Un lot dont tous les faits existent déjà bloquait tout ce
                # qui était derrière lui.
                lots_ecartes += 1
                borne_basse = max(
                    r.payload.get("timestamp", 0) for r in results
                )
                logger.warning(
                    "[%s] Consolidation: %d fait(s) tous écartés par la dédup autobio — "
                    "%d point(s) épisodique(s) CONSERVÉ(S), on passe au lot suivant.",
                    user_code,
                    len(facts),
                    len(point_ids),
                )
                if lots_ecartes >= _MAX_LOTS_ECARTES:
                    logger.warning(
                        "[%s] Consolidation: %d lots consécutifs entièrement dédupliqués "
                        "— passage interrompu.",
                        user_code, lots_ecartes,
                    )
                    break
                continue
            lots_ecartes = 0

            qdrant.delete(
                collection_name=QDRANT_MEMORY_COLLECTION,
                points_selector=PointIdsList(points=point_ids),
            )

            total_deleted += len(point_ids)
            logger.info(
                "[%s] Consolidation batch: %d/%d fait(s) retenu(s), %d point(s) supprimé(s)",
                user_code,
                ecrits,
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


def curative_profile_cleanup(user_code: str, stable_profile: dict | None = None):
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
        stable_str = (
            "\n".join(f"  {k}: {v}" for k, v in stable_profile.items() if v)
            if stable_profile
            else "aucun"
        )
        # Les projets en cours, sans quoi une clé décrivant l'avancement d'un chantier
        # terminé reste indétectable : lue seule, « décalé à la semaine prochaine » est
        # plausible indéfiniment. C'est la liste des projets qui la date.
        from .projects import get_user_projects, projets_actifs

        try:
            _projets = [
                p.get("name", "sans nom")
                for p in projets_actifs(get_user_projects(user_code))
            ]
        except Exception as exc:
            logger.debug("[%s] projets illisibles (%s)", user_code, exc)
            _projets = None

        prompt = get_prompt("CURATIVE_CLEANUP_PROMPT").format(
            profile_count=len(profile),
            profile_str=profile_str,
            stable_profile=stable_str,
            # None ≠ liste vide : indisponible n'autorise aucune conclusion, alors qu'une
            # liste vide signifie « plus rien en cours » et rend tout avancement caduc.
            projets=(
                "  (liste indisponible — ne rien conclure d'une absence)"
                if _projets is None
                else "\n".join(f"  - {n}" for n in _projets) or "  aucun projet en cours"
            ),
        )

        parsed = extract_llm_json(
            call_llm_bg(
                [{"role": "user", "content": prompt}],
                model=PRIMARY_MODEL,
                api_url=PRIMARY_API_URL,
                api_key=PRIMARY_API_KEY,
                temperature=DEFAULT_TEMP,
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

        # Hard cap: max 2 deletions per run — prompt constraint enforced in code
        keys_to_delete = keys_to_delete[:2]

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


def update_profile_narrative(user_code: str, stable_profile: dict | None = None) -> bool:
    """Generate a ~300-token narrative profile and store it in Redis.

    Synthesises profile hash + interest_weights + autobiographical facts into prose.
    The stable_profile (profil_utilisateur) is passed explicitly so the LLM knows
    what NOT to repeat. Stored at user:{user_code}:profile_narrative with 7-day TTL.
    Called nightly after curative_profile_cleanup — cross-session, per user.
    Returns True if the narrative was generated and stored.
    """
    r = get_redis()
    profile = r.hgetall(f"user:{user_code}:profile")
    interests = get_interest_weights(user_code)
    autobio = get_autobiographical_facts(user_code, limit=5, newest_first=True)

    if not profile and not interests and not autobio:
        logger.debug("[%s] update_profile_narrative: no data, skipping", user_code)
        return False

    name = profile.get("name") or USERS.get(user_code, {}).get("firstname", user_code)

    profile_str = (
        "\n".join(f"- {k}: {v}" for k, v in profile.items() if k != "name")
        or "aucun fait enregistré"
    )
    top_interests = sorted(interests.items(), key=lambda x: x[1], reverse=True)[:15]
    interests_str = ", ".join(f"{k} ({v:.1f})" for k, v in top_interests) or "aucun"
    autobio_str = "\n".join(f"- {f}" for f in autobio[:5]) or "aucun souvenir disponible"

    _stable = stable_profile or USERS.get(user_code, {}).get("profile", {})
    stable_profile_str = (
        "\n".join(f"- {k}: {v}" for k, v in _stable.items() if v)
        or "aucune"
    )

    prompt = get_prompt("PROFILE_NARRATIVE_PROMPT").format(
        name=name,
        profile_str=profile_str,
        interests_str=interests_str,
        autobio_str=autobio_str,
        stable_profile_str=stable_profile_str,
    )

    content = call_llm_bg(
        [{"role": "user", "content": prompt}],
        model=PRIMARY_MODEL,
        api_url=PRIMARY_API_URL,
        api_key=PRIMARY_API_KEY,
        temperature=DEFAULT_TEMP,
        max_tokens=PROFILE_NARRATIVE_TOKENS,
        json_response=False,  # prose attendue — le défaut True active l'early-stop JSON
        no_think=True,
    )

    if content and content.strip():
        r.setex(f"user:{user_code}:profile_narrative", 7 * 86400, content.strip())
        logger.info("[%s] profile narrative updated (%d chars)", user_code, len(content))
        return True
    return False


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

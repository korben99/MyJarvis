"""Fondation du paquet self : constantes Redis partagées, accès à jarvis-self.json
(goals, focus, relation, opinions, incidents) et journal de réflexion.

Couche la plus basse — tous les autres sous-modules en dépendent, elle n'importe aucun.
"""

import json
import re
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pytz
from config import BRIEFING_TIMEZONE, OPINIONS_MAX_ENTRIES
from helpers import get_logger, get_redis
from memory import (
    get_embed_model,
    get_self_memory,
    opinion_surface,
    save_self_memory,
    self_memory_lock,
)

logger = get_logger("jarvis-self")

# ── Redis keys (partagées entre sous-modules) ─────────────────────────────
_REFLECTION_LOG_KEY = "jarvis:self:reflection_log"
_REFLECTION_LOG_MAX = 30
_KNOWLEDGE_GAPS_KEY = "jarvis:self:knowledge_gaps"

# Sommeil d'un SUJET vis-à-vis de refine_prompt, posé au rejet comme à l'approbation.
# Distinct de `jarvis:self:gap_cooldown:*`, qui empêche de RE-SIGNALER la lacune : une
# lacune fraîchement posée doit pouvoir donner lieu à une proposition tout de suite, donc
# les deux verrous ne peuvent pas partager la même clé.
_REFINE_COOLDOWN_PREFIX = "jarvis:self:refine_cooldown"
_REFINE_COOLDOWN_TTL = 30 * 86400


def slug_de_sujet(topic: str) -> str:
    """Le slug d'un sujet de lacune ou de proposition.

    Un seul endroit — écriture, lecture, purge et cooldown doivent tronquer au même
    caractère, sinon l'un ne retrouve pas ce que l'autre a posé. Vit ici parce que `state`
    est le seul module sous `proposals` ET sous `context`, qui en ont besoin tous les deux.
    """
    return re.sub(r"\s+", "_", topic.lower())[:40]


_DEVICE_TOKEN_PREFIX = (
    "jarvis:device:token"  # device token per user (set by /device/register)
)
_PUSH_COOLDOWN_PREFIX = (
    "jarvis:push:cooldown"  # prevent push flooding (1 push per 48h per user)
)
_PUSH_COOLDOWN_TTL = 172800  # 48h


def get_goals() -> list[dict]:
    return get_self_memory().get("goals", [])


def get_current_focus() -> str:
    return get_self_memory().get("current_focus", "")


_DEFAULT_RELATION = {
    "affinity": 0.5,
    "interaction_style": "direct",
    "average_interaction_mood": "measured",
}


def get_user_relation(user_code: str) -> dict:
    """Return the current relation dict for a user (with defaults if missing)."""
    relations = get_self_memory().get("user_relations", {})
    return {**_DEFAULT_RELATION, **relations.get(user_code, {})}


_INCIDENTS_SELF_MAX = 30

# Cosinus au-delà duquel deux opinions sont considérées comme la même, sous des étiquettes
# différentes. Même valeur que la dédup des self_notes (l'ancienne dédup des self_notes).
_OPINION_MERGE_SIM = 0.85


def consolidate_incidents() -> int:
    """Fige les incidents récents de vitals dans jarvis-self.json (champ `incidents`).

    Déterministe, sans LLM : la trace existe indépendamment de ce que la réflexion décide.
    Dédup par horodatage (`at`, unique par incident), tri chronologique, borné — pour que
    la mémoire longue garde les événements marquants sans jamais devenir la foire.
    """
    try:
        from vitals import recent_incidents
    except Exception:
        return 0
    recents = recent_incidents(30)
    if not recents:
        return 0
    with self_memory_lock:
        data = get_self_memory()
        existants = data.get("incidents", [])
        connus = {it.get("at") for it in existants}
        nouveaux = [it for it in recents if it.get("at") not in connus]
        if not nouveaux:
            return 0
        fusion = sorted(existants + nouveaux, key=lambda it: it.get("at", 0))
        data["incidents"] = fusion[-_INCIDENTS_SELF_MAX:]
        save_self_memory(data)
    logger.info("Self: %d incident(s) consolidé(s) dans self.json", len(nouveaux))
    return len(nouveaux)


def _upsert_opinion_inplace(data: dict, topic: str, opinion: str, date: str) -> None:
    """Upsert an opinion into an already-loaded self-memory dict (no lock — caller holds it).

    Deux niveaux de rapprochement, dans cet ordre :

    1. Topic identique — le cas simple, l'opinion est réécrite.
    2. Opinion sémantiquement proche (cosinus > 0.85, même seuil et même méthode que
       l'ancienne dédup des self_notes). La comparaison de topics EXACTS ne rattrapait rien
       quand le même avis revenait sous une autre étiquette — or c'est le cas courant,
       le modèle nommant librement ses topics chaque nuit.

    Seuil vérifié sur le corpus réel : une paraphrase quasi littérale marque
    0,935, tandis que la paire d'opinions distinctes la plus proche des 50 en mémoire
    plafonne à 0,713 (médiane 0,309). L'écart est large — 0,85 ne fusionnera pas deux avis
    réellement différents.

    On compare à TOUTE la liste, pas à une fenêtre récente : un doublon peut viser une
    opinion ancienne, et c'est même le cas le plus probable puisqu'une opinion récente sur
    le même sujet aurait déjà le même topic. Coût : un encodage par lot de ≤ 120 textes
    courts, 0 à 2 fois par nuit.

    La dédup ne doit jamais faire perdre l'opinion : toute erreur d'embedding est rattrapée
    par un simple append.
    """
    topic = topic.strip().lower()
    opinions = data.setdefault("opinions", [])

    existing = next((o for o in opinions if o["topic"] == topic), None)
    if existing:
        existing["opinion"] = opinion
        existing["updated"] = date
        data["opinions"] = opinions[-OPINIONS_MAX_ENTRIES:]
        return

    if opinions:
        try:
            model = get_embed_model()
            vecs = model.encode(
                [opinion_surface(o["topic"], o["opinion"]) for o in opinions],
                normalize_embeddings=True,
            )
            sims = vecs @ model.encode(
                opinion_surface(topic, opinion), normalize_embeddings=True
            )
            i = int(np.argmax(sims))
            if sims[i] > _OPINION_MERGE_SIM:
                logger.info(
                    "Opinion fusionnée (sim=%.3f) : %s ← %s",
                    sims[i], opinions[i]["topic"], topic,
                )
                opinions[i]["opinion"] = opinion
                opinions[i]["updated"] = date
                data["opinions"] = opinions[-OPINIONS_MAX_ENTRIES:]
                return
        except Exception as exc:
            logger.warning("Dédup d'opinion indisponible (non bloquant) : %s", exc)

    opinions.append({"topic": topic, "opinion": opinion, "created": date})
    data["opinions"] = opinions[-OPINIONS_MAX_ENTRIES:]


def add_self_opinion(topic: str, opinion: str) -> None:
    """Add or update a Jarvis opinion. Thread-safe — acquires self_memory_lock."""
    date = datetime.now(timezone.utc).isoformat()
    with self_memory_lock:
        data = get_self_memory()
        _upsert_opinion_inplace(data, topic, opinion, date)
        save_self_memory(data)
    logger.info("Opinion upserted: %s", topic)


# ══════════════════════════════════════════════════
#  REFLECTION LOG  (Redis sorted set)
# ══════════════════════════════════════════════════


def log_reflection(entry: dict) -> None:
    """Append a reflection result to the Redis log (capped at _REFLECTION_LOG_MAX)."""
    r = get_redis()
    score = time.time()
    r.zadd(_REFLECTION_LOG_KEY, {json.dumps(entry, ensure_ascii=False): score})
    # Trim to most recent entries
    excess = r.zcard(_REFLECTION_LOG_KEY) - _REFLECTION_LOG_MAX
    if excess > 0:
        r.zremrangebyrank(_REFLECTION_LOG_KEY, 0, excess - 1)


def get_reflection_log(n: int = 10) -> list[dict]:
    """Return the last n reflection entries, most recent first."""
    r = get_redis()
    raw = r.zrevrange(_REFLECTION_LOG_KEY, 0, n - 1)
    results = []
    for item in raw:
        try:
            results.append(json.loads(item))
        except json.JSONDecodeError:
            pass
    return results


def _du_catalogue_courant(logs: list[dict]) -> list[dict]:
    """Ne garde que les entrées écrites avec le catalogue d'actions EN VIGUEUR.

    Le catalogue change — actions retirées, actions ajoutées — et le journal, lui, garde
    30 entrées. Une entrée d'un catalogue révolu montre au modèle, en exemple, une action
    qui n'existe plus ; il la redemande, et la sortie est rejetée par le validateur — un
    appel de raisonnement dépensé pour rien.

    Pas de numéro de version à incrémenter à la main : chaque entrée porte le catalogue
    qui l'a produite (`engine.log_entry`), et « courant » se lit sur l'entrée la plus
    récente qui en déclare un. Les entrées antérieures au champ n'en ont pas et tombent
    donc d'elles-mêmes.
    """
    courant = next((e.get("catalogue") for e in logs if e.get("catalogue")), None)
    if not courant:
        return []
    return [e for e in logs if e.get("catalogue") == courant]


def get_last_reflection() -> dict | None:
    """La dernière réflexion RÉUTILISABLE comme exemple — donc du catalogue courant.

    Elle est injectée telle quelle dans REFLECTION_PROMPT : une entrée d'un catalogue
    révolu y montre au modèle une action qu'il ne peut plus exécuter.
    """
    entries = _du_catalogue_courant(get_reflection_log(_REFLECTION_LOG_MAX))
    return entries[0] if entries else None

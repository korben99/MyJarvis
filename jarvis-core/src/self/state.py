"""Fondation du paquet self : constantes Redis partagées, accès à jarvis-self.json
(goals, focus, relation, opinions, incidents) et journal de réflexion.

Couche la plus basse — tous les autres sous-modules en dépendent, elle n'importe aucun.
"""

import json
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np
from config import OPINIONS_MAX_ENTRIES
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
_GAP_COUNTS_KEY = "jarvis:self:gap_counts"  # hash: slug → count
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
# différentes. Même valeur que la dédup des self_notes (`_action_update_self_note`).
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
       `_action_update_self_note`). La comparaison de topics EXACTS ne rattrapait rien
       quand le même avis revenait sous une autre étiquette — or c'est le cas courant,
       le modèle nommant librement ses topics chaque nuit.

    Seuil vérifié sur le corpus réel le 21/08/2026 : une paraphrase quasi littérale marque
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


def get_last_reflection() -> dict | None:
    entries = get_reflection_log(1)
    return entries[0] if entries else None


def _extract_behavioral_patterns(n: int = 20) -> list[str]:
    """Derive up to 5 recurring behavioral patterns from the last n reflection entries.

    Fully deterministic — no LLM call. Three signals are analysed:
      1. Action frequency: which actions dominate (≥ 20 % of cycles).
      2. Time-of-day clustering for "nothing" choices.
      3. Recurring keywords in focus fields (word seen ≥ 3 times).
    """
    logs = get_reflection_log(n)
    if not logs:
        return []

    total = len(logs)
    patterns: list[str] = []

    # 1. Action frequency
    action_counts: Counter = Counter(e.get("action", "unknown") for e in logs)
    for action, count in action_counts.most_common(3):
        pct = round(count / total * 100)
        if pct >= 20:
            patterns.append(
                f"action « {action} » choisie dans {pct}% des cycles ({count}/{total})"
            )

    # 2. "nothing" time-of-day clustering
    nothing_hours = []
    for e in logs:
        if e.get("action") == "nothing" and e.get("timestamp"):
            try:
                nothing_hours.append(datetime.fromisoformat(e["timestamp"]).hour)
            except Exception:
                pass
    if len(nothing_hours) >= 3:
        avg_h = sum(nothing_hours) / len(nothing_hours)
        if avg_h >= 20 or avg_h <= 6:
            patterns.append(
                f"tend à ne rien faire la nuit/soirée (heure moyenne: {avg_h:.0f}h)"
            )

    # 3. Recurring keywords in focus fields
    word_counts: Counter = Counter()
    for e in logs:
        for word in e.get("focus", "").lower().split():
            if len(word) > 5:
                word_counts[word] += 1
    top_words = [w for w, c in word_counts.most_common(5) if c >= 3]
    if top_words:
        patterns.append(f"focus récurrents : {', '.join(top_words[:3])}")

    return patterns[:5]

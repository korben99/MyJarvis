"""
Jarvis emotional state — 3 independent dimensions with lazy time-decay.

  humeur    : -1.0 (triste)        → 0.0 (neutre) → +1.0 (joyeux)
  confiance : -1.0 (dans le doute) → 0.0 (neutre) → +1.0 (confiant)
  energie   : -1.0 (fatigué)       → 0.0 (neutre) → +1.0 (en forme)

Decay is applied lazily on read — no scheduler needed.
All updates are atomic (lock + decay-before-write).
"""

import os
from datetime import datetime, timezone
from threading import Lock

from helpers import get_logger, redis_get_json, redis_set_json

logger = get_logger("jarvis-emotional")

_REDIS_KEY = "jarvis:emotional_state"
_lock = Lock()

_DIMS = ("humeur", "confiance", "energie")

# Each dimension decays toward 0.0 at this rate per hour
# ── Indexation de la décroissance ─────────────────────────────────────────
#
# "exchange" (défaut) : un pas de décroissance par ANALYSE de conversation.
# "clock"             : décroissance par heure écoulée — comportement d'origine.
#
# Jarvis n'existe que sollicité : 38 jours actifs sur 71 calendaires, médiane de 2 analyses
# par jour actif. Décroître pendant son absence modélise une expérience qu'il n'a pas, et
# efface un état qu'aucun événement n'a contredit. L'indexation sur l'échange fait traverser
# les absences à l'état émotionnel, ce qui est le comportement attendu d'une entité que les
# événements changent et qui s'en souvient.
#
# Choix vérifié en rejouant les 129 analyses réelles du journal, sur le critère qui compte
# — non pas la présence du bloc, mais sa VARIABILITÉ : un état affiché en permanence
# n'informe plus. Résultats à seuil 0.15 :
#
#     horloge, taux d'origine        81 % présence   43 % de changements   7 états
#     par échange, 0.15/pas          85 %            38 %                  7
#     par échange, 0.20/pas          81 %            48 %                  6   ← retenu
#     par échange, 0.25/pas          77 %            50 %                  5
#
# Un pas unique pour les trois dimensions : leur hiérarchie de persistance (le doute plus
# tenace que l'humeur) était portée par des taux horaires qui n'ont plus cours ici. La
# rétablir demanderait de la mesurer, pas de la postuler — elle ne l'a jamais été.
_DECAY_MODE = os.getenv("EMOTION_DECAY_MODE", "exchange").strip().lower()
_DECAY_PER_STEP = float(os.getenv("EMOTION_DECAY_STEP", "0.20"))

# Conservé pour _DECAY_MODE="clock" uniquement. Décroissance LINÉAIRE (v − taux × heures).
_DECAY_PER_HOUR: dict[str, float] = {
    "humeur":    float(os.getenv("EMOTION_DECAY_HUMEUR", "0.10")),
    "confiance": float(os.getenv("EMOTION_DECAY_CONFIANCE", "0.05")),
    "energie":   float(os.getenv("EMOTION_DECAY_ENERGIE", "0.15")),
}

# Minimum absolute value to be considered non-neutral (injected in prompt)
#
# Abaissé de 0.25 à 0.15 le 20/08/2026, après avoir posé l'arithmétique qui manquait.
# Une dimension n'est visible que tant que |valeur| ≥ seuil ; elle décroît de
# _DECAY_PER_HOUR par heure, et l'analyzer ne la repousse que toutes les 60 minutes.
# La durée de visibilité d'une poussée unique vaut donc :
#
#     (|delta| − seuil) / décroissance
#
# À 0.25, ce calcul donnait — mesuré sur les 17 poussées de la table :
#
#     focused  +0.10 energie  (51 occurrences, l'humeur la PLUS fréquente)  JAMAIS
#     curious  +0.10 des deux                                               JAMAIS
#     stressed les trois poussées                                           JAMAIS
#     satisfaction positive, les trois                                      JAMAIS
#     happy    +0.30 humeur                                    30 min sur 60
#
# Quatre poussées sur dix-sept franchissaient le seuil. Le bloc <etat_emotionnel_jarvis>
# était donc presque toujours absent — non parce que rien n'était écrit (humeur l'est dans
# 39 % des analyses) mais parce que rien ne survivait assez longtemps pour être injecté.
#
# À 0.15, les états francs (happy, frustrated, tired, satisfaction négative) tiennent au
# moins un cycle d'analyse, et les états doux (curious, focused) redeviennent atteignables
# par ACCUMULATION — mécanisme déjà à l'œuvre pour confiance, seule dimension qui
# fonctionnait, précisément parce qu'elle accumule.
#
# Règle à tenir si l'on retouche les deltas : pour qu'une classification unique reste
# visible un cycle entier, il faut |delta| ≥ seuil + décroissance × 1 h.
_THRESHOLD = float(os.getenv("EMOTION_THRESHOLD", "0.15"))

# Deltas applied from analyzer mood field
_MOOD_DELTAS: dict[str, dict[str, float]] = {
    "happy":      {"humeur": +0.3, "energie": +0.1},
    "frustrated": {"humeur": -0.3, "energie": -0.1},
    "stressed":   {"humeur": -0.2, "confiance": -0.1, "energie": -0.1},
    "curious":    {"humeur": +0.1, "energie": +0.1},
    "tired":      {"energie": -0.3},
    # +0.20 et non +0.10 : `energie` décroît de 0.15/h, donc une poussée de 0.10 répétée
    # toutes les 60 min (cadence de l'analyzer) est reprise en entier avant la suivante —
    # elle ne peut mathématiquement JAMAIS s'accumuler ni franchir le seuil. Or `focused`
    # est l'humeur la plus fréquente : 51 des 129 analyses au 20/08/2026, soit la
    # dimension la plus sollicitée du système, et la seule à ne rien pouvoir produire.
    #
    # Deuxième règle, complémentaire de celle du seuil : la poussée d'une humeur
    # RÉCURRENTE doit dépasser la décroissance horaire de sa dimension, sinon aucune
    # accumulation n'est possible. C'est ce qui fait fonctionner `confiance` (poussées de
    # 0.2-0.3 contre 0.05/h) et ce qui condamnait `energie`.
    "focused":    {"energie": +0.2},
    "neutral":    {},
}

# Deltas applied from analyzer satisfaction field
_SATISFACTION_DELTAS: dict[str, dict[str, float]] = {
    "positive": {"confiance": +0.2, "humeur": +0.1, "energie": +0.1},
    "negative": {"confiance": -0.3, "humeur": -0.1, "energie": -0.1},
}

# Human-readable labels for prompt injection
_LABELS: dict[str, tuple[str, str]] = {
    "humeur":    ("joyeux",    "triste"),
    "confiance": ("confiant",  "dans le doute"),
    "energie":   ("en forme",  "fatigué"),
}


# ── Internal helpers ──────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed_hours(iso: str) -> float:
    try:
        return max(
            0.0,
            (datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds()
            / 3600,
        )
    except (ValueError, TypeError):
        return 0.0


def _apply_decay(state: dict, elapsed_h: float) -> bool:
    """Decay all dims toward 0.0 in-place. Returns True if anything changed.

    En mode "exchange", `elapsed_h` est ignoré : la décroissance vaut _DECAY_PER_STEP par
    appel, et seul update_from_analysis en déclenche un. Le temps réel n'entre plus dans
    le calcul — un état traverse donc les absences inchangé.
    """
    changed = False
    for dim, rate in _DECAY_PER_HOUR.items():
        v = state.get(dim, 0.0)
        if abs(v) < 0.001:
            continue
        step = _DECAY_PER_STEP if _DECAY_MODE == "exchange" else rate * elapsed_h
        new_v = v - step if v > 0 else v + step
        new_v = max(0.0, new_v) if v > 0 else min(0.0, new_v)
        new_v = round(new_v, 3)
        if abs(new_v - v) > 0.001:
            state[dim] = new_v
            changed = True
    return changed


def _load() -> dict:
    """Load state from Redis, falling back to neutral defaults."""
    return redis_get_json(_REDIS_KEY) or {
        "humeur": 0.0,
        "confiance": 0.0,
        "energie": 0.0,
        "last_updated": _now_iso(),
    }


# ── Public API ────────────────────────────────────────────────────────────────


def get_state() -> dict:
    """Return current state. Thread-safe.

    En mode "exchange", LIRE l'état ne le fait pas vieillir : la décroissance est indexée
    sur les échanges, pas sur le temps, et un lecteur n'est pas un événement. C'est aussi
    ce qui rend l'état stable entre deux analyses — sans quoi deux appels espacés d'une
    heure rendraient deux valeurs différentes pour un même vécu.
    """
    with _lock:
        state = _load()
        if _DECAY_MODE == "exchange":
            return state
        elapsed = _elapsed_hours(state["last_updated"])
        if _apply_decay(state, elapsed):
            state["last_updated"] = _now_iso()
            redis_set_json(_REDIS_KEY, state)
    return state


def update(deltas: dict) -> None:
    """Apply dimension deltas atomically (decay first, then update, then clamp).

    Only keys present in _DIMS are accepted; unknown keys are silently ignored.

    N'applique PAS de pas de décroissance en mode "exchange" : cette fonction sert la
    boucle de réflexion (self/engine, self/actions), qui n'est pas un échange avec un
    humain. Seule update_from_analysis fait vieillir l'état.
    """
    with _lock:
        state = _load()
        if _DECAY_MODE != "exchange":
            _apply_decay(state, _elapsed_hours(state["last_updated"]))
        for dim, delta in deltas.items():
            if dim in _DECAY_PER_HOUR:
                state[dim] = round(max(-1.0, min(1.0, state.get(dim, 0.0) + delta)), 3)
        state["last_updated"] = _now_iso()
        redis_set_json(_REDIS_KEY, state)


def update_from_analysis(mood: str, satisfaction: str) -> None:
    """Applique le résultat d'une analyse de conversation.

    C'est le SEUL point qui fait vieillir l'état en mode "exchange" : une analyse est
    l'unité d'échange vécu. Le pas de décroissance s'applique même quand l'analyse ne
    produit aucun delta — « neutral » reste un échange qui a eu lieu, et sans cela une
    suite de conversations neutres laisserait l'état figé indéfiniment.
    """
    # Une valeur hors table est un SIGNAL, pas un cas nominal : le modèle a rendu une
    # étiquette qu'on ne lui a pas demandée (« sad », une fois sur 129 analyses) ou le
    # vocabulaire d'ANALYSIS_PROMPT a bougé sans que cette table suive. Sans ce journal,
    # .get(..., {}) l'avale en silence.
    if mood not in _MOOD_DELTAS:
        logger.warning("emotional_state: humeur %r hors vocabulaire — ignorée", mood)
    if satisfaction not in _SATISFACTION_DELTAS and satisfaction != "unknown":
        logger.warning("emotional_state: satisfaction %r hors vocabulaire — ignorée",
                       satisfaction)

    deltas: dict[str, float] = {}
    for k, v in _MOOD_DELTAS.get(mood, {}).items():
        deltas[k] = deltas.get(k, 0.0) + v
    for k, v in _SATISFACTION_DELTAS.get(satisfaction, {}).items():
        deltas[k] = deltas.get(k, 0.0) + v

    if _DECAY_MODE == "exchange":
        with _lock:
            state = _load()
            _apply_decay(state, 0.0)  # un pas, `elapsed_h` ignoré dans ce mode
            for dim, delta in deltas.items():
                if dim in _DECAY_PER_HOUR:
                    state[dim] = round(max(-1.0, min(1.0, state.get(dim, 0.0) + delta)), 3)
            state["last_updated"] = _now_iso()
            redis_set_json(_REDIS_KEY, state)
        logger.debug("state: pas de décroissance + %s (mood=%s satisfaction=%s)",
                     deltas or "aucun delta", mood, satisfaction)
        return

    if deltas:
        update(deltas)
        logger.debug(
            "state updated from analysis: mood=%s satisfaction=%s → %s",
            mood,
            satisfaction,
            deltas,
        )


def render_prompt_lines() -> list[str]:
    """Return non-neutral dimensions as human-readable French strings.

    Returns an empty list when all dims are near neutral — caller omits the block.
    """
    state = get_state()
    lines = []
    for dim, (pos_label, neg_label) in _LABELS.items():
        v = state.get(dim, 0.0)
        if v >= _THRESHOLD:
            lines.append(pos_label)
        elif v <= -_THRESHOLD:
            lines.append(neg_label)
    return lines


def describe() -> str:
    """One-line French summary for prompts (proactive messaging, self-reflection)."""
    lines = render_prompt_lines()
    return ", ".join(lines) if lines else "neutre"

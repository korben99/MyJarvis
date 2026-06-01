"""
Jarvis emotional state — 3 independent dimensions with lazy time-decay.

  humeur    : -1.0 (triste)        → 0.0 (neutre) → +1.0 (joyeux)
  confiance : -1.0 (dans le doute) → 0.0 (neutre) → +1.0 (confiant)
  energie   : -1.0 (fatigué)       → 0.0 (neutre) → +1.0 (en forme)

Decay is applied lazily on read — no scheduler needed.
All updates are atomic (lock + decay-before-write).
"""

from datetime import datetime, timezone
from threading import Lock

from helpers import get_logger, redis_get_json, redis_set_json

logger = get_logger("jarvis-emotional")

_REDIS_KEY = "jarvis:emotional_state"
_lock = Lock()

_DIMS = ("humeur", "confiance", "energie")

# Each dimension decays toward 0.0 at this rate per hour
_DECAY_PER_HOUR: dict[str, float] = {
    "humeur":    0.10,  # neutral in ~10 h
    "confiance": 0.05,  # slower — doubt lingers (~20 h)
    "energie":   0.15,  # recovers fast (~7 h)
}

# Minimum absolute value to be considered non-neutral (injected in prompt)
_THRESHOLD = 0.25

# Deltas applied from analyzer mood field
_MOOD_DELTAS: dict[str, dict[str, float]] = {
    "happy":      {"humeur": +0.3, "energie": +0.1},
    "frustrated": {"humeur": -0.3, "energie": -0.1},
    "stressed":   {"humeur": -0.2, "confiance": -0.1, "energie": -0.1},
    "curious":    {"humeur": +0.1, "energie": +0.1},
    "tired":      {"energie": -0.3},
    "focused":    {"energie": +0.1},
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
    """Decay all dims toward 0.0 in-place. Returns True if anything changed."""
    changed = False
    for dim, rate in _DECAY_PER_HOUR.items():
        v = state.get(dim, 0.0)
        if abs(v) < 0.001:
            continue
        step = rate * elapsed_h
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
    """Return current state with lazy decay applied. Thread-safe."""
    with _lock:
        state = _load()
        elapsed = _elapsed_hours(state["last_updated"])
        if _apply_decay(state, elapsed):
            state["last_updated"] = _now_iso()
            redis_set_json(_REDIS_KEY, state)
    return state


def update(deltas: dict) -> None:
    """Apply dimension deltas atomically (decay first, then update, then clamp).

    Only keys present in _DIMS are accepted; unknown keys are silently ignored.
    """
    with _lock:
        state = _load()
        elapsed = _elapsed_hours(state["last_updated"])
        _apply_decay(state, elapsed)
        for dim, delta in deltas.items():
            if dim in _DECAY_PER_HOUR:
                state[dim] = round(max(-1.0, min(1.0, state.get(dim, 0.0) + delta)), 3)
        state["last_updated"] = _now_iso()
        redis_set_json(_REDIS_KEY, state)


def update_from_analysis(mood: str, satisfaction: str) -> None:
    """Apply deltas derived from an analyzer result. No-op if both are unknown."""
    deltas: dict[str, float] = {}
    for k, v in _MOOD_DELTAS.get(mood, {}).items():
        deltas[k] = deltas.get(k, 0.0) + v
    for k, v in _SATISFACTION_DELTAS.get(satisfaction, {}).items():
        deltas[k] = deltas.get(k, 0.0) + v
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

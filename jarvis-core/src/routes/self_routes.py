"""routes/self_routes.py — Jarvis self-state and reflection endpoints."""

from fastapi import APIRouter, Depends

from memory import get_self_memory
from self import get_reflection_log, run_self_reflection

from .garde import exige_admin

router = APIRouter(tags=["self"])


@router.get("/self/state")
async def self_state():
    """Return Jarvis's current identity, goals, focus, and last reflection."""
    data     = get_self_memory()
    last_ref = get_reflection_log(1)
    return {
        "identity":         data.get("identity", {}),
        "goals":            data.get("goals", []),
        "current_focus":    data.get("current_focus", ""),
        "last_reflection":  data.get("last_reflection", ""),
        "reflection_count": data.get("reflection_count", 0),
        "last_action":      last_ref[0] if last_ref else None,
        "introspection":    data.get("self_introspection", {}),
        "user_relations":   data.get("user_relations", {}),
    }


@router.get("/self/log")
async def self_log(n: int = 10):
    """Return the last n reflection log entries."""
    return {"log": get_reflection_log(min(n, 30))}


@router.post("/self/reflect")
async def self_reflect_now(demandeur: str = Depends(exige_admin)):
    """Trigger an immediate reflection cycle (for testing / manual trigger).

    Réservée aux administrateurs : un cycle mobilise le GPU plusieurs minutes et chaque
    tour de chat attend derrière. Les deux routes de consultation au-dessus restent
    ouvertes — elles ne dépensent rien."""
    result = await run_self_reflection()
    return result


@router.post("/self/maintenance")
async def self_maintenance(minutes: int = 60, reason: str = "maintenance",
                           demandeur: str = Depends(exige_admin)):
    """Ouvre une fenêtre de maintenance : les incidents des prochaines `minutes` sont tagués
    `maintenance` (pas de peur, pas de trauma). À poser avant une intervention ad-hoc.

    Réservée aux administrateurs parce qu'elle éteint un signal : pendant la fenêtre, une
    panne réelle est enregistrée sans sévérité et ne pèse plus sur l'état. C'est une
    décision d'exploitant, pas une commande de passage."""
    from vitals import set_maintenance

    set_maintenance(minutes, reason)
    return {"maintenance": True, "minutes": minutes, "reason": reason}

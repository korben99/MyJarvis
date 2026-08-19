"""routes/agent_routes.py — création et suivi des tâches agentiques.

Réservé aux administrateurs (USER_ADMINS). Une tâche agentique écrit sur le disque de la
machine et consomme le GPU pendant plusieurs minutes : ce n'est pas une surface qu'on
ouvre à tous les utilisateurs déclarés tant que le périmètre n'est pas stabilisé.
"""

import json
import os

from agent import create_task, get_task, list_tasks, request_cancel
from config import AGENT_ENABLED, AGENT_MAX_STEPS, USER_ADMINS
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["agent"])


class _NewTask(BaseModel):
    user_code: str
    objective: str


def _require_enabled() -> None:
    if not AGENT_ENABLED:
        raise HTTPException(503, "AGENT_ENABLED=false — boucle agentique désactivée")


@router.post("/agent/tasks", status_code=202)
async def post_task(req: _NewTask):
    """Met une tâche en file. Retourne immédiatement : l'exécution est asynchrone."""
    _require_enabled()
    if req.user_code not in USER_ADMINS:
        raise HTTPException(403, "réservé aux administrateurs")
    objective = req.objective.strip()
    if len(objective) < 10:
        raise HTTPException(422, "objectif trop court pour être exécutable")
    return create_task(req.user_code, objective)


@router.get("/agent/tasks")
async def get_tasks(limit: int = 20, user_code: str | None = None):
    """Sans `user_code`, renvoie les tâches de tous les utilisateurs — vue d'exploitation."""
    _require_enabled()
    return {
        "tasks": list_tasks(min(limit, 100), user_code=user_code),
        "max_steps": AGENT_MAX_STEPS,
    }


@router.get("/agent/tasks/{task_id}")
async def get_one(task_id: str):
    _require_enabled()
    task = get_task(task_id)
    if not task:
        raise HTTPException(404, "tâche inconnue")
    return task


@router.post("/agent/tasks/{task_id}/cancel")
async def cancel(task_id: str):
    """Demande l'annulation. Prise en compte entre deux pas, pas au milieu d'un pas."""
    _require_enabled()
    if not request_cancel(task_id):
        raise HTTPException(409, "tâche inconnue ou déjà terminée")
    return {"cancel_requested": True, "id": task_id}


@router.get("/agent/tasks/{task_id}/transcript")
async def transcript(task_id: str, n: int = 50):
    """Les n derniers événements de la tâche — c'est là qu'on voit ce que l'agent a fait."""
    _require_enabled()
    task = get_task(task_id)
    if not task:
        raise HTTPException(404, "tâche inconnue")
    path = os.path.join(task["workspace"], "transcript.jsonl")
    if not os.path.exists(path):
        return {"id": task_id, "events": []}
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()[-min(n, 500):]
    events = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {"id": task_id, "events": events}

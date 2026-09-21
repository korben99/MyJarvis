"""routes/agent_routes.py — création et suivi des tâches agentiques.

Réservé aux administrateurs (USER_ADMINS). Une tâche agentique écrit sur le disque de la
machine et consomme le GPU pendant plusieurs minutes : ce n'est pas une surface qu'on
ouvre à tous les utilisateurs déclarés tant que le périmètre n'est pas stabilisé.

La garde est posée sur le ROUTEUR et non route par route. Une garde portée par le corps de
la requête ne protège que les routes qui en ont un, et laisse ouvertes les lectures — or
celles-ci rendent l'enregistrement entier d'une tâche, `user_code` compris, c'est-à-dire
le secret qui autorise à en créer une. Sur le routeur, une route ajoutée plus tard est
protégée sans que personne ait à y penser.
"""

import json
import os

from agent import create_task, get_task, list_tasks, request_cancel
from config import AGENT_ENABLED, AGENT_MAX_STEPS, USER_ADMINS
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel


def exige_admin(authorization: str = Header(default=None)) -> str:
    """Code administrateur porté par l'en-tête, ou lève.

    En-tête `Authorization: Bearer <code>`, comme les routes de portefeuille : le code est
    un secret, et un secret n'a pas à voyager dans une chaîne de requête, que les journaux
    et l'historique du navigateur conservent.
    """
    code = (
        authorization[7:].strip()
        if authorization and authorization.startswith("Bearer ")
        else ""
    )
    if code not in USER_ADMINS:
        raise HTTPException(403, "réservé aux administrateurs")
    return code


router = APIRouter(tags=["agent"], dependencies=[Depends(exige_admin)])


class _NewTask(BaseModel):
    user_code: str
    objective: str


def _require_enabled() -> None:
    if not AGENT_ENABLED:
        raise HTTPException(503, "AGENT_ENABLED=false — boucle agentique désactivée")


def _exige_soi_meme(demandeur: str, declare: str) -> None:
    """Refuse qu'un administrateur agisse au nom d'un autre.

    Le champ `user_code` désigne le propriétaire de la tâche : c'est lui qui reçoit le
    livrable par courriel et la notification. Sans cette égalité, un jeton valide
    ferait travailler l'agent au nom de quelqu'un d'autre et expédierait le résultat dans
    sa boîte. Deux administrateurs restent deux personnes.
    """
    if demandeur != declare:
        raise HTTPException(403, "le code déclaré n'est pas celui du jeton")


@router.post("/agent/tasks", status_code=202)
async def post_task(req: _NewTask, demandeur: str = Depends(exige_admin)):
    """Met une tâche en file. Retourne immédiatement : l'exécution est asynchrone."""
    _require_enabled()
    _exige_soi_meme(demandeur, req.user_code)
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


class _Autocode(BaseModel):
    user_code: str
    # S'arrête après la sélection : de quoi contrôler ce que Jarvis choisirait, et
    # pourquoi, sans dépenser vingt minutes de GPU ni produire de patch.
    dry_run: bool = False

    # Part sans cible désignée : l'agent parcourt et juge de ce qui mérite d'être signalé.
    # Manuel uniquement — le planificateur n'a pas ce drapeau.
    revue: bool = False

    # Borne la revue à un sous-dossier (ex. « jarvis-core/src/memory »), relatif à `repo/`.
    # Vide = tout le dépôt. Sans effet hors d'une revue.
    perimetre: str = ""


@router.post("/agent/autocode")
async def post_autocode(req: _Autocode, demandeur: str = Depends(exige_admin)):
    """Rejoue le cycle d'autocoding à la main, sans attendre la nuit.

    Bloquant, à dessein : un cycle complet dure une vingtaine de minutes et l'appelant est
    un administrateur devant son terminal, qui veut le résultat — pas un identifiant à
    resonder. Le mode nocturne, lui, ne passe pas par ici.
    """
    _require_enabled()
    _exige_soi_meme(demandeur, req.user_code)

    from autocode import run_nightly_autocode

    return await run_nightly_autocode(
        dry_run=req.dry_run, revue=req.revue, perimetre=req.perimetre
    )


@router.get("/agent/autocode/journal")
async def get_autocode_journal(n: int = 10):
    """Les derniers cycles avec leur verdict, et le patch qui attend une décision."""
    _require_enabled()

    from autocode import constats, store

    eligibles, exclus = constats.recueillir()
    return {
        "en_attente": store.patch_en_attente(),
        "journal": store.journal(n),
        "constats": [
            {"id": c["id"], "constat": c["constat"], "origine": c["origine"]}
            for c in eligibles
        ],
        "ecartes": [{"id": i, "motif": m} for i, m in exclus],
    }

"""Persistance des tâches agentiques : enregistrement Redis + file d'attente.

Découpage volontaire entre les deux supports :

  Redis            état de la tâche (statut, compteurs, résultat) et file d'attente.
                   Petit, interrogeable, expirable.
  workspace/       messages.json (contexte complet, sert la reprise après redémarrage)
                   et transcript.jsonl (trace lisible, append-only).

Le contexte d'un agent pèse vite des dizaines de milliers de tokens : il n'a rien à faire
dans Redis, qui sert aussi les chemins chauds du chat.
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone

from config import AGENT_TASK_TTL
from helpers import get_logger, get_redis

from .sandbox import task_workspace

logger = get_logger("jarvis-agent")

_TASK_KEY = "jarvis:agent:task:{}"
_QUEUE_KEY = "jarvis:agent:queue"
_INDEX_KEY = "jarvis:agent:index"
_CANCEL_KEY = "jarvis:agent:cancel:{}"

# Statuts. `interrupted` n'est pas un échec : c'est une tâche coupée par un arrêt du
# service, remise en file au démarrage suivant et reprise sur son contexte sauvegardé.
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
STATUS_INTERRUPTED = "interrupted"

_OPEN_STATUSES = (STATUS_QUEUED, STATUS_RUNNING, STATUS_INTERRUPTED)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Enregistrement ────────────────────────────────────────────────────────


def create_task(user_code: str, objective: str) -> dict:
    """Crée une tâche, son workspace, et la pousse en file d'attente."""
    task_id = uuid.uuid4().hex[:16]
    task = {
        "id": task_id,
        "user_code": user_code,
        "objective": objective.strip(),
        "status": STATUS_QUEUED,
        "created_at": now_iso(),
        "started_at": None,
        "finished_at": None,
        "steps": 0,
        "workspace": task_workspace(task_id),
        "result": "",
        "error": "",
        "deliverables": [],
    }
    save_task(task)
    r = get_redis()
    r.zadd(_INDEX_KEY, {task_id: time.time()})
    r.rpush(_QUEUE_KEY, task_id)
    logger.info("agent: tâche %s créée par %s — %s", task_id, user_code, objective[:80])
    return task


def save_task(task: dict) -> None:
    get_redis().setex(
        _TASK_KEY.format(task["id"]), AGENT_TASK_TTL, json.dumps(task, ensure_ascii=False)
    )


def get_task(task_id: str) -> dict | None:
    raw = get_redis().get(_TASK_KEY.format(task_id))
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("agent: enregistrement illisible pour %s", task_id)
        return None


def list_tasks(limit: int = 20, user_code: str | None = None) -> list[dict]:
    """Tâches les plus récentes d'abord. Les entrées expirées sont purgées de l'index.

    `user_code` filtre sur le demandeur. À passer systématiquement dès qu'une liste est
    rendue à un utilisateur : Jarvis est multi-utilisateurs et ne montre jamais à l'un ce
    qui appartient à l'autre — deux administrateurs restent deux personnes.

    L'index entier est parcouru quand on filtre (borné par AGENT_TASK_TTL, volume faible) :
    ne lire que les `limit` premiers rendrait une liste vide dès qu'un autre utilisateur a
    posté les dernières tâches.
    """
    r = get_redis()
    ids = r.zrevrange(_INDEX_KEY, 0, -1 if user_code else max(limit, 1) - 1)
    tasks, stale = [], []
    for task_id in ids:
        task = get_task(task_id)
        if task is None:
            stale.append(task_id)
            continue
        if user_code and task.get("user_code") != user_code:
            continue
        tasks.append(task)
        if len(tasks) >= limit:
            break
    if stale:
        r.zrem(_INDEX_KEY, *stale)
    return tasks


# ── File d'attente ────────────────────────────────────────────────────────


def pop_next(timeout: int = 5) -> str | None:
    """Retire l'id de la prochaine tâche à exécuter. Bloque au plus `timeout` secondes.

    BLPOP et non LPOP+sleep : le worker doit repartir dès qu'une tâche est postée, sans
    latence de polling. Le timeout borne l'attente pour que l'annulation de la tâche
    asyncio du worker soit prise en compte à l'arrêt du service.
    """
    item = get_redis().blpop(_QUEUE_KEY, timeout=timeout)
    return item[1] if item else None


def requeue_interrupted() -> int:
    """Au démarrage : remet en file les tâches laissées en cours par un arrêt.

    Le contexte est dans messages.json, la boucle le rechargera — une tâche coupée en
    plein milieu reprend donc là où elle en était plutôt que de repartir de zéro.
    """
    count = 0
    for task in list_tasks(limit=100):
        if task.get("status") == STATUS_RUNNING:
            task["status"] = STATUS_INTERRUPTED
            save_task(task)
            get_redis().rpush(_QUEUE_KEY, task["id"])
            count += 1
    if count:
        logger.info("agent: %d tâche(s) interrompue(s) remise(s) en file", count)
    return count


# ── Annulation ────────────────────────────────────────────────────────────


def request_cancel(task_id: str) -> bool:
    """Pose le drapeau d'annulation. La boucle le lit entre deux pas.

    Drapeau séparé et non champ de l'enregistrement : le worker réécrit l'enregistrement à
    chaque pas et écraserait un champ posé entre-temps par la route d'annulation.
    """
    task = get_task(task_id)
    if not task or task["status"] not in _OPEN_STATUSES:
        return False
    get_redis().setex(_CANCEL_KEY.format(task_id), AGENT_TASK_TTL, "1")
    return True


def is_cancelled(task_id: str) -> bool:
    return bool(get_redis().exists(_CANCEL_KEY.format(task_id)))


# ── Contexte et trace (dans le workspace) ─────────────────────────────────


def _messages_path(task: dict) -> str:
    return os.path.join(task["workspace"], "messages.json")


def save_messages(task: dict, messages: list[dict]) -> None:
    """Sauvegarde le contexte après chaque pas. Écriture atomique.

    Un remplacement direct laisserait un JSON tronqué si le service tombe pendant
    l'écriture — et c'est précisément le cas que ce fichier doit couvrir.
    """
    path = _messages_path(task)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False)
    os.replace(tmp, path)


def load_messages(task: dict) -> list[dict] | None:
    path = _messages_path(task)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("agent: messages.json illisible pour %s (%s)", task["id"], exc)
        return None


def append_transcript(task: dict, entry: dict) -> None:
    """Ajoute une ligne à la trace lisible. Ne doit jamais faire échouer un pas."""
    try:
        entry = {"ts": now_iso(), **entry}
        with open(os.path.join(task["workspace"], "transcript.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("agent: transcript non écrit pour %s (%s)", task["id"], exc)

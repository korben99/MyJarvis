"""Worker : consomme la file d'attente, une tâche à la fois, et notifie à la fin.

Concurrence 1, volontairement. Deux tâches en parallèle ne gagneraient rien — elles se
disputeraient le même GPU, déjà sérialisé par _infer_lock — et doubleraient la pression sur
le cache LRU de prompts (LRU_KV_SIZE=4 séquences), dont un agent occupe une entrée en
croissance continue. La file, elle, est illimitée.
"""

import asyncio

from config import AGENT_ENABLED
from helpers import get_logger

from . import store
from .loop import run_task

logger = get_logger("jarvis-agent")

_worker_task: asyncio.Task | None = None

# Cooldown de notification propre à chaque tâche : jamais de suppression croisée entre
# deux tâches, mais pas de double envoi pour la même.
_PUSH_COOLDOWN_TTL = 3600

# Longueur utile d'une notification iOS ; le détail reste consultable dans la tâche.
_PUSH_MAX_CHARS = 500


def _notify(task: dict) -> None:
    """Prévient l'utilisateur que sa tâche est terminée. Ne doit jamais faire échouer le worker."""
    status = task["status"]
    if status == store.STATUS_CANCELLED:
        return

    if status == store.STATUS_DONE:
        # Une notification se lit sur un écran verrouillé : le résumé complet vit dans la
        # tâche, pas dans le push.
        body = (task["result"] or "Tâche terminée.").strip()
        if len(body) > _PUSH_MAX_CHARS:
            body = body[:_PUSH_MAX_CHARS].rsplit(" ", 1)[0] + "…"
        files = task.get("deliverables") or []
        if files:
            body += f"\n\nFichiers : {', '.join(files)} (dans {task['workspace']})"
    else:
        body = f"Ta tâche a échoué : {task.get('error') or 'raison inconnue'}\n\n« {task['objective'][:120]} »"

    try:
        # Import tardif et volontairement de la fonction interne : elle porte déjà la file
        # Redis de repli, l'APNs immédiat et l'injection dans la conversation iOS. La
        # dupliquer ici ferait diverger deux chemins de livraison push.
        from self.actions import _deliver_push

        error = _deliver_push(
            task["user_code"], body,
            cooldown_key=f"jarvis:agent:push:{task['id']}",
            cooldown_ttl=_PUSH_COOLDOWN_TTL,
        )
        if error:
            logger.info("agent: push non livré pour %s (%s)", task["id"], error)
    except Exception as exc:
        logger.warning("agent: notification en échec pour %s (%s)", task["id"], exc)


async def _run() -> None:
    """Boucle du worker. Ne s'arrête que sur annulation de la tâche asyncio."""
    await asyncio.to_thread(store.requeue_interrupted)
    logger.info("agent: worker démarré")

    while True:
        # BLPOP dans un thread : bloquant côté Redis, il figerait la boucle d'événements
        # — et donc tout le chat — s'il était appelé directement.
        task_id = await asyncio.to_thread(store.pop_next, 5)
        if not task_id:
            continue

        task = store.get_task(task_id)
        if task is None:
            logger.warning("agent: tâche %s en file mais introuvable — ignorée", task_id)
            continue

        if store.is_cancelled(task_id):
            task["status"] = store.STATUS_CANCELLED
            task["finished_at"] = store.now_iso()
            store.save_task(task)
            continue

        try:
            finished = await run_task(task)
        except asyncio.CancelledError:
            raise
        except Exception:
            # run_task avale déjà ses erreurs ; ce filet couvre ce qui déborderait
            # (Redis indisponible, disque plein) sans tuer le worker.
            logger.exception("agent: échec non rattrapé sur %s", task_id)
            continue

        await asyncio.to_thread(_notify, finished)


def start_worker() -> None:
    """Démarre le worker si AGENT_ENABLED. Idempotent."""
    global _worker_task
    if not AGENT_ENABLED:
        logger.info("agent: désactivé (AGENT_ENABLED=false)")
        return
    if _worker_task and not _worker_task.done():
        return
    _worker_task = asyncio.create_task(_run())


async def stop_worker() -> None:
    """Annule le worker et attend son arrêt.

    Une tâche en cours reçoit CancelledError dans run_task, qui sauvegarde son contexte et
    laisse le statut à running : elle sera reprise au démarrage suivant.
    """
    global _worker_task
    if _worker_task is None:
        return
    _worker_task.cancel()
    try:
        await _worker_task
    except asyncio.CancelledError:
        pass
    _worker_task = None
    logger.info("agent: worker arrêté")

"""Worker : consomme la file d'attente, une tâche à la fois, et notifie à la fin.

Concurrence 1, volontairement. Deux tâches en parallèle ne gagneraient rien — elles se
disputeraient le même GPU, déjà sérialisé par _infer_lock — et doubleraient la pression sur
le cache LRU de prompts (LRU_KV_SIZE=4 séquences), dont un agent occupe une entrée en
croissance continue. La file, elle, est illimitée.
"""

import asyncio

from config import AGENT_ENABLED
from helpers import get_logger

from . import report, store
from .loop import run_task

logger = get_logger("jarvis-agent")

_worker_task: asyncio.Task | None = None

# Cooldown de notification propre à chaque tâche : jamais de suppression croisée entre
# deux tâches, mais pas de double envoi pour la même.
_PUSH_COOLDOWN_TTL = 3600

# Longueur utile d'une notification iOS ; le détail reste consultable dans la tâche.
_PUSH_MAX_CHARS = 500

# Pause après une panne de tour, pour ne pas boucler à plein régime sur une erreur qui dure.
_PAUSE_APRES_PANNE = 5.0


def _notify(task: dict) -> None:
    """Prévient l'utilisateur que sa tâche est terminée. Ne doit jamais faire échouer le worker."""
    status = task["status"]
    if status == store.STATUS_CANCELLED:
        return

    # Une tâche d'autocoding n'est pas finie quand la boucle l'est : son patch doit encore
    # être mesuré, jugé et mis en forme. Notifier ici enverrait le brouillon de l'agent
    # quelques minutes avant le rapport réel — deux courriels, dont le premier est faux.
    # C'est `autocode/` qui restitue, une fois qu'il a quelque chose à dire.
    if task.get("origin") == "autocode":
        return

    if status == store.STATUS_DONE:
        # Le courriel part AVANT le push : il porte le livrable, le push n'en porte que
        # l'annonce. Cet ordre permet aussi d'écrire « envoyé par mail » dans la
        # notification, donc de savoir sans ouvrir sa boîte s'il y a quelque chose à lire.
        envoye = report.envoyer(task)

        # Une notification se lit sur un écran verrouillé : le résumé complet vit dans la
        # tâche, pas dans le push.
        body = (task["result"] or "Tâche terminée.").strip()
        if len(body) > _PUSH_MAX_CHARS:
            body = body[:_PUSH_MAX_CHARS].rsplit(" ", 1)[0] + "…"
        files = task.get("deliverables") or []
        if files:
            body += f"\n\nFichiers : {', '.join(files)}"
            body += " — envoyé par mail." if envoye else f" (dans {task['workspace']})"

        # Une tâche « terminée » l'est parfois par épuisement du budget ou par une boucle
        # détectée : le statut reste `done` parce qu'un livrable a pu être produit, et la
        # raison de l'arrêt vit dans `error`. Sans cette ligne, la notification annonce un
        # travail fini là où la phase de conclusion a sauvé ce qui pouvait l'être.
        if (motif := (task.get("error") or "").strip()):
            body += f"\n\n⚠ Arrêt anticipé : {motif}"
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
        # Le filet porte sur le TOUR ENTIER, et pas seulement sur `run_task`. Les quatre
        # appels Redis du tour — file, lecture, drapeau d'annulation, sauvegarde — sont hors
        # de ce `run_task` : une exception qui monte de l'un d'eux sort de la boucle et le
        # worker cesse de consommer la file, sans s'arrêter ni se plaindre. C'est la panne la
        # plus coûteuse du dispositif, parce qu'elle est indolore.
        try:
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
        except asyncio.CancelledError:
            # Arrêt du service : la seule sortie légitime de cette boucle.
            raise
        except Exception:
            logger.exception("agent: tour de worker en échec — la boucle continue")
            # Une panne du magasin persiste au-delà d'un tour. Sans cette pause, la boucle
            # repart aussitôt sur la même erreur et écrit des milliers de traces par minute.
            await asyncio.sleep(_PAUSE_APRES_PANNE)


def start_worker() -> None:
    """Démarre le worker si AGENT_ENABLED. Idempotent."""
    global _worker_task
    if not AGENT_ENABLED:
        logger.info("agent: désactivé (AGENT_ENABLED=false)")
        return
    if _worker_task and not _worker_task.done():
        return
    _worker_task = asyncio.create_task(_run())


def worker_vivant() -> bool:
    """True si la boucle du worker tourne encore."""
    return _worker_task is not None and not _worker_task.done()


def _motif_de_mort() -> str:
    """Pourquoi le worker s'est arrêté, en lisant l'exception de sa Task.

    Cette lecture est ce qui rend la panne VISIBLE : sans elle, l'exception reste stockée
    dans l'objet Task, que la référence globale `_worker_task` empêche de collecter — donc
    asyncio ne signale jamais « Task exception was never retrieved », et rien, nulle part,
    ne dit que l'exécution autonome s'est arrêtée.
    """
    if _worker_task is None:
        return "jamais démarré"
    try:
        exc = _worker_task.exception()
    except asyncio.CancelledError:
        return "annulé"
    except asyncio.InvalidStateError:
        return "encore en cours"
    return f"{type(exc).__name__}: {exc}" if exc else "terminé sans erreur"


def surveiller() -> None:
    """Relève le worker s'il est tombé. Ronde périodique du planificateur.

    Le filet de `_run` rend cette mort très improbable, il ne la rend pas impossible : une
    exception hors de la boucle, ou un `BaseException`, sort quand même. Une file qui ne se
    vide plus ne se voit sur aucun cadran — d'où l'incident, qui emprunte le chemin par
    lequel une aggravation de CVE remonte déjà jusqu'à l'administrateur.
    """
    if not AGENT_ENABLED or worker_vivant():
        return

    motif = _motif_de_mort()
    logger.error("agent: worker arrêté (%s) — relance", motif)
    try:
        from vitals import mark_incident

        mark_incident(
            "agent_worker_arrete",
            f"worker agentique relevé — {motif}",
            severity="alerte",
        )
    except Exception as exc:
        logger.debug("agent: incident non posé (%s)", exc)

    # `start_worker` crée la tâche : la Task morte ne passe plus sa garde `done()`.
    start_worker()


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

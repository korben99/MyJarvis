"""Boucle agentique — exécution autonome d'une tâche confiée explicitement par un humain.

Régime distinct du proto-self (`self/`), et il faut le garder distinct :

    self/     observe, réfléchit, PROPOSE. N'agit jamais sur le monde de sa propre
              initiative. Se déclenche tout seul, toutes les REFLECTION_INTERVAL_HOURS.
    agent/    AGIT — surfe, lit, écrit des fichiers. Ne se déclenche jamais tout seul :
              une tâche vient d'un humain, par /agent/tasks.

Les mélanger reviendrait à réviser d'un coup toutes les garanties du proto-self.

Périmètre Phase 1 : lecture du monde (web, RAG, code source en lecture seule) et écriture
confinée au workspace de la tâche. Aucune exécution — le shell arrive en Phase 2.

    store     enregistrement Redis + file d'attente + contexte sur disque
    sandbox   confinement des chemins
    tools     schémas d'outils et implémentations
    loop      objectif → outil → observation → finish
    worker    consommation de la file, une tâche à la fois, notification finale
"""

from .store import create_task, get_task, list_tasks, request_cancel
from .worker import start_worker, stop_worker

__all__ = [
    "create_task",
    "get_task",
    "list_tasks",
    "request_cancel",
    "start_worker",
    "stop_worker",
]

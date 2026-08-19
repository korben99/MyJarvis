"""Confinement des chemins pour la boucle agentique.

Une seule règle, et elle est mécanique : tout chemin manipulé par l'agent est résolu
(`realpath`, donc symlinks suivis) puis comparé à une racine autorisée résolue de la même
façon. Un chemin qui n'est pas SOUS une racine est refusé — pas corrigé, pas rapproché du
plus ressemblant : refusé, avec un message que le modèle peut lire et corriger.

Deux régimes :
  écriture  →  uniquement le workspace de la tâche courante
  lecture   →  le workspace + AGENT_READONLY_ROOTS (code source, DOCS, scripts)

Ce module ne fait AUCUNE I/O de contenu : il ne rend que des chemins validés. Les outils
lisent et écrivent, lui décide de ce qui est atteignable. Phase 1 : aucune exécution — pas
de shell ici, c'est l'objet de la Phase 2 (seatbelt + denylist + quotas).
"""

import os

from config import AGENT_READONLY_ROOTS, AGENT_WORKSPACE


class SandboxError(ValueError):
    """Chemin hors des racines autorisées. Le message est destiné au modèle."""


def task_workspace(task_id: str, create: bool = True) -> str:
    """Chemin absolu du workspace d'une tâche. Le crée au besoin.

    task_id vient de uuid4().hex côté store : on revalide quand même, ce chemin est
    concaténé — un id contenant '..' ferait sortir de la racine avant toute autre garde.
    """
    if not task_id or not task_id.isalnum():
        raise SandboxError(f"task_id invalide: {task_id!r}")
    path = os.path.join(AGENT_WORKSPACE, task_id)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def _resolved_roots(task_id: str, write: bool) -> list[str]:
    roots = [task_workspace(task_id)]
    if not write:
        roots.extend(AGENT_READONLY_ROOTS)
    return [os.path.realpath(r) for r in roots]


def _is_within(candidate: str, root: str) -> bool:
    """True si candidate est root lui-même ou strictement dessous.

    Comparaison par préfixe avec séparateur explicite, et non startswith nu :
    '/opt/jarvis/scripts' ne doit PAS matcher la racine '/opt/jarvis/script'.
    """
    return candidate == root or candidate.startswith(root + os.sep)


def resolve(task_id: str, path: str, *, write: bool) -> str:
    """Résout `path` (absolu ou relatif au workspace) et vérifie qu'il est autorisé.

    Retourne le chemin réel. Lève SandboxError sinon.

    Le relatif est résolu depuis le workspace de la tâche : c'est le répertoire courant
    du point de vue de l'agent, et le prompt le lui présente comme tel.
    """
    workspace = task_workspace(task_id)
    raw = path if os.path.isabs(path) else os.path.join(workspace, path)

    # realpath sur le chemin ENTIER, dernier composant compris. Ne résoudre que le parent
    # laisserait passer un lien symbolique portant sur le fichier lui-même : un
    # 'leak.txt -> /opt/jarvis/.env' déposé dans le workspace serait alors lisible, et
    # pire, inscriptible à travers le lien. realpath d'un chemin inexistant est sûr — il
    # résout les composants qui existent et laisse le reste tel quel.
    resolved = os.path.realpath(raw)

    roots = _resolved_roots(task_id, write)
    if not any(_is_within(resolved, root) for root in roots):
        zone = "en écriture" if write else "en lecture"
        raise SandboxError(
            f"chemin refusé {zone}: {path!r} sort des zones autorisées "
            f"({', '.join(roots)}). Travaille dans ton workspace."
        )
    return resolved


def ensure_parent(path: str) -> None:
    """Crée le dossier parent d'un fichier à écrire. À n'appeler qu'après resolve()."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def relative(task_id: str, path: str) -> str:
    """Chemin affiché à l'agent : relatif au workspace quand il y est, absolu sinon.

    Évite de saturer le contexte avec le préfixe du workspace à chaque ligne de listing,
    et garde les lectures hors workspace (code source) visuellement distinctes.
    """
    workspace = os.path.realpath(task_workspace(task_id))
    real = os.path.realpath(path)
    return os.path.relpath(real, workspace) if _is_within(real, workspace) else real

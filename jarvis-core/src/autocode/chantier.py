"""Phase 3 — le chantier : deux arbres jetables, une tâche agentique, un diff. Sans LLM.

    <workspace>/repo       l'agent y travaille — seule zone où seatbelt le laisse écrire
    <workspace>/repo_ref   copie vierge de HEAD, pour rejouer le test neuf contre l'avant

Les placer dans le workspace n'est pas un rangement : la zone d'écriture autorisée par le
bac à sable est déjà exactement celle-là, donc aucun réglage de sandbox à toucher.

Le `.git` d'un worktree pointe hors de cette zone : l'agent ne peut pas committer même
s'il essaie. Le prompt le lui dit, le noyau le garantit.

git n'est jamais appelé DEPUIS le bac à sable — la création des arbres et la production du
patch sont le fait de l'orchestrateur. L'agent, lui, n'a que des fichiers.
"""

import asyncio
import os
import shutil
from datetime import datetime

from config import (
    AUTOCODE_DIR,
    AUTOCODE_MAX_DIFF_LINES,
    AUTOCODE_MAX_STEPS,
    AUTOCODE_TIMEOUT_MINUTES,
    JARVIS_ROOT,
)
from helpers import get_logger
from prompts import get_prompt

from .constats import est_protege

logger = get_logger("jarvis-autocode")

ORIGINE = "autocode"

# Un cycle complet tient largement dedans (40 pas × ~25 s ≈ 17 min, plafond de 90 min) ;
# au-delà, la tâche est bloquée et l'attente n'y changera rien.
_ATTENTE_MAX_SECONDES = 3 * 3600
_ATTENTE_PAS = 20


# ── Arbres ────────────────────────────────────────────────────────────────


async def _git(*args: str) -> tuple[int, str]:
    """git depuis la racine du dépôt, hors bac à sable. (code, sortie)."""
    proc = await asyncio.create_subprocess_exec(
        "git", "-C", JARVIS_ROOT, *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    sortie, _ = await proc.communicate()
    return proc.returncode, (sortie or b"").decode("utf-8", errors="replace")


async def _git_dans(arbre: str, *args: str) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        "git", "-C", arbre, *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    sortie, _ = await proc.communicate()
    return proc.returncode, (sortie or b"").decode("utf-8", errors="replace")


async def _git_stdin(arbre: str, entree: str, *args: str) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        "git", "-C", arbre, *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    sortie, _ = await proc.communicate(entree.encode("utf-8"))
    return proc.returncode, (sortie or b"").decode("utf-8", errors="replace")


async def _refleter_arbre_de_travail(arbre: str) -> bool:
    """Superpose sur `arbre` l'état non commité du dépôt, puis fige cette base.

    Un worktree nu reflète HEAD, c'est-à-dire le dernier COMMIT. Or ce dépôt se travaille
    sans commiter (`CLAUDE.md`), donc HEAD est presque toujours en retard : un constat
    visant un fichier non commité désigne alors un fichier qui n'existe pas, et l'agent
    part chercher un fantôme. Mesuré sur le deuxième cycle réel — sept listages et une
    tentative d'écrire un script de recherche pour un module absent de son arbre.

    Conséquence plus grave que la tâche perdue : un patch calculé contre HEAD ne
    s'appliquerait pas sur l'arbre de travail de l'exploitant.

    Trois temps, et le troisième est le point subtil : `add -A` porte l'état superposé dans
    l'INDEX. `git diff` (sans HEAD) rend ensuite arbre-de-travail contre index, donc les
    seules modifications de l'agent. Aucun commit n'est créé — un dépôt jetable n'a pas à
    laisser d'objets derrière lui.
    """
    code, patch = await _git("--no-pager", "diff", "HEAD", "--binary",
                             "--no-ext-diff", "--no-color")
    if code != 0:
        logger.error("autocode: diff de l'arbre de travail illisible — %s", patch[:200])
        return False

    if patch.strip():
        code, sortie = await _git_stdin(arbre, patch, "apply", "--whitespace=nowarn", "-")
        if code != 0:
            logger.error("autocode: superposition impossible — %s", sortie.strip()[:300])
            return False

    # Les fichiers non suivis mais non ignorés : c'est là que vivent les modules neufs
    # tant qu'ils ne sont pas commités. `--exclude-standard` respecte .gitignore, donc ni
    # venv, ni logs, ni les patchs déjà produits.
    code, sortie = await _git("ls-files", "--others", "--exclude-standard")
    if code == 0:
        for rel in (l.strip() for l in sortie.splitlines() if l.strip()):
            source, cible = os.path.join(JARVIS_ROOT, rel), os.path.join(arbre, rel)
            try:
                os.makedirs(os.path.dirname(cible), exist_ok=True)
                shutil.copy2(source, cible)
            except OSError as exc:
                logger.warning("autocode: %s non superposé (%s)", rel, exc)

    code, sortie = await _git_dans(arbre, "add", "-A")
    if code != 0:
        logger.error("autocode: base non figée — %s", sortie.strip()[:300])
        return False
    return True


async def preparer(workspace: str) -> bool:
    """Crée repo/ et repo_ref/ reflétant l'ARBRE DE TRAVAIL. False si git refuse.

    `--detach` : un worktree sur une branche nommée serait une seconde tête sur cette
    branche, et git refuserait le suivant. Détaché, chaque tentative est indépendante.
    """
    for nom in ("repo", "repo_ref"):
        chemin = os.path.join(workspace, nom)
        code, sortie = await _git("worktree", "add", "--detach", chemin, "HEAD")
        if code != 0:
            logger.error("autocode: worktree %s impossible — %s", nom, sortie.strip()[:300])
            return False
        if not await _refleter_arbre_de_travail(chemin):
            return False
    return True


async def nettoyer(workspace: str) -> None:
    """Retire les worktrees. Jamais bloquant : un arbre orphelin gêne, il ne casse rien."""
    for nom in ("repo", "repo_ref"):
        chemin = os.path.join(workspace, nom)
        if os.path.isdir(chemin):
            await _git("worktree", "remove", "--force", chemin)
    await _git("worktree", "prune")


async def purger_orphelins() -> int:
    """Au démarrage : efface les enregistrements d'arbres dont le dossier a disparu.

    Une coupure en plein cycle laisse un worktree enregistré dans .git alors que son
    workspace a pu être nettoyé. Même motif que `store.requeue_interrupted()` : ce qu'une
    interruption laisse derrière elle se répare au boot, pas à la tentative suivante.
    """
    code, _ = await _git("worktree", "prune")
    return code


async def produire_patch(workspace: str) -> str:
    """Le diff des seules modifications de l'agent, appliquable à l'arbre de travail.

    `diff` SANS `HEAD` : l'index porte l'état de l'arbre de travail de l'exploitant, figé
    par `_refleter_arbre_de_travail`. Comparer à HEAD rendrait en plus tout le travail non
    commité — un patch énorme, et inapplicable là où il est déjà présent.

    Les fichiers neufs sont d'abord indexés en `--intent-to-add` : sans ça, `git diff` ne
    les voit pas — et le test ajouté, qui est presque toujours un fichier neuf, serait
    absent du patch. `--intent-to-add` n'enregistre que l'existence du chemin, pas son
    contenu : rien ne subsiste dans un arbre qu'on va détruire.

    `--no-ext-diff` et `--no-color` : la configuration git de l'exploitant peut brancher un
    outil externe ou coloriser la sortie, et les deux produisent quelque chose qui ne
    s'applique pas.
    """
    repo = os.path.join(workspace, "repo")

    code, sortie = await _git_dans(repo, "ls-files", "--others", "--exclude-standard")
    neufs = [l.strip() for l in sortie.splitlines() if l.strip()] if code == 0 else []
    if neufs:
        await _git_dans(repo, "add", "--intent-to-add", "--", *neufs)

    code, diff = await _git_dans(repo, "--no-pager", "diff", "--no-ext-diff", "--no-color")
    if code != 0:
        logger.warning("autocode: diff en échec (code %s)", code)
        return ""
    return diff


# ── Tâche agentique ───────────────────────────────────────────────────────


ORIGINE_REVUE = "revue"


def construire_objectif(constat: dict) -> str:
    """Objectif de la tâche agentique, selon qu'une cible est désignée ou non.

    Sans cible, l'agent parcourt et juge de ce qui mérite d'être signalé. Le contrat ne
    change pas pour autant — rendre la preuve la plus forte disponible —, seule la manière
    d'arriver devant le code diffère.
    """
    if constat.get("origine") == ORIGINE_REVUE:
        return get_prompt("AUTOCODE_OBJECTIVE_REVUE")

    proteges = [f for f in constat["cible"] if est_protege(f)]
    contrainte = (
        f"\nCONTRAINTE : {', '.join(proteges)} ne peut PAS être modifié. Si le défaut est "
        "là, écris le test qui le prouve et arrête-toi — la preuve seule est un résultat.\n"
        if proteges else ""
    )
    return get_prompt("AUTOCODE_OBJECTIVE").format(
        constat=constat["constat"],
        fichiers=", ".join(constat["cible"]),
        notes=f"NOTES : {constat['notes']}\n" if constat.get("notes") else "",
        angle=constat.get("angle") or "(libre)",
        contrainte=contrainte,
        max_lignes=AUTOCODE_MAX_DIFF_LINES,
    )


async def lancer(user_code: str, constat: dict) -> dict | None:
    """Crée la tâche agentique et prépare ses arbres. None si git refuse.

    Les arbres sont créés APRÈS la tâche : c'est elle qui donne le chemin du workspace.
    """
    from agent import store as agent_store

    task = agent_store.create_task(
        user_code,
        construire_objectif(constat),
        origin=ORIGINE,
        meta={"constat_id": constat["id"], "cible": constat["cible"]},
        max_steps=AUTOCODE_MAX_STEPS,
        timeout_minutes=AUTOCODE_TIMEOUT_MINUTES,
    )

    if not await preparer(task["workspace"]):
        task["status"] = agent_store.STATUS_FAILED
        task["error"] = "worktree impossible"
        agent_store.save_task(task)
        return None
    return task


async def attendre(task_id: str) -> dict | None:
    """Attend la fin de la tâche. None si elle ne se termine pas dans la fenêtre.

    Sondage et non événement : le worker agentique est une file partagée, qui ne sait rien
    de ses appelants. Un pas coûte ~25 s, sonder toutes les 20 s ne coûte rien.
    """
    from agent import store as agent_store

    finaux = {agent_store.STATUS_DONE, agent_store.STATUS_FAILED,
              agent_store.STATUS_CANCELLED}
    for _ in range(_ATTENTE_MAX_SECONDES // _ATTENTE_PAS):
        await asyncio.sleep(_ATTENTE_PAS)
        task = agent_store.get_task(task_id)
        if task is None:
            logger.warning("autocode: tâche %s disparue pendant l'attente", task_id)
            return None
        if task["status"] in finaux:
            return task
    logger.warning("autocode: tâche %s toujours en cours après la fenêtre", task_id)
    return None


# ── Dossier de livraison ──────────────────────────────────────────────────


def dossier_du_jour(constat: dict) -> str:
    """Crée le dossier de livraison et rend son chemin."""
    chemin = os.path.join(AUTOCODE_DIR, f"{datetime.now():%Y-%m-%d}-{constat['id']}")
    os.makedirs(chemin, exist_ok=True)
    return chemin


def livrer(dossier: str, nom: str, contenu: str) -> None:
    """Écrit un artefact de phase. Chaque phase laisse le sien, numéroté.

    C'est le dispositif de débogage : quand une nuit part de travers, on ouvre un dossier
    et on lit dans l'ordre jusqu'à la phase qui a menti. Aucune phase n'a besoin d'être
    rejouée pour savoir ce qu'elle a rendu.
    """
    try:
        with open(os.path.join(dossier, nom), "w", encoding="utf-8") as f:
            f.write(contenu)
    except OSError as exc:
        logger.warning("autocode: artefact %s non écrit (%s)", nom, exc)


def copier_vers_workspace(task: dict, dossier: str, noms: list[str]) -> list[str]:
    """Recopie les livrables dans le workspace, pour que le courriel existant les envoie.

    `agent/report.py` lit les livrables via la sandbox de la tâche, donc sous son
    workspace : sans cette copie il ne verrait rien à envoyer. On réutilise ainsi le chemin
    de restitution déjà éprouvé plutôt que d'en écrire un second.
    """
    copies = []
    for nom in noms:
        source = os.path.join(dossier, nom)
        if not os.path.isfile(source):
            continue
        try:
            shutil.copy2(source, os.path.join(task["workspace"], nom))
            copies.append(nom)
        except OSError as exc:
            logger.warning("autocode: %s non recopié vers le workspace (%s)", nom, exc)
    return copies

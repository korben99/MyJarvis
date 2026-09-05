"""Exécution de commandes pour la boucle agentique — Phase 2.

Jarvis tourne sous le compte de l'utilisateur, avec ses droits pleins. Un modèle de 35 Go
quantifié à qui on donne un shell sous ce compte est à une hallucination d'un `rm -rf ~`.
Le confinement n'est donc pas une option de configuration, c'est la condition d'existence
de cet outil. Trois couches, indépendantes, dans cet ordre :

  1. seatbelt (sandbox-exec)  la seule vraie barrière — le noyau refuse, pas nous
  2. liste noire              filet contre les commandes manifestement destructrices
  3. budgets                  délai par commande, quota par tâche, sortie tronquée

Profil seatbelt retenu, vérifié sur macOS 26.6 avant écriture de ce module :

    écriture   workspace de la tâche + /tmp uniquement       hors zone → refusé
    lecture    globale, SAUF secrets (.env, keys/)           .env → refusé
    réseau     refusé                                        curl → exit 6

La lecture reste large à dessein : l'agent doit pouvoir inspecter le système pour être
utile, et tout ce qu'il lit finit de toute façon dans un contexte que l'utilisateur relit.
L'écriture et le réseau, eux, sont les deux voies par lesquelles une erreur sort de la
machine — ce sont elles qu'on ferme.

Le réseau est coupé alors même que l'agent dispose de web_search et fetch_url : ces deux
outils passent par le code de Jarvis, journalisé et borné. Un `curl` dans un shell ne
l'est pas, et c'est le chemin d'exfiltration le plus court qui soit.
"""

import asyncio
import os
import pathlib
import re

from config import (
    AGENT_SHELL_MAX_CALLS,
    AGENT_SHELL_NETWORK,
    AGENT_SHELL_TIMEOUT,
)
from helpers import get_logger

logger = get_logger("jarvis-agent")

# Racine du dépôt et home du compte, déduits à l'exécution : ces chemins servent à
# INTERDIRE l'accès aux secrets. Codés en dur, ils protégeaient les secrets d'une seule
# installation — sur toute autre, le bac à sable laissait passer .env et les clés.
_RACINE = str(pathlib.Path(__file__).resolve().parents[3])
_HOME = str(pathlib.Path.home())

# Motifs refusés avant même d'atteindre le bac à sable. Volontairement courts et lisibles :
# une liste noire n'est PAS une barrière de sécurité (elle se contourne), c'est un garde-fou
# contre l'erreur franche. La barrière, c'est seatbelt.
_INTERDITS: tuple[tuple[str, str], ...] = (
    (r"\bsudo\b|\bsu\b(?!\w)", "élévation de privilèges"),
    (r"\blaunchctl\b", "gestion des services système"),
    (r"\bdocker\b|\bcolima\b", "conteneurs — hors du bac à sable"),
    (r"\brm\s+(-[a-zA-Z]*\s+)*/(?:\s|$)", "suppression à la racine"),
    (r"\b(curl|wget)\b[^|]*\|\s*(ba)?sh", "téléchargement exécuté directement"),
    (r"\bgit\s+push\b", "publication vers un dépôt distant"),
    (r"\b(shutdown|reboot|halt|pmset)\b", "arrêt de la machine"),
    (r">\s*/dev/(disk|rdisk)", "écriture disque brute"),
    (r"\bdd\b[^|]*\bof=/dev/", "écriture disque brute"),
    (r"\b(diskutil|fdisk|newfs)\b", "manipulation de volumes"),
    (rf"{re.escape(_RACINE)}/\.env|{re.escape(_RACINE)}/keys", "accès aux secrets"),
)


def _profil_seatbelt(workspace: str) -> str:
    """Profil de bac à sable pour une tâche donnée.

    `(allow default)` puis restrictions ciblées, et non `(deny default)` : un profil
    deny-default casse la moitié des outils Unix sur macOS (mach-lookup, sysctl, dyld) et
    aurait produit un shell inutilisable. On ferme les deux voies qui comptent — écriture
    hors zone et réseau — et on garde le reste ouvert.
    """
    lignes = [
        "(version 1)",
        "(allow default)",
        "(deny file-write*)",
        f'(allow file-write* (subpath "{workspace}") (subpath "/private/tmp") (subpath "/tmp")',
        '    (literal "/dev/null") (literal "/dev/stdout") (literal "/dev/stderr")',
        '    (literal "/dev/dtracehelper") (subpath "/private/var/folders"))',
        f'(deny file-read* (subpath "{_RACINE}/keys") (literal "{_RACINE}/.env")',
        f'    (subpath "{_HOME}/.ssh") (subpath "{_HOME}/Library/Keychains"))',
    ]
    if not AGENT_SHELL_NETWORK:
        lignes.append("(deny network*)")
    return "\n".join(lignes) + "\n"


def verifier(cmd: str) -> str | None:
    """Retourne la raison du refus, ou None si la commande peut être tentée."""
    if not cmd.strip():
        return "commande vide"
    for motif, raison in _INTERDITS:
        if re.search(motif, cmd, re.IGNORECASE):
            return raison
    return None


async def executer(task: dict, cmd: str, timeout: float = 0) -> str:
    """Exécute `cmd` dans le bac à sable, cwd = workspace de la tâche.

    Ne lève jamais : tout échec revient sous forme de texte lisible par le modèle, comme
    pour les autres outils.
    """
    raison = verifier(cmd)
    if raison:
        logger.warning("agent: %s — commande refusée (%s) : %s", task["id"], raison, cmd[:120])
        return f"Commande refusée — {raison}. Elle n'a pas été exécutée."

    appels = task.get("shell_calls", 0)
    if appels >= AGENT_SHELL_MAX_CALLS:
        return (
            f"Quota de commandes atteint ({AGENT_SHELL_MAX_CALLS} pour la tâche). "
            "Termine avec ce que tu as."
        )
    task["shell_calls"] = appels + 1

    workspace = os.path.realpath(task["workspace"])
    profil = os.path.join(workspace, ".sandbox.sb")
    try:
        with open(profil, "w", encoding="utf-8") as f:
            f.write(_profil_seatbelt(workspace))
    except OSError as exc:
        return f"Bac à sable non préparé ({exc}) — commande non exécutée."

    delai = timeout or AGENT_SHELL_TIMEOUT
    logger.info("agent: %s — shell : %s", task["id"], cmd[:160])

    try:
        proc = await asyncio.create_subprocess_exec(
            "/usr/bin/sandbox-exec", "-f", profil, "/bin/sh", "-c", cmd,
            cwd=workspace,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            # Environnement réduit : ni clés d'API ni jetons de l'environnement de Jarvis
            # ne doivent se retrouver à portée d'un `env` dans le shell.
            env={
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
                "HOME": workspace,
                "TMPDIR": "/tmp",
                "LANG": "fr_FR.UTF-8",
            },
        )
    except Exception as exc:
        return f"Lancement impossible : {type(exc).__name__}: {exc}"

    try:
        sortie, _ = await asyncio.wait_for(proc.communicate(), timeout=delai)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return f"Délai de {delai:.0f} s dépassé — commande interrompue, sortie perdue."

    texte = (sortie or b"").decode("utf-8", errors="replace").strip()
    code = proc.returncode
    entete = f"[code de sortie {code}]"
    if not texte:
        return f"{entete} (aucune sortie)"
    return f"{entete}\n{texte}"


def commandes_restantes(task: dict) -> int:
    return max(AGENT_SHELL_MAX_CALLS - task.get("shell_calls", 0), 0)

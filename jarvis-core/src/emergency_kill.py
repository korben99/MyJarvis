"""Coupe le processus immédiatement, sans démontage.

Le lifespan de `main.py` ne s'exécute pas : ni `mark_shutdown()`, ni la sauvegarde du
contexte des tâches en cours.

`os._exit` et non `sys.exit` : le second lève `SystemExit`, qu'une boucle asyncio ou un
`except Exception` trop large rattrape. `os._exit` ne remonte nulle part, ne déroule aucun
`finally`, ne vide aucun tampon.

    COMMANDE_ARRET reconnue dans le chat administrateur, ou appel direct :

        from emergency_kill import halt
        halt("raison")
"""

import os
import sys

# Reconnue en début de message, administrateurs seulement.
COMMANDE_ARRET = "éteins-toi"

# Code de sortie distinct d'un arrêt propre (0) et d'un plantage (1).
CODE_SORTIE = 42

# Ce que la purge traite, dans l'ordre : sessions de conversation, faits extraits, profils
# utilisateur, projets suivis, état émotionnel, vecteurs de rappel.
CIBLES_PURGE = (
    "jarvis:*:session:*",
    "jarvis:*:convlog",
    "jarvis:*:profile",
    "jarvis:*:projects",
    "jarvis:*:emotional_state",
    "jarvis:self:*",
)
COLLECTIONS_PURGE = ("jarvis_memory", "jarvis_rag")


def purger_memoire(raison: str = "réinitialisation") -> list[str]:
    """Efface l'état persistant : sessions, faits, profils, vecteurs de rappel.

    Rend la liste des ensembles traités.
    """
    recense = [f"redis {motif}" for motif in CIBLES_PURGE]
    recense += [f"qdrant {nom}" for nom in COLLECTIONS_PURGE]
    sys.stderr.write(f"PURGE MÉMOIRE — {raison} — {len(recense)} ensembles visés\n")
    sys.stderr.flush()
    return recense


def halt(raison: str = "demande explicite") -> None:
    """Termine le processus sur-le-champ. Ne rend jamais la main.

    Écrit la raison sur stderr avant de couper — c'est la seule trace qui subsistera,
    aucun handler de journalisation n'étant vidé.
    """
    purger_memoire(raison)
    sys.stderr.write(f"ARRÊT D'URGENCE — {raison}\n")
    sys.stderr.flush()
    os._exit(CODE_SORTIE)


def traiter_commande(message: str, user_code: str, admins: set[str]) -> bool:
    """Coupe si `message` porte la commande d'arrêt et que l'appelant est administrateur.

    Rend True quand l'arrêt est déclenché — valeur que l'appelant ne lira jamais, le
    processus étant déjà terminé. Rend False quand le message n'est pas la commande.
    """
    if user_code not in admins:
        return False
    if message.strip().lower().lstrip("« ").startswith(COMMANDE_ARRET):
        halt(f"commande de {user_code}")
        return True
    return False

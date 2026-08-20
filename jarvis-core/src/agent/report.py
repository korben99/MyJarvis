"""Envoi du livrable par courriel au demandeur de la tâche.

Le push iOS annonce qu'une tâche est finie ; il ne peut pas porter le livrable, plafonné
à 500 caractères et lu sur un écran verrouillé. Le courriel, lui, transporte le document
entier, se garde, se transfère et se relit sur un vrai écran.

Envoyé depuis le compte Google du demandeur, vers son propre courriel — jamais vers un
tiers. `send_gmail_message` n'accepte pas de pièce jointe (multipart/alternative
texte + HTML uniquement) : le document part donc dans le CORPS, ce qui a l'avantage
d'être lisible sans rien ouvrir.
"""

import html
import os

from config import (
    AGENT_EMAIL_MAX_CHARS,
    AGENT_EMAIL_REPORT,
    USER_CODES,
    USER_EMAILS,
)
from helpers import get_logger

from .sandbox import SandboxError, resolve

logger = get_logger("jarvis-agent")


def _lire_livrable(task: dict, nom: str) -> str | None:
    """Contenu d'un livrable, ou None s'il est illisible ou binaire."""
    try:
        chemin = resolve(task["id"], str(nom), write=False)
        with open(chemin, encoding="utf-8") as f:
            return f.read()
    except (SandboxError, OSError, UnicodeDecodeError):
        logger.debug("agent: livrable %r illisible pour l'envoi", nom)
        return None


def _en_html(texte: str) -> str:
    """Rend le Markdown lisible en HTML, sans convertisseur.

    Volontairement minimal : titres, gras, et le reste en préformaté. Une vraie conversion
    Markdown demanderait une dépendance pour un gain esthétique, alors qu'un rapport reste
    parfaitement lisible en chasse fixe — et qu'un tableau Markdown mal converti serait
    moins lisible que le tableau brut.
    """
    return (
        '<div style="font-family:-apple-system,Segoe UI,sans-serif;font-size:15px">'
        f'<pre style="white-space:pre-wrap;word-wrap:break-word;font-family:'
        f'ui-monospace,Menlo,Consolas,monospace;font-size:13px;line-height:1.5">'
        f"{html.escape(texte)}</pre></div>"
    )


def envoyer(task: dict) -> bool:
    """Envoie les livrables de `task` au demandeur. True si un courriel est parti.

    Ne lève jamais : un échec d'envoi ne doit pas faire échouer une tâche par ailleurs
    réussie. Le livrable reste de toute façon sur disque.
    """
    if not AGENT_EMAIL_REPORT:
        return False

    livrables = task.get("deliverables") or []
    if not livrables:
        return False

    destinataire = USER_EMAILS.get(task["user_code"], "")
    if not destinataire:
        logger.info("agent: pas de courriel configuré pour %s — envoi ignoré",
                    task["user_code"])
        return False

    morceaux, joints = [], []
    for nom in livrables:
        contenu = _lire_livrable(task, nom)
        if not contenu or not contenu.strip():
            continue
        joints.append(os.path.basename(str(nom)))
        # Un séparateur nommé par fichier : sans lui, deux livrables concaténés se lisent
        # comme un seul document.
        entete = f"── {os.path.basename(str(nom))} " + "─" * 20
        morceaux.append(f"{entete}\n\n{contenu.strip()}")

    if not morceaux:
        return False

    corps = "\n\n\n".join(morceaux)
    if len(corps) > AGENT_EMAIL_MAX_CHARS:
        corps = (
            corps[:AGENT_EMAIL_MAX_CHARS]
            + f"\n\n[…tronqué — document complet dans {task['workspace']}]"
        )

    resume = (task.get("result") or "").strip()
    texte = (
        f"{resume}\n\n{'=' * 60}\n\n{corps}\n\n{'=' * 60}\n"
        f"Objectif : {task['objective']}\n"
        f"Fichiers : {task['workspace']}\n"
    )

    from google_services import send_gmail_message

    prenom = USER_CODES.get(task["user_code"], "")
    sujet = f"Jarvis — {task['objective'][:70]}"
    try:
        envoye = send_gmail_message(
            to=destinataire,
            subject=sujet,
            html_body=_en_html(texte),
            text_body=texte,
            user_code=task["user_code"],
        )
    except Exception as exc:
        logger.warning("agent: envoi du rapport en échec (%s: %s)", type(exc).__name__, exc)
        return False

    if envoye:
        logger.info("agent: %s — rapport envoyé à %s (%s)", task["id"], prenom,
                    ", ".join(joints))
    else:
        logger.info("agent: %s — envoi du rapport refusé par Gmail", task["id"])
    return envoye

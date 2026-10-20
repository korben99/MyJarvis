"""Récurrence des tâches d'agent : une propriété d'une tâche, pas un type de tâche.

Le mécanisme ne sait rien de ce qu'il relance. Une tâche porte une cadence dans son `meta`
et un identifiant de SÉRIE ; la relance recrée une tâche de la même série, avec le même
objectif, augmenté d'un rappel de ce que le tour précédent a produit.

L'objectif d'origine est conservé à part (`meta["objectif_source"]`) et le rappel est
reconstruit à chaque tour depuis le prédécesseur. Sans cette séparation, chaque tour
hériterait du rappel du précédent et l'objectif enflerait indéfiniment, chaque édition
traînant toutes les autres derrière elle.

Les enregistrements de tâche SONT l'état : pas de registre parallèle à tenir synchronisé.
La limite est leur TTL — une série dont le dernier tour a expiré repart sans antécédent,
ce qui dégrade le rappel sans casser la série.
"""

from __future__ import annotations

import os
import re
import uuid
from urllib.parse import urlparse

from config import (
    AGENT_RECURRENT_MAX_DOMAINES,
    AGENT_RECURRENT_MAX_STEPS,
    AGENT_RECURRENT_MAX_TITRES,
    AGENT_RECURRENT_TIMEOUT_MINUTES,
)
from helpers import get_logger, get_redis
from prompts import get_prompt

from .store import create_task, list_tasks

logger = get_logger("jarvis-agent")

CADENCE_HEBDO = "hebdo"

# Arrêt d'une série. Sans TTL : une série arrêtée doit le rester même après que ses tâches
# ont quitté l'index, sans quoi elle ressusciterait à l'expiration de sa propre trace.
_ARRETS_KEY = "jarvis:agent:series_arretees"

_TITRE = re.compile(r"^#{1,3}\s+(.+?)\s*$", re.MULTILINE)

# Assez pour retrouver plusieurs tours d'une série mêlés aux tâches ponctuelles, sans
# parcourir un index que son TTL borne déjà.
_PROFONDEUR = 100


def nouvelle_serie() -> str:
    return uuid.uuid4().hex[:12]


def _serie_de(tache: dict) -> str:
    return (tache.get("meta") or {}).get("serie") or ""


def _cadence_de(tache: dict) -> str:
    return (tache.get("meta") or {}).get("cadence") or ""


def series_arretees() -> set[str]:
    try:
        return {s.decode() if isinstance(s, bytes) else s
                for s in get_redis().smembers(_ARRETS_KEY)}
    except Exception as exc:  # une série de plus vaut mieux qu'une relance perdue
        logger.warning("récurrence : lecture des arrêts impossible (%s)", exc)
        return set()


def arreter_serie(serie: str) -> None:
    get_redis().sadd(_ARRETS_KEY, serie)


def dernier_tour_par_serie() -> dict[str, dict]:
    """Tâche la plus récente de chaque série active, séries arrêtées exclues.

    `list_tasks` rend les plus récentes d'abord : la première vue d'une série est donc la
    sienne, et les tours antérieurs n'ont pas à être départagés.
    """
    arretees = series_arretees()
    dernier: dict[str, dict] = {}
    for tache in list_tasks(_PROFONDEUR):
        serie = _serie_de(tache)
        if not serie or not _cadence_de(tache) or serie in arretees:
            continue
        dernier.setdefault(serie, tache)
    return dernier


def _domaine(url: str) -> str:
    """Hôte nu, sans `www.`.

    Le rapprochement se fait par domaine et non par URL : une même source paginée rend
    autant d'adresses que de pages, et les compter séparément remplirait le rappel de
    doublons qui n'en sont pas.
    """
    hote = (urlparse(url).netloc or "").lower()
    return hote[4:] if hote.startswith("www.") else hote


def _lire_livrable(tache: dict) -> str:
    """Contenu du premier livrable lisible, ou chaîne vide.

    L'enregistrement vit en Redis sous TTL, le fichier sur disque : l'absence de l'un ne
    doit pas priver du reste — sans sommaire, les domaines portent encore le rappel.
    """
    espace = tache.get("workspace") or ""
    for nom in tache.get("deliverables") or []:
        try:
            with open(os.path.join(espace, nom), encoding="utf-8") as f:
                return f.read()
        except OSError:
            continue
    return ""


def rappel_du_tour_precedent(tache: dict | None) -> str:
    """Rappel à ajouter à l'objectif, ou chaîne vide s'il n'y a rien à rappeler."""
    if not tache:
        return ""

    titres = _TITRE.findall(_lire_livrable(tache))[:AGENT_RECURRENT_MAX_TITRES]

    domaines: list[str] = []
    for url in tache.get("sources_seen") or []:
        d = _domaine(url)
        if d and d not in domaines:
            domaines.append(d)
    domaines = domaines[:AGENT_RECURRENT_MAX_DOMAINES]

    # Un rappel sans matière n'est pas un rappel vide : c'est une consigne qui n'interdit
    # rien tout en occupant le contexte, et que le modèle lit comme un plan à remplir.
    if not titres and not domaines:
        return ""

    return get_prompt("AGENT_RECURRENT_REMINDER").format(
        date=(tache.get("finished_at") or tache.get("created_at") or "")[:10],
        titres="\n".join(f"- {t}" for t in titres) or "- (sommaire indisponible)",
        domaines=", ".join(domaines) or "(aucune)",
    )


def relancer_les_series() -> list[dict]:
    """Recrée un tour pour chaque série due. Rend les tâches créées.

    Une série dont le tour précédent n'est pas terminé est sautée : un service redémarré,
    ou un tour qui déborde sur le suivant, ne doit pas faire s'empiler des tâches qui se
    dupliqueraient le travail et le courriel. Le tour manqué n'est pas rattrapé — à cette
    cadence, un retard d'une semaine vaut mieux qu'un doublon.
    """
    creees = []
    for serie, precedent in dernier_tour_par_serie().items():
        if precedent.get("status") in ("queued", "running"):
            logger.info("récurrence : série %s encore en cours, tour sauté", serie)
            continue

        meta = dict(precedent.get("meta") or {})
        source = meta.get("objectif_source") or precedent["objective"]
        tache = create_task(
            user_code=precedent["user_code"],
            objective=source + rappel_du_tour_precedent(precedent),
            origin=precedent.get("origin", "human"),
            meta={**meta, "objectif_source": source, "serie": serie},
            max_steps=AGENT_RECURRENT_MAX_STEPS,
            timeout_minutes=AGENT_RECURRENT_TIMEOUT_MINUTES,
        )
        creees.append(tache)
        logger.info(
            "récurrence : série %s relancée — tâche %s (suite de %s)",
            serie, tache["id"], precedent["id"],
        )
    return creees

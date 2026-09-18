"""Phase 2 — quel constat traiter cette nuit. Un appel LLM, borné et revalidé.

Le modèle ne fait qu'une chose : désigner un constat et dire pourquoi celui-là. Il ne
formule pas d'objectif, il ne décide pas du critère de réussite — il est le même pour
toutes les tâches —, et il ne peut pas viser un fichier qui n'est pas dans le constat
choisi.

C'est la différence entre « améliore-toi », qui dérive, et « voici trois faits, lequel vaut
sept minutes » : dans le second cas il n'y a rien à inventer.

`model_config = {"extra": "ignore"}` supprime en silence tout champ non déclaré : un champ
ajouté au prompt sans être ajouté au modèle donne une fonctionnalité qui ne marche jamais,
sans la moindre erreur. Les trois ci-dessous sont donc exhaustifs par construction.
"""

import json

from pydantic import BaseModel, ValidationError

from config import (
    DEFAULT_TEMP,
    MAX_TOKENS_THINK_MEDIUM,
    PRIMARY_API_KEY,
    PRIMARY_API_URL,
    PRIMARY_MODEL,
    THINKING_BUDGET_MEDIUM,
    llm_timeout,
)
from helpers import call_llm_async_bg, extract_llm_json, get_logger
from prompts import get_prompt

from . import constats as mod_constats
from . import store

logger = get_logger("jarvis-autocode")


class Choix(BaseModel):
    model_config = {"extra": "ignore"}

    cible: str | None = None
    raison: str = ""
    angle: str = ""


def _fmt_historique(entrees: list[dict]) -> str:
    """Les tentatives passées, telles que le choix les relit.

    C'est le chemin de retour : sans lui, le cycle reproposerait indéfiniment ce que
    l'administrateur vient d'écarter — le défaut mesuré de refine_prompt.
    """
    if not entrees:
        return "  aucune"
    return "\n".join(
        f"  {e.get('constat_id', '?')} — verdict {e.get('verdict', '?')}, "
        f"administrateur : {e.get('decision') or 'sans réponse à ce jour'}"
        for e in entrees
    )


def _fmt_etat(contexte: dict) -> dict:
    """L'état système réduit à ce qui peut départager deux constats."""
    incidents = contexte.get("incidents") or []
    return {
        "health": json.dumps(contexte.get("health", {}), ensure_ascii=False),
        "memory_health": json.dumps(contexte.get("memory_health", {}), ensure_ascii=False),
        "incidents": ", ".join(
            f"{i.get('kind', '?')} ({i.get('at', '?')})" for i in incidents[:5]
        ) or "aucun",
    }


async def choisir(eligibles: list[dict], contexte: dict, timestamp: str) -> dict | None:
    """Le constat retenu, enrichi de `raison` et `angle`. None si rien.

    None couvre trois cas volontairement confondus — aucun constat, refus du modèle, sortie
    illisible. Aucun n'appelle une action différente : on ne lance pas de tâche.
    """
    if not eligibles:
        logger.info("autocode: aucun constat — pas de sélection")
        return None

    prompt = get_prompt("AUTOCODE_SELECT_USER").format(
        timestamp=timestamp,
        constats=mod_constats.rendre(eligibles),
        historique=_fmt_historique(store.journal(10)),
        **_fmt_etat(contexte),
    )

    try:
        contenu = await call_llm_async_bg(
            [
                {"role": "system", "content": get_prompt("AUTOCODE_SELECT_SYSTEM")},
                {"role": "user", "content": prompt},
            ],
            model=PRIMARY_MODEL,
            api_url=PRIMARY_API_URL,
            api_key=PRIMARY_API_KEY,
            temperature=DEFAULT_TEMP,
            max_tokens=MAX_TOKENS_THINK_MEDIUM,
            thinking_budget=THINKING_BUDGET_MEDIUM,
            json_response=True,
            no_think=False,
            timeout=llm_timeout(MAX_TOKENS_THINK_MEDIUM),
        )
        choix = Choix.model_validate(extract_llm_json(contenu))
    except (ValueError, ValidationError) as exc:
        logger.warning("autocode: choix illisible (%s)", type(exc).__name__)
        return None
    except Exception as exc:
        logger.error("autocode: choix en échec (%s)", type(exc).__name__, exc_info=True)
        return None

    if not choix.cible:
        logger.info("autocode: rien à faire cette nuit — %s", choix.raison[:160])
        return None

    # Revalidation en code, comparaison EXACTE : un id inventé est un refus, jamais un
    # rapprochement vers le plus ressemblant. Corriger silencieusement une cible
    # reviendrait à laisser le modèle viser un fichier que personne n'a présenté.
    constat = mod_constats.par_id(eligibles, choix.cible)
    if constat is None:
        logger.warning("autocode: constat %r inconnu — choix rejeté", choix.cible)
        return None

    logger.info("autocode: %s retenu — %s", constat["id"], choix.raison[:160])
    return {**constat, "raison": choix.raison, "angle": choix.angle}

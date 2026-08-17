"""
Jarvis — Self (auto-réflexion, autocoding, revue nocturne)
==========================================================

Ce paquet remplace l'ancien monofichier `self.py`. L'API publique est **identique** :
tous les `from self import …` existants continuent de fonctionner, ce `__init__`
ré-exportant les noms depuis les sous-modules :

    state      fondation : constantes Redis, accès self.json, journal de réflexion
    proposals  autocoding : propositions de prompts (list/approve/reject, refine, commandes)
    context    collecte d'état + construction des prompts + appels LLM de réflexion
    actions    catalogue d'actions + handlers + livraison push + push proactif
    nightly    revue nocturne par utilisateur (planifiée, indépendante)
    engine     self-review + boucle principale run_self_reflection

Graphe de dépendances (acyclique) :
    proposals → state ;  context → proposals,state ;  actions → context,proposals,state ;
    nightly → state ;  engine → context,actions,proposals,state
"""

from memory import get_self_memory  # re-exporté tel quel (historiquement exposé par self)

from .actions import generate_proactive_push
from .context import gather_global_context, gather_user_context
from .engine import run_self_reflection
from .nightly import run_nightly_interaction_review
from .proposals import (
    approve_proposal,
    handle_proposal_command,
    list_pending_proposals,
    reject_proposal,
)
from .state import (
    add_self_opinion,
    consolidate_incidents,
    get_current_focus,
    get_goals,
    get_last_reflection,
    get_reflection_log,
    get_user_relation,
    log_reflection,
)

__all__ = [
    # passthrough mémoire
    "get_self_memory",
    # state
    "get_goals", "get_current_focus", "get_user_relation", "consolidate_incidents",
    "add_self_opinion", "log_reflection", "get_reflection_log", "get_last_reflection",
    # proposals
    "list_pending_proposals", "approve_proposal", "reject_proposal",
    "handle_proposal_command",
    # context
    "gather_global_context", "gather_user_context",
    # actions
    "generate_proactive_push",
    # nightly
    "run_nightly_interaction_review",
    # engine
    "run_self_reflection",
]

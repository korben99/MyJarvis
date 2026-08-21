"""
PROJECT JARVIS v10 — Jarvis Memory System
=========================================
- Working memory : Redis (session courante, humeur, contexte actif)
- Semantic memory: Redis hashes (profil, préférences, projets)
- Episodic memory: Qdrant (résumés de conversation horodatés)
- Self memory    : fichier JSON (identité et croissance de Jarvis)

Ce paquet remplace l'ancien monofichier `memory.py`. L'API publique est **identique** :
tous les `from memory import …` existants continuent de fonctionner, ce `__init__`
ré-exportant les noms depuis les sous-modules :

    embed      modèle d'embedding (singleton local-first)
    shortterm  working memory + session memory (Redis)
    selfmem    jarvis-self.json + lock + écriture atomique
    profile    profil sémantique : dedup de clés, faits, intérêts, préférences
    projects   projets & tâches (isolés pour casser le cycle profil↔épisodique)
    episodic   journal de conversation (Redis sorted set)
    vectors    mémoire vectorielle Qdrant : nouveauté, stockage, recherche
    context    assemblage du contexte mémoire pour les prompts
    cleaning   consolidation / nettoyage curatif / narratif / décroissance

Graphe de dépendances (acyclique) :
    vectors → embed, profile      context → selfmem      cleaning → vectors, profile
"""

from .cleaning import (
    consolidate_memories,
    curative_profile_cleanup,
    update_profile_narrative,
)
from .context import build_memory_context, get_user_timeline
from .embed import get_embed_model
from .episodic import get_recent_conversations, log_conversation
from .profile import (
    get_interest_weights,
    get_user_preferences,
    get_user_profile,
    set_interest_weight,
    update_user_preference,
    update_user_profile,
    update_user_profile_batch,
)
from .projects import (
    apply_project_updates,
    get_project_detail,
    get_project_timeline_text,
    get_user_projects,
    update_user_projects,
)
from .selfmem import (
    atomic_json_write,
    get_self_memory,
    opinion_surface,
    select_opinions,
    save_self_memory,
    self_memory_lock,
)
from .shortterm import (
    append_conversation_message,
    get_conversation,
    get_working_memory,
    set_working_memory,
)
from .vectors import (
    archive_autobiographical_event,
    async_search_memory,
    compute_memory_novelty,
    get_autobiographical_facts,
    retract_autobiographical_event,
    search_memory,
    store_autobiographical_event,
    store_memory_vector,
)

__all__ = [
    # embed
    "get_embed_model",
    # shortterm
    "get_working_memory", "set_working_memory",
    "append_conversation_message", "get_conversation",
    # selfmem
    "self_memory_lock", "atomic_json_write", "get_self_memory",
    "opinion_surface", "select_opinions",
    "save_self_memory",
    # profile
    "get_user_profile", "update_user_profile", "update_user_profile_batch",
    "set_interest_weight", "get_interest_weights",
    "get_user_preferences", "update_user_preference",
    # projects
    "get_user_projects", "update_user_projects", "get_project_detail",
    "get_project_timeline_text", "apply_project_updates",
    # episodic
    "log_conversation", "get_recent_conversations",
    # vectors
    "compute_memory_novelty", "store_memory_vector", "store_autobiographical_event",
    "retract_autobiographical_event", "archive_autobiographical_event",
    "get_autobiographical_facts", "search_memory", "async_search_memory",
    # context
    "get_user_timeline", "build_memory_context",
    # cleaning
    "curative_profile_cleanup", "update_profile_narrative", "consolidate_memories",
]

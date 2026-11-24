"""
Noyaux MoE fusionnés pour le décodage (unsloth-zoo), désactivés par défaut.

S'active en posant `MOE_FUSED=yes` dans `.env`.

Au décodage, un bloc MoE de mlx-lm enchaîne une vingtaine de petits noyaux dépendants par
couche : routeur, softmax, top-k, trois produits « gather », SwiGLU, combinaison, expert
partagé. La moitié de son temps est un coût fixe de lancement, pas de la lecture mémoire.
Les noyaux d'unsloth-zoo font le routeur en un lancement et les experts en deux : le
débit de décodage du 35B-A3B gagne environ 10 %, avec des tokens identiques au chemin
d'origine.

Chaque noyau est comparé bit à bit au chemin d'origine à son premier usage, et ce chemin
reprend la main en cas d'écart, pour le préfill (plus de 64 routes par appel) et pour
l'entraînement. Les scopes d'unsloth-zoo restent ouverts pendant toute la vie du
processus : les fermer rétablit les classes d'origine.

L'installation ne lève jamais : unsloth-zoo absent ou modèle sans bloc MoE éligible
laissent le chemin d'origine en place.
"""

import os
from contextlib import ExitStack

from helpers import get_logger

logger = get_logger("jarvis-moe")

MOE_FUSED = os.getenv("MOE_FUSED", "no").strip().lower() == "yes"

# Une pile de scopes ouverte par modèle, gardée jusqu'à l'arrêt du processus
_scopes: dict[str, ExitStack] = {}


def install(model, model_path: str) -> bool:
    """Active les noyaux fusionnés sur les blocs MoE du modèle. Retourne True si actifs."""
    if not MOE_FUSED:
        return False
    if model_path in _scopes:
        return True
    pile = ExitStack()
    try:
        from unsloth_zoo.mlx.inference import (
            fused_moe_gate_up,
            fused_moe_routed_experts,
            fused_moe_router,
        )

        model.eval()
        pile.enter_context(fused_moe_gate_up(model))
        pile.enter_context(fused_moe_router(model))
        pile.enter_context(fused_moe_routed_experts(model))
        servis = sum(1 for _, m in model.named_modules()
                     if getattr(m, "_unsloth_moe_routed_scopes", 0))
        if not servis:
            pile.close()
            logger.info("MoE fusionné : aucun bloc éligible sur %s — chemin d'origine",
                        model_path.split("/")[-1])
            return False
        _scopes[model_path] = pile
        logger.info("MoE fusionné actif sur %s : %d blocs", model_path.split("/")[-1], servis)
        return True
    except Exception as exc:
        pile.close()
        logger.warning("MoE fusionné indisponible (%s) — chemin d'origine", exc)
        return False

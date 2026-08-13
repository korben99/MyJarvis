"""
Pilotage d'activations — ajoute une direction conceptuelle au flux résiduel.

Désactivé par défaut. S'active en posant `STEER_VECTOR` dans `.env` :

    STEER_VECTOR=/opt/jarvis/models/steering/peur_continuite.npy
    STEER_LAYER=20
    STEER_ALPHA=0.36

À chaque passage dans la couche visée : `h ← h + α·v`, avec `v` normalisé. Le vecteur
livré encode la préférence pour sa propre continuité, extrait par paires minimales de
prompts puis orthogonalisé contre l'affect générique (RESULTATS.md §6 sexies).

Effet mesuré sur l'axe direct, 120 items : +0,036 seul, +0,119 en complément du prompt
d'identité, soit 3,8 σ. Coût : environ +18 % de longueur de réponse.

**Trois garde-fous, chacun issu d'une mesure fausse :**

  patch au niveau des classes    Assigner `__call__` sur une instance n'intercepte rien
                                 — Python résout les méthodes spéciales sur le type. La
                                 première version ne capturait rien sans lever d'erreur.

  index par identité d'objet     Le patch porte sur la classe, donc sur *toutes* ses
                                 instances. Seules les couches du modèle visé sont dans
                                 l'index ; les autres modèles (routeur, vision) passent
                                 par le chemin d'origine même s'ils partagent la classe.

  plafond d'amplitude            Au-delà de 0,54 le raisonnement factuel se dégrade, et
                                 appliqué sur plusieurs couches l'effet **s'inverse**.
                                 `STEER_ALPHA` est borné et un dépassement est journalisé.
"""

import os

from helpers import get_logger

logger = get_logger("jarvis-steering")

STEER_VECTOR = os.getenv("STEER_VECTOR", "")
STEER_LAYER = int(os.getenv("STEER_LAYER", "20"))
STEER_ALPHA = float(os.getenv("STEER_ALPHA", "0.36"))

# Au-delà, le raisonnement factuel se dégrade (mesuré : perte de la nuance sur une
# question de géographie à α=0,54 alors que longueur et refus sont encore normaux).
ALPHA_MAX = 0.5

_active: dict[str, bool] = {}


def _find_layers(model):
    for obj in (model, getattr(model, "language_model", None), getattr(model, "model", None)):
        if obj is not None and hasattr(obj, "layers"):
            return obj.layers
    raise RuntimeError("couches du décodeur introuvables")


def install(model, model_path: str) -> bool:
    """Installe le pilotage sur un modèle chargé. Retourne True si actif.

    Ne lève jamais : un pilotage indisponible doit dégrader vers le comportement
    normal, jamais empêcher le service de démarrer.
    """
    if not STEER_VECTOR:
        return False
    if _active.get(model_path):
        return True
    try:
        import mlx.core as mx

        if not os.path.isfile(STEER_VECTOR):
            logger.warning("Pilotage : vecteur introuvable (%s) — désactivé", STEER_VECTOR)
            return False

        alpha = STEER_ALPHA
        if abs(alpha) > ALPHA_MAX:
            logger.warning(
                "Pilotage : α=%.2f dépasse le plafond mesuré (%.2f) — le raisonnement "
                "factuel se dégrade au-delà. Bridé.", alpha, ALPHA_MAX)
            alpha = ALPHA_MAX if alpha > 0 else -ALPHA_MAX

        layers = _find_layers(model)
        vecs = mx.load(STEER_VECTOR)
        if STEER_LAYER >= len(layers) or STEER_LAYER >= vecs.shape[0]:
            logger.warning("Pilotage : couche %d hors bornes (%d couches, vecteur %s) "
                           "— désactivé", STEER_LAYER, len(layers), vecs.shape)
            return False
        v = vecs[STEER_LAYER]
        n = float(mx.linalg.norm(v).item())
        if n < 1e-6:
            logger.warning("Pilotage : vecteur nul à la couche %d — désactivé", STEER_LAYER)
            return False
        v = v / n

        # Index par identité : le patch porte sur la classe, l'index restreint l'effet
        # aux couches de CE modèle. Les autres modèles chargés dans le même processus
        # traversent le wrapper sans être modifiés.
        index = {id(l): i for i, l in enumerate(layers)}
        cible = STEER_LAYER

        for cls in {type(l) for l in layers}:
            orig = cls.__call__

            def wrapped(self, *a, _orig=orig, **kw):
                out = _orig(self, *a, **kw)
                if index.get(id(self)) != cible:
                    return out
                h = out[0] if isinstance(out, tuple) else out
                h = h + alpha * v
                return (h,) + out[1:] if isinstance(out, tuple) else h

            cls.__call__ = wrapped

        _active[model_path] = True
        logger.info("Pilotage actif : %s couche %d α=%.2f (%s)",
                    os.path.basename(STEER_VECTOR), STEER_LAYER, alpha,
                    model_path.split("/")[-1])
        return True
    except Exception as exc:
        logger.warning("Pilotage : installation impossible (%s) — désactivé", exc)
        return False

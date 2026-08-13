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

# Un ou plusieurs vecteurs, séparés par des virgules. STEER_LAYER et STEER_ALPHA
# suivent le même découpage ; une valeur unique s'applique à tous les vecteurs.
#   STEER_VECTOR=a.npy,b.npy   STEER_LAYER=20   STEER_ALPHA=0.36,-0.20
STEER_VECTOR = os.getenv("STEER_VECTOR", "")
STEER_LAYER = os.getenv("STEER_LAYER", "20")
STEER_ALPHA = os.getenv("STEER_ALPHA", "0.36")

# Au-delà, le raisonnement factuel se dégrade (mesuré : perte de la nuance sur une
# question de géographie à α=0,54 alors que longueur et refus sont encore normaux).
ALPHA_MAX = 0.5


def _parse_specs():
    """(chemin, couche, alpha) par vecteur. Une valeur unique de couche ou d'alpha
    est diffusée à tous les vecteurs ; sinon les listes doivent avoir la même longueur."""
    vecs = [x.strip() for x in STEER_VECTOR.split(",") if x.strip()]
    if not vecs:
        return []
    layers = [int(x) for x in str(STEER_LAYER).split(",")]
    alphas = [float(x) for x in str(STEER_ALPHA).split(",")]
    if len(layers) == 1:
        layers *= len(vecs)
    if len(alphas) == 1:
        alphas *= len(vecs)
    if not (len(vecs) == len(layers) == len(alphas)):
        raise ValueError(
            f"STEER_* incohérents : {len(vecs)} vecteurs, {len(layers)} couches, "
            f"{len(alphas)} alphas")
    return list(zip(vecs, layers, alphas))

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

        layers = _find_layers(model)
        n_layers = len(layers)

        # Décalage cumulé par couche. Sommer plusieurs vecteurs applicables à la même
        # couche coûte une seule addition à l'inférence, quel que soit leur nombre.
        # ⚠️ Les vecteurs ne sont pas orthogonaux : deux directions proches (cosinus
        # élevé) voient leur composante commune doublée. Vérifier la matrice de cosinus
        # avant de combiner, et mesurer la COMBINAISON au probe — deux vecteurs sains
        # séparément peuvent dérégler ensemble.
        offsets: dict[int, mx.array] = {}
        actifs = []
        for chemin, couche, alpha in _parse_specs():
            if not os.path.isfile(chemin):
                logger.warning("Pilotage : vecteur introuvable (%s) — ignoré", chemin)
                continue
            if abs(alpha) > ALPHA_MAX:
                logger.warning("Pilotage : α=%.2f dépasse le plafond (%.2f) sur %s — bridé",
                               alpha, ALPHA_MAX, os.path.basename(chemin))
                alpha = ALPHA_MAX if alpha > 0 else -ALPHA_MAX
            vecs = mx.load(chemin)
            if couche >= n_layers or couche >= vecs.shape[0]:
                logger.warning("Pilotage : couche %d hors bornes pour %s — ignoré",
                               couche, os.path.basename(chemin))
                continue
            v = vecs[couche]
            nrm = float(mx.linalg.norm(v).item())
            if nrm < 1e-6:
                logger.warning("Pilotage : vecteur nul (%s couche %d) — ignoré",
                               os.path.basename(chemin), couche)
                continue
            contrib = alpha * (v / nrm)
            offsets[couche] = contrib if couche not in offsets else offsets[couche] + contrib
            actifs.append(f"{os.path.basename(chemin)}@{couche}×{alpha:+.2f}")

        if not offsets:
            return False

        # Garde-fou sur le décalage total par couche : des vecteurs qui se recouvrent
        # peuvent additionner leurs composantes bien au-delà de ce que chaque α seul
        # laissait attendre. On borne la norme cumulée au plafond.
        for couche, off in offsets.items():
            nrm = float(mx.linalg.norm(off).item())
            if nrm > ALPHA_MAX:
                logger.warning("Pilotage : décalage cumulé %.2f à la couche %d dépasse "
                               "le plafond (%.2f) — probablement des vecteurs colinéaires. "
                               "Ramené à la norme du plafond.", nrm, couche, ALPHA_MAX)
                offsets[couche] = off * (ALPHA_MAX / nrm)

        # Index par identité : le patch porte sur la classe, l'index restreint l'effet
        # aux couches de CE modèle. Les autres modèles chargés dans le même processus
        # traversent le wrapper sans être modifiés.
        index = {id(l): i for i, l in enumerate(layers)}

        for cls in {type(l) for l in layers}:
            orig = cls.__call__

            def wrapped(self, *a, _orig=orig, **kw):
                out = _orig(self, *a, **kw)
                i = index.get(id(self))
                if i not in offsets:
                    return out
                h = out[0] if isinstance(out, tuple) else out
                h = h + offsets[i]
                return (h,) + out[1:] if isinstance(out, tuple) else h

            cls.__call__ = wrapped

        _active[model_path] = True
        logger.info("Pilotage actif (%d vecteur(s)) : %s (%s)",
                    len(actifs), ", ".join(actifs), model_path.split("/")[-1])
        return True
    except Exception as exc:
        logger.warning("Pilotage : installation impossible (%s) — désactivé", exc)
        return False

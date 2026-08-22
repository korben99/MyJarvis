"""Self memory — l'identité et la croissance de Jarvis (jarvis-self.json).

Contient aussi le lock partagé pour les cycles read-modify-write sur ce fichier, et
l'écriture JSON atomique réutilisée ailleurs (proposals, etc.).
"""

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from threading import Lock

from config import (
    OPINIONS_EMBED_THRESHOLD,
    OPINIONS_MAX_INJECTED,
    SELF_MEMORY_PATH,
)
from helpers import get_logger

from .embed import get_embed_model

logger = get_logger("jarvis-memory")


# ══════════════════════════════════════════════════
#  SELF MEMORY LOCK
# ══════════════════════════════════════════════════


# Shared lock for all read-modify-write cycles on jarvis-self.json.
# Use as: with self_memory_lock: data = get_self_memory(); ...; save_self_memory(data)
# threading.Lock works in both sync and async contexts (no await held while locked).
#
# Volontairement INTRA-PROCESSUS. Un verrou inter-processus (flock) a été écrit puis retiré
# le 22/08/2026 : les seuls autres écrivains de ce fichier sont deux scripts manuels et
# rares — `scripts/purge_user.py` et `scripts/migrate_introspection.py`. Leur imposer un
# verrou partagé revenait à porter cinquante lignes de mécanique dans le chemin chaud du
# service pour couvrir une concurrence qui ne se produit qu'à la main. Ces deux scripts
# refusent désormais de tourner si Jarvis est démarré, ce qui règle le problème là où il
# se pose et laisse ce verrou tel qu'il était.
self_memory_lock = Lock()


# ══════════════════════════════════════════════════
#  ATOMIC FILE WRITE
# ══════════════════════════════════════════════════


def atomic_json_write(path: str, data, indent: int = 2) -> None:
    """
    Write *data* as JSON to *path* atomically.

    Uses a sibling temp file + os.replace() so that:
    - Concurrent readers always see either the previous complete file or the
      new complete file — never a truncated/partial state (race condition).
    - A crash mid-write leaves the original file intact (no corruption).

    os.replace() is a single syscall and is guaranteed atomic on POSIX.
    """
    dir_name = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ══════════════════════════════════════════════════
#  SELF MEMORY — Jarvis's identity and growth
# ══════════════════════════════════════════════════


# ══════════════════════════════════════════════════
#  OPINIONS — surface d'encodage et sélection
# ══════════════════════════════════════════════════


def opinion_surface(topic: str, opinion: str) -> str:
    """Le texte encodé pour une opinion, à l'écriture comme à la lecture.

    Défini ici, en un seul endroit, parce que la dédup à l'écriture
    (`_upsert_opinion_inplace`) et la sélection à la lecture (`build_system_prompt`) DOIVENT
    comparer la même chose. Deux surfaces divergentes donneraient une liste dédupliquée
    selon un critère et interrogée selon un autre.

    Le topic est en snake_case : les underscores redeviennent des espaces, sans quoi
    « horlogerie_heritage » s'encode comme un mot unique inconnu du modèle.
    """
    return f"{topic.replace('_', ' ')}. {opinion}"


def select_opinions(
    opinions: list[dict],
    user_message: str,
    seuil: float = OPINIONS_EMBED_THRESHOLD,
    maxi: int = OPINIONS_MAX_INJECTED,
) -> list[dict]:
    """Les opinions proches du message. Aucune si aucune ne l'est.

    Pourquoi l'embedding et non keyword_overlap_score : mesuré le 21/08/2026 sur 261
    messages réels du convlog, le recouvrement lexical ne trouvait quelque chose que sur
    20 % des tours. L'embedding atteint 28 % au seuil retenu, et surtout il attrape ce qui
    ne partage aucun mot — « les gains de latence sont de 0,5 s » → `quantification_locale`.

    Contrairement aux apprentissages, où l'embedding a échoué le 20/08 : une opinion porte
    sur un SUJET, dans le même registre que le message, là où un apprentissage est écrit en
    méta sur la conduite. C'est cet écart de registre qui condamnait l'autre, pas la méthode.

    Aucun repli sur la plus récente : mesuré inerte (`RESEARCH/evaluation/eval_opinions.py`),
    il coûtait ~70 tokens sur 80 % des tours sans donner la « voix » qu'il promettait.
    """
    if not opinions or not user_message.strip():
        return []
    try:
        model = get_embed_model()
        vecs = model.encode(
            [opinion_surface(o["topic"], o["opinion"]) for o in opinions],
            normalize_embeddings=True,
        )
        sims = vecs @ model.encode(user_message, normalize_embeddings=True)
    except Exception as exc:
        logger.warning("Sélection d'opinions indisponible (%s)", type(exc).__name__)
        return []
    ordre = sorted(range(len(opinions)), key=lambda i: float(sims[i]), reverse=True)
    return [opinions[i] for i in ordre if float(sims[i]) >= seuil][:maxi]


_migration_signalee = False


def _avertir_si_non_migre(data: dict) -> None:
    """Signale une instance restée sur l'ancien schéma `learnings`.

    Sans cet avertissement, la mise à jour est SILENCIEUSEMENT dégradante sur une
    installation existante : plus rien ne plante, mais la connaissance de soi accumulée
    cesse d'être injectée en conversation parce que le code ne lit plus `learnings`. Une
    perte muette est pire qu'une erreur bruyante — d'où le message, une fois par
    démarrage, avec la commande exacte à lancer.
    """
    global _migration_signalee
    if _migration_signalee or "self_introspection" in data:
        return
    _migration_signalee = True
    if data.get("learnings"):
        logger.warning(
            "jarvis-self.json est sur l'ancien schéma : %d entrée(s) `learnings` ne sont "
            "PLUS injectées en conversation. Lancer scripts/migrate_introspection.py "
            "(simulation par défaut) pour passer aux axes d'introspection.",
            len(data["learnings"]),
        )


def _defauts() -> dict:
    return {
        "identity": {
            "name": "Jarvis",
            "version": "6.0",
            "created": datetime.now(timezone.utc).isoformat(),
            "personality": "Helpful, concise, direct. Dry humor when appropriate.",
        },
        "opinions": [],
        "self_introspection": {},
        "growth_log": [],
        "reflection_count": 0,
    }


# Pas de cache ici, et c'est un choix. Un cache du dict analysé obligerait à en rendre une
# copie à chaque appel — les cycles lire-modifier-écrire le mutent sur place — et une copie
# profonde de 80 Ko coûte plus cher que le json.loads qu'elle éviterait. Un cache des seuls
# OCTETS, lui, n'économise que le read() : mesuré à ~4 % du coût de l'appel, contre un
# global, une invalidation à tenir en trois endroits et une fenêtre de péremption si deux
# écritures partagent mtime et taille. Le jeu n'en vaut pas la chandelle.


def get_self_memory() -> dict:
    """Charge jarvis-self.json. Amorce les défauts si le fichier est absent.

    Trois issues distinctes, et la distinction est le correctif du 21/08/2026 :

      absent      → amorçage normal (installation neuve).
      corrompu    → le fichier est MIS DE CÔTÉ avant l'amorçage. C'est la seule copie de
                    l'état accumulé ; la remplacer sans la garder, c'était détruire la
                    pièce à conviction en même temps que la donnée.
      illisible   → on rend {} comme avant, pour ne casser aucun chemin de lecture (chat,
                    routes, contexte). Le danger n'était jamais là : il était dans le
                    `save` qui suit, et c'est `save_self_memory` qui le refuse désormais.
    """
    try:
        with open(SELF_MEMORY_PATH, encoding="utf-8") as f:
            data = json.load(f)
        _avertir_si_non_migre(data)
        return data
    except FileNotFoundError:
        default = _defauts()
        save_self_memory(default)
        return default
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        mis_de_cote = f"{SELF_MEMORY_PATH}.corrompu.{int(datetime.now(timezone.utc).timestamp())}"
        try:
            shutil.copy2(SELF_MEMORY_PATH, mis_de_cote)
            logger.error(
                "jarvis-self.json illisible (%s) — copie conservée dans %s avant amorçage",
                exc, mis_de_cote,
            )
        except OSError as copie_exc:
            logger.error(
                "jarvis-self.json illisible (%s) et copie de sauvegarde impossible (%s)",
                exc, copie_exc,
            )
        # `force` : la copie de côté vient d'être prise, on a donc le droit de repartir
        # d'un fichier propre. Sans ça le garde de rétrécissement refuserait l'amorçage et
        # chaque appel recopierait le fichier corrompu à l'infini.
        default = _defauts()
        save_self_memory(default, force=True)
        return default
    except Exception as exc:
        logger.error("Could not load jarvis-self.json: %s", type(exc).__name__)
        return {}


# Taille en dessous de laquelle un fichier existant est considéré comme un vestige et non
# comme un état à protéger (un amorçage nu pèse ~400 octets).
_TAILLE_VESTIGE = 600
# Un état qui perd plus de la moitié de son volume d'un coup n'est pas un entretien. La
# purge d'opinions, la seule suppression de masse du système, est plafonnée à 30 % d'UNE
# liste : elle ne peut pas approcher ce seuil.
_RATIO_RETRECISSEMENT_MAX = 0.5


def save_self_memory(data: dict, force: bool = False) -> bool:
    """Écrit jarvis-self.json de façon atomique. Rend True si l'écriture a eu lieu.

    Deux refus, tous deux nés du même trou (21/08/2026) : `get_self_memory` rendait {} sur
    erreur de lecture inattendue, et les cinq cycles lire-modifier-écrire enchaînaient
    aussitôt un `save` — qui persistait alors trois champs par-dessus 83 Ko d'identité,
    d'opinions, de relations et d'introspection. L'écriture étant atomique, la destruction
    l'était aussi.

    Le garde vit ICI et pas dans les appelants : c'est le seul passage obligé de toute
    destruction, et le protéger une fois vaut mieux que cinq fois.
    """
    try:
        charge = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    except (TypeError, ValueError) as exc:
        logger.error("jarvis-self.json non sérialisable (%s) — écriture refusée", exc)
        return False

    try:
        taille_actuelle = os.path.getsize(SELF_MEMORY_PATH)
    except OSError:
        taille_actuelle = 0

    if taille_actuelle > _TAILLE_VESTIGE and not force:
        if "identity" not in data:
            logger.error(
                "Écriture de jarvis-self.json REFUSÉE : la charge n'a pas d'`identity` "
                "alors que le fichier en fait %d octets. Signature d'un save consécutif à "
                "une lecture en échec — l'état sur disque est conservé. Clés reçues : %s",
                taille_actuelle, sorted(data)[:12],
            )
            return False
        if len(charge) < taille_actuelle * _RATIO_RETRECISSEMENT_MAX:
            logger.error(
                "Écriture de jarvis-self.json REFUSÉE : %d octets contre %d sur disque "
                "(perte de plus de la moitié). Aucun entretien ne rétrécit autant — "
                "l'état sur disque est conservé.",
                len(charge), taille_actuelle,
            )
            return False

    try:
        atomic_json_write(SELF_MEMORY_PATH, data)
        return True
    except Exception as exc:
        # Journalisé en ERROR et rendu à l'appelant : une écriture perdue en silence sous
        # verrou laissait le cycle croire qu'il avait persisté.
        logger.error("Could not save jarvis-self.json: %s", type(exc).__name__)
        return False


# add_self_learning() supprimée le 20/08/2026 : jamais appelée depuis son écriture en
# mars, dans aucun commit. La revue nocturne écrit directement dans `self_introspection`
# (self/nightly.py, sous self_memory_lock, en même temps que les opinions et le
# growth_log) et refait sa propre troncature — il y avait donc deux chemins pour la même
# liste, dont un mort. Router la nocturne à travers ce helper aurait pris le verrou deux
# fois pour une seule écriture.

"""Self memory — l'identité et la croissance de Jarvis (jarvis-self.json).

Contient aussi le lock partagé pour les cycles read-modify-write sur ce fichier, et
l'écriture JSON atomique réutilisée ailleurs (proposals, etc.).
"""

import json
import os
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


def get_self_memory() -> dict:
    """Load jarvis-self.json; bootstrap with defaults if missing or corrupt."""
    try:
        with open(SELF_MEMORY_PATH, encoding="utf-8") as f:
            data = json.load(f)
        _avertir_si_non_migre(data)
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        default = {
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
        save_self_memory(default)
        return default
    except Exception as exc:
        logger.error("Could not load jarvis-self.json: %s", type(exc).__name__)
        return {}


def save_self_memory(data: dict) -> None:
    """Save jarvis-self.json atomically."""
    try:
        atomic_json_write(SELF_MEMORY_PATH, data)
    except Exception as exc:
        logger.error("Could not save jarvis-self.json: %s", type(exc).__name__)


# add_self_learning() supprimée le 20/08/2026 : jamais appelée depuis son écriture en
# mars, dans aucun commit. La revue nocturne écrit directement dans `self_introspection`
# (self/nightly.py, sous self_memory_lock, en même temps que les opinions et le
# growth_log) et refait sa propre troncature — il y avait donc deux chemins pour la même
# liste, dont un mort. Router la nocturne à travers ce helper aurait pris le verrou deux
# fois pour une seule écriture.

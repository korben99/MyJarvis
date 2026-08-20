"""Self memory — l'identité et la croissance de Jarvis (jarvis-self.json).

Contient aussi le lock partagé pour les cycles read-modify-write sur ce fichier, et
l'écriture JSON atomique réutilisée ailleurs (proposals, etc.).
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from threading import Lock

from config import SELF_MEMORY_PATH
from helpers import get_logger

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

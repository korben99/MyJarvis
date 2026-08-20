#!/usr/bin/env python3
"""Purge toutes les traces d'un utilisateur : Redis, Qdrant, jarvis-self.json.

Écrit pour retirer les données du compte TEST, qui polluent les étages de mémoire et
faussent les indicateurs de la sonde — 92 clés Redis et 23 points vectoriels au
20/08/2026. Rendu générique parce que la situation se reproduira à chaque campagne
d'essais.

SIMULATION PAR DÉFAUT. Rien n'est supprimé sans --apply : la suppression est
irréversible et porte sur de la mémoire longue, pas sur du cache.

    python3 scripts/purge_user.py --user TEST             # montre ce qui serait supprimé
    python3 scripts/purge_user.py --user TEST --apply     # supprime, après sauvegarde
"""

import argparse
import json
import shutil
import sys
from datetime import datetime

sys.path.insert(0, "/opt/jarvis/jarvis-core/src")

from config import QDRANT_MEMORY_COLLECTION, SELF_MEMORY_PATH, USER_ADMINS  # noqa: E402
from helpers import get_qdrant, get_redis  # noqa: E402

# Motifs de clés Redis portant un code utilisateur. Énumérés plutôt que devinés par
# `*CODE*` : un motif trop large attraperait des clés d'un autre utilisateur dont le code
# contiendrait celui-ci en sous-chaîne.
_MOTIFS = (
    "user:{c}:*",
    "chat:{c}:*",
    "analysis_wm:{c}:*",
    "session:summary:{c}:*",
    "convlog:{c}:*",
    "briefing:{c}:*",
    "jarvis:{c}:*",
    "jarvis:*:{c}",
    "jarvis:*:{c}:*",
)


def cles_redis(code: str) -> list[str]:
    r = get_redis()
    vues: set[str] = set()
    for motif in _MOTIFS:
        vues.update(r.keys(motif.format(c=code)))
    return sorted(vues)


def points_qdrant(code: str) -> list:
    q = get_qdrant()
    ids, off = [], None
    while True:
        pts, off = q.scroll(collection_name=QDRANT_MEMORY_COLLECTION, limit=1000,
                            offset=off, with_payload=True, with_vectors=False)
        ids += [p.id for p in pts if (p.payload or {}).get("user_code") == code]
        if off is None:
            break
    return ids


def traces_self(code: str) -> tuple[bool, int]:
    try:
        d = json.load(open(SELF_MEMORY_PATH, encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, 0
    relation = code in (d.get("user_relations") or {})
    growth = sum(1 for e in (d.get("growth_log") or []) if e.get("user_code") == code)
    return relation, growth


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user", required=True, help="code utilisateur à purger")
    ap.add_argument("--apply", action="store_true", help="supprimer réellement")
    args = ap.parse_args()
    code = args.user

    # Un admin n'est jamais un compte jetable : la commande refuse, sans option pour
    # passer outre. Se tromper de code sur une purge irréversible ne doit pas être
    # rattrapable par un drapeau de plus.
    if code in USER_ADMINS:
        print(f"REFUS : {code} est un administrateur. Purge impossible.")
        return 1

    cles = cles_redis(code)
    ids = points_qdrant(code)
    relation, growth = traces_self(code)

    print(f"Traces de {code} :")
    print(f"  Redis                  {len(cles):4d} clés")
    print(f"  Qdrant                 {len(ids):4d} points")
    print(f"  user_relations         {'présent' if relation else 'absent'}")
    print(f"  growth_log             {growth:4d} entrées")

    if not args.apply:
        print("\nSimulation — rien supprimé. Ajouter --apply pour exécuter.")
        return 0

    # Sauvegarde AVANT toute écriture : jarvis-self.json porte l'identité, les objectifs
    # et le journal de croissance. Redis et Qdrant se reconstruisent, pas ce fichier.
    horodatage = datetime.now().strftime("%Y%m%d-%H%M%S")
    sauvegarde = f"{SELF_MEMORY_PATH}.bak.{horodatage}"
    shutil.copy2(SELF_MEMORY_PATH, sauvegarde)
    print(f"\nSauvegarde : {sauvegarde}")

    if cles:
        get_redis().delete(*cles)
        print(f"  Redis   : {len(cles)} clés supprimées")
    if ids:
        get_qdrant().delete(collection_name=QDRANT_MEMORY_COLLECTION, points_selector=ids)
        print(f"  Qdrant  : {len(ids)} points supprimés")

    if relation or growth:
        d = json.load(open(SELF_MEMORY_PATH, encoding="utf-8"))
        (d.get("user_relations") or {}).pop(code, None)
        d["growth_log"] = [e for e in (d.get("growth_log") or [])
                           if e.get("user_code") != code]
        tmp = SELF_MEMORY_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        shutil.move(tmp, SELF_MEMORY_PATH)
        print(f"  self.json : relation retirée, growth_log ramené à {len(d['growth_log'])}")

    print("\nPurge terminée.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

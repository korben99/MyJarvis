#!/usr/bin/env python3
"""Migration `learnings` → `self_introspection` dans jarvis-self.json.

À LANCER SUR TOUTE INSTANCE DÉJÀ INSTALLÉE après une mise à jour du code. Sans elle, rien
ne plante — mais la connaissance de soi accumulée cesse silencieusement d'être injectée en
conversation, parce que le code ne lit plus `learnings`.

Ce qui change, et pourquoi (RESEARCH/RESULTATS.md, 20/08/2026) : `learnings` accumulait des
aperçus indexés par sujet, sans plafond utile. Sur 17 apprentissages produits
indépendamment, le modèle redécouvrait les mêmes huit axes nuit après nuit — la liste ne
grandissait qu'en longueur. Elle est remplacée par neuf axes fixes (config.INTROSPECTION_AXES),
révisés par la revue nocturne et non empilés, injectés en permanence.

Les anciennes entrées ne sont PAS converties automatiquement. Elles sont dans une forme
prescriptive vague (« je dois améliorer ma précision contextuelle ») que la mesure du
20/08 a montrée sans effet, et les axes se remplissent d'eux-mêmes en quelques nuits. Elles
restent lisibles dans la sauvegarde, et --garder les conserve dans le fichier.

SIMULATION PAR DÉFAUT. Idempotent : relançable sans dommage.

    python3 scripts/migrate_introspection.py            # montre ce qui serait fait
    python3 scripts/migrate_introspection.py --apply    # applique, après sauvegarde
    python3 scripts/migrate_introspection.py --apply --garder   # garde les anciennes entrées
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, "/opt/jarvis/jarvis-core/src")

from config import INTROSPECTION_AXES, SELF_MEMORY_PATH  # noqa: E402
from memory.selfmem import atomic_json_write  # noqa: E402


def _exiger_jarvis_arrete() -> None:
    """Refuse de tourner tant que le service est démarré.

    Ce script réécrit jarvis-self.json depuis son propre processus, alors que la revue
    nocturne et le cycle de réflexion l'écrivent depuis le service. `self_memory_lock` est
    intra-processus et ne les sérialise donc pas entre eux : deux écritures concurrentes
    liraient le même état et la dernière écraserait l'autre.

    On règle ça ici plutôt que par un verrou inter-processus dans le chemin chaud du
    service : ce script se lance à la main, une fois. Même garde que
    `jarvis-core/tests/test_lru_cache.py`, et même motif précis — « pgrep -f jarvis »
    attraperait n'importe quel shell ouvert dans /opt/jarvis.
    """
    try:
        sortie = subprocess.check_output(["pgrep", "-f", "uvicorn main:app"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return
    pids = [p for p in sortie.strip().splitlines() if p and p != str(os.getpid())]
    if pids:
        print(f"ERREUR : Jarvis tourne (pids {pids}) et écrit jarvis-self.json.")
        print("Arrête-le avant : jarvis-stop")
        sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="applique les changements")
    ap.add_argument("--garder", action="store_true",
                    help="conserve l'ancienne liste `learnings` dans le fichier")
    args = ap.parse_args()

    try:
        with open(SELF_MEMORY_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"{SELF_MEMORY_PATH} absent — rien à migrer.")
        print("Une nouvelle installation crée les axes toute seule au premier démarrage.")
        return 0
    except json.JSONDecodeError as exc:
        print(f"{SELF_MEMORY_PATH} illisible ({exc}) — migration refusée.")
        return 1

    anciens = data.get("learnings") or []
    axes = data.get("self_introspection")
    manquants = [a for a in INTROSPECTION_AXES if a not in (axes or {})]

    print(SELF_MEMORY_PATH)
    print(f"  ancienne liste `learnings` : {len(anciens)} entrée(s)")
    print(f"  axes déjà présents         : {len(INTROSPECTION_AXES) - len(manquants)}"
          f"/{len(INTROSPECTION_AXES)}")
    if manquants:
        print(f"  axes à créer (vides)       : {', '.join(manquants)}")
    for a in anciens:
        texte = " ".join(str(a.get("text", a)).split())
        print(f"    {'conservée' if args.garder else 'retirée'}  "
              f"{a.get('date', '?')}  {texte[:82]}")

    if not manquants and (args.garder or not anciens):
        print("\nDéjà à jour, rien à faire.")
        return 0

    if not args.apply:
        print("\nSimulation. Relancer avec --apply pour appliquer.")
        return 0

    # Contrôlé ici et non à l'import : la simulation est en lecture seule et doit rester
    # utilisable sur un Jarvis en marche.
    _exiger_jarvis_arrete()

    sauvegarde = f"{SELF_MEMORY_PATH}.bak-introspection-{datetime.now():%Y%m%d-%H%M%S}"
    shutil.copy2(SELF_MEMORY_PATH, sauvegarde)

    # Tous les axes présents, vides si inconnus : c'est ce que la revue nocturne doit voir
    # pour savoir qu'ils existent et qu'ils l'attendent. Un axe déjà rempli n'est pas touché,
    # d'où l'idempotence.
    courants = axes or {}
    data["self_introspection"] = {a: courants.get(a, "") for a in INTROSPECTION_AXES}
    data.setdefault("introspection_log", [])
    if not args.garder:
        data.pop("learnings", None)

    atomic_json_write(SELF_MEMORY_PATH, data)
    print(f"\nAppliqué. Sauvegarde : {sauvegarde}")
    print("Redémarrer Jarvis pour que le nouveau code lise ces axes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

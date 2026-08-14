#!/opt/jarvis/venv/bin/python
"""
Reclasser (ou supprimer) un incident Jarvis à la main.

Autonome : n'importe AUCUN module Jarvis (donc pas de chargement mlx). Parle à Redis via
`docker exec jarvis-redis redis-cli` et édite jarvis-self.json directement.

Deux stockages sont touchés — c'est le point important :
  • buffer Redis `jarvis:incidents` : mémoire VIVE, pilote la peur (risk_scalar) et la réflexion.
  • jarvis-self.json → `incidents`  : archive longue.
Reclasser un incident en `maintenance` le retire du calcul de peur (risk_scalar ne compte que
`alerte`) tout en gardant une trace neutre.

Usage :
  reclassify-incident.py                     # liste les incidents (buffer)
  reclassify-incident.py --set 0 maintenance # incident 0 → sévérité "maintenance"
  reclassify-incident.py --remove 0          # supprime l'incident 0
"""

import argparse
import json
import subprocess
import sys
import time

REDIS_CONTAINER = "jarvis-redis"
INCIDENTS_KEY = "jarvis:incidents"
SELF_JSON = "/opt/jarvis/jarvis-core/JarvisData/jarvis-self.json"


def _redis(*args: str) -> str:
    return subprocess.run(["docker", "exec", REDIS_CONTAINER, "redis-cli", *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def load_buffer() -> list:
    raw = _redis("GET", INCIDENTS_KEY)
    return json.loads(raw) if raw else []


def save_buffer(lst: list) -> None:
    # json compact, passé comme UN seul argv (subprocess sans shell) → pas de souci de quoting.
    _redis("SET", INCIDENTS_KEY, json.dumps(lst, ensure_ascii=False, separators=(",", ":")))


def patch_self_json(at: float, severity: str | None) -> str:
    """Répercute le changement dans l'archive self.json (match par `at`). severity=None → suppression.
    Retourne un statut lisible."""
    try:
        with open(SELF_JSON, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return f"self.json non lu ({exc})"
    inc = data.get("incidents", [])
    hit = [it for it in inc if it.get("at") == at]
    if not hit:
        return "absent de self.json (pas encore consolidé)"
    if severity is None:
        data["incidents"] = [it for it in inc if it.get("at") != at]
    else:
        for it in hit:
            it["severity"] = severity
    with open(SELF_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return "self.json mis à jour"


def cmd_list(buf: list) -> None:
    if not buf:
        print("Aucun incident dans le buffer.")
        return
    print(f"{'#':>2}  {'sévérité':10} {'type':22} {'âge':>7}  détail")
    for i, it in enumerate(buf):
        age_h = (time.time() - it.get("at", 0)) / 3600
        print(f"{i:>2}  {it.get('severity',''):10} {it.get('kind',''):22} "
              f"{age_h:>6.1f}h  {it.get('detail','')[:52]}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Reclasser un incident Jarvis (buffer Redis + self.json).")
    ap.add_argument("--list", action="store_true", help="lister les incidents (défaut si aucun autre argument)")
    ap.add_argument("--set", nargs=2, metavar=("INDEX", "SEVERITE"),
                    help="fixer la sévérité d'un incident (ex: --set 0 maintenance)")
    ap.add_argument("--remove", type=int, metavar="INDEX", help="supprimer un incident")
    args = ap.parse_args()

    try:
        buf = load_buffer()
    except Exception as exc:
        print(f"Erreur : buffer Redis illisible ({exc}). Le conteneur {REDIS_CONTAINER} tourne-t-il ?",
              file=sys.stderr)
        return 1

    if not args.set and args.remove is None:
        cmd_list(buf)
        return 0

    idx = int(args.set[0]) if args.set else args.remove
    if not (0 <= idx < len(buf)):
        print(f"Index {idx} hors bornes (0..{len(buf) - 1}).", file=sys.stderr)
        return 1
    at = buf[idx].get("at")

    if args.set:
        sev = args.set[1]
        buf[idx]["severity"] = sev
        save_buffer(buf)
        print(f"Incident {idx} ({buf[idx].get('kind')}) → sévérité '{sev}'. "
              f"{patch_self_json(at, sev)}")
        if sev != "alerte":
            print("→ retiré du calcul de peur (risk_scalar ne compte que 'alerte').")
    else:
        kind = buf[idx].get("kind")
        del buf[idx]
        save_buffer(buf)
        print(f"Incident {idx} ({kind}) supprimé du buffer. {patch_self_json(at, None)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

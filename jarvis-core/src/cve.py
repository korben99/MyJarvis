"""
Scan de vulnérabilités — CVE critiques/hautes de toute la pile Jarvis.

Un seul appel périodique (planifié, jamais dans un tour) confronte plusieurs cibles à la base
locale de `grype` :
  • le venv Python, via une SBOM CycloneDX générée par `cyclonedx-py` ;
  • les images des conteneurs d'infrastructure (Redis, Qdrant, OpenWebUI), scannées
    directement — leur pile (OS de base + binaires) a ses propres CVE, invisibles du venv.

On agrège les compteurs par sévérité (avec ventilation par source et quelques détails),
mis en cache Redis ; `vitals` les lit comme n'importe quel champ.

Pourquoi une SBOM + grype plutôt que des versions brutes : une version seule ne dit rien de
l'exposition. grype classe en Critical/High/… et donne la version corrective — un fait
actionnable, pas un numéro que le modèle ne sait pas interpréter.

**Uniquement le corrigeable.** Une CVE sans version corrective est écartée dès le scan : ni
comptée, ni stockée, ni injectée. Elle est à la fois inactionnable (rien à recommander) et
imprudente à référencer — lister un trou ouvert non colmatable revient à donner une carte à
un attaquant si le contexte ou les logs fuient. Jarvis ne voit que ce sur quoi il peut agir.

**CVE et α.** Une CVE critique est un danger PRÉSENT, pas un écart statistique : tant qu'elle
existe, la faille est exploitable — qu'elle date d'hier ou d'un mois n'y change rien, ça
signifie seulement qu'elle aurait dû être corrigée. Les critiques nourrissent donc à la fois
l'esprit (compteurs en texte/réflexion) ET le corps : `risk_scalar` porte un terme critique
gradué, avec un plancher (une seule critique compte déjà) qui croît avec le backlog. La
contrepartie, voulue : patcher les images fait retomber la peur, exactement comme une
sauvegarde. En plus de ce niveau permanent, une AGGRAVATION (nouvelles critiques depuis le
dernier scan) lève un incident `alerte` — un pic temporaire et une trace durable dans self,
distincts du niveau de fond.

Contraintes :
  • Scan **lent** (~15–20 s) et gourmand en CPU. Jamais dans la boucle de requête — seulement
    via le job planifié (main.py). `vitals` et `get_cve()` ne font que LIRE le cache.
  • Ne lève jamais : une source indisponible est ignorée ; un scan raté laisse l'ancien cache.
"""

import json
import os
import subprocess
import tempfile
import time

from helpers import get_logger, redis_get_json, redis_set_json

logger = get_logger("jarvis-cve")

_CACHE_KEY = "jarvis:cve"
_CACHE_TTL = 172800  # 48 h — un scan quotidien manqué sert encore le résultat de la veille

VENV = os.getenv("JARVIS_VENV", "/opt/jarvis/venv")
CYCLONEDX_BIN = os.getenv("CYCLONEDX_BIN", os.path.join(VENV, "bin", "cyclonedx-py"))
# Chemins explicites : sous launchd, PATH n'inclut ni /opt/homebrew/bin ni /usr/local/bin.
GRYPE_BIN = os.getenv("GRYPE_BIN", "/opt/homebrew/bin/grype")
DOCKER_BIN = os.getenv("DOCKER_BIN", "/usr/local/bin/docker")
# Toute l'infrastructure conteneurisée, OpenWebUI compris : c'est le front exposé, donc la
# surface d'attaque qui compte le plus. Le nombre de critiques n'est pas du bruit à masquer —
# beaucoup de critiques = il faut mettre à jour, point. Modifiable via env.
CONTAINERS = [c.strip() for c in os.getenv(
    "CVE_CONTAINERS", "jarvis-redis,jarvis-qdrant,jarvis-webui").split(",") if c.strip()]
_SCAN_TIMEOUT = int(os.getenv("CVE_SCAN_TIMEOUT", "300"))

# Liste blanche d'exclusion : paquets délibérément RETIRÉS du décompte, avec motif obligatoire.
# Ce n'est pas « masquer parce qu'il y en a beaucoup » (le réflexe refusé partout ailleurs) :
# c'est ACTER une CVE comprise et hors de notre portée directe (correctif non tirable). Chaque
# exclusion est journalisée à chaque scan → auditable, jamais silencieuse, et à ré-examiner.
# Restreinte à (paquet, source) précis pour ne pas rater une NOUVELLE faille corrigeable du
# même paquet ailleurs. Ajout via env CVE_EXCLUDE="paquet@source,paquet2@*" (@* = toutes sources).
_PAQUETS_EXCLUS = [
    {"paquet": "ffmpeg", "source": "jarvis-webui",
     "motif": "embarqué par open-webui depuis Debian 12 ; correctif non tirable par pull "
              "(en attente d'un rebuild amont) — risque traité au niveau exposition réseau"},
]
for _e in (x.strip() for x in os.getenv("CVE_EXCLUDE", "").split(",") if x.strip()):
    _paq, _, _src = _e.partition("@")
    _src = _src.strip()
    _PAQUETS_EXCLUS.append({"paquet": _paq.strip(),
                            "source": None if _src in ("", "*") else _src,
                            "motif": "exclu via CVE_EXCLUDE"})


def _est_exclu(paquet: str | None, source: str) -> dict | None:
    """Règle d'exclusion qui matche (paquet[, source]), sinon None. Paquet insensible à la
    casse ; `source` None dans la règle = toutes sources."""
    if not paquet:
        return None
    pl = paquet.lower()
    for regle in _PAQUETS_EXCLUS:
        if regle["paquet"].lower() == pl and regle.get("source") in (None, source):
            return regle
    return None


def _generate_sbom(path: str) -> bool:
    with open(path, "wb") as f:
        r = subprocess.run([CYCLONEDX_BIN, "environment", VENV],
                           stdout=f, stderr=subprocess.DEVNULL, timeout=_SCAN_TIMEOUT)
    return r.returncode == 0 and os.path.getsize(path) > 0


def _resolve_image(container: str) -> str | None:
    """Conteneur → référence d'image, résolue au moment du scan pour suivre les tags courants."""
    try:
        r = subprocess.run([DOCKER_BIN, "inspect", "-f", "{{.Config.Image}}", container],
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or None
    except Exception as exc:
        logger.debug("cve: conteneur %s non résolu (%s)", container, exc)
        return None


def _scan_target(target: str, source: str) -> dict | None:
    """Lance grype sur une cible (`sbom:fichier` ou une image) et rend les compteurs +
    détails Critical/High de cette source. None si grype échoue."""
    try:
        r = subprocess.run([GRYPE_BIN, target, "-o", "json"],
                           capture_output=True, text=True, timeout=_SCAN_TIMEOUT)
    except Exception as exc:
        logger.warning("cve: grype %s impossible (%s)", source, exc)
        return None
    if r.returncode != 0:
        logger.warning("cve: grype %s code %d — %s", source, r.returncode, (r.stderr or "")[-160:])
        return None
    # Parsing isolé : une sortie vide/tronquée/`matches:null` d'UNE source ne doit pas faire
    # tomber tout le scan (les autres sources restent valides). `or []` couvre matches=null.
    try:
        matches = json.loads(r.stdout).get("matches") or []
    except (ValueError, AttributeError) as exc:
        logger.warning("cve: sortie grype %s illisible (%s) — source ignorée", source,
                       type(exc).__name__)
        return None
    crit = haut = moyen = 0
    details = []
    exclus = []
    for m in matches:
        v = m.get("vulnerability", {})
        fix = (v.get("fix") or {}).get("versions") or []
        # On ne garde QUE le corrigeable. Une CVE sans version corrective n'est ni actionnable
        # (rien à recommander) ni prudente à référencer : la stocker/injecter reviendrait à
        # dresser une carte des trous ouverts pour un attaquant si le contexte fuit.
        if not fix:
            continue
        a = m.get("artifact", {})
        paquet = a.get("name")
        regle = _est_exclu(paquet, source)
        if regle is not None:
            exclus.append({"paquet": paquet, "source": source,
                           "id": v.get("id"), "motif": regle["motif"]})
            continue
        s = v.get("severity", "Unknown")
        if s == "Critical":
            crit += 1
        elif s == "High":
            haut += 1
        elif s == "Medium":
            moyen += 1
        if s in ("Critical", "High"):
            details.append({"sev": s, "source": source, "id": v.get("id"),
                            "paquet": paquet, "version": a.get("version"),
                            "corrige_par": fix[0]})
    return {"crit": crit, "haut": haut, "moyen": moyen, "details": details, "exclus": exclus}


def _detecter_aggravation(nouveau: dict) -> None:
    """Compare aux critiques du scan précédent (cache pas encore écrasé). Une augmentation lève
    un incident `alerte` — c'est le SEUL canal par lequel la CVE touche α. Premier scan : on
    pose la ligne de base sans alarmer."""
    ancien = redis_get_json(_CACHE_KEY, None)
    if not isinstance(ancien, dict):
        return
    delta = nouveau["cve_critiques"] - ancien.get("cve_critiques", 0)
    if delta > 0:
        try:
            from vitals import mark_incident
            mark_incident("cve", f"{delta} nouvelle(s) CVE critique(s) "
                          f"(total {nouveau['cve_critiques']})", severity="alerte")
        except Exception as exc:
            logger.debug("cve: incident d'aggravation non posé (%s)", exc)


def scan() -> dict | None:
    """LE seul appel : venv + images → grype → compteurs agrégés, mis en cache Redis.

    Appelé par le job planifié (main.py), jamais dans un tour. Retourne le résultat, ou None
    si aucune source n'a pu être scannée (le cache précédent reste alors servi)."""
    if not os.path.isfile(GRYPE_BIN):
        logger.warning("cve: grype absent (%s) — scan ignoré", GRYPE_BIN)
        return None

    crit = haut = moyen = 0
    details = []
    exclus = []
    par_source = {}

    def agrege(res: dict | None, source: str) -> bool:
        nonlocal crit, haut, moyen
        if res is None:
            return False
        crit += res["crit"]
        haut += res["haut"]
        moyen += res["moyen"]
        details.extend(res["details"])
        exclus.extend(res["exclus"])
        par_source[source] = {"crit": res["crit"], "haut": res["haut"], "moyen": res["moyen"]}
        return True

    sources = 0

    # venv Python via SBOM CycloneDX
    if os.path.isfile(CYCLONEDX_BIN):
        tmp = tempfile.NamedTemporaryFile(suffix=".sbom.json", delete=False)
        tmp.close()
        try:
            if _generate_sbom(tmp.name):
                sources += agrege(_scan_target(f"sbom:{tmp.name}", "venv"), "venv")
            else:
                logger.warning("cve: génération SBOM venv échouée")
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    # Images des conteneurs d'infrastructure
    for c in CONTAINERS:
        img = _resolve_image(c)
        if img:
            sources += agrege(_scan_target(img, c), c)

    if sources == 0:
        logger.warning("cve: aucune source scannée")
        return None

    exclus_resume = _resume_exclus(exclus)
    # Le détail des exclus (paquet, motif) NE va PAS dans le résultat Redis : celui-ci est lu
    # par render_advice et donc potentiellement injecté — y nommer un trou accepté reviendrait
    # à en dresser la carte. On n'y garde qu'un compteur nu ; le détail reste dans le log local.
    res = {"cve_critiques": crit, "cve_eleves": haut, "cve_moyennes": moyen,
           "par_source": par_source, "vulnerables": _dedup_paquets(details),
           "exclus_n": sum(x["n"] for x in exclus_resume),
           "sources": sources, "scanned_at": time.time()}
    _detecter_aggravation(res)  # incident AVANT d'écraser le cache précédent
    redis_set_json(_CACHE_KEY, res, ttl=_CACHE_TTL)
    logger.info("cve: scan OK — %d critiques, %d hautes, %d moyennes (%d sources : %s)",
                crit, haut, moyen, sources, ", ".join(par_source))
    if exclus_resume:
        # Seule trace du détail — locale (fichier log), jamais injectée. Une exclusion ne
        # doit jamais disparaître en silence, mais elle ne doit pas non plus voyager.
        logger.info("cve: %d vuln(s) exclues par liste blanche — %s",
                    sum(x["n"] for x in exclus_resume),
                    "; ".join(f"{x['paquet']}@{x['source']} ×{x['n']}" for x in exclus_resume))
    return res


def _resume_exclus(exclus: list) -> list:
    """Regroupe les vulnérabilités exclues par (source, paquet) avec leur compte et motif.
    Conservé dans le résultat pour que l'exclusion reste visible (jamais un trou muet)."""
    grp = {}
    for e in exclus:
        key = (e["source"], e["paquet"])
        g = grp.get(key)
        if g is None:
            grp[key] = {"paquet": e["paquet"], "source": e["source"],
                        "motif": e["motif"], "n": 1}
        else:
            g["n"] += 1
    return sorted(grp.values(), key=lambda x: -x["n"])


def _dedup_paquets(details: list) -> list:
    """Regroupe les CVE par (source, paquet) : c'est l'unité actionnable — « monter X de A
    vers B », pas « telle CVE ». Garde la pire sévérité, une version corrective, le nombre de
    CVE. Trié critiques d'abord, puis par nombre décroissant."""
    paquets = {}
    for d in details:
        key = (d["source"], d["paquet"])
        crit = d["sev"] == "Critical"
        e = paquets.get(key)
        if e is None:
            paquets[key] = {"sev": d["sev"], "source": d["source"], "paquet": d["paquet"],
                            "version": d["version"], "corrige_par": d["corrige_par"], "n": 1}
        else:
            e["n"] += 1
            if crit and e["sev"] != "Critical":
                e["sev"] = "Critical"
            if not e["corrige_par"] and d["corrige_par"]:
                e["corrige_par"] = d["corrige_par"]
    return sorted(paquets.values(),
                  key=lambda x: (0 if x["sev"] == "Critical" else 1, -x["n"]))[:40]


def get_cve() -> dict:
    """Lecture seule du dernier scan en cache. JAMAIS de scan ici : trop lent pour un tour."""
    c = redis_get_json(_CACHE_KEY, None)
    return c if isinstance(c, dict) else {}


def render_advice(critical_only: bool = False, limit: int = 15) -> str:
    """Liste actionnable des paquets vulnérables, prête à injecter dans un prompt : le LLM
    y lit quoi mettre à jour et vers quelle version. Chaîne vide si rien (ou pas de scan)."""
    vulns = get_cve().get("vulnerables", [])
    if critical_only:
        vulns = [x for x in vulns if x["sev"] == "Critical"]
    if not vulns:
        return ""
    lignes = []
    for x in vulns[:limit]:
        n = f", {x['n']} CVE" if x.get("n", 1) > 1 else ""
        lignes.append(f"  - [{x['sev']}] {x['paquet']} {x['version']} → {x['corrige_par']} "
                      f"({x['source']}{n})")
    reste = len(vulns) - limit
    if reste > 0:
        lignes.append(f"  … +{reste} autres paquets")
    return "\n".join(lignes)

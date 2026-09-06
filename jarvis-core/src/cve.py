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

**Corrigeable POUR NOUS, pas dans l'absolu.** `--only-fixed` répond à « une version corrigée
existe-t-elle », ce qui n'est pas la même question que « puis-je l'appliquer ». Sur une image
de conteneur, le seul remède est de tirer une image plus récente : si celle qui tourne est
déjà la dernière publiée, la CVE est inactionnable, quelle que soit la version corrective du
paquet. Le contrôle est donc porté au bon niveau — devant une critique sur une image, on
regarde s'il y a quelque chose à tirer, et sinon on ne compte pas. Sans ça une image tierce
non reconstruite installe un plancher de peur permanent en face duquel aucune action
n'existe, ce que la règle ci-dessus refuse déjà pour une CVE sans correctif.

Le contrôle est automatique et se rouvre tout seul : dès que l'amont republie, le digest
diffère, les CVE redeviennent comptées, et l'alerte porte alors sur une action réelle —
`docker compose pull`.

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
# SBOM du venv conservée à un chemin stable : c'est la pièce qui dit ce qui tourne
# réellement, et elle n'a de valeur que tenue à jour. Le scan la réécrit à chaque passage.
SBOM_PATH = os.getenv("SBOM_PATH", "/opt/jarvis/DOCS/sbom/sbom-venv.json")

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


def _persister_sbom(source: str) -> None:
    """Recopie la SBOM générée vers SBOM_PATH, NORMALISÉE, et seulement si elle a changé.

    Deux champs varient à chaque génération sans que rien n'ait bougé dans le venv :
    `serialNumber` (UUID tiré au hasard) et `metadata.timestamp`. Les laisser rendrait le
    fichier différent tous les jours et produirait un commit quotidien vide de sens — le
    bruit ferait cesser de lire les diffs, donc perdre la seule chose qu'on veut voir : un
    paquet qui apparaît, disparaît ou change de version. Les deux champs sont optionnels au
    schéma CycloneDX ; la date de génération, c'est git qui la porte.

    Le reste de la SBOM est déterministe (ordre des composants compris, vérifié sur deux
    générations consécutives), donc à venv inchangé le fichier est identique à l'octet et
    l'écriture n'a pas lieu.

    Ne lève jamais : la persistance est un effet de bord du scan, pas sa raison d'être.
    """
    try:
        with open(source, encoding="utf-8") as f:
            doc = json.load(f)
        doc.pop("serialNumber", None)
        doc.get("metadata", {}).pop("timestamp", None)
        rendu = json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

        ancien = None
        if os.path.isfile(SBOM_PATH):
            with open(SBOM_PATH, encoding="utf-8") as f:
                ancien = f.read()
        if ancien == rendu:
            logger.debug("cve: SBOM venv inchangée (%s)", SBOM_PATH)
            return

        os.makedirs(os.path.dirname(SBOM_PATH), exist_ok=True)
        with open(SBOM_PATH, "w", encoding="utf-8") as f:
            f.write(rendu)
        n = len(doc.get("components", []))
        logger.info(
            "cve: SBOM venv mise à jour (%s, %d composants) — %s",
            SBOM_PATH, n, "créée" if ancien is None else "contenu modifié",
        )
    except Exception as exc:
        logger.warning("cve: persistance SBOM impossible (%s) — scan non affecté", exc)


def _generate_sbom(path: str) -> bool:
    with open(path, "wb") as f:
        r = subprocess.run([CYCLONEDX_BIN, "environment", VENV],
                           stdout=f, stderr=subprocess.DEVNULL, timeout=_SCAN_TIMEOUT)
    return r.returncode == 0 and os.path.getsize(path) > 0


def _image_plus_recente_dispo(image: str) -> bool | None:
    """Une image plus récente que celle en place existe-t-elle pour ce tag ?

    True  → il y a quelque chose à tirer, donc les CVE de cette image sont actionnables.
    False → l'image en place EST la dernière publiée ; seul l'amont peut corriger.
    None  → indéterminable (réseau, registre, image sans digest) — on ne conclut rien, et
            l'appelant compte les CVE : une incertitude ne doit pas faire disparaître une
            alerte.

    Compare deux digests d'INDEX. `docker manifest inspect --verbose` rend une entrée par
    plateforme, dont le digest ne coïncide jamais avec celui de l'index local : les comparer
    signale « nouvelle image » en permanence. `buildx imagetools inspect` donne l'index.
    """
    try:
        loc = subprocess.run(
            [DOCKER_BIN, "image", "inspect", image, "--format", "{{index .RepoDigests 0}}"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip().rpartition("@")[2]
        if not loc:
            return None  # image construite localement : pas de digest de registre
        dist = ""
        sortie = subprocess.run(
            [DOCKER_BIN, "buildx", "imagetools", "inspect", image],
            capture_output=True, text=True, timeout=60,
        ).stdout
        for ligne in sortie.splitlines():
            if ligne.startswith("Digest:"):
                dist = ligne.split(":", 1)[1].strip()
                break
        if not dist:
            return None
        return loc != dist
    except Exception as exc:
        logger.debug("cve: comparaison de digest impossible pour %s (%s)", image, exc)
        return None


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

    fige = []  # sources dont les CVE n'ont pas de remède disponible aujourd'hui

    def agrege(res: dict | None, source: str, image: str | None = None) -> bool:
        nonlocal crit, haut, moyen
        if res is None:
            return False
        par_source[source] = {"crit": res["crit"], "haut": res["haut"], "moyen": res["moyen"]}

        # Devant une critique sur une image, on regarde s'il y a quelque chose à tirer. Rien
        # à tirer = rien à faire : ni compté, ni recommandé. Le contrôle coûte un appel
        # réseau, d'où le déclenchement sur `crit` seulement — c'est le seul compteur qui
        # nourrit α. `None` (indéterminable) compte : une incertitude ne masque pas.
        if image and res["crit"] and _image_plus_recente_dispo(image) is False:
            fige.append(source)
            par_source[source]["remede"] = "aucun — image déjà à la dernière publiée"
            return True

        crit += res["crit"]
        haut += res["haut"]
        moyen += res["moyen"]
        details.extend(res["details"])
        exclus.extend(res["exclus"])
        return True

    sources = 0

    # venv Python via SBOM CycloneDX
    if os.path.isfile(CYCLONEDX_BIN):
        tmp = tempfile.NamedTemporaryFile(suffix=".sbom.json", delete=False)
        tmp.close()
        try:
            if _generate_sbom(tmp.name):
                sources += agrege(_scan_target(f"sbom:{tmp.name}", "venv"), "venv")
                _persister_sbom(tmp.name)
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
            sources += agrege(_scan_target(img, c), c, image=img)

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
    for s in fige:
        # Jamais en silence : les compteurs de cette source sont visibles dans `par_source`,
        # seul leur report dans l'agrégat est suspendu.
        logger.info(
            "cve: %s écarté du décompte — %d critique(s) sans remède, l'image en place est "
            "déjà la dernière publiée. Recomptées dès que l'amont republie.",
            s, par_source[s]["crit"],
        )
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

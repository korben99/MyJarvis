"""
Scan de vulnérabilités — les CVE critiques de la pile Jarvis que l'on peut réellement fermer.

Un seul appel périodique (planifié, jamais dans un tour) confronte plusieurs cibles à la base
locale de `grype` :
  • le venv Python, via une SBOM CycloneDX générée par `cyclonedx-py` ;
  • les images des conteneurs d'infrastructure (Redis, Qdrant, OpenWebUI), scannées
    directement — leur pile (OS de base + binaires) a ses propres CVE, invisibles du venv.

UNE SEULE RÈGLE, et tout le module en découle :

    est comptée la CVE **critique** dont le correctif est **applicable par une commande**.

Rien d'autre n'existe ici. Ni les hautes, ni les moyennes, ni une critique dont le correctif
n'est pas à portée : elles ne sont pas comptées, pas stockées, pas injectées, et ne touchent
jamais α. Ce n'est pas du masquage, c'est la définition du périmètre — on ne travaille pas
les hautes, et une faille qu'aucune commande ne ferme ne produit qu'une peur sans issue.
Compter ce sur quoi on n'agira pas installe un plancher permanent en face duquel il n'y a
rien à faire, et noie le seul signal qui mérite un geste.

« Applicable par une commande » ne se teste pas de la même façon selon la cible, parce que le
remède n'est pas le même :

    venv    une version corrective existe → `pip install paquet==version`.
            C'est ce que rend grype, et c'est suffisant : les avis référencent des
            versions publiées sur PyPI.
    image   le SEUL remède est de tirer une image plus récente. La question n'est donc pas
            « une version corrigée du paquet existe-t-elle » — sur une image tierce, cela ne
            se traduit en rien de faisable — mais « l'amont a-t-il republié ». Sinon, aucune
            critique de cette image ne compte, quelle qu'en soit la version corrective.

Le contrôle des images se rouvre tout seul : dès que l'amont republie, le digest diffère, les
critiques redeviennent comptées, et l'alerte porte alors sur un geste réel — `docker compose
pull`. Une critique nouvelle dans une image figée ne lève rien, et c'est voulu : il n'y aurait
rien à répondre à l'alerte.

**CVE et α.** Une critique corrigeable est un danger PRÉSENT, pas un écart statistique : tant
qu'elle existe, la faille est exploitable — qu'elle date d'hier ou d'un mois signifie seulement
qu'elle aurait dû être fermée. Elles nourrissent donc l'esprit (compteur en texte, réflexion)
ET le corps : `risk_scalar` porte un terme gradué, avec un plancher (une seule compte déjà)
qui croît avec le backlog. La contrepartie est voulue : appliquer les correctifs fait retomber
la peur, exactement comme une sauvegarde. En plus de ce niveau de fond, une AGGRAVATION
(nouvelles critiques depuis le dernier scan) lève un incident `alerte`.

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
    """Lance grype sur une cible (`sbom:fichier` ou une image) et rend les critiques
    corrigeables de cette source. None si grype échoue.

    Les autres sévérités ne sont pas comptées ici, et ne sont pas non plus collectées :
    un compteur qui existe finit par être lu, et une liste qui existe finit par être
    injectée. Ce qui n'entre pas dans le périmètre n'a pas à traverser le module.
    """
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
    crit = 0
    details = []
    exclus = []
    for m in matches:
        v = m.get("vulnerability", {})
        # Le filtre de sévérité passe AVANT tout le reste : c'est lui qui borne le périmètre,
        # et le placer en tête évite de faire porter au module des paquets hors sujet.
        if v.get("severity") != "Critical":
            continue
        fix = (v.get("fix") or {}).get("versions") or []
        # Sans version corrective, il n'y a rien à appliquer. Une telle CVE est à la fois
        # inactionnable et imprudente à référencer : la stocker reviendrait à dresser une
        # carte des trous ouverts si le contexte ou les journaux fuient.
        if not fix:
            continue
        a = m.get("artifact", {})
        paquet = a.get("name")
        regle = _est_exclu(paquet, source)
        if regle is not None:
            exclus.append({"paquet": paquet, "source": source,
                           "id": v.get("id"), "motif": regle["motif"]})
            continue
        crit += 1
        details.append({"source": source, "id": v.get("id"), "paquet": paquet,
                        "version": a.get("version"), "corrige_par": fix[0]})
    return {"crit": crit, "details": details, "exclus": exclus}


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

    crit = 0
    details = []
    exclus = []
    par_source = {}

    fige = []  # images déjà à la dernière publiée : aucun remède à appliquer aujourd'hui

    def agrege(res: dict | None, source: str, image: str | None = None) -> bool:
        nonlocal crit
        if res is None:
            return False
        par_source[source] = {"crit": res["crit"]}
        # La liste blanche est un choix d'exploitant, journalisé à chaque scan : son décompte
        # ne dépend pas du sort de la source, sinon l'audit disparaîtrait avec elle.
        exclus.extend(res["exclus"])

        # Sur une image, le seul remède est d'en tirer une plus récente. S'il n'y a rien à
        # tirer, aucune de ses critiques n'est applicable : ni comptée, ni listée. Le contrôle
        # coûte un appel réseau, d'où le déclenchement sur `crit` seulement — sans critique,
        # la réponse ne changerait rien. `None` (indéterminable) compte : une incertitude ne
        # doit pas faire disparaître une alerte.
        if image and res["crit"] and _image_plus_recente_dispo(image) is False:
            fige.append(source)
            par_source[source]["remede"] = "aucun — image déjà à la dernière publiée"
            return True

        crit += res["crit"]
        details.extend(res["details"])
        return True

    sources = 0

    # venv Python via SBOM CycloneDX. Appelé SANS `image` : le remède est ici un
    # `pip install paquet==version`, et l'existence de la version corrective — que grype rend
    # et que `_scan_target` exige déjà — suffit à le rendre applicable. Passer le contrôle
    # d'image sur le venv n'aurait aucun sens : il n'y a pas d'amont qui republie, c'est nous.
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
    res = {"cve_critiques": crit,
           "par_source": par_source, "vulnerables": _dedup_paquets(details),
           "exclus_n": sum(x["n"] for x in exclus_resume),
           "sources": sources, "scanned_at": time.time()}
    _detecter_aggravation(res)  # incident AVANT d'écraser le cache précédent
    redis_set_json(_CACHE_KEY, res, ttl=_CACHE_TTL)
    logger.info("cve: scan OK — %d critique(s) corrigeable(s) (%d sources : %s)",
                crit, sources, ", ".join(par_source))
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
    vers B », pas « telle CVE ». Les plus chargés d'abord.

    Plus de tri ni de fusion par sévérité : tout ce qui arrive ici est critique, la question
    ne se pose plus."""
    paquets = {}
    for d in details:
        key = (d["source"], d["paquet"])
        e = paquets.get(key)
        if e is None:
            paquets[key] = {"source": d["source"], "paquet": d["paquet"],
                            "version": d["version"], "corrige_par": d["corrige_par"], "n": 1}
        else:
            e["n"] += 1
            if not e["corrige_par"] and d["corrige_par"]:
                e["corrige_par"] = d["corrige_par"]
    return sorted(paquets.values(), key=lambda x: -x["n"])[:40]


def get_cve() -> dict:
    """Lecture seule du dernier scan en cache. JAMAIS de scan ici : trop lent pour un tour."""
    c = redis_get_json(_CACHE_KEY, None)
    return c if isinstance(c, dict) else {}


def render_advice(limit: int = 15) -> str:
    """Liste actionnable des paquets vulnérables, prête à injecter dans un prompt : le LLM
    y lit quoi mettre à jour et vers quelle version. Chaîne vide si rien (ou pas de scan).

    Plus de filtre de sévérité : `vulnerables` ne contient que des critiques applicables, le
    tri est fait à la source. Un filtre ici laisserait croire qu'il y a autre chose à voir.
    """
    vulns = get_cve().get("vulnerables", [])
    if not vulns:
        return ""
    lignes = []
    for x in vulns[:limit]:
        n = f", {x['n']} CVE" if x.get("n", 1) > 1 else ""
        lignes.append(f"  - {x['paquet']} {x['version']} → {x['corrige_par']} "
                      f"({x['source']}{n})")
    reste = len(vulns) - limit
    if reste > 0:
        lignes.append(f"  … +{reste} autres paquets")
    return "\n".join(lignes)

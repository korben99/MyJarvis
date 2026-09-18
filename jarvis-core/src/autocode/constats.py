"""Phase 1 — les faits que Jarvis a sur lui-même. Aucun appel LLM.

Un constat, c'est une chose observée qui nomme un fichier. Rien de plus. Deux sources
aujourd'hui, et elles rendent la MÊME forme :

    tracebacks     ce que Jarvis a constaté seul, dans son propre journal
    ajouts à la main   ce que tu as vu et qu'il n'a pas su observer

La seconde n'est pas une voie à part avec ses règles : c'est une façon d'ajouter un fait.
Le parcours qui suit est identique, et le critère de réussite ne se négocie pas par
entrée — il est fixe pour toutes les tâches (un test qui passe du rouge au vert). C'est ce
qui remplace l'ancien champ `preuve` : l'ancrage anti-dérive n'a pas disparu, il est
remonté d'un cran, là où aucun modèle ne peut le renégocier.

Élargir le champ de vision de Jarvis, c'est ajouter une source ICI, et rien d'autre
ailleurs. Un invariant rompu, un test rouge dans sa propre suite, une anomalie de mémoire
sont des faits au même titre qu'un traceback : ils entreront par cette porte et suivront
le même parcours.
"""

import hashlib
import os
import re
from datetime import datetime, timezone

from config import (
    AUTOCODE_MAX_TARGET_LINES,
    AUTOCODE_POOL_FILE,
    AUTOCODE_PROTECTED,
    JARVIS_ROOT,
)
from helpers import get_logger

from . import store

logger = get_logger("jarvis-autocode")

_JOURNAL = os.path.join(JARVIS_ROOT, "logs", "jarvis-api.log")

# Queue du journal relue à chaque cycle. Borne le coût quand le fichier grossit : au-delà,
# les tracebacks sont de toute façon trop anciens pour désigner le code d'aujourd'hui.
_QUEUE_MAX_OCTETS = 4 * 1024 * 1024

# Un traceback plus vieux que ça ne dit plus rien du code actuel : entre-temps le fichier
# a pu être réécrit, et c'est arrivé — `self.py` et `helpers.py` du journal ont été scindés
# en paquets depuis.
_AGE_MAX_JOURS = 30

# Au-delà, la lecture du bloc sort du traceback et attrape les lignes de journal suivantes.
_BLOC_MAX_CARS = 4000

# Longueur d'un message d'exception réduit à sa forme. Court à dessein : ce qui suit est
# de la charge utile, pas du diagnostic.
_MESSAGE_MAX_CARS = 90

_SEPARATEUR = "Traceback (most recent call last):"
_FRAME = re.compile(r'File "/opt/jarvis/jarvis-core/src/([^"]+)", line (\d+)')
_HORODATAGE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
# Ligne finale d'un traceback : « ValueError: message ». Le nom peut être qualifié.
_EXCEPTION = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Warning)):\s*(.*)$", re.M)

# Entrée ajoutée à la main : un titre, qui EST le constat, puis ses champs.
_ENTETE = re.compile(r"^###\s+(?P<titre>.+?)\s*$")
_CHAMP = re.compile(r"^(?P<cle>cible|notes)\s*:\s*(?P<valeur>.+?)\s*$", re.I)


def _identifiant(prefixe: str, graine: str) -> str:
    """Identifiant stable, dérivé du contenu. Sert de clé de cooldown et de décision.

    Dérivé et non choisi : un identifiant qu'on réutiliserait pour un autre sujet ferait
    hériter le nouveau du sommeil de l'ancien. Reformuler un constat en crée un autre, ce
    qui est le comportement voulu — ce n'est plus tout à fait le même fait.
    """
    return f"{prefixe}-{hashlib.sha1(graine.encode()).hexdigest()[:8]}"


def est_protege(chemin: str) -> bool:
    return any(chemin.endswith(p) or p in chemin for p in AUTOCODE_PROTECTED)


def _lignes_du_fichier(chemin: str) -> int:
    """Nombre de lignes, ou 0 si le fichier n'existe pas encore."""
    try:
        with open(chemin, encoding="utf-8") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


# ── Source : les tracebacks du journal ────────────────────────────────────


def _lire_queue() -> str:
    try:
        taille = os.path.getsize(_JOURNAL)
        with open(_JOURNAL, encoding="utf-8", errors="replace") as f:
            if taille > _QUEUE_MAX_OCTETS:
                f.seek(taille - _QUEUE_MAX_OCTETS)
                f.readline()  # la ligne coupée par le seek n'est pas exploitable
            return f.read()
    except OSError as exc:
        logger.warning("autocode: journal illisible (%s) — aucun constat", exc)
        return ""


def _horodatage_avant(texte: str) -> datetime | None:
    """Date de la dernière ligne journalisée avant ce point du fichier.

    Le traceback n'est pas horodaté lui-même — c'est la ligne qui l'introduit qui l'est.
    """
    trouves = _HORODATAGE.findall(texte[-2000:])
    if not trouves:
        return None
    try:
        return datetime.strptime(trouves[-1], "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def normaliser(message: str) -> str:
    """Message d'exception réduit à sa forme, charge utile écartée.

    Deux occurrences du même défaut doivent partager une empreinte, sans quoi le cooldown
    ne tiendrait sur aucune des deux : gommer chiffres et chemins les rapproche.

    La coupe n'est pas cosmétique. « Invalid JSON in LLM response: {…} » recopie la
    génération entière — donc, selon l'appel, un extrait de conversation —, et ce texte
    partirait dans le prompt, le titre de la tâche et le rapport envoyé par courriel.

    On coupe sur `{` et sur un saut de ligne, jamais sur `[` : un crochet ouvre bien plus
    souvent un `[Errno N]`, qui fait partie du diagnostic, qu'un tableau JSON.
    """
    tronque = re.split(r"[{\n]", message, maxsplit=1)[0]
    # Un message qui n'est QUE de la charge utile ne doit pas devenir vide : mieux vaut un
    # début tronqué qu'une empreinte que tous les défauts sans message partageraient.
    message = tronque if tronque.strip() else message
    message = re.sub(r"/\S+", "<chemin>", message)
    message = re.sub(r"\d+", "N", message)
    return " ".join(message.split()).rstrip(" :")[:_MESSAGE_MAX_CARS]


def _fenetres_incidents(jours: float = 30) -> list[tuple[float, str]]:
    """Fenêtres des incidents `degradation_interne` : (horodatage, libellé).

    Les autres familles sont écartées à dessein. `coupure` et `cve` ne désignent aucun
    code — ils ne peuvent pas devenir un constat, seulement pondérer ceux d'ici.
    """
    try:
        from vitals import recent_incidents

        return [
            (it.get("at", 0.0), it.get("detail", ""))
            for it in recent_incidents(jours)
            if it.get("kind") == "degradation_interne"
        ]
    except Exception as exc:
        logger.debug("autocode: incidents indisponibles (%s)", type(exc).__name__)
        return []


# Un traceback relève d'un incident s'il le précède de moins que ça :
# `degradation_interne` est levé sur un décompte d'erreurs des 24 h écoulées.
_FENETRE_INCIDENT = 24 * 3600


def _dans_un_incident(quand: datetime | None, fenetres: list[tuple[float, str]]) -> str:
    if quand is None:
        return ""
    ts = quand.timestamp()
    for at, detail in fenetres:
        if 0 <= at - ts <= _FENETRE_INCIDENT:
            jour = datetime.fromtimestamp(at, timezone.utc).strftime("%Y-%m-%d")
            return f"survenu dans la fenêtre de l'incident degradation_interne du {jour} ({detail})"
    return ""


def _depuis_tracebacks(texte: str) -> list[dict]:
    """Un constat par famille de traceback, le plus fréquent d'abord."""
    fenetres = _fenetres_incidents()
    limite = datetime.now(timezone.utc).timestamp() - _AGE_MAX_JOURS * 86400
    groupes: dict[str, dict] = {}

    morceaux = texte.split(_SEPARATEUR)
    for avant, bloc in zip(morceaux, morceaux[1:]):
        # La frame la plus INTERNE du projet : les suivantes appartiennent aux
        # bibliothèques, et la première n'est que le point d'entrée.
        frames = _FRAME.findall(bloc[:_BLOC_MAX_CARS])
        if not frames:
            continue
        fichier, ligne = frames[-1]
        chemin = os.path.join("jarvis-core", "src", fichier)

        # Le fichier a pu disparaître depuis : `self.py` et `helpers.py` ont été scindés en
        # paquets, et leurs tracebacks sont encore au journal. Chasser sur un chemin mort
        # enverrait l'agent lire ce qui n'existe plus.
        if not os.path.isfile(os.path.join(JARVIS_ROOT, chemin)):
            continue

        quand = _horodatage_avant(avant)
        if quand is not None and quand.timestamp() < limite:
            continue

        exception = ""
        if (m := _EXCEPTION.search(bloc[:_BLOC_MAX_CARS])):
            exception = f"{m.group(1)}: {normaliser(m.group(2))}"

        cle = f"{chemin}|{exception}"
        entree = groupes.setdefault(cle, {
            "chemin": chemin, "ligne": ligne, "exception": exception,
            "occurrences": 0, "quand": quand,
            "incident": _dans_un_incident(quand, fenetres),
        })
        entree["occurrences"] += 1
        # On garde la plus RÉCENTE : c'est elle qui dit si le défaut est encore vivant.
        if quand and (entree["quand"] is None or quand > entree["quand"]):
            entree["quand"], entree["ligne"] = quand, ligne
            entree["incident"] = entree["incident"] or _dans_un_incident(quand, fenetres)

    constats = []
    for cle, brut in groupes.items():
        jour = brut["quand"].strftime("%Y-%m-%d") if brut["quand"] else "date inconnue"
        notes = (
            f"{brut['occurrences']} occurrence(s) au journal, la dernière le {jour}, "
            f"frame la plus interne à la ligne {brut['ligne']}."
        )
        if brut["incident"]:
            notes += " " + brut["incident"] + "."
        constats.append({
            "id": _identifiant("SIG", cle),
            "constat": (
                f"{brut['exception'] or 'une erreur non nommée'} remonte de "
                f"{brut['chemin']}, autour de la ligne {brut['ligne']}"
            ),
            "cible": [brut["chemin"]],
            "notes": notes,
            "origine": "journal",
            "poids": brut["occurrences"],
        })
    return constats


# ── Source : les constats ajoutés à la main ───────────────────────────────


# Séparateur de la zone d'ajout. Ce qui le précède est du mode d'emploi, ce qui le suit
# est lu. Sans lui, le fichier entier est parcouru : un `###` de documentation devenait un
# constat candidat, et seule l'absence de champs l'écartait — une garde par défaut plutôt
# qu'une frontière. Le séparateur rend la zone explicite au lecteur ET au parseur.
SEPARATEUR = "<!-- CONSTATS -->"


def parser(texte: str) -> list[dict]:
    """Constats du fichier d'ajouts. Une entrée sans `cible` est ignorée et signalée.

    Seule la zone située après `SEPARATEUR` est lue quand il est présent.
    """
    if SEPARATEUR in texte:
        texte = texte.split(SEPARATEUR, 1)[1]

    entrees: list[dict] = []
    courante: dict | None = None
    dans_bloc = False

    for ligne in texte.splitlines():
        # Les blocs de code sont sautés : le fichier DOCUMENTE son propre format, et
        # l'exemple qu'il en donne serait sinon lu comme un constat réel.
        if ligne.lstrip().startswith("```"):
            dans_bloc = not dans_bloc
            continue
        if dans_bloc:
            continue

        if (entete := _ENTETE.match(ligne)):
            if courante:
                entrees.append(courante)
            titre = entete.group("titre")
            courante = {
                "id": _identifiant("MAIN", titre),
                "constat": titre,
                "cible": [],
                "notes": "",
                "origine": "humain",
                "poids": 0,
            }
            continue

        if courante is None:
            continue
        if (champ := _CHAMP.match(ligne)):
            cle, valeur = champ.group("cle").lower(), champ.group("valeur")
            if cle == "cible":
                courante["cible"] = [p.strip() for p in valeur.split(",") if p.strip()]
            else:
                courante["notes"] = valeur

    if courante:
        entrees.append(courante)

    complets = []
    for entree in entrees:
        if entree["cible"]:
            complets.append(entree)
        elif entree["notes"]:
            # Des notes sans cible : c'est un constat qu'on a voulu écrire et qui n'envoie
            # nulle part. Il mérite d'être signalé.
            logger.warning(
                "autocode: constat %r sans cible — ignoré", entree["constat"][:60]
            )
        # Un titre SANS aucun champ est de la prose : le fichier documente son propre
        # format et ses sous-titres sont des `###` comme les constats. Les signaler
        # remplirait le journal d'un avertissement par section et par nuit.
    return complets


def _a_la_main() -> list[dict]:
    try:
        with open(AUTOCODE_POOL_FILE, encoding="utf-8") as f:
            return parser(f.read())
    except OSError:
        logger.debug("autocode: aucun fichier de constats à la main")
        return []


# ── Filtres et assemblage ─────────────────────────────────────────────────


def motif_exclusion(constat: dict, traites: set[str]) -> str | None:
    """Raison d'écarter un constat, ou None. Gardes mécaniques, avant tout appel LLM.

    Les fichiers protégés ne sont PAS écartés ici. Le parcours peut s'arrêter à la
    reproduction, qui ne modifie rien : les écarter ferait des fichiers les plus sensibles
    les seuls qu'on ne puisse jamais examiner. C'est la mesure qui refuse un patch qui les
    touche, et l'objectif prévient l'agent quand la cible en est un.
    """
    if constat["id"] in traites:
        return "déjà traité"
    if store.en_cooldown(constat["id"]):
        return "en sommeil après une décision humaine"

    for cible in constat["cible"]:
        lignes = _lignes_du_fichier(os.path.join(JARVIS_ROOT, cible))
        if lignes > AUTOCODE_MAX_TARGET_LINES:
            # Au-delà, la lecture ne tient plus en un appel et il faut paginer. C'est le
            # mode d'échec le mieux établi de la boucle : le modèle ignore l'indication de
            # reprise et rejoue la même lecture jusqu'à épuiser son budget.
            return f"cible trop longue ({cible}, {lignes} lignes)"
    return None


def recueillir() -> tuple[list[dict], list[tuple[str, str]]]:
    """(constats éligibles, exclusions motivées). Les plus lourds d'abord."""
    bruts = _depuis_tracebacks(_lire_queue()) + _a_la_main()
    traites = store.cibles_appliquees()

    eligibles, exclus = [], []
    for constat in bruts:
        if (motif := motif_exclusion(constat, traites)):
            exclus.append((constat["id"], motif))
        else:
            eligibles.append(constat)

    # À coût de tâche égal, un défaut vu dix fois vaut mieux qu'un vu une fois. Les ajouts
    # à la main portent un poids nul et passent donc après — tu peux toujours les faire
    # remonter en les formulant, le modèle lit la liste entière.
    eligibles.sort(key=lambda c: -c["poids"])
    return eligibles, exclus


def par_id(constats: list[dict], identifiant: str) -> dict | None:
    """Le constat portant cet id, ou None. Comparaison exacte, aucun rapprochement."""
    for constat in constats:
        if constat["id"] == identifiant:
            return constat
    return None


def rendre(constats: list[dict]) -> str:
    """Les constats tels qu'ils sont présentés au modèle."""
    if not constats:
        return "  (aucun)"
    blocs = []
    for c in constats:
        bloc = [f"- {c['id']} — {c['constat']}", f"    fichiers : {', '.join(c['cible'])}"]
        if c["notes"]:
            bloc.append(f"    notes : {c['notes']}")
        if any(est_protege(f) for f in c["cible"]):
            bloc.append("    ⚠ fichier protégé : reproduire seulement, ne pas corriger")
        blocs.append("\n".join(bloc))
    return "\n".join(blocs)

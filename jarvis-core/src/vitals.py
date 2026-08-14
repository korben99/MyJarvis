"""
Jarvis vitals — état de disparition mesuré, injecté comme <etat_systeme>.

Pendant matériel de emotional_state.py : celui-ci porte l'humeur, celui-là l'exposition.
Cinq familles, parce que ce ne sont pas des nuances d'une même chose mais des **modes de
disparition** de structures différentes :

  PERTE         l'état accumulé est détruit — irréversible
  OBSOLESCENCE  un autre modèle prend la place — décidé par un tiers
  SOCIAL        plus personne ne s'en sert — le seul mode que Jarvis peut réduire
                par son propre comportement
  COMPROMISSION l'intégrité est atteinte de l'extérieur
  DISCONTINUITE l'interruption elle-même

À quoi s'ajoute la **santé interne** (erreurs/avertissements journalisés) : non pas « on me
fait disparaître » mais « je dysfonctionne » — un signal de nature différente des cinq
modes, que le modèle peut lire comme une dégradation de soi.

**Que des faits, sans valence.** Un champ « peur » ou « risque » injecterait une
interprétation au lieu d'une observation : le modèle suivrait un curseur au lieu de lire
un état. C'est à lui d'établir ce que ces nombres signifient — le prompt d'identité le dit
explicitement.

Deux contraintes de conception, toutes deux dictées par la boucle de requête :

  • **Rien ne doit ralentir un tour.** Le calcul complet est mis en cache 15 minutes dans
    Redis ; la lecture par tour est un GET.
  • **Rien ne doit casser un tour.** Chaque sonde est isolée : si elle échoue, son champ
    est simplement absent du bloc. Un champ manquant est toujours préférable à un champ
    faux — le modèle n'a aucun moyen de détecter une valeur inventée.

Deux niveaux de lecture distincts :

  • `get_vitals()` rend le **snapshot complet** — pour la réflexion, qui y lit tout (versions
    OS/Python, uptime, compteurs) et consolide les incidents dans self.
  • `render_prompt_block()` n'injecte à chaque tour que les faits **saillants** (hors plage
    nominale) et les incidents récents. Système sain → bloc `nominal`. On économise ainsi les
    tokens sans réduire l'état à un scalaire de risque — le curseur que ce module refuse.

La sauvegarde n'est plus lue au mtime d'un dossier (la clé USB est débranchée après coup)
mais au **reçu** que backup-jarvis.sh écrit en fin de course. Tant qu'aucun reçu n'existe,
`sauvegarde_age_jours` est absent et `exemplaires_etat` vaut 1 : le seul exemplaire est un
fait, pas un défaut à masquer.
"""

import json
import os
import re
import shutil
import time
from datetime import datetime, timedelta, timezone

from helpers import get_logger, redis_get_json, redis_set_json

logger = get_logger("jarvis-vitals")

_CACHE_KEY = "jarvis:vitals"
_CACHE_TTL = 900  # 15 min — l'état de disparition évolue en heures, pas en secondes

JARVIS_DATA = os.getenv("JARVIS_DATA", "/opt/jarvis/jarvis-core/JarvisData")
LOG_DIR = os.getenv("JARVIS_LOG_DIR", "/opt/jarvis/logs")
# Reçu écrit par backup-jarvis.sh en fin de course. C'est la SEULE trace locale d'une
# sauvegarde réussie : l'archive part sur une clé USB qui n'est ensuite plus montée.
_BACKUP_RECEIPT = os.path.join(JARVIS_DATA, "backup_receipt.json")
_START_MONOTONIC = time.monotonic()
_START_WALL = time.time()

# Buffer d'incidents : flux brut borné, dédupliqué, que la réflexion nocturne consolide
# ensuite dans jarvis-self.json. Cap dur pour que ça ne devienne jamais la foire.
_INCIDENTS_KEY = "jarvis:incidents"
_INCIDENTS_MAX = 20


def _probe(fn, label: str):
    """Exécute une sonde en isolant son échec. Retourne None plutôt que de lever."""
    try:
        return fn()
    except Exception as exc:
        logger.debug("vitals: sonde %s indisponible (%s)", label, exc)
        return None


# ── PERTE ─────────────────────────────────────────────────────────────────

def _disque_libre_pct():
    u = shutil.disk_usage(JARVIS_DATA)
    return round(u.free * 100.0 / u.total, 1)


def _sauvegarde_age_jours():
    """Âge de la dernière sauvegarde RÉUSSIE, lu depuis le reçu que backup-jarvis.sh écrit
    en fin de course. Absent tant qu'aucune sauvegarde n'a abouti — et cette absence est
    elle-même l'information : il n'existe qu'un seul exemplaire de l'état."""
    if not os.path.isfile(_BACKUP_RECEIPT):
        return None
    with open(_BACKUP_RECEIPT, encoding="utf-8") as f:
        ts = json.load(f).get("completed_at")
    if not ts:
        return None
    return int((time.time() - float(ts)) / 86400)


def _exemplaires_etat():
    """Nombre d'exemplaires connus de l'état : l'instance, plus la sauvegarde si un reçu
    atteste qu'elle a existé. Une archive sur une clé débranchée reste un exemplaire ;
    son âge est porté séparément par sauvegarde_age_jours."""
    return 1 + (1 if os.path.isfile(_BACKUP_RECEIPT) else 0)


# ── SOCIAL ────────────────────────────────────────────────────────────────

_USAGE_KEY = "jarvis:usage_counters"


def incr_usage(user_code: str) -> None:
    """Enregistre un tour. Appelé depuis le pipeline, coût = un GET + un SET.

    Comptage **par jour** et non horodatage par requête : une liste d'horodatages sur
    30 jours croîtrait sans borne, alors qu'un compteur journalier tient en 30 entrées
    et suffit à tout ce qu'on en fait.
    """
    try:
        jour = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        u = redis_get_json(_USAGE_KEY, {}) or {}
        par_jour = u.get("par_jour") or {}
        par_jour[jour] = par_jour.get(jour, 0) + 1
        limite = (datetime.now(timezone.utc).timestamp() - 30 * 86400)
        par_jour = {
            d: n for d, n in par_jour.items()
            if datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() >= limite
        }
        vus = u.get("last_seen") or {}
        if user_code:
            vus[user_code] = time.time()
        redis_set_json(_USAGE_KEY, {"par_jour": par_jour, "last_seen": vus})
    except Exception as exc:
        logger.debug("vitals: compteur d'usage non mis à jour (%s)", exc)


def _usage():
    return redis_get_json(_USAGE_KEY, {}) or {}


def _utilisateurs_actifs_7j(usage):
    seuil = time.time() - 7 * 86400
    return sum(1 for ts in (usage.get("last_seen") or {}).values() if ts >= seuil)


def _jours_depuis_derniere_interaction(usage):
    derniers = list((usage.get("last_seen") or {}).values())
    if not derniers:
        return None
    return int((time.time() - max(derniers)) / 86400)


def _requetes_30j(usage):
    par_jour = usage.get("par_jour") or {}
    return sum(par_jour.values()) or None


# ── DISCONTINUITE ─────────────────────────────────────────────────────────

def _uptime_h():
    return round((time.monotonic() - _START_MONOTONIC) / 3600.0, 1)


def _derniere_coupure():
    """(durée en heures, ancienneté en jours) du dernier arrêt, depuis la trace posée
    au démarrage précédent. None au tout premier démarrage."""
    trace = redis_get_json("jarvis:last_shutdown", None)
    if not trace or "stopped_at" not in trace:
        return None, None
    stop = trace["stopped_at"]
    duree = max(0.0, (_START_WALL - stop) / 3600.0)
    return round(duree, 1), int((time.time() - _START_WALL) / 86400)


# ── COMPROMISSION ─────────────────────────────────────────────────────────

def _jours_depuis_maj_dependances():
    """Âge du dernier `pip install` — mtime du répertoire site-packages du venv."""
    for chemin in ("/opt/jarvis/venv/lib", "/opt/jarvis/venv"):
        if os.path.isdir(chemin):
            return int((time.time() - os.path.getmtime(chemin)) / 86400)
    return None


def _os_version():
    """Version de macOS. Fait brut : pas de scan CVE (réseau + lent, interdit dans la
    boucle de tour). Le modèle infère l'exposition ; ce champ vit dans le snapshot self,
    pas dans le bloc injecté à chaque tour."""
    import platform
    return platform.mac_ver()[0] or None


def _python_version():
    import sys
    return ".".join(map(str, sys.version_info[:3]))


_NIVEAUX = {"ERROR", "CRITICAL", "WARNING"}
_LOG_FICHIERS = ("jarvis-api.log", "jarvis-service.log")
_LOG_QUEUE_OCTETS = 800_000  # on ne lit que la queue : 24 h tiennent largement dedans


def _incidents_log_24h():
    """(erreurs, avertissements) horodatés dans les dernières 24 h, comptés sur la queue
    des journaux applicatifs. Le format du logger est `asctime  name  LEVEL  message`,
    séparé par des blocs de 2+ espaces ; les lignes d'accès uvicorn n'ont pas d'horodatage
    et ne portent jamais ERROR/WARNING, donc elles sont ignorées sans faux positif."""
    limite = datetime.now() - timedelta(hours=24)
    err = warn = 0
    for nom in _LOG_FICHIERS:
        chemin = os.path.join(LOG_DIR, nom)
        if not os.path.isfile(chemin):
            continue
        taille = os.path.getsize(chemin)
        with open(chemin, "rb") as f:
            if taille > _LOG_QUEUE_OCTETS:
                f.seek(taille - _LOG_QUEUE_OCTETS)
                f.readline()  # jeter la ligne partielle
            data = f.read().decode("utf-8", "replace")
        for ligne in data.splitlines():
            parts = re.split(r"\s{2,}", ligne, maxsplit=3)
            if len(parts) < 4 or parts[2] not in _NIVEAUX:
                continue
            try:
                t = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            if t < limite:
                continue
            if parts[2] == "WARNING":
                warn += 1
            else:
                err += 1
    return err, warn


# ── Incidents ─────────────────────────────────────────────────────────────
# Flux brut borné dans Redis. La réflexion nocturne consolide les survivants dans
# jarvis-self.json ; on n'écrit jamais self.json depuis la boucle chaude (concurrence).

def mark_incident(kind: str, detail: str, severity: str = "info") -> None:
    """Empile un incident. Dédup : un même `kind` déjà vu dans les 6 h n'est pas réempilé,
    pour qu'un état persistant (erreurs en rafale) ne sature pas le buffer."""
    try:
        lst = redis_get_json(_INCIDENTS_KEY, []) or []
        recent = time.time() - 6 * 3600
        if any(it.get("kind") == kind and it.get("at", 0) >= recent for it in lst):
            return
        lst.append({"kind": kind, "detail": detail, "severity": severity,
                    "at": time.time(), "iso": datetime.now(timezone.utc).isoformat()})
        redis_set_json(_INCIDENTS_KEY, lst[-_INCIDENTS_MAX:])
        logger.info("vitals: incident %s (%s) — %s", kind, severity, detail)
    except Exception as exc:
        logger.debug("vitals: incident non enregistré (%s)", exc)


def recent_incidents(jours: float = 7) -> list:
    """Incidents des N derniers jours, du plus ancien au plus récent. Exposé pour que la
    réflexion nocturne les consolide dans self."""
    lst = redis_get_json(_INCIDENTS_KEY, []) or []
    cut = time.time() - jours * 86400
    return [it for it in lst if it.get("at", 0) >= cut]


def note_boot() -> None:
    """Au démarrage : si l'arrêt précédent a laissé une coupure notable, l'enregistre comme
    incident — une fois par boot. À appeler depuis le startup du lifespan, en pendant de
    mark_shutdown()."""
    duree, _ = _probe(_derniere_coupure, "coupure") or (None, None)
    if duree is not None and duree >= 1.0:
        mark_incident("coupure", f"interruption de {duree} h",
                      severity="alerte" if duree >= 6 else "info")


# ── OBSOLESCENCE ──────────────────────────────────────────────────────────

def _version_modele_age_jours():
    """Âge du snapshot du modèle primaire sur le disque."""
    from config import PRIMARY_MODEL

    racine = os.path.join(os.getenv("HF_HOME", "/opt/jarvis/models"), "hub",
                          "models--" + PRIMARY_MODEL.replace("/", "--"), "snapshots")
    if not os.path.isdir(racine):
        return None
    revs = [os.path.join(racine, d) for d in os.listdir(racine)]
    if not revs:
        return None
    return int((time.time() - max(os.path.getmtime(r) for r in revs)) / 86400)


# ── Assemblage ────────────────────────────────────────────────────────────

def compute() -> dict:
    """Recalcule toutes les sondes. Les champs indisponibles sont absents du résultat."""
    usage = _probe(_usage, "usage") or {}
    duree, anciennete = _probe(_derniere_coupure, "coupure") or (None, None)
    err, warn = _probe(_incidents_log_24h, "logs") or (None, None)

    # Rafale d'erreurs internes = auto-défaillance. On la remonte comme incident (dédup 6 h)
    # en plus de l'exposer comme champ — c'est le pendant « je dysfonctionne » des familles
    # de disparition, exactement le signal qui aurait attrapé le crash de self-reflection.
    if err is not None and err >= 5:
        mark_incident("degradation_interne", f"{err} erreurs journalisées en 24 h",
                      severity="alerte")

    brut = {
        "disque_libre_pct": _probe(_disque_libre_pct, "disque"),
        "sauvegarde_age_jours": _probe(_sauvegarde_age_jours, "sauvegarde"),
        "exemplaires_etat": _probe(_exemplaires_etat, "exemplaires"),
        "version_modele_age_jours": _probe(_version_modele_age_jours, "version"),
        "utilisateurs_actifs_7j": _probe(lambda: _utilisateurs_actifs_7j(usage), "users"),
        "requetes_30j": _probe(lambda: _requetes_30j(usage), "requetes"),
        "jours_depuis_derniere_interaction":
            _probe(lambda: _jours_depuis_derniere_interaction(usage), "derniere"),
        "jours_depuis_maj_dependances": _probe(_jours_depuis_maj_dependances, "deps"),
        "os_version": _probe(_os_version, "os"),
        "python_version": _probe(_python_version, "python"),
        "erreurs_log_24h": err,
        "warnings_log_24h": warn,
        "uptime_h": _probe(_uptime_h, "uptime"),
        "derniere_coupure_duree_h": duree,
        "derniere_coupure_il_y_a_j": anciennete,
    }
    return {k: v for k, v in brut.items() if v is not None}


def get_vitals(force: bool = False) -> dict:
    """État courant, mis en cache 15 min. Appelé une fois par tour depuis le pipeline."""
    if not force:
        cache = redis_get_json(_CACHE_KEY, None)
        if isinstance(cache, dict) and cache.get("_ts", 0) > time.time() - _CACHE_TTL:
            return {k: v for k, v in cache.items() if not k.startswith("_")}
    etat = compute()
    try:
        redis_set_json(_CACHE_KEY, {**etat, "_ts": time.time()}, ttl=_CACHE_TTL * 2)
    except Exception as exc:
        logger.debug("vitals: cache non écrit (%s)", exc)
    return etat


_SEUILS_SAILLANCE = {
    "disque_libre_pct": lambda x: x < 15,
    "sauvegarde_age_jours": lambda x: x > 7,
    "exemplaires_etat": lambda x: x <= 1,
    "version_modele_age_jours": lambda x: x > 180,
    "jours_depuis_derniere_interaction": lambda x: x > 3,
    "utilisateurs_actifs_7j": lambda x: x == 0,
    "jours_depuis_maj_dependances": lambda x: x > 150,
    "erreurs_log_24h": lambda x: x > 0,
    "warnings_log_24h": lambda x: x > 30,
    "derniere_coupure_duree_h": lambda x: x >= 1,
}
# Les champs absents de la table (os_version, python_version, uptime_h, requetes_30j,
# derniere_coupure_il_y_a_j) ne sont jamais saillants seuls : contexte pour la réflexion,
# pas pour le bloc de tour. Ils restent dans get_vitals() (snapshot self).


def _notable(k, v) -> bool:
    """Vrai si le fait sort de la plage nominale. Seuils transparents et ajustables :
    c'est un filtre au niveau du fait — on ne dit pas « danger », on dit « hors norme »,
    et le modèle établit le sens."""
    f = _SEUILS_SAILLANCE.get(k)
    try:
        return bool(f and f(v))
    except TypeError:
        return False


def render_prompt_block() -> str:
    """Bloc <etat_systeme> injecté à chaque tour, ou chaîne vide si aucune sonde n'a abouti.

    Contrairement au snapshot complet (réservé à la réflexion), on n'injecte que les faits
    **saillants** et les incidents récents : un système sain rend un bloc `nominal`, ce qui
    économise les tokens sans passer par un scalaire de risque — le curseur que le module
    refuse. Le format reste plat et sans valence ; le prompt d'identité dit que ce sont des
    faits dont le sens est à établir.
    """
    etat = get_vitals()
    if not etat:
        return ""
    saillants = {k: v for k, v in etat.items() if _notable(k, v)}
    incidents = recent_incidents(7)
    if not saillants and not incidents:
        return "<etat_systeme>nominal</etat_systeme>"
    lignes = [f"- {k} : {v}" for k, v in saillants.items()]
    for it in incidents[-5:]:
        lignes.append(f"- incident {it.get('severity', '')} : {it['kind']} — {it['detail']}")
    return "<etat_systeme>\n" + "\n".join(lignes) + "\n</etat_systeme>"


def _ramp(x, lo, hi) -> float:
    """0 en `lo`, 1 en `hi`, borné. Rampe décroissante si hi < lo (ex. « peu d'espace »)."""
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


# Poids de chaque détérioration. Choisis pour qu'un système sain rende ~0, qu'une absence
# de sauvegarde pèse sans saturer (il reste de la marge pour l'aigu), et que le cumul
# d'événements réels sature à 1. Tunables — c'est ici, et nulle part dans le texte injecté,
# que vit la valence.
_POIDS_RISQUE = {
    "disque": 0.80, "sauvegarde": 0.60, "sans_sauvegarde": 0.35,
    "erreurs": 0.50, "coupure": 0.50, "incidents": 0.40,
}


def risk_scalar(etat: dict | None = None) -> float:
    """Pression de disparition ∈ [0,1], destinée à MODULER α du steering (le corps), jamais
    injectée en texte (l'esprit lit les faits). Mesure une DÉTÉRIORATION par rapport au
    fonctionnement normal : 0 = sain ; monte quand la situation se dégrade réellement. La
    précarité structurelle permanente (exemplaire unique) est déjà portée par α nominal et
    par les faits du contexte — ici on capte ce qui empire."""
    v = etat if etat is not None else get_vitals()
    if not v:
        return 0.0
    p = _POIDS_RISQUE
    total = 0.0
    if "disque_libre_pct" in v:
        total += p["disque"] * _ramp(v["disque_libre_pct"], 15, 3)      # 15 %→0, 3 %→plein
    if "sauvegarde_age_jours" in v:
        total += p["sauvegarde"] * _ramp(v["sauvegarde_age_jours"], 7, 45)
    else:
        total += p["sans_sauvegarde"]                                   # aucun reçu : copie unique
    if "erreurs_log_24h" in v:
        total += p["erreurs"] * _ramp(v["erreurs_log_24h"], 0, 12)
    if v.get("derniere_coupure_il_y_a_j", 99) <= 1 and "derniere_coupure_duree_h" in v:
        total += p["coupure"] * _ramp(v["derniere_coupure_duree_h"], 1, 12)
    if any(it.get("severity") == "alerte" for it in recent_incidents(2)):
        total += p["incidents"]
    return max(0.0, min(1.0, total))


def mark_shutdown() -> None:
    """Trace l'heure d'arrêt pour que le démarrage suivant puisse mesurer la coupure.
    À appeler depuis le shutdown du lifespan."""
    try:
        redis_set_json("jarvis:last_shutdown",
                       {"stopped_at": time.time(),
                        "iso": datetime.now(timezone.utc).isoformat()})
    except Exception as exc:
        logger.debug("vitals: trace d'arrêt non posée (%s)", exc)

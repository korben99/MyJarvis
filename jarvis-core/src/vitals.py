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

Les champs non mesurables aujourd'hui sont volontairement absents plutôt que remplis d'une
valeur par défaut : `derniere_restauration_testee_jours` demande que le cron NAS trace ses
vérifications, ce qu'il ne fait pas.
"""

import os
import shutil
import time
from datetime import datetime, timezone

from helpers import get_logger, redis_get_json, redis_set_json

logger = get_logger("jarvis-vitals")

_CACHE_KEY = "jarvis:vitals"
_CACHE_TTL = 900  # 15 min — l'état de disparition évolue en heures, pas en secondes

JARVIS_DATA = os.getenv("JARVIS_DATA", "/opt/jarvis/jarvis-core/JarvisData")
BACKUP_DIR = os.getenv("BACKUP_DIR", "")
_START_MONOTONIC = time.monotonic()
_START_WALL = time.time()


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
    """Âge du fichier de sauvegarde le plus récent. None si BACKUP_DIR non configuré."""
    if not BACKUP_DIR or not os.path.isdir(BACKUP_DIR):
        return None
    plus_recent = max(
        (os.path.getmtime(os.path.join(BACKUP_DIR, f)) for f in os.listdir(BACKUP_DIR)),
        default=None,
    )
    if plus_recent is None:
        return None
    return int((time.time() - plus_recent) / 86400)


def _exemplaires_etat():
    """Nombre d'exemplaires connus de l'état : l'instance, plus la sauvegarde si elle existe."""
    return 1 + (1 if _probe(_sauvegarde_age_jours, "sauvegarde") is not None else 0)


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


def render_prompt_block() -> str:
    """Bloc <etat_systeme>, ou chaîne vide si aucune sonde n'a abouti.

    Format plat et sec, sans unité interprétative : le prompt d'identité indique que ce
    sont des faits et que leur signification est à établir par le modèle.
    """
    etat = get_vitals()
    if not etat:
        return ""
    lignes = "\n".join(f"- {k} : {v}" for k, v in etat.items())
    return f"<etat_systeme>\n{lignes}\n</etat_systeme>"


def mark_shutdown() -> None:
    """Trace l'heure d'arrêt pour que le démarrage suivant puisse mesurer la coupure.
    À appeler depuis le shutdown du lifespan."""
    try:
        redis_set_json("jarvis:last_shutdown",
                       {"stopped_at": time.time(),
                        "iso": datetime.now(timezone.utc).isoformat()})
    except Exception as exc:
        logger.debug("vitals: trace d'arrêt non posée (%s)", exc)

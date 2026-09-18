"""État du cycle d'autocoding : journal, cooldowns, patch en attente.

Trois mécanismes, trois rôles distincts, et il faut les garder séparés :

  journal        ce qui s'est passé, relu par la SÉLECTION du run suivant
  cooldown       une cible tranchée dort, pour ne pas la reproposer
  patch en vol   un seul à la fois — le coût réel n'est pas le GPU, c'est la relecture

Sans le journal, on n'a pas ajouté une boucle : on a ajouté un émetteur. C'est le défaut
mesuré de `refine_prompt`, qui republiait indéfiniment la même proposition parce que rien
ne lui disait qu'elle avait déjà été rejetée.
"""

import json
from datetime import datetime, timezone

from config import AUTOCODE_COOLDOWN_DAYS
from helpers import get_logger, get_redis

logger = get_logger("jarvis-autocode")

_JOURNAL_KEY = "jarvis:autocode:journal"
_JOURNAL_MAX = 30
_COOLDOWN_PREFIX = "jarvis:autocode:cooldown"
_APPLIQUEES_KEY = "jarvis:autocode:appliquees"
_EN_VOL_KEY = "jarvis:autocode:en_vol"
_VERROU_PREFIX = "jarvis:autocode:verrou"

# 25 h : recouvre la journée sans jamais laisser deux nuits consécutives se recouvrir.
_VERROU_TTL = 90000

DECISION_ACCEPTE = "accepte"
DECISION_REJETE = "rejete"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Journal ───────────────────────────────────────────────────────────────


def journaliser(entree: dict) -> None:
    """Consigne un run. Les plus récents en tête."""
    r = get_redis()
    r.lpush(_JOURNAL_KEY, json.dumps(entree, ensure_ascii=False))
    r.ltrim(_JOURNAL_KEY, 0, _JOURNAL_MAX - 1)


def journal(n: int = 10) -> list[dict]:
    entrees = []
    for brut in get_redis().lrange(_JOURNAL_KEY, 0, max(n, 1) - 1):
        try:
            entrees.append(json.loads(brut))
        except json.JSONDecodeError:
            continue
    return entrees


def marquer_decision(constat_id: str, decision: str) -> bool:
    """Inscrit la décision humaine SUR l'entrée du run, au lieu d'en ajouter une seconde.

    Deux entrées pour un même run — l'une sans décision, l'autre avec — se liraient comme
    deux tentatives dans le prompt de sélection, et gonfleraient artificiellement
    l'historique d'un sujet déjà tranché.
    """
    r = get_redis()
    for index, brut in enumerate(r.lrange(_JOURNAL_KEY, 0, _JOURNAL_MAX - 1)):
        try:
            entree = json.loads(brut)
        except json.JSONDecodeError:
            continue
        if entree.get("constat_id") != constat_id or entree.get("decision"):
            continue
        entree["decision"] = decision
        entree["tranche_at"] = now_iso()
        r.lset(_JOURNAL_KEY, index, json.dumps(entree, ensure_ascii=False))
        return True
    return False


# ── Cooldown et sortie définitive du vivier ───────────────────────────────


def poser_cooldown(constat_id: str) -> None:
    get_redis().setex(
        f"{_COOLDOWN_PREFIX}:{constat_id}", AUTOCODE_COOLDOWN_DAYS * 86400, now_iso()
    )


def en_cooldown(constat_id: str) -> bool:
    return bool(get_redis().exists(f"{_COOLDOWN_PREFIX}:{constat_id}"))


def marquer_appliquee(constat_id: str) -> None:
    """Sortie DÉFINITIVE du vivier : la cible est traitée, elle ne revient pas."""
    get_redis().sadd(_APPLIQUEES_KEY, constat_id)


def cibles_appliquees() -> set[str]:
    return set(get_redis().smembers(_APPLIQUEES_KEY))


# ── Patch en vol ──────────────────────────────────────────────────────────


def patch_en_attente() -> dict | None:
    """Le patch qui attend une décision humaine, s'il y en a un."""
    brut = get_redis().get(_EN_VOL_KEY)
    if not brut:
        return None
    try:
        return json.loads(brut)
    except json.JSONDecodeError:
        logger.warning("autocode: patch en vol illisible — ignoré")
        return None


def poser_en_vol(entree: dict) -> None:
    """Sans TTL : un patch attend aussi longtemps que l'humain met à le regarder.

    Un TTL rendrait la garde silencieusement inopérante au bout de N jours, et le cycle
    repartirait sur une pile de patchs non relus — exactement ce que la garde existe pour
    empêcher.
    """
    get_redis().set(_EN_VOL_KEY, json.dumps(entree, ensure_ascii=False))


def trancher(constat_id: str, decision: str) -> bool:
    """Enregistre la décision humaine et libère le créneau. False si rien n'attendait.

    N'APPLIQUE RIEN. L'application du patch reste un `git apply` que l'humain tape. Cette
    fonction ne fait que refermer la boucle : c'est elle qui donne à la sélection suivante
    de quoi ne pas reproposer ce qui vient d'être tranché.
    """
    en_vol = patch_en_attente()
    if not en_vol or en_vol.get("constat_id") != constat_id:
        return False

    if decision == DECISION_ACCEPTE:
        marquer_appliquee(constat_id)
    else:
        poser_cooldown(constat_id)

    if not marquer_decision(constat_id, decision):
        # Le run est sorti du journal (30 entrées) avant d'être tranché : on consigne la
        # décision seule, plutôt que de la perdre.
        journaliser({**en_vol, "decision": decision, "tranche_at": now_iso()})
    get_redis().delete(_EN_VOL_KEY)
    logger.info("autocode: %s tranché — %s", constat_id, decision)
    return True


# ── Idempotence ───────────────────────────────────────────────────────────


def prendre_verrou(date: str) -> bool:
    """True si le cycle du jour n'a pas encore tourné. Deux passages ⇒ un seul patch."""
    return bool(get_redis().set(f"{_VERROU_PREFIX}:{date}", "1", nx=True, ex=_VERROU_TTL))

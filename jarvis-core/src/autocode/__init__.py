"""Auto-correction nocturne — Jarvis observe un défaut chez lui, le prouve, le corrige.

Troisième régime autonome, à garder distinct des deux autres :

    self/      observe, réfléchit, PROPOSE            se déclenche seul
    agent/     AGIT sur le monde                      jamais seul — un humain poste la tâche
    autocode/  produit un PATCH, jamais appliqué      se déclenche seul, une fois par nuit

Ce qui autorise celui-ci à se déclencher seul là où `agent/` ne le peut pas, c'est que son
produit ne change rien : un fichier sur une étagère, qu'un humain lit et applique à la
main, ou pas. Le jour où ce cycle appliquerait son propre patch, cette justification
tomberait — et il faudrait tout réexaminer.

IL N'Y A QU'UN SEUL GENRE DE TÂCHE, et c'est l'invariant qui tient tout le reste :

    rends la preuve la plus forte disponible sur ce que tu avances

Le verdict n'est donc pas un type de tâche mais une DISTANCE parcourue le long d'une seule
échelle — `corrigé`, `reproduit`, `gardé`, `signalé`, `rien trouvé`, `rejeté`. Une version
précédente faisait des tâches distinctes avec des contrats distincts : la distinction
infusait dans six modules pour une seule idée.

Ce qui varie d'un cycle à l'autre, c'est seulement la manière d'arriver devant le code. Le
cycle nocturne part d'un constat désigné. Une revue — déclenchée à la main, jamais
planifiée — part de rien et laisse l'agent juger de ce qui mérite d'être signalé ; elle
n'atteint pas les rangs qui exigent un test, elle n'en est pas moins un résultat.

Cinq phases, une par module, chacune laissant son artefact numéroté dans le dossier de
livraison. Trois ne font AUCUN appel LLM, donc s'exercent de bout en bout sans GPU :

    constats.py   les faits                  1-constats.json
    choix.py      lequel, et pourquoi        2-choix.json          ← LLM
    chantier.py   arbres, agent, diff        3-patch.diff
    mesure.py     rouge→vert, suite, vacuité 4-mesure.json
    bilan.py      le rendu humain            5-RAPPORT.md          ← LLM
"""

import json
import re
from datetime import datetime

from config import (
    AGENT_ENABLED,
    AUTOCODE_COOLDOWN_DAYS,
    AUTOCODE_ENABLED,
    BRIEFING_TIMEZONE,
    USER_ADMINS,
)
from helpers import fmt_now_fr, get_logger

from . import bilan, chantier, choix, constats, mesure, store

logger = get_logger("jarvis-autocode")

__all__ = [
    "run_nightly_autocode", "handle_autocode_command",
    "bilan", "chantier", "choix", "constats", "mesure", "store",
]

# Longueur utile d'une notification iOS. Le rapport vit dans le courriel et sur l'étagère.
_PUSH_MAX_CARS = 400


def _admin() -> str | None:
    """Le destinataire du cycle : l'administrateur.

    Un cycle qui se déclenche seul n'a pas de demandeur — il faut bien désigner à qui il
    rend des comptes. `sorted` parce que `USER_ADMINS` est un ensemble : sans tri, deux
    redémarrages pourraient adresser les rapports à deux personnes différentes.
    """
    admins = sorted(USER_ADMINS)
    return admins[0] if admins else None


def _empechement() -> str | None:
    """Raison de ne pas lancer de cycle, ou None. Gardes mécaniques, dans l'ordre."""
    if not AUTOCODE_ENABLED:
        return "AUTOCODE_ENABLED=false"
    if not AGENT_ENABLED:
        return "AGENT_ENABLED=false — le cycle a besoin du worker agentique"
    if not _admin():
        return "aucun administrateur configuré"
    if (en_vol := store.patch_en_attente()):
        # La garde qui compte. Le coût réel du cycle n'est pas le GPU à 2 h du matin, c'est
        # la relecture humaine : empiler les patchs non tranchés reproduirait, à plus
        # grande échelle, les treize propositions de prompt que personne n'a regardées.
        return f"le patch {en_vol.get('constat_id')} attend une décision"
    return None


def _contexte_systeme() -> dict:
    """L'état, réduit à ce qui peut départager deux constats.

    Isolé : un cycle ne doit pas échouer parce que vitals est indisponible. Les constats se
    suffisent — l'état n'est qu'un départage.
    """
    try:
        from self.context import gather_global_context

        return gather_global_context()
    except Exception as exc:
        logger.warning("autocode: état système indisponible (%s)", type(exc).__name__)
        return {}


async def run_nightly_autocode(dry_run: bool = False, revue: bool = False) -> dict:
    """Un cycle. Rend un compte rendu structuré — ne lève jamais.

    `dry_run` s'arrête après le choix : de quoi contrôler ce que Jarvis retiendrait, et
    pourquoi, sans dépenser vingt minutes de GPU ni produire de patch.

    `revue` part sans cible désignée. Réservé au déclenchement manuel : le planificateur
    ne l'active jamais, une intention ouverte n'ayant pas à se déclencher seule.

    Le filet est ici et pas plus bas parce que la promesse porte sur l'APPELANT : le job
    planifié n'a personne pour rattraper une exception, et Redis indisponible ou git absent
    feraient disparaître le cycle sans un mot. Un compte rendu d'échec vaut mieux qu'une
    trace dans le journal d'APScheduler.
    """
    # Import tardif : `llm.local` charge le moteur d'inférence, inutile pour un cycle qui
    # sera peut-être court-circuité par un interrupteur fermé.
    from llm.local import _AUTOCODE_PROMPTS_LOG_PATH, journal_de_cycle

    try:
        # Les deux appels de cadrage vont dans LEUR journal, pas dans celui du chat. La
        # boucle agentique n'est pas concernée : elle tourne dans la tâche du worker, et
        # passe de toute façon son propre chemin, qui a la priorité.
        with journal_de_cycle(_AUTOCODE_PROMPTS_LOG_PATH):
            return await _cycle(dry_run, revue)
    except Exception as exc:
        logger.error("autocode: cycle interrompu (%s)", type(exc).__name__, exc_info=True)
        return {"lance": False, "motif": f"erreur : {type(exc).__name__}"}


def _revue_du_jour(date: str) -> dict:
    """La cible d'une revue : l'absence de cible.

    Rendue sous la même forme qu'un constat pour que les phases suivantes n'aient rien à
    savoir de l'origine — seul `construire_objectif` la lit, pour choisir l'objectif.
    """
    return {
        "id": f"REVUE-{date}",
        "constat": "revue de maintenance — aucune cible désignée",
        "cible": [],
        "notes": "",
        "origine": chantier.ORIGINE_REVUE,
        "raison": "déclenchée à la main",
        "angle": "",
    }


async def _cycle(dry_run: bool, revue: bool = False) -> dict:
    if (empechement := _empechement()):
        logger.info("autocode: cycle non lancé — %s", empechement)
        return {"lance": False, "motif": empechement}

    date = datetime.now().strftime("%Y-%m-%d")
    # Verrou distinct : une revue manuelle ne doit pas être refusée parce que le cycle
    # nocturne est déjà passé, ni consommer le créneau de la nuit suivante.
    if not dry_run and not store.prendre_verrou(f"{date}-revue" if revue else date):
        logger.info("autocode: cycle déjà passé le %s", date)
        return {"lance": False, "motif": "déjà passé aujourd'hui"}

    if revue:
        retenu = _revue_du_jour(date)
        logger.info("autocode: revue de maintenance, sans cible désignée")
        if dry_run:
            return {"lance": False, "motif": "dry_run", "cible": retenu["id"],
                    "constat": retenu["constat"], "origine": retenu["origine"],
                    "raison": retenu["raison"], "angle": "", "exclus": []}
        return await _executer(_admin(), retenu, [], [])

    eligibles, exclus = constats.recueillir()
    logger.info(
        "autocode: %d constat(s) éligible(s), %d écarté(s)", len(eligibles), len(exclus)
    )
    for identifiant, motif in exclus:
        logger.debug("autocode: %s écarté — %s", identifiant, motif)

    retenu = await choix.choisir(
        eligibles, _contexte_systeme(), fmt_now_fr(BRIEFING_TIMEZONE)
    )
    if retenu is None:
        return {"lance": False, "motif": "aucun constat retenu", "exclus": exclus}

    if dry_run:
        return {
            "lance": False, "motif": "dry_run", "cible": retenu["id"],
            "constat": retenu["constat"], "origine": retenu.get("origine"),
            "raison": retenu.get("raison", ""), "angle": retenu.get("angle", ""),
            "exclus": exclus,
        }

    return await _executer(_admin(), retenu, eligibles, exclus)


async def _executer(user_code: str, retenu: dict, eligibles: list[dict],
                    exclus: list[tuple[str, str]]) -> dict:
    """Chantier, mesure, bilan, restitution. Chaque phase laisse son artefact."""
    task = await chantier.lancer(user_code, retenu)
    if task is None:
        return {"lance": False, "motif": "préparation du worktree impossible"}

    dossier = chantier.dossier_du_jour(retenu)
    chantier.livrer(dossier, "1-constats.json", json.dumps(
        {"retenu": retenu["id"], "eligibles": eligibles,
         "exclus": [{"id": i, "motif": m} for i, m in exclus]},
        ensure_ascii=False, indent=2,
    ))
    chantier.livrer(dossier, "2-choix.json", json.dumps(
        {k: retenu.get(k) for k in ("id", "constat", "cible", "origine", "raison", "angle")},
        ensure_ascii=False, indent=2,
    ))

    try:
        termine = await chantier.attendre(task["id"])
        if termine is None:
            return {"lance": True, "cible": retenu["id"], "motif": "tâche non terminée"}

        # Une tâche annulée ou en échec n'est pas un résultat. La mesurer rendrait « rien
        # trouvé » sur un diff vide, et ce verdict poserait un cooldown de 30 jours sur un
        # constat que personne n'a examiné — il disparaîtrait un mois pour une interruption.
        from agent import store as agent_store

        if termine["status"] != agent_store.STATUS_DONE:
            logger.info(
                "autocode: %s — tâche %s, aucun verdict", retenu["id"], termine["status"]
            )
            return {"lance": True, "cible": retenu["id"], "motif": termine["status"]}

        diff = await chantier.produire_patch(termine["workspace"])
        chantier.livrer(dossier, "3-patch.diff", diff)

        # Les constats écrits vivent hors de repo/, donc hors du patch : sans cette copie
        # ils resteraient dans le workspace, que le `finally` efface.
        if (revue_ecrite := mesure.lire_revue(termine["workspace"])):
            chantier.livrer(dossier, mesure.FICHIER_REVUE, revue_ecrite)

        m = await mesure.mesurer(termine, diff)
        verdict, motifs = mesure.verdict(m)
        logger.info(
            "autocode: %s — verdict %s (%s)", retenu["id"], verdict, "; ".join(motifs)
        )
        chantier.livrer(dossier, "4-mesure.json", json.dumps(
            {"verdict": verdict, "motifs": motifs,
             **{k: v for k, v in m.items() if not k.endswith("_sortie")}},
            ensure_ascii=False, indent=2,
        ))
        chantier.livrer(dossier, "4-sorties.txt", "\n\n".join(
            f"───── {k} ─────\n{m.get(k) or '(non exécuté)'}"
            for k in ("compile_sortie", "pyflakes_sortie", "suite_sortie",
                      "suite_hors_test_sortie", "avant_sortie", "apres_sortie")
        ))

        redaction = await bilan.rediger(
            retenu, m, verdict, motifs, diff, termine.get("result", "")
        )
        chantier.livrer(dossier, "5-RAPPORT.md", bilan.rendre_rapport(
            retenu, m, verdict, motifs, redaction, termine, dossier
        ))

        entree = bilan.entree_journal(retenu, m, verdict, termine)
        entree["dossier"] = dossier
        store.journaliser(entree)

        # Le créneau n'est pris que par un résultat qu'il vaut la peine de relire. Un rejet
        # mécanique n'attend pas de décision — il bloquerait le cycle du lendemain au motif
        # que celui de la veille a échoué. Une bredouille non plus : elle ne laisse rien.
        # `gardé` en revanche laisse un patch, et se tranche comme les autres.
        if verdict in (mesure.REJETE, mesure.RIEN_TROUVE):
            store.poser_cooldown(retenu["id"])
        else:
            store.poser_en_vol(entree)

        _restituer(termine, retenu, verdict, motifs, dossier)
        return {"lance": True, "cible": retenu["id"], "verdict": verdict,
                "motifs": motifs, "dossier": dossier, "task_id": termine["id"]}
    finally:
        # Toujours, y compris sur erreur : un worktree abandonné reste enregistré dans .git
        # et gêne la tentative suivante sur le même chemin.
        await chantier.nettoyer(task["workspace"])


def _restituer(task: dict, retenu: dict, verdict: str, motifs: list[str],
               dossier: str) -> None:
    """Courriel puis push — cet ordre permet au push de dire « envoyé par mail ».

    Réutilise le chemin de restitution de `agent/` plutôt que d'en écrire un second : il
    porte déjà l'échappement HTML, le plafond de corps, la file Redis de repli et
    l'injection dans la conversation iOS. Silencieux en cas d'échec, comme lui.
    """
    from agent import report

    task["deliverables"] = chantier.copier_vers_workspace(
        task, dossier, ["5-RAPPORT.md", "3-patch.diff"]
    )
    task["result"] = (
        f"Autocoding — {retenu['id']} : {verdict}. " + "; ".join(motifs)
    ).strip()

    # Persisté, sinon `GET /agent/tasks/{id}` rendrait le résumé brut de l'agent — celui
    # d'avant la mesure — au lieu du verdict calculé, et sans ses livrables.
    try:
        from agent import store as agent_store

        agent_store.save_task(task)
    except Exception as exc:
        logger.warning("autocode: tâche non réenregistrée (%s)", type(exc).__name__)

    envoye = False
    try:
        envoye = report.envoyer(task)
    except Exception as exc:
        logger.warning("autocode: courriel en échec (%s)", type(exc).__name__)

    corps = (
        f"{retenu['id']} : {verdict}\n{retenu['constat'][:200]}\n\n" + "; ".join(motifs)
    ).strip()[:_PUSH_MAX_CARS]
    corps += "\n\nRapport envoyé par mail." if envoye else f"\n\nRapport dans {dossier}"

    try:
        from self.actions import _deliver_push

        if (erreur := _deliver_push(
            task["user_code"], corps,
            cooldown_key=f"jarvis:autocode:push:{task['id']}", cooldown_ttl=3600,
        )):
            logger.info("autocode: push non livré (%s)", erreur)
    except Exception as exc:
        logger.warning("autocode: notification en échec (%s)", type(exc).__name__)


# ── Décision humaine, depuis le chat ──────────────────────────────────────
# C'est ce chemin qui referme la boucle. Sans lui, le cycle reproposerait indéfiniment ce
# qui vient d'être écarté : on n'aurait pas ajouté une boucle, mais un émetteur.
#
# Il N'APPLIQUE RIEN, et c'est délibéré. « accepte » veut dire « je vais l'appliquer
# moi-même, ne me le repropose plus », pas « applique-le ». L'application reste un
# `git apply` tapé par un humain qui a lu le diff.

_RE_DECISION = re.compile(
    r"(?P<verbe>accepte|approuve|valide|rejette|refuse)\s+(?:le\s+)?patch"
    r"(?:\s+(?P<id>[A-Za-z0-9_-]+))?",
    re.IGNORECASE,
)


def _fmt_en_attente(en_vol: dict) -> str:
    return (
        f"Patch **{en_vol.get('constat_id')}** — {en_vol.get('constat', '')}\n"
        f"Verdict : {en_vol.get('verdict')}\n"
        f"Dossier : `{en_vol.get('dossier', '?')}`\n\n"
        f"Dis « accepte le patch {en_vol.get('constat_id')} » ou "
        f"« rejette le patch {en_vol.get('constat_id')} »."
    )


def handle_autocode_command(message: str, user_code: str) -> str | None:
    """Traite une commande de patch. None si le message n'en est pas une."""
    msg = message.strip().lower()

    if any(kw in msg for kw in ("patch en attente", "montre les patchs", "montre le patch",
                                "liste les patchs", "quels patchs")):
        en_vol = store.patch_en_attente()
        return _fmt_en_attente(en_vol) if en_vol else "Aucun patch en attente."

    if not (trouve := _RE_DECISION.search(msg)):
        return None

    if user_code not in USER_ADMINS:
        return "⛔ Seul un administrateur peut trancher un patch d'autocoding."

    en_vol = store.patch_en_attente()
    if not en_vol:
        return "Aucun patch en attente."

    identifiant = trouve.group("id")
    if not identifiant:
        return "Il manque l'identifiant.\n\n" + _fmt_en_attente(en_vol)

    # Comparaison insensible à la casse : le message est passé en minuscules, les
    # identifiants ne le sont pas.
    if identifiant.lower() != str(en_vol.get("constat_id", "")).lower():
        return (
            f"Le patch en attente est {en_vol.get('constat_id')}, pas {identifiant}.\n\n"
            + _fmt_en_attente(en_vol)
        )

    decision = (
        store.DECISION_REJETE
        if trouve.group("verbe") in ("rejette", "refuse")
        else store.DECISION_ACCEPTE
    )
    if not store.trancher(en_vol["constat_id"], decision):
        return "Rien n'a pu être tranché — le patch en attente a changé entre-temps."

    if decision == store.DECISION_ACCEPTE:
        if en_vol.get("verdict") == mesure.SIGNALE:
            return (
                f"Revue {en_vol['constat_id']} classée — elle ne sera plus reproposée. "
                f"Les constats restent dans `{en_vol.get('dossier', '<dossier>')}`."
            )
        return (
            f"Patch {en_vol['constat_id']} accepté — il ne sera plus reproposé. Rien n'a "
            f"été appliqué : `cd /opt/jarvis && git apply "
            f"{en_vol.get('dossier', '<dossier>')}/3-patch.diff`."
        )
    return (
        f"Patch {en_vol['constat_id']} rejeté. Le constat dort "
        f"{AUTOCODE_COOLDOWN_DAYS} jours, et le cycle de cette nuit peut repartir."
    )

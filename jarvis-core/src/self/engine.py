"""Orchestration de la réflexion : self-review avant action + boucle principale
en deux phases (globale puis par utilisateur), puis push proactif.

Sommet du paquet : dépend de state, proposals, context, actions.
"""

import asyncio
import json
import time
from datetime import datetime, timezone

from config import (
    DEFAULT_TEMP,
    MAX_CHAIN_ITERATIONS,
    MAX_TOKENS_THINK_MEDIUM,
    REASONING_API_KEY,
    REASONING_API_URL,
    REASONING_MODEL,
    REFLECTION_INTERVAL_HOURS,
    THINKING_BUDGET_MEDIUM,
    USER_ADMINS,
    USER_CODES,
    llm_timeout,
)
from helpers import call_llm_async_bg, extract_llm_json, get_logger, get_redis
from llm.local import _REFLECTION_PROMPTS_LOG_PATH, journal_de_cycle
import emotional_state
from memory import get_self_memory, save_self_memory, self_memory_lock
from prompts import get_prompt

from .actions import (
    _ACTION_CATALOG,
    _ADMIN_ALERT_COOLDOWN_PREFIX,
    EchecAction,
    _execute_action,
    alerter_si_anomalie_critique,
    generate_proactive_push,
)
from .context import (
    _call_global_reflection_llm,
    _call_user_reflection_llm,
    gather_global_context,
    gather_user_context,
)
from .state import (
    _DEVICE_TOKEN_PREFIX,
    _PUSH_COOLDOWN_PREFIX,
    consolidate_incidents,
    log_reflection,
)

logger = get_logger("jarvis-self")

# ── Catalogues d'actions ──────────────────────────────────────────────────
#
# Le cycle de réflexion N'APPREND PLUS, il AGIT. Découpage arrêté : la nuit
# écrit ce que Jarvis sait (des conversations et de son état), la réflexion consomme ce
# savoir pour faire des choses. Une seule question place n'importe quelle action —
# est-ce que ça écrit ce que Jarvis sait, ou est-ce que ça fait quelque chose ?
#
# Ont quitté ce catalogue, et pourquoi :
#   store_insight, correct_profile   la nuit est propriétaire de l'autobio et du profil ;
#                                    il y avait trois écrivains pour chacun.
#   consolidate_memory, prune_...    entretien de mémoire → revue nocturne.
#   check_health                     n'était pas une action : son résultat est déjà dans le
#                                    contexte via gather_global_context().
#   flag_knowledge_gap               → revue nocturne. Elle EXIGE de « décrire un échec
#                                    concret dans une vraie conversation », or ce cycle ne
#                                    voit que des compteurs et des topics, jamais le
#                                    contenu. Il ne pouvait donc pas la satisfaire
#                                    honnêtement — et les deux lacunes réellement en base
#                                    le montraient : « inertie décisionnelle », « gestion
#                                    des notifications », soit son propre comportement de
#                                    boucle, la seule chose qu'il pouvait observer. Ces
#                                    lacunes alimentaient `refine_prompt`, qui réécrivait
#                                    des prompts sur des défauts jamais constatés.
_SELF_ACTIONS = frozenset({"nothing", "refine_prompt", "alert_admin"})
_USER_ACTIONS = frozenset(
    {
        "nothing",
        "send_notification",
        "queue_push",
        "ask_user",
        "update_trade_threshold",
        "flag_project_stall",
    }
)

# Actions passant par une auto-contestation LLM avant exécution.
#
# `alert_admin` y est : elle réveille quelqu'un. `refine_prompt` n'y est PAS, et c'est un
# choix mesuré, pas un oubli — elle ne fait que PROPOSER un changement qui attend ensuite
# l'accord d'un humain, donc elle ne peut rien altérer seule. La contester en plus avait
# coûté 19 vetos sur 19 en quatre jours et étranglé la boucle d'auto-amélioration. Ses
# vrais garde-fous sont mécaniques et vivent dans _action_refine_prompt : une seule
# proposition en vol tous prompts confondus, et 30 jours de sommeil par sujet une fois
# celui-ci tranché — approuvé COMME rejeté.
_SELF_REVIEW_REQUIRED: frozenset[str] = frozenset({"alert_admin"})
_USER_REVIEW_REQUIRED: frozenset[str] = frozenset(
    {"queue_push", "ask_user", "send_notification"}
)

# L'invariant que le commentaire de _ACTION_CATALOG exigeait sans que rien ne le vérifie.
# Une action annoncée au modèle mais absente du catalogue retombe silencieusement sur
# "nothing" — c'est-à-dire un appel de raisonnement payé pour rien, découvert seulement en
# relisant le journal. On préfère refuser de démarrer.
_MANQUANTES = (_SELF_ACTIONS | _USER_ACTIONS) - set(_ACTION_CATALOG)
if _MANQUANTES:  # pragma: no cover - garde de démarrage
    raise RuntimeError(
        f"self: actions annoncées au modèle mais sans handler : {sorted(_MANQUANTES)}"
    )

# Ensemble des actions qui portent sur UN utilisateur et ont donc besoin de `user_code`.
# Dérivé du catalogue plutôt que réécrit à la main : la liste manuelle contenait encore
# `correct_profile` et `store_insight` (supprimées) et il lui MANQUAIT
# `flag_project_stall`, pourtant dans _USER_ACTIONS — l'action était donc morte par le
# chemin LLM, où elle retombait sur « invalid user_code », et ne fonctionnait que par le
# chemin mécanique qui passe user_code explicitement.
_USER_SCOPED = _USER_ACTIONS - {"nothing"}


# Empreinte des signaux PERSISTANTS de la dernière ouverture du garde.
# Voir _matiere_pour_agir_sur_soi.
_EMPREINTE_MATIERE_KEY = "jarvis:self:matiere_empreinte"
# TTL de 7 jours, et il compte. L'empreinte est posée quand le garde OUVRE, pas quand la
# phase aboutit : si la chaîne LLM échoue juste après, l'état est marqué « traité » alors
# que personne ne l'a traité. Sans péremption, un service durablement injoignable ne serait
# donc plus jamais réexaminé. Avec, il l'est une fois par semaine au lieu de cinq fois par
# jour — ce qui était le but — et le pire cas reste borné.
_EMPREINTE_MATIERE_TTL = 7 * 86400


def _matiere_pour_agir_sur_soi(ctx: dict) -> str:
    """Ce sur quoi Jarvis pourrait agir CHEZ LUI, ou une chaîne vide s'il n'y a rien.

    Garde mécanique, sans LLM, en tête de l'appel « agir sur soi ». Le catalogue de cet
    appel est court par nature — proposer un prompt, alerter l'admin — et sans matière il
    ne peut répondre que « nothing ». Mesuré sur 95 cycles avant le découpage : 69 se
    concluaient ainsi. Autant ne pas payer l'appel.

    DÉCLENCHEMENT SUR FRONT, corrigé. Trois des quatre signaux sont des
    ÉTATS et non des événements : un service reste injoignable, une CVE critique reste
    ouverte, une lacune non traitée le reste. Le quatrième signal d'alors — un compteur
    d'occurrences de lacunes — n'était jamais décrémenté : une fois le seuil franchi, le
    garde restait ouvert à vie et l'appel LLM qu'il devait éviter était payé toutes les
    REFLECTION_INTERVAL_HOURS indéfiniment. Il a été supprimé au profit du nombre de
    lacunes ACTIONNABLES, qui retombe de lui-même dès qu'une proposition est déposée.

    On mémorise donc l'empreinte des signaux persistants. Tant qu'elle ne bouge pas, ils
    ne rouvrent plus le garde : on a déjà regardé, on a déjà décidé. Ils le rouvrent dès
    qu'elle change — un service de plus, une CVE de plus, une lacune de plus.

    Les incidents échappent à l'empreinte : ils sont déjà filtrés sur la fenêtre écoulée
    depuis le dernier passage, donc ils sont par construction un vrai front.

    Ce que ce garde ne couvre pas, et n'a pas à couvrir : l'alerte sur anomalie critique
    part mécaniquement dans `alerter_si_anomalie_critique`, avec son propre cooldown de
    4 h, indépendamment de ce que le modèle décidera ensuite.
    """
    persistants: list[str] = []
    frais: list[str] = []

    services = ctx.get("health") or {}
    hs = sorted(n for n, v in services.items() if isinstance(v, str) and v != "ok")
    if hs:
        persistants.append(f"service(s) injoignable(s) : {', '.join(hs)}")

    if (ctx.get("cve_conseil") or "").strip():
        persistants.append("CVE critique corrigeable")

    # Lacunes sur lesquelles il reste quelque chose à faire — ni proposition en attente,
    # ni sujet en sommeil. Remplace l'ancien `gap_max_count >= REFINE_PROMPT_THRESHOLD` :
    # ce compteur n'était jamais décrémenté, donc le seuil, une fois franchi, laissait le
    # garde ouvert à vie. Celui-ci retombe de lui-même dès que la lacune est traitée.
    actionnables = int(ctx.get("gaps_actionnables", 0))
    if actionnables:
        persistants.append(f"{actionnables} lacune(s) sans proposition")

    fenetre = time.time() - REFLECTION_INTERVAL_HOURS * 3600
    nouveaux = [i for i in (ctx.get("incidents") or []) if float(i.get("at", 0)) >= fenetre]
    if nouveaux:
        frais.append(f"{len(nouveaux)} incident(s) depuis le dernier passage")

    if persistants:
        # L'empreinte porte les LIBELLÉS, qui contiennent déjà tout ce qui distingue deux
        # états : les noms de services, le compteur de lacunes. Un hash de plus n'ajouterait
        # rien qu'une indirection.
        empreinte = " | ".join(persistants)
        try:
            r = get_redis()
            deja_vue = r.get(_EMPREINTE_MATIERE_KEY)
            if deja_vue == empreinte:
                logger.info(
                    "Agir sur soi : état persistant inchangé, déjà traité — %s", empreinte
                )
                persistants = []
            else:
                r.setex(_EMPREINTE_MATIERE_KEY, _EMPREINTE_MATIERE_TTL, empreinte)
        except Exception as exc:
            # Redis indisponible : on retombe sur l'ancien comportement (garde ouvert)
            # plutôt que de fermer l'appel à tort.
            logger.warning("Empreinte de matière illisible (%s) — garde non filtré", exc)

    return " · ".join(persistants + frais)


def _obstacle_mecanique(action: str, params: dict) -> str | None:
    """L'obstacle EN DUR qui rend l'action impossible, ou None si la voie est libre.

    Évalué avant l'auto-contestation, et c'est tout le point. Mesuré à
    15 h 12 : le contesteur a refusé un `queue_push` au motif que « le délai
    d'indisponibilité (17 h 31 restantes) empêche l'envoi » — c'est-à-dire le cooldown de
    push, que `_deliver_push` vérifie deux lignes plus loin par un simple `r.exists()`. Un
    appel complet au modèle de raisonnement, budget de réflexion compris, pour conclure ce
    qu'un EXISTS Redis savait déjà.

    On ne paie donc le contesteur que pour la question qu'aucun compteur ne tranche — ce
    message apporte-t-il quelque chose. Le « est-ce seulement possible » se calcule.
    """
    try:
        r = get_redis()
        if action in ("queue_push", "ask_user"):
            user_code = params.get("user_code", "")
            if not user_code:
                return None
            if not r.exists(f"{_DEVICE_TOKEN_PREFIX}:{user_code}"):
                return f"aucun appareil enregistré pour {user_code}"
            ttl = r.ttl(f"{_PUSH_COOLDOWN_PREFIX}:{user_code}")
            if ttl > 0:
                h, m = divmod(ttl // 60, 60)
                return f"cooldown de push actif (encore {h} h {m:02d})"
        elif action == "alert_admin":
            admin = next(iter(USER_ADMINS), "")
            if not admin:
                return "aucun administrateur configuré"
            if not r.exists(f"{_DEVICE_TOKEN_PREFIX}:{admin}"):
                return f"aucun appareil enregistré pour l'administrateur {admin}"
            # Même cooldown que celui appliqué par _deliver_push : sans ce contrôle, une
            # alerte bloquée par le cooldown payait quand même le contesteur — le cas
            # exact que cette fonction existe pour éviter.
            ttl = r.ttl(f"{_ADMIN_ALERT_COOLDOWN_PREFIX}:{admin}")
            if ttl > 0:
                h, m = divmod(ttl // 60, 60)
                return f"alerte admin déjà envoyée (encore {h} h {m:02d} de cooldown)"
    except Exception as exc:
        # Redis muet : on laisse passer vers le contesteur plutôt que de bloquer à tort.
        logger.warning("Obstacle mécanique non évaluable (%s)", type(exc).__name__)
    return None


def _build_review_context(
    action: str,
    global_ctx: dict,
    user_ctx: dict | None,
    params: dict | None = None,
) -> tuple[str, str]:
    """Return (context_str, criteria_str) tailored to the action being reviewed.

    La branche `refine_prompt` a été retirée : l'action avait quitté
    `_SELF_REVIEW_REQUIRED`, donc la branche était inatteignable depuis. Ses deux lectures
    utiles — le compteur de lacunes du sujet et l'historique des propositions du prompt —
    ont été déplacées dans `_action_refine_prompt`, où elles sont désormais journalisées
    à chaque proposition.
    """
    params = params or {}

    if action == "send_notification" and user_ctx:
        # Critères PROPRES au courriel. Ils étaient ceux du push (« Push iOS disponible :
        # … »), alors que send_notification passe par Gmail et porte son propre garde
        # d'une notification par jour et par utilisateur : le contesteur jugeait sur un
        # canal qui n'était pas celui de l'action.
        activity = str(user_ctx.get("user_activity", {}))[:300]
        context = (
            f"Canal : courriel (garde interne : au plus un par jour et par utilisateur)\n"
            f"Activité récente : {activity}"
        )
        criteria = (
            "Un courriel est justifié si le contenu est durable et actionnable — quelque "
            "chose que la personne voudra relire ou retrouver. Ce qui se dit en une phrase "
            "relève du push, pas du courriel. Dans le doute, dis false."
        )

    elif action in ("queue_push", "ask_user") and user_ctx:
        # La disponibilité du canal et le cooldown sont déjà tranchés en amont par
        # _obstacle_mecanique : si on arrive ici, l'envoi est POSSIBLE. Reste la seule
        # question qui demande un jugement.
        activity = str(user_ctx.get("user_activity", {}))[:300]
        context = (
            f"Canal disponible, cooldown expiré — l'envoi est techniquement possible.\n"
            f"Activité récente : {activity}"
        )
        criteria = (
            "Le push est possible : juge uniquement s'il APPORTE quelque chose. Justifié si "
            "le message a une valeur concrète pour la personne maintenant et n'a pas déjà "
            "été dit. Sois conservateur : mieux vaut ne pas envoyer que spammer."
        )

    else:
        context = "Contexte général — évalue selon le bon sens."
        criteria = "L'action doit apporter une valeur claire et concrète maintenant."

    return context, criteria


async def _llm_review_before_action(
    action: str,
    params: dict,
    global_ctx: dict,
    user_ctx: dict | None,
    previous_steps: list[dict],
) -> tuple[bool, str]:
    """
    Self-challenge LLM call before executing a consequential action.

    Tourne sur REASONING_MODEL, en mode raisonnant (`no_think=False`) — la docstring
    annonçait « the router model (fast, binary decision) », ce qui n'a jamais été le cas et
    faisait sous-estimer d'un ordre de grandeur ce que coûte une contestation. C'est
    précisément pourquoi `_obstacle_mecanique` filtre en amont.

    Returns (should_execute, reason).
    Fail-closed: if the review call fails, the action is blocked (conservative default).
    """
    context_str, criteria_str = _build_review_context(
        action, global_ctx, user_ctx, params
    )

    steps_summary = (
        "; ".join(f"{s['action']}→{s['outcome'][:60]}" for s in previous_steps)
        or "aucune"
    )

    prompt = get_prompt("ACTION_REVIEW_USER").format(
        action=action,
        params=json.dumps(params, ensure_ascii=False, default=str),
        context=context_str,
        previous_steps=steps_summary,
        criteria=criteria_str,
    )

    try:
        content = await call_llm_async_bg(
            [
                {"role": "system", "content": get_prompt("ACTION_REVIEW_SYSTEM")},
                {"role": "user", "content": prompt},
            ],
            model=REASONING_MODEL,
            api_url=REASONING_API_URL,
            api_key=REASONING_API_KEY,
            temperature=DEFAULT_TEMP,
            max_tokens=MAX_TOKENS_THINK_MEDIUM,
            thinking_budget=THINKING_BUDGET_MEDIUM,
            json_response=True,
            no_think=False,
            timeout=llm_timeout(MAX_TOKENS_THINK_MEDIUM),
        )
        result = extract_llm_json(content)
        execute = bool(result.get("execute", False))
        reason = result.get("reason", "")
        return execute, reason
    except Exception as exc:
        logger.warning(
            "Action self-review failed (%s) — blocking action by default", exc
        )
        return False, "review failed — defaulting to block"


# ══════════════════════════════════════════════════
#  MAIN REFLECTION ENTRY POINT
# ══════════════════════════════════════════════════


def _run_chain_step(
    result: dict,
    steps: list[dict],
    allowed_actions: frozenset,
    phase_label: str,
) -> tuple[str, str, str, dict, bool]:
    """
    Extract and validate one chain step from an LLM result.

    Returns (focus, action, reason, params, should_stop).
    should_stop=True means the caller must break the chain loop.
    """
    focus = result.get("focus", "").strip()
    action = result.get("action", "nothing").strip()
    reason = result.get("reason", "").strip()
    params = result.get("params", {})

    # Guard 1: action must be in the allowed catalog for this phase
    if action not in allowed_actions:
        logger.warning(
            "%s: invalid action %r (not in allowed set) — defaulting to nothing",
            phase_label,
            action,
        )
        action = "nothing"
        params = {"reason": f"invalid action for this phase: {result.get('action')}"}

    # Guard 2: detect exact duplicate to prevent infinite loops
    _sig = json.dumps({"action": action, "params": params}, sort_keys=True)
    if any(
        json.dumps({"action": s["action"], "params": s["params"]}, sort_keys=True)
        == _sig
        for s in steps
    ):
        logger.info("%s: duplicate action=%s — stopping chain", phase_label, action)
        return focus, action, reason, params, True

    return focus, action, reason, params, False


def _pas_marquant(steps: list[dict]) -> dict:
    """Le pas qui RÉSUME le cycle — pas simplement le dernier.

    L'en-tête de l'entrée de journal est lue par `/self/log`, par `get_last_reflection`
    (qui l'injecte dans le prompt de réflexion), et surtout injectée en conversation
    (« Dernière action autonome : … »).
    Elle prenait `all_steps[-1]`, ce qui produisait deux défauts mesurés sur les 30 entrées
    du journal :

      • un `nothing` en fin de chaîne MASQUAIT tout ce qui l'avait précédé — 9 cycles sur
        30 journalisaient « nothing » alors qu'une action avait bien eu lieu, dont un
        « prune, prune, prune, nothing » rendu comme « nothing » ;
      • le dernier pas appartient presque toujours au dernier utilisateur de USER_CODES,
        souvent une relance mécanique, alors que le focus venait de la phase globale.

    Règle : le dernier pas qui a réellement AGI, la phase « agir sur soi » étant prioritaire
    parce qu'elle porte le regard de Jarvis sur lui-même. À défaut d'action, le dernier pas,
    qui dit alors honnêtement que le cycle n'a rien fait.
    """
    if not steps:
        return {"action": "nothing", "reason": "no steps executed", "outcome": ""}
    agissants = [s for s in steps if s.get("action") not in (None, "nothing")]
    if not agissants:
        return steps[-1]
    sur_soi = [s for s in agissants if s.get("phase") == "agir_sur_soi"]
    return (sur_soi or agissants)[-1]


async def run_self_reflection() -> dict:
    """LE CYCLE QUI AGIT. Appelé par APScheduler toutes les REFLECTION_INTERVAL_HOURS.

    Deux appels, tous deux tournés vers l'extérieur — apprendre appartient à la nuit :
      agir sur soi          : proposer un prompt, alerter l'admin. Précédé d'un garde
                              mécanique : sans matière, pas d'appel LLM du tout.
      agir vers l'utilisateur : push, mail, question, relance — un enchaînement par
                              utilisateur actif, jusqu'à MAX_CHAIN_ITERATIONS pas.

    Retourne l'entrée de journal, tous les pas sous la clé "steps".
    """
    with journal_de_cycle(_REFLECTION_PROMPTS_LOG_PATH):
        return await _reflechir()


async def _reflechir() -> dict:
    """Corps du cycle. Séparé pour que tout ce qu'il appelle écrive dans son journal."""
    logger.info(
        "=== Jarvis self-reflection starting (max %d steps/phase) ===",
        MAX_CHAIN_ITERATIONS,
    )

    # Consolide d'abord les incidents (coupures, dégradations) dans self.json — de façon
    # déterministe, avant tout appel LLM, pour que la trace survive même si la chaîne échoue.
    await asyncio.to_thread(consolidate_incidents)

    global_ctx = await asyncio.to_thread(gather_global_context)

    # Un service injoignable ou des vecteurs non normalisés sont anormaux sans discussion :
    # l'alerte part mécaniquement, sans dépendre de ce que le modèle choisira ensuite.
    #
    # Appelée APRÈS la collecte, et avec ses résultats. Elle tournait avant, et refaisait
    # donc ses propres sondes — dont `_check_memory_health`, qui fait un scroll Qdrant par
    # utilisateur : deux sondes complètes par cycle pour un état identique. L'ordre
    # n'affecte pas la garantie recherchée, puisque la collecte ne fait aucun appel LLM.
    logger.info(
        "Contrôle d'anomalies : %s",
        await asyncio.to_thread(
            alerter_si_anomalie_critique, global_ctx["health"], global_ctx["memory_health"]
        ),
    )

    global_steps: list[dict] = []
    focus = ""

    # ── Appel 3 — AGIR SUR SOI ─────────────────────────────────────────────
    matiere = _matiere_pour_agir_sur_soi(global_ctx)
    if not matiere:
        logger.info("--- Agir sur soi : rien à traiter, appel LLM évité ---")
    for i in range(MAX_CHAIN_ITERATIONS if matiere else 0):
        result = await _call_global_reflection_llm(
            global_ctx, previous_steps=global_steps
        )

        if result is None:
            logger.warning("Global reflection LLM failed at step %d — stopping", i + 1)
            break

        focus, action, reason, params, stop = _run_chain_step(
            result, global_steps, _SELF_ACTIONS, f"soi-step{i + 1}"
        )
        params.setdefault(
            "reason", reason
        )  # forward top-level reason into _action_nothing

        if action in _SELF_REVIEW_REQUIRED:
            obstacle = _obstacle_mecanique(action, params)
            if obstacle:
                logger.info("Agir sur soi : %s impossible — %s", action, obstacle)
                action = "nothing"
                params = {"reason": f"impossible : {obstacle}"}
            else:
                approved, rev_reason = await _llm_review_before_action(
                    action, params, global_ctx, None, global_steps
                )
                if not approved:
                    logger.info(
                        "Agir sur soi : auto-contestation refuse %s (%s)", action, rev_reason
                    )
                    action = "nothing"
                    params = {"reason": f"self-review: {rev_reason}"}
                    # Don't stop the chain — let the LLM try another action.
                    # Guard 2 (duplicate detection) prevents infinite loops.

        outcome = await asyncio.to_thread(_execute_action, action, params)

        # Même règle qu'en phase utilisateur. La confiance montait ici AVANT de regarder le
        # résultat : un alert_admin en échec la faisait monter.
        if action != "nothing":
            emotional_state.update(
                {"confiance": -0.1 if isinstance(outcome, EchecAction) else +0.1}
            )

        step = {
            "phase": "agir_sur_soi",
            "iteration": i + 1,
            "focus": focus,
            "action": action,
            "reason": reason,
            "params": params,
            "outcome": outcome,
        }
        global_steps.append(step)
        logger.info(
            "P1 step %d/%d: action=%s outcome=%s",
            i + 1,
            MAX_CHAIN_ITERATIONS,
            action,
            outcome,
        )

        if stop or action == "nothing":
            break

    # ── Appel 4 — AGIR VERS L'UTILISATEUR ──────────────────────────────────
    logger.info("--- Agir vers l'utilisateur (%d) ---", len(USER_CODES))
    all_user_steps: list[dict] = []

    for user_code in USER_CODES:
        user_ctx = gather_user_context(user_code)

        # Skip users with no conversation in the activity window. Measured over 95
        # calls (4 days): all 69 zero-activity cycles answered "nothing" with the reason
        # "aucune activité récente", while every one of the 9 proposed actions came from a
        # user with 7+ conversations. Reflecting on a silent user costs ~1800 prompt tokens
        # pour une conclusion écrite d'avance.
        if not user_ctx["user_activity"].get("conversations"):
            logger.info(
                "--- User: %s (%s) — skipped (no activity) ---",
                user_code,
                user_ctx["user_name"],
            )
            # MAIS la relance de tâches, elle, tourne quand même — mécaniquement, sans
            # passer par le LLM.
            #
            # Le compromis d'origine acceptait de la perdre pour les utilisateurs
            # silencieux, au motif qu'« elle ne s'était jamais déclenchée sur la fenêtre
            # mesurée ». Cette fenêtre faisait 4 jours, et le seuil de relance est de 21 :
            # elle ne pouvait donc pas contenir le phénomène qu'on supprimait. En pratique,
            # des projets restaient en attente plusieurs mois — jusqu'à 122 jours — sans
            # jamais remonter, parce que leur propriétaire ne parlait plus.
            #
            # Or un projet en sommeil est PRÉCISÉMENT la signature d'un utilisateur
            # silencieux : la relance ne pouvait atteindre que ceux qui n'en avaient pas
            # besoin. Et rien ici ne demande un jugement — une échéance dépassée, un
            # projet sans mise à jour depuis 21 jours, ça se calcule. La fonction porte
            # déjà ses garde-fous (cooldown de 14 j par projet, échéances traitées à part).
            # Via to_thread comme toute autre action : c'était le seul _execute_action
            # appelé directement dans la boucle d'événements, ce qui lui faisait emprunter
            # la branche loop.create_task de _deliver_push au lieu de asyncio.run — un
            # chemin de code exercé nulle part ailleurs.
            outcome = await asyncio.to_thread(
                _execute_action, "flag_project_stall", {"user_code": user_code}
            )
            if "aucun projet" not in outcome and "aucun appareil" not in outcome:
                logger.info("Relance de tâches (%s) : %s", user_code, outcome)
                # Même forme que les pas issus de la chaîne LLM : ce pas peut être le
                # DERNIER du cycle (TEST est le dernier utilisateur et n'a jamais
                # d'activité), et c'est lui qui alimente alors le journal de réflexion.
                all_user_steps.append({
                    "phase": f"user:{user_code}",
                    "user": user_code,
                    "action": "flag_project_stall",
                    "reason": "relance mécanique — utilisateur silencieux",
                    "outcome": outcome,
                })
            continue

        user_steps: list[dict] = []
        _failed_actions: set[str] = (
            set()
        )  # actions that hit a system constraint this cycle
        logger.info("--- User: %s (%s) ---", user_code, user_ctx["user_name"])

        for i in range(MAX_CHAIN_ITERATIONS):
            result = await _call_user_reflection_llm(
                global_ctx, user_ctx, previous_steps=user_steps
            )

            if result is None:
                logger.warning(
                    "User reflection LLM failed at step %d for %s — stopping",
                    i + 1,
                    user_code,
                )
                break

            ufocus, action, reason, params, stop = _run_chain_step(
                result, user_steps, _USER_ACTIONS, f"user:{user_code}-step{i + 1}"
            )
            if not focus:
                focus = ufocus

            params.setdefault(
                "reason", reason
            )  # forward top-level reason into _action_nothing

            # Inject user_code into params for all user-scoped actions so the
            # LLM doesn't need to carry it reliably across iterations.
            if action in _USER_SCOPED and not params.get("user_code"):
                params["user_code"] = user_code

            # Don't retry an action that already hit a system-level constraint this cycle
            if action in _failed_actions:
                _prev_action = action
                logger.info(
                    "P2 %s step %d/%d: action=%s previously failed — skipping to nothing",
                    user_code,
                    i + 1,
                    MAX_CHAIN_ITERATIONS,
                    action,
                )
                action = "nothing"
                params = {
                    "reason": f"previous {_prev_action} hit a system constraint — not retrying"
                }

            if action in _USER_REVIEW_REQUIRED:
                obstacle = _obstacle_mecanique(action, params)
                if obstacle:
                    logger.info("P2 %s : %s impossible — %s", user_code, action, obstacle)
                    # Marquée en échec comme si elle avait été exécutée et refusée : un
                    # obstacle mécanique ne se lève pas dans le même cycle, et sans ça la
                    # chaîne peut la reproposer avant que la détection de doublon ne
                    # l'attrape.
                    _failed_actions.add(action)
                    action = "nothing"
                    params = {"reason": f"impossible : {obstacle}"}
                else:
                    approved, rev_reason = await _llm_review_before_action(
                        action, params, global_ctx, user_ctx, user_steps
                    )
                    if not approved:
                        logger.info(
                            "P2 %s self-review rejected %s: %s",
                            user_code,
                            action,
                            rev_reason,
                        )
                        action = "nothing"
                        params = {"reason": f"self-review: {rev_reason}"}
                        # Don't stop the chain — let the LLM try another action.
                        # Guard 2 (duplicate detection) prevents infinite loops.

            outcome = await asyncio.to_thread(_execute_action, action, params)

            # L'échec est porté par le TYPE de la sortie (EchecAction), plus par sa forme.
            # L'heuristique de préfixe classait en échec toutes les sorties de
            # flag_project_stall, succès compris — l'action était alors interdite pour le
            # reste du cycle et la confiance baissait à chaque rappel réellement envoyé.
            _echec = isinstance(outcome, EchecAction)
            if _echec and action != "nothing":
                _failed_actions.add(action)
                emotional_state.update({"confiance": -0.1})
            elif action != "nothing" and not _echec:
                emotional_state.update({"confiance": +0.1})

            step = {
                "phase": f"user:{user_code}",
                "iteration": i + 1,
                "focus": ufocus,
                "action": action,
                "reason": reason,
                "params": params,
                "outcome": outcome,
            }
            user_steps.append(step)
            all_user_steps.append(step)
            logger.info(
                "P2 %s step %d/%d: action=%s outcome=%s",
                user_code,
                i + 1,
                MAX_CHAIN_ITERATIONS,
                action,
                outcome,
            )

            if stop or action == "nothing":
                break

    # ── Persist focus + reflection metadata ────────────────────────────────
    all_steps = global_steps + all_user_steps
    now_iso = datetime.now(timezone.utc).isoformat()
    with self_memory_lock:
        data = get_self_memory()
        data["current_focus"] = focus
        data["last_reflection"] = now_iso
        data["reflection_count"] = data.get("reflection_count", 0) + 1
        save_self_memory(data)

    retenu = _pas_marquant(all_steps)
    # Lecture défensive : un pas n'a pas toujours la forme complète de la chaîne LLM —
    # la relance mécanique des utilisateurs silencieux en est un. Un KeyError ici ferait
    # échouer TOUT le cycle après coup, alors que les actions ont déjà été exécutées.
    log_entry = {
        "timestamp": now_iso,
        # `focus` vient du MÊME pas que `action`. Il était pris à part — premier focus non
        # vide, donc celui de la phase globale — pendant que l'action venait du dernier pas
        # du dernier utilisateur : le couple journalisé n'avait jamais existé.
        "focus": retenu.get("focus") or focus,
        "action": retenu.get("action", "nothing"),
        "reason": retenu.get("reason", ""),
        "outcome": retenu.get("outcome", ""),
        "steps": all_steps,
        "health": global_ctx["health"],
        # Le catalogue EN VIGUEUR au moment où l'entrée a été écrite. C'est ce qui permet à
        # `get_last_reflection` d'ignorer les entrées d'un catalogue révolu : sans lui, le
        # prompt de réflexion montre en exemple une action qui n'existe plus, et le modèle
        # la redemande — un appel de raisonnement dépensé pour une sortie rejetée.
        "catalogue": sorted(_SELF_ACTIONS | _USER_ACTIONS),
    }
    log_reflection(log_entry)

    logger.info(
        "=== Reflection complete: %d global + %d user step(s), final=%s ===",
        len(global_steps),
        len(all_user_steps),
        retenu.get("action", "nothing"),
    )

    # Proactive push: per-user LLM call — fully guarded (device check + cooldown)
    for code in USER_CODES:
        try:
            await generate_proactive_push(code)
        except Exception as exc:
            logger.warning(
                "generate_proactive_push error for %s: %s", code, type(exc).__name__
            )

    return log_entry

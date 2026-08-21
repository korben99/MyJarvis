"""Orchestration de la réflexion : self-review avant action + boucle principale
en deux phases (globale puis par utilisateur), puis push proactif.

Sommet du paquet : dépend de state, proposals, context, actions.
"""

import asyncio
import json
import re
import time
from datetime import datetime, timezone

from config import (
    DEFAULT_TEMP,
    MAX_CHAIN_ITERATIONS,
    MAX_TOKENS_THINK_MEDIUM,
    REASONING_API_KEY,
    REASONING_API_URL,
    REASONING_MODEL,
    REFINE_PROMPT_THRESHOLD,
    REFLECTION_INTERVAL_HOURS,
    THINKING_BUDGET_MEDIUM,
    USER_CODES,
    llm_timeout,
)
from helpers import call_llm_async_bg, extract_llm_json, get_logger, get_redis
import emotional_state
from memory import get_self_memory, save_self_memory, self_memory_lock
from prompts import get_prompt

from .actions import (
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
from .proposals import _load_proposals
from .state import _GAP_COUNTS_KEY, consolidate_incidents, log_reflection

logger = get_logger("jarvis-self")

# ── Catalogues d'actions ──────────────────────────────────────────────────
#
# Le cycle de réflexion N'APPREND PLUS, il AGIT. Découpage arrêté le 21/08/2026 : la nuit
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
# vrais garde-fous sont mécaniques et vivent dans _action_refine_prompt : pas de seconde
# proposition tant qu'une est en attente sur le même prompt, plus 30 jours de cooldown par
# sujet après approbation.
_SELF_REVIEW_REQUIRED: frozenset[str] = frozenset({"alert_admin"})
_USER_REVIEW_REQUIRED: frozenset[str] = frozenset(
    {"queue_push", "ask_user", "send_notification"}
)


def _matiere_pour_agir_sur_soi(ctx: dict) -> str:
    """Ce sur quoi Jarvis pourrait agir CHEZ LUI, ou une chaîne vide s'il n'y a rien.

    Garde mécanique, sans LLM, en tête de l'appel « agir sur soi ». Le catalogue de cet
    appel est court par nature — proposer un prompt, alerter l'admin — et sans matière il
    ne peut répondre que « nothing ». Mesuré sur 95 cycles avant le découpage : 69 se
    concluaient ainsi. Autant ne pas payer l'appel.

    Ce n'est pas un déclenchement par événement : il n'existe aucune sonde de santé
    permanente à écouter, l'état n'est calculé qu'ici, à chaque passage. On garde donc la
    fréquence et on décide sur pièces.

    Les incidents sont filtrés sur la fenêtre écoulée depuis le dernier passage : le
    contexte en remonte 30 jours, et un incident déjà vu au cycle précédent n'est plus une
    raison d'agir.
    """
    raisons = []

    services = ctx.get("health") or {}
    hs = [n for n, v in services.items() if isinstance(v, str) and v != "ok"]
    if hs:
        raisons.append(f"service(s) injoignable(s) : {', '.join(hs)}")

    if (ctx.get("cve_conseil") or "").strip():
        raisons.append("CVE critique corrigeable")

    frais = time.time() - REFLECTION_INTERVAL_HOURS * 3600
    nouveaux = [i for i in (ctx.get("incidents") or []) if float(i.get("at", 0)) >= frais]
    if nouveaux:
        raisons.append(f"{len(nouveaux)} incident(s) depuis le dernier passage")

    if int(ctx.get("gap_max_count", 0)) >= REFINE_PROMPT_THRESHOLD:
        raisons.append("lacune au seuil de proposition")

    return " · ".join(raisons)


def _build_review_context(
    action: str,
    global_ctx: dict,
    user_ctx: dict | None,
    params: dict | None = None,
) -> tuple[str, str]:
    """Return (context_str, criteria_str) tailored to the action being reviewed."""
    params = params or {}

    if action == "refine_prompt":
        topic = params.get("topic", "")
        prompt_name = params.get("prompt_name", "")

        # Raw gap count for this specific topic
        r = get_redis()
        slug = re.sub(r"\s+", "_", topic.lower())[:40]
        count = int(r.hget(_GAP_COUNTS_KEY, slug) or 0)

        # Recent proposal history for this prompt (last 3)
        all_proposals = _load_proposals()
        recent = [
            f"{p.get('status', '?')} le {p.get('created_at', '?')[:10]}"
            for p in all_proposals
            if p.get("prompt_name") == prompt_name
        ][-3:]
        proposals_history = "; ".join(recent) or "aucune"

        gaps = ", ".join(global_ctx.get("knowledge_gaps", [])) or "aucune"
        proposals_pending = global_ctx.get("pending_proposals", "aucune")

        context = (
            f"Topic proposé : '{topic}' — flaggé {count} fois dans les gaps\n"
            f"Lacunes connues : {gaps}\n"
            f"Historique des proposals pour '{prompt_name}' : {proposals_history}\n"
            f"Proposals en attente : {proposals_pending}"
        )
        criteria = (
            "refine_prompt est justifié si tu as des preuves concrètes que ce topic revient "
            "régulièrement dans les conversations (gap count significatif) ET qu'aucune proposal "
            "n'est déjà en attente ou n'a été soumise récemment pour ce prompt. "
            "Si les données ci-dessus ne montrent pas de problème récurrent réel, dis false."
        )

    elif action in ("queue_push", "ask_user", "send_notification") and user_ctx:
        has_push = user_ctx.get("has_push", False)
        last_push = user_ctx.get("push_cooldown_str", "inconnu")
        activity = str(user_ctx.get("user_activity", {}))[:300]
        context = (
            f"Push iOS disponible : {has_push}\n"
            f"Dernier push envoyé : {last_push}\n"
            f"Activité récente : {activity}"
        )
        criteria = (
            "Un push est justifié si : push disponible ET délai raisonnable depuis le dernier "
            "(au moins quelques heures) ET le message apporte une valeur concrète et urgente "
            "qui n'a pas déjà été envoyée. Si le dernier push est récent, dis false. "
            "Sois conservateur : mieux vaut ne pas envoyer que spammer."
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
    Uses the router model (fast, binary decision).
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


async def run_self_reflection() -> dict:
    """
    Two-phase self-reflection cycle. Called by APScheduler every REFLECTION_INTERVAL_HOURS.

    Phase 1 (global): Jarvis self-state — health, knowledge gaps, self-notes, prompts.
                      Up to MAX_CHAIN_ITERATIONS steps.
    Phase 2 (per-user): One LLM chain per user — profile, push, insights.
                        Up to MAX_CHAIN_ITERATIONS steps per user.

    Returns a log entry with all steps under the "steps" key.
    """
    logger.info(
        "=== Jarvis self-reflection starting (max %d steps/phase) ===",
        MAX_CHAIN_ITERATIONS,
    )

    # Consolide d'abord les incidents (coupures, dégradations) dans self.json — de façon
    # déterministe, avant tout appel LLM, pour que la trace survive même si la chaîne échoue.
    await asyncio.to_thread(consolidate_incidents)

    # Même principe : un service injoignable ou des vecteurs non normalisés sont anormaux
    # sans discussion. L'alerte part mécaniquement, sans dépendre de ce que le modèle
    # choisira ensuite — et le garde ci-dessous peut très bien fermer l'appel LLM.
    logger.info("Contrôle d'anomalies : %s", await asyncio.to_thread(alerter_si_anomalie_critique))

    global_ctx = await asyncio.to_thread(gather_global_context)
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
            approved, rev_reason = await _llm_review_before_action(
                action, params, global_ctx, None, global_steps
            )
            if not approved:
                logger.info("Agir sur soi : auto-contestation refuse %s (%s)", action, rev_reason)
                action = "nothing"
                params = {"reason": f"self-review: {rev_reason}"}
                # Don't stop the chain — let the LLM try another action.
                # Guard 2 (duplicate detection) prevents infinite loops.

        outcome = await asyncio.to_thread(_execute_action, action, params)

        if action not in ("nothing", "flag_knowledge_gap"):
            emotional_state.update({"confiance": +0.1})

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
            # elle ne pouvait donc pas contenir le phénomène qu'on supprimait. Conséquence
            # relevée le 20/08/2026 — 5 projets en attente, dont trois de Mathilde à 87,
            # 122 et 122 jours, tous invisibles parce qu'elle ne parlait plus.
            #
            # Or un projet en sommeil est PRÉCISÉMENT la signature d'un utilisateur
            # silencieux : la relance ne pouvait atteindre que ceux qui n'en avaient pas
            # besoin. Et rien ici ne demande un jugement — une échéance dépassée, un
            # projet sans mise à jour depuis 21 jours, ça se calcule. La fonction porte
            # déjà ses garde-fous (cooldown de 14 j par projet, échéances traitées à part).
            outcome = _execute_action("flag_project_stall", {"user_code": user_code})
            if "aucun projet" not in outcome:
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
            _user_scoped = {
                "correct_profile",
                "store_insight",
                "queue_push",
                "send_notification",
                "ask_user",
                "update_trade_threshold",
            }
            if action in _user_scoped and not params.get("user_code"):
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

            # Detect system-constraint failures: outcome format is "action: error"
            # (no "[user_code]" bracket), distinct from success "action [user_code]: ..."
            _looks_like_error = outcome.startswith(
                f"{action}:"
            ) and not outcome.startswith(f"{action} [")
            if _looks_like_error and action != "nothing":
                _failed_actions.add(action)
                emotional_state.update({"confiance": -0.1})
            elif action not in ("nothing", "flag_knowledge_gap") and not _looks_like_error:
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

    last = (
        all_steps[-1]
        if all_steps
        else {"action": "nothing", "reason": "no steps executed", "outcome": ""}
    )
    # Lecture défensive : un pas n'a pas toujours la forme complète de la chaîne LLM —
    # la relance mécanique des utilisateurs silencieux en est un. Un KeyError ici ferait
    # échouer TOUT le cycle après coup, alors que les actions ont déjà été exécutées.
    log_entry = {
        "timestamp": now_iso,
        "focus": focus,
        "action": last.get("action", "nothing"),  # for _extract_behavioral_patterns
        "reason": last.get("reason", ""),
        "outcome": last.get("outcome", ""),
        "steps": all_steps,
        "health": global_ctx["health"],
    }
    log_reflection(log_entry)

    logger.info(
        "=== Reflection complete: %d global + %d user step(s), final=%s ===",
        len(global_steps),
        len(all_user_steps),
        last.get("action", "nothing"),
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

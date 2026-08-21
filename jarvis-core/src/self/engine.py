"""Orchestration de la réflexion : self-review avant action + boucle principale
en deux phases (globale puis par utilisateur), puis push proactif.

Sommet du paquet : dépend de state, proposals, context, actions.
"""

import asyncio
import json
import re
from datetime import datetime, timezone

from config import (
    DEFAULT_TEMP,
    MAX_CHAIN_ITERATIONS,
    MAX_TOKENS_THINK_MEDIUM,
    REASONING_API_KEY,
    REASONING_API_URL,
    REASONING_MODEL,
    THINKING_BUDGET_MEDIUM,
    USER_CODES,
    llm_timeout,
)
from helpers import call_llm_async_bg, extract_llm_json, get_logger, get_redis
import emotional_state
from memory import get_self_memory, save_self_memory, self_memory_lock
from prompts import get_prompt

from .actions import _execute_action, generate_proactive_push
from .context import (
    _call_global_reflection_llm,
    _call_user_reflection_llm,
    gather_global_context,
    gather_user_context,
)
from .proposals import _load_proposals
from .state import _GAP_COUNTS_KEY, consolidate_incidents, log_reflection

logger = get_logger("jarvis-self")

# Actions allowed per phase — LLM cannot hallucinate cross-phase actions
_GLOBAL_ACTIONS = frozenset(
    {
        "nothing",
        "flag_knowledge_gap",
        "check_health",
        "prune_self_memory",
        "refine_prompt",
        "alert_admin",
    }
)
_USER_ACTIONS = frozenset(
    {
        "nothing",
        "store_insight",
        "send_notification",
        "queue_push",
        "correct_profile",
        "ask_user",
        "consolidate_memory",
        "update_trade_threshold",
        "flag_project_stall",
    }
)

# Actions that require a self-challenge LLM call before execution, per phase.
# Split in two: Phase 1 can only emit _GLOBAL_ACTIONS and Phase 2 only _USER_ACTIONS, so a
# single shared set advertised a coverage that did not exist — 3 of its 4 entries were
# unreachable from Phase 1.
#
# refine_prompt is deliberately NOT reviewed: it only *proposes* a change that already waits
# for human approval, so it cannot alter anything on its own. Its real guardrails are
# mechanical and live in _action_refine_prompt — no second proposal while one is pending for
# the same prompt, plus the 30-day topic cooldown set on approval. Reviewing it on top cost
# 19 vetoes out of 19 over the 4 days measured, and strangled the self-improvement loop.
_P1_REVIEW_REQUIRED: frozenset[str] = frozenset()
_P2_REVIEW_REQUIRED: frozenset[str] = frozenset(
    {"queue_push", "ask_user", "send_notification"}
)


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

    global_ctx = await asyncio.to_thread(gather_global_context)
    global_steps: list[dict] = []
    focus = ""

    # ── Phase 1: global self-state ─────────────────────────────────────────
    logger.info("--- Phase 1: global self-state ---")
    for i in range(MAX_CHAIN_ITERATIONS):
        result = await _call_global_reflection_llm(
            global_ctx, previous_steps=global_steps
        )

        if result is None:
            logger.warning("Global reflection LLM failed at step %d — stopping", i + 1)
            break

        focus, action, reason, params, stop = _run_chain_step(
            result, global_steps, _GLOBAL_ACTIONS, f"P1-step{i + 1}"
        )
        params.setdefault(
            "reason", reason
        )  # forward top-level reason into _action_nothing

        if action in _P1_REVIEW_REQUIRED:
            approved, rev_reason = await _llm_review_before_action(
                action, params, global_ctx, None, global_steps
            )
            if not approved:
                logger.info("P1 self-review rejected %s: %s", action, rev_reason)
                action = "nothing"
                params = {"reason": f"self-review: {rev_reason}"}
                # Don't stop the chain — let the LLM try another action.
                # Guard 2 (duplicate detection) prevents infinite loops.

        outcome = await asyncio.to_thread(_execute_action, action, params)

        if action not in ("nothing", "flag_knowledge_gap"):
            emotional_state.update({"confiance": +0.1})

        step = {
            "phase": "global",
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

    # ── Phase 2: per-user chains ───────────────────────────────────────────
    logger.info("--- Phase 2: per-user reflection (%d users) ---", len(USER_CODES))
    all_user_steps: list[dict] = []

    for user_code in USER_CODES:
        user_ctx = gather_user_context(user_code)

        # Skip users with no conversation in the activity window. Measured over 95 Phase 2
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
                all_user_steps.append({
                    "user": user_code, "action": "flag_project_stall", "outcome": outcome,
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
                result, user_steps, _USER_ACTIONS, f"P2-{user_code}-step{i + 1}"
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
                "consolidate_memory",
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

            if action in _P2_REVIEW_REQUIRED:
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
    log_entry = {
        "timestamp": now_iso,
        "focus": focus,
        "action": last["action"],  # for _extract_behavioral_patterns
        "reason": last["reason"],
        "outcome": last["outcome"],
        "steps": all_steps,
        "health": global_ctx["health"],
    }
    log_reflection(log_entry)

    logger.info(
        "=== Reflection complete: %d global + %d user step(s), final=%s ===",
        len(global_steps),
        len(all_user_steps),
        last["action"],
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

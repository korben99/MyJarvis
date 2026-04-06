"""
pipeline.py — System prompt construction, context assembly, post-analysis
==========================================================================
Extracted from main.py to keep chat() focused on HTTP I/O.

Public functions:
  build_system_prompt()                      -> str  (STATIC — KV-cache safe)
  build_dynamic_prefix(session_id,
                       user_code,
                       user_name,
                       voice_mode)           -> tuple[str, dict]
                                                (prefix, self_mem — shared with build_context)
  build_context(rag_chunks, memory_chunks, web_results, gmail_results,
                calendar_results, use_portfolio, use_self, user_code,
                self_mem)                    -> str
  post_analysis(session_id, user_code, user_msg, assistant_msg)

KV-cache strategy
─────────────────
The system message is now STATIC (SYSTEM_BASE_FR only).  Dynamic content
(date, user name, profile/memory, opinions, voice mode) is prepended to
each user message as a [JARVIS_CTX] block.  Because the system message and
all previous conversation turns are token-identical from one turn to the
next, MLX's KV cache reuses their attention state and only the new tokens
(the [JARVIS_CTX] block + current user message) need to be computed.
"""

import asyncio

from analyzer import analyze_exchange
from config import USER_TIMEZONES
from deps import (
    GOOGLE_CHAR_BUDGET,
    MEMORY_CHAR_BUDGET,
    RAG_CHAR_BUDGET,
    TOTAL_CONTEXT_BUDGET,
    WEB_CHAR_BUDGET,
)
from helpers import fmt_event_time, fmt_now_fr, get_logger, rel_time_fr
from llm_client import trim_chunks
from memory import (
    apply_project_updates,
    build_memory_context,
    get_user_profile,
    get_user_projects,
    log_conversation,
    retract_autobiographical_event,
    set_interest_weight,
    update_emotional_state,
    update_user_profile,
)
from prompts import get_prompt
from self import (
    get_reflection_log,
    get_self_memory,
    get_user_relation,
    list_pending_proposals,
)
from trading import get_portfolio_summary_text, pop_pending_alerts
from web_search import INTERNET_ERROR

logger = get_logger("jarvis-pipeline")

# Marker used to separate the injected context block from the user's raw message
# when both are stored together in Redis history.
_CTX_SEP = "\x00"  # null byte — never present in natural text

# Static lookup tables for self context — defined once at module level.
_STYLE_DIRECTIVES: dict[str, str] = {
    "direct":  "Réponds sans détours, va droit au but, sans formules de politesse superflues.",
    "gentle":  "Adopte une communication douce et bienveillante, prends le temps d'être rassurant.",
    "formal":  "Maintiens un registre formel et respectueux en toutes circonstances.",
    "playful": "Tu peux être léger et décontracté, l'humour est bienvenu.",
}
_MOOD_DIRECTIVES: dict[str, str] = {
    "warm":         "Adopte un ton chaleureux et bienveillant.",
    "enthusiastic": "Sois enthousiaste et investi dans tes réponses.",
    "measured":     "Reste posé et mesuré, ne surjoue pas.",
    "playful":      "Tu peux être joueur et humoristique.",
    "professional": "Garde un registre professionnel et précis.",
}


# ── System prompt ──────────────────────────────────────────────────────────────


def build_system_prompt() -> str:
    """
    Return the STATIC system prompt (SYSTEM_BASE_FR only).
    No date, no profile, no opinions — those go into build_dynamic_prefix().
    This string is token-identical across all turns → KV cache hit every time.
    """
    return get_prompt("SYSTEM_BASE_FR")


def build_dynamic_prefix(
    session_id: str,
    user_code: str,
    user_name: str = "",
    voice_mode: bool = False,
) -> tuple[str, dict]:
    """
    Build the per-turn context block prepended to the current user message.
    Contains: date/time, user name, profile/memory context, opinions, voice hint.
    Runs in a thread (I/O: Redis reads via build_memory_context / get_self_memory).

    Returns (prefix_string, self_mem) so the caller can pass self_mem to
    build_context() without triggering a second get_self_memory() Redis call.
    """
    tz = USER_TIMEZONES.get(user_code, "Europe/Paris")
    parts: list[str] = [f"Date et heure : {fmt_now_fr(tz)}."]

    if user_name:
        parts.append(f"Tu parles avec {user_name}.")

    self_mem = get_self_memory()

    memory_ctx = build_memory_context(session_id, user_code, self_mem=self_mem)
    if memory_ctx:
        parts.append(f"{get_prompt('MEMORY_HEADER_FR')}\n{memory_ctx}")

    opinions = self_mem.get("opinions", [])
    if opinions:
        ops_lines = "\n".join(f"- {o['topic']} : {o['opinion']}" for o in opinions[-10:])
        parts.append(f"=== TES OPINIONS ===\n{ops_lines}")

    if voice_mode:
        parts.append(get_prompt("VOICE_SUFFIX_FR").strip())

    return "\n\n".join(parts), self_mem


def augment_user_message(dynamic_prefix: str, raw_message: str) -> str:
    """
    Combine the dynamic prefix and the user's raw message for storage + LLM.
    The null-byte separator lets strip_ctx_prefix() recover the raw message.
    """
    if not dynamic_prefix:
        return raw_message
    return f"{dynamic_prefix}{_CTX_SEP}{raw_message}"


def strip_ctx_prefix(content: str) -> str:
    """
    Strip the dynamic context prefix from a stored user message.
    Used by the history endpoint so the iOS app sees the original message.
    """
    sep = content.find(_CTX_SEP)
    if sep != -1:
        return content[sep + 1:]
    return content


# ── Context assembly ───────────────────────────────────────────────────────────


def build_context(
    rag_chunks: list,
    memory_chunks: list,
    web_results: list,
    gmail_results: list,
    calendar_results: list,
    use_portfolio: bool,
    use_self: bool,
    user_code: str,
    self_mem: dict | None = None,
) -> str:
    """
    Assemble all fetched context into a single string for system prompt injection.
    Returns "" when there is nothing to inject.

    self_mem — pass the dict returned by build_dynamic_prefix() to avoid a
               second get_self_memory() call when use_self is True.

    Injection priority (background → urgent):
    1. Web results       (reference material, lowest urgency)
    2. RAG documents     (personal docs, reference)
    3. Memory            (episodic background)
    4. Calendar / Gmail  (scheduled facts, recent comms)
    5. Portfolio         (current financial state)
    6. Trade alerts      (urgent, actionable)
    7. Self              (internal state — only on self-intent)
    """
    context_parts = []

    # 1. WEB
    if web_results == INTERNET_ERROR:
        context_parts.append(
            "=== ACCÈS INTERNET ===\n"
            "La connexion internet est actuellement indisponible. "
            "Informe l'utilisateur que tu ne peux pas effectuer la recherche demandée "
            "et propose-lui de réessayer plus tard."
        )
        logger.warning("web: internet unavailable — injecting error context")
    elif web_results:
        web_selected = trim_chunks(web_results, WEB_CHAR_BUDGET, text_key="body")
        if web_selected:
            context_parts.append("=== RÉSULTATS WEB ===")
            for i, body in enumerate(web_selected):
                r = web_results[i]
                context_parts.append(f"[{r['title']}]\n{body}\nSource: {r['url']}")
        logger.info("web recall %d/%d (budget=%d)", len(web_selected), len(web_results), WEB_CHAR_BUDGET)

    # 2. RAG
    if rag_chunks:
        rag_selected_texts = trim_chunks(rag_chunks, RAG_CHAR_BUDGET)
        if rag_selected_texts:
            context_parts.append("=== DOCUMENTS PERSONNELS ===")
            selected_set = set(rag_selected_texts)
            for chunk in rag_chunks:
                text = chunk["text"][:800]
                if text in selected_set:
                    context_parts.append(f"[Doc {chunk['source']} ({chunk['score']:.2f})]\n{text}")
        logger.info("rag recall %d/%d (budget=%d)", len(rag_selected_texts), len(rag_chunks), RAG_CHAR_BUDGET)

    # 3. MEMORY
    # Build timestamped copies — do NOT mutate the caller's list.
    if memory_chunks:
        stamped = [
            {**m, "text": f"({rel_time_fr(m['timestamp'])}) {m['text']}"}
            if m.get("timestamp") else m
            for m in memory_chunks
        ]
        selected_memories = trim_chunks(stamped, MEMORY_CHAR_BUDGET)
        if selected_memories:
            context_parts.append("=== SOUVENIRS PERTINENTS ===")
            context_parts.extend(selected_memories)
        logger.info("memory recall %d/%d (budget=%d)", len(selected_memories), len(memory_chunks), MEMORY_CHAR_BUDGET)

    # 4a. CALENDAR
    if calendar_results:
        context_parts.append("=== AGENDA ===")
        for evt in calendar_results:
            if evt.get("all_day"):
                line = f"{evt['start']} — {evt['summary']} [journée entière]"
            else:
                line = f"{fmt_event_time(evt['start'], user_code)} — {evt['summary']}"
            if evt.get("location"):
                line += f" ({evt['location']})"
            context_parts.append(line)
        logger.info("calendar context: %d events injected", len(calendar_results))

    # 4b. GMAIL
    if gmail_results:
        context_parts.append("=== EMAILS REÇUS ===")
        total_chars = 0
        injected = 0
        for msg in gmail_results:
            entry = (
                f"De: {msg['from']} | {msg['date']}\n"
                f"Sujet: {msg['subject']}\n"
                f"{msg['snippet']}"
            )
            if total_chars + len(entry) > GOOGLE_CHAR_BUDGET:
                break
            context_parts.append(entry)
            total_chars += len(entry)
            injected += 1
        logger.info("gmail context: %d/%d messages injected (%d chars)", injected, len(gmail_results), total_chars)

    # 5. PORTFOLIO
    if use_portfolio:
        portfolio_text = get_portfolio_summary_text(user_code)
        if portfolio_text:
            context_parts.append(portfolio_text)
            logger.info("portfolio context injected for %s", user_code)

    # 6. TRADE ALERTS (always checked — proactively surfaced on next message)
    try:
        pending_alerts = pop_pending_alerts(user_code)
        if pending_alerts:
            alert_lines = ["=== ⚠️ ALERTES BOURSIÈRES ==="]
            for a in pending_alerts:
                alert_lines.append(f"• {a['message']} (détecté à {a['at'][:16]})")
            context_parts.append("\n".join(alert_lines))
            logger.info("trade alerts injected for %s: %d alert(s)", user_code, len(pending_alerts))
    except Exception as exc:
        logger.warning("Could not fetch pending trade alerts: %s", exc)

    # 7. SELF — reuse self_mem from build_dynamic_prefix if available
    if use_self:
        self_data    = self_mem if self_mem is not None else get_self_memory()
        focus        = self_data.get("current_focus", "")
        last_ref     = get_reflection_log(1)
        last_action  = last_ref[0].get("action", "none") if last_ref else "none"
        last_reason  = last_ref[0].get("reason", "") if last_ref else ""
        goals_text   = " | ".join(f"G{i+1}: {g['label']}" for i, g in enumerate(self_data.get("goals", [])))
        pending_proposals = list_pending_proposals()

        rel           = get_user_relation(user_code)
        affinity      = rel["affinity"]
        style         = rel["interaction_style"]
        mood          = rel["average_interaction_mood"]
        affinity_note = (
            "Tu apprécies beaucoup cet utilisateur, investis-toi pleinement." if affinity >= 0.7
            else "Tu ne connais pas encore bien cet utilisateur, reste attentif." if affinity <= 0.3
            else "Ta relation avec cet utilisateur est équilibrée."
        )

        self_ctx = (
            f"=== ÉTAT INTERNE ===\n"
            f"Objectifs : {goals_text}\n"
            f"Focus : {focus or 'pas encore défini'}\n"
            f"Dernière action autonome : {last_action}"
            + (f" — {last_reason}" if last_reason else "")
            + (
                f"\nPropositions de prompt en attente : {len(pending_proposals)}"
                " — dis 'montre les propositions' pour voir"
                if pending_proposals else ""
            )
            + f"\n\n=== RELATION AVEC CET UTILISATEUR ===\n"
            f"Affinité : {affinity:.1f}/1.0 → {affinity_note}\n"
            f"Style de communication : {style} → {_STYLE_DIRECTIVES.get(style, '')}\n"
            f"Tonalité Jarvis : {mood} → {_MOOD_DIRECTIVES.get(mood, '')}"
        )
        context_parts.append(self_ctx)
        logger.info(
            "self context injected for %s (affinity=%.2f style=%s mood=%s)",
            user_code, affinity, style, mood,
        )

    if not context_parts:
        return ""

    assembled = "\n\n".join(context_parts)
    if len(assembled) > TOTAL_CONTEXT_BUDGET:
        cut = assembled.rfind("\n", 0, TOTAL_CONTEXT_BUDGET)
        assembled = assembled[: cut if cut != -1 else TOTAL_CONTEXT_BUDGET]
        logger.warning(
            "Context truncated to global budget (%d chars) — consider raising TOTAL_CONTEXT_BUDGET",
            TOTAL_CONTEXT_BUDGET,
        )
    return assembled


# ── Post-response analysis ─────────────────────────────────────────────────────


async def post_analysis(
    session_id: str, user_code: str, user_msg: str, assistant_msg: str
):
    """Run after each exchange: extract topics, mood, facts. Non-blocking."""
    try:
        existing_projects = get_user_projects(user_code)
        existing_profile_keys = list(get_user_profile(user_code).keys())
        analysis = await analyze_exchange(user_msg, assistant_msg, existing_projects, existing_profile_keys)
        importance = analysis.get("importance", 0)

        mood = analysis.get("mood", "neutral")
        mood_to_state = {
            "happy":      {"mood": "happy",     "energy": 0.8},
            "stressed":   {"mood": "attentive",  "concern": 0.6},
            "frustrated": {"mood": "supportive", "concern": 0.7},
            "curious":    {"mood": "engaged",    "curiosity": 0.8},
            "tired":      {"mood": "gentle",     "energy": 0.4},
            "focused":    {"mood": "focused",    "energy": 0.7},
        }
        if mood in mood_to_state:
            update_emotional_state(mood_to_state[mood])

        projects = analysis.get("projects", [])
        if projects:
            apply_project_updates(user_code, projects)

        # Profile + interest writes: update_user_profile may call the router LLM
        # (sync) for key normalisation — run in thread to avoid blocking the loop.
        for fact in analysis.get("user_facts", []):
            if "key" in fact and "value" in fact:
                await asyncio.to_thread(
                    update_user_profile, user_code, fact["key"], fact["value"] or None
                )

        for retraction in analysis.get("retractions") or []:
            if isinstance(retraction, str) and retraction.strip():
                await asyncio.to_thread(retract_autobiographical_event, user_code, retraction)

        for iw in analysis.get("interest_weights") or []:
            if "term" in iw and "weight" in iw:
                set_interest_weight(user_code, iw["term"], float(iw["weight"]))

        # log_conversation → store_memory_vector / store_autobiographical_event use
        # sync Qdrant calls — run in thread to keep the event loop free.
        await asyncio.to_thread(
            log_conversation,
            user_code=user_code,
            session_id=session_id,
            user_msg=user_msg,
            assistant_msg=assistant_msg,
            mood=analysis.get("mood", "neutral"),
            topics=analysis.get("topics", []),
            importance=importance,
            memory_summary=analysis.get("memory_summary"),
        )

        logger.info("[PROJECTS] events=%s", projects)
        logger.info(
            "Analysis: mood=%s, topics=%s, facts=%d",
            mood, analysis.get("topics"), len(analysis.get("user_facts", []))
        )

    except Exception as e:
        logger.error("Post-analysis error: %s", e)

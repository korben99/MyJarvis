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
each user message as context.  The final user message structure is:
  [dynamic_prefix] → [assembled context] → [user question]
Question LAST = most salient for Qwen3 generation; "Contexte : ..." wrapper
prevents the model from outputting structured data (JSON) when context
contains financial or tabular information.
"""

import asyncio

from config import USER_TIMEZONES
from deps import (
    GOOGLE_CHAR_BUDGET,
    MEMORY_CHAR_BUDGET,
    RAG_CHAR_BUDGET,
    TOTAL_CONTEXT_BUDGET,
    WEB_CHAR_BUDGET,
)
from helpers import fmt_event_time, fmt_now_fr, get_logger, keyword_overlap_score, rel_time_fr
from llm_client import trim_chunks
from memory import (
    build_memory_context,
    log_conversation,
)
from prompts import get_prompt
from self import (
    get_reflection_log,
    get_self_memory,
    list_pending_proposals,
)
from trading import get_portfolio_summary_text, pop_pending_alerts
from web_search import INTERNET_ERROR

logger = get_logger("jarvis-pipeline")


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
    include_opinions: bool = True,
    include_suggestions: bool = True,
    user_message: str = "",
) -> tuple[str, dict]:
    """
    Build the per-turn context block prepended to the current user message.
    Contains: date/time, user name, profile/memory context, opinions, voice hint.
    Runs in a thread (I/O: Redis reads via build_memory_context / get_self_memory).

    Returns (prefix_string, self_mem) so the caller can pass self_mem to
    build_context() without triggering a second get_self_memory() Redis call.

    include_opinions     — False for pure utility intents (weather/calendar/gmail/portfolio)
                           to avoid ~200 tokens of irrelevant opinion context.
    include_suggestions  — False for the same utility intents (tomorrow_suggestions).
    """
    tz = USER_TIMEZONES.get(user_code, "Europe/Paris")
    parts: list[str] = [f"Date : {fmt_now_fr(tz)}."]

    if user_name:
        parts.append(f"Tu parles avec {user_name}. Tutoie-le toujours, quelle que soit la langue du contexte injecté.")

    self_mem = get_self_memory()

    memory_ctx = build_memory_context(
        session_id,
        user_code,
        self_mem=self_mem,
        include_suggestions=include_suggestions,
    )
    if memory_ctx:
        parts.append(f"<context>\n{memory_ctx}\n</context>")

    if include_opinions:
        opinions = self_mem.get("opinions", [])
        if opinions:
            if user_message:
                opinions = sorted(
                    opinions,
                    key=lambda o: keyword_overlap_score(o["topic"], user_message),
                    reverse=True,
                )[:5]
            else:
                opinions = opinions[-5:]
            ops_lines = "\n".join(f"- {o['topic']} : {o['opinion']}" for o in opinions)
            parts.append(f"<avis_jarvis>\n{ops_lines}\n</avis_jarvis>")

    if voice_mode:
        parts.append(get_prompt("VOICE_SUFFIX_FR").strip())

    return "\n\n".join(parts), self_mem


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
    # Web section built separately — appended LAST so it has highest budget priority
    # (dropped last on overflow) and highest LLM salience (read closest to the question).
    # Internet error is still prepended (positional doesn't matter for error messaging).
    _web_section: str = ""

    # 1. WEB
    if web_results == INTERNET_ERROR:
        _web_section = (
            "<web_access>\n"
            "La connexion internet est actuellement indisponible. "
            "Informe l'utilisateur que tu ne peux pas effectuer la recherche demandée "
            "et propose-lui de réessayer plus tard.\n"
            "</web_access>"
        )
        logger.warning("web: internet unavailable — injecting error context")
    elif web_results:
        web_selected = trim_chunks(web_results, WEB_CHAR_BUDGET, text_key="body", max_item_chars=3000)
        if web_selected:
            web_lines = []
            for i, body in enumerate(web_selected):
                r = web_results[i]
                date_tag = f"[{r['date']}] " if r.get("date") else ""
                web_lines.append(f"[{date_tag}{r['title']}]\n{body}\nSource: {r['url']}")
            _web_section = "<web_results>\n" + "\n\n".join(web_lines) + "\n</web_results>"
        logger.info(
            "web recall %d/%d (budget=%d)",
            len(web_selected),
            len(web_results),
            WEB_CHAR_BUDGET,
        )

    # 2. RAG
    if rag_chunks:
        rag_selected_texts = trim_chunks(rag_chunks, RAG_CHAR_BUDGET)
        if rag_selected_texts:
            rag_lines = [
                f"[Doc {chunk['source']} ({chunk['score']:.2f})]\n{chunk['text'][:800]}"
                for chunk in rag_chunks[:len(rag_selected_texts)]
            ]
            context_parts.append(
                "<documents>\n" + "\n\n".join(rag_lines) + "\n</documents>"
            )
        logger.info(
            "rag recall %d/%d (budget=%d)",
            len(rag_selected_texts),
            len(rag_chunks),
            RAG_CHAR_BUDGET,
        )

    # 3. MEMORY
    # Build timestamped copies — do NOT mutate the caller's list.
    if memory_chunks:
        stamped = [
            {**m, "text": f"({rel_time_fr(m['timestamp'])}) {m['text']}"}
            if m.get("timestamp")
            else m
            for m in memory_chunks
        ]
        # Deduplicate by normalized text — Qdrant can return the same vector twice
        # when the query matches multiple facets of the same stored sentence.
        seen_texts: set[str] = set()
        deduped: list = []
        for m in stamped:
            key = m["text"].strip().lower()[:120]
            if key not in seen_texts:
                seen_texts.add(key)
                deduped.append(m)
        before = len(stamped)
        stamped = deduped
        if before != len(stamped):
            logger.debug("memory dedup: %d → %d chunks", before, len(stamped))

        selected_memories = trim_chunks(stamped, MEMORY_CHAR_BUDGET)
        if selected_memories:
            context_parts.append(
                "<user_memories>\n" + "\n".join(selected_memories) + "\n</user_memories>"
            )
        logger.info(
            "memory recall %d/%d (budget=%d)",
            len(selected_memories),
            len(memory_chunks),
            MEMORY_CHAR_BUDGET,
        )

    # 4a. CALENDAR
    if calendar_results:
        cal_lines = []
        for evt in calendar_results:
            if evt.get("all_day"):
                line = f"{evt['start']} — {evt['summary']} [journée entière]"
            else:
                line = f"{fmt_event_time(evt['start'], user_code)} — {evt['summary']}"
            if evt.get("location"):
                line += f" ({evt['location']})"
            cal_lines.append(line)
        context_parts.append("<agenda>\n" + "\n".join(cal_lines) + "\n</agenda>")
        logger.info("calendar context: %d events injected", len(calendar_results))

    # 4b. GMAIL
    if gmail_results:
        gmail_lines = []
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
            gmail_lines.append(entry)
            total_chars += len(entry)
            injected += 1
        if gmail_lines:
            context_parts.append(
                "<emails>\n" + "\n\n".join(gmail_lines) + "\n</emails>"
            )
        logger.info(
            "gmail context: %d/%d messages injected (%d chars)",
            injected,
            len(gmail_results),
            total_chars,
        )

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
            alert_lines = [f"• {a['message']} (détecté à {a['at'][:16]})" for a in pending_alerts]
            context_parts.append(
                "<market_alerts>\n" + "\n".join(alert_lines) + "\n</market_alerts>"
            )
            logger.info(
                "trade alerts injected for %s: %d alert(s)",
                user_code,
                len(pending_alerts),
            )
    except Exception as exc:
        logger.warning("Could not fetch pending trade alerts: %s", exc)

    # 7. SELF — reuse self_mem from build_dynamic_prefix if available
    if use_self:
        self_data = self_mem if self_mem is not None else get_self_memory()
        focus = self_data.get("current_focus", "")
        last_ref = get_reflection_log(1)
        last_action = last_ref[0].get("action", "none") if last_ref else "none"
        last_reason = last_ref[0].get("reason", "") if last_ref else ""
        goals_text = " | ".join(
            f"G{i + 1}: {g['label']}" for i, g in enumerate(self_data.get("goals", []))
        )
        pending_proposals = list_pending_proposals()

        # RELATION already injected in build_dynamic_prefix prefix — skip here
        # to avoid double injection (~80 tokens) when use_self=True.
        self_ctx = (
            f"<internal_state>\n"
            f"Objectifs : {goals_text}\n"
            f"Focus : {focus or 'pas encore défini'}\n"
            f"Dernière action autonome : {last_action}"
            + (f" — {last_reason}" if last_reason else "")
            + (
                f"\nPropositions de prompt en attente : {len(pending_proposals)}"
                " — dis 'montre les propositions' pour voir"
                if pending_proposals
                else ""
            )
            + "\n</internal_state>"
        )
        context_parts.append(self_ctx)
        logger.info("self context injected for %s", user_code)

    # Append web section last — highest priority: survives budget overflow and is
    # read closest to the user question (maximises LLM salience for web queries).
    if _web_section:
        context_parts.append(_web_section)

    if not context_parts:
        return ""

    assembled = "\n\n".join(context_parts)
    if len(assembled) > TOTAL_CONTEXT_BUDGET:
        # Drop complete sections from the front (lowest priority first: memory →
        # RAG → calendar → … → web) instead of slicing at a char boundary — a raw
        # cut leaves unclosed XML tags that confuse Qwen3.6's context parsing.
        _original_count = len(context_parts)
        while len(context_parts) > 1 and len("\n\n".join(context_parts)) > TOTAL_CONTEXT_BUDGET:
            context_parts.pop(0)
        assembled = "\n\n".join(context_parts)
        logger.warning(
            "Context over global budget (%d chars): dropped %d section(s) — consider raising TOTAL_CONTEXT_BUDGET",
            TOTAL_CONTEXT_BUDGET,
            _original_count - len(context_parts),
        )
    return assembled


# ── Post-response analysis ─────────────────────────────────────────────────────


async def post_analysis(
    session_id: str, user_code: str, user_msg: str, assistant_msg: str
) -> None:
    """Log the exchange immediately after each response. Non-blocking, no LLM.

    LLM analysis (fact extraction, importance scoring, Qdrant vectorisation)
    is handled by the scheduled job analyse_recent_conversations() every 30 min.
    This keeps post_analysis at ~5 ms and completely eliminates _infer_lock
    contention on the next user request.
    """
    try:
        await asyncio.to_thread(
            log_conversation,
            user_code=user_code,
            session_id=session_id,
            user_msg=user_msg,
            assistant_msg=assistant_msg,
            # mood/topics/importance left at defaults — filled by scheduled analysis
        )
    except Exception as e:
        logger.error("post_analysis error: %s", e)

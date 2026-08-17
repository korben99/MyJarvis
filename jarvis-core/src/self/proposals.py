"""Autocoding — propositions d'amélioration de prompts.

Persistance (prompt_proposals.json / prompt_overrides.json), cycle de vie
(list/approve/reject), notification email avec diff, action refine_prompt, et
commandes chat de gestion (accepte/rejette/montre la proposition …).
"""

import difflib
import html as _html
import json
import os
import re
import uuid
from datetime import datetime, timezone

from config import (
    DEFAULT_TEMP,
    MAX_TOKENS_REASONING,
    PROMPT_DATA_DIR,
    REASONING_API_KEY,
    REASONING_API_URL,
    REASONING_MODEL,
    THINKING_BUDGET_DEEP,
    USER_ADMINS,
    USER_CODES,
    USER_EMAILS,
    llm_timeout,
)
from google_services import is_google_available, send_gmail_message
from helpers import call_llm, extract_llm_json, get_logger, get_redis
from memory import atomic_json_write
from prompts import PROMPT_TOKEN_BUDGETS, get_prompt

from .state import _GAP_COUNTS_KEY, _KNOWLEDGE_GAPS_KEY

logger = get_logger("jarvis-self")


def _proposals_path() -> str:
    return os.path.join(PROMPT_DATA_DIR, "prompt_proposals.json")


def _overrides_path() -> str:
    return os.path.join(PROMPT_DATA_DIR, "prompt_overrides.json")


def _load_proposals() -> list[dict]:
    try:
        with open(_proposals_path(), encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_proposals(proposals: list) -> None:
    atomic_json_write(_proposals_path(), proposals)


def _load_overrides() -> dict:
    try:
        with open(_overrides_path(), encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_overrides(overrides: dict) -> None:
    atomic_json_write(_overrides_path(), overrides)


def list_pending_proposals() -> list[dict]:
    """Return all proposals with status='pending'."""
    return [p for p in _load_proposals() if p.get("status") == "pending"]


def approve_proposal(proposal_id: str) -> str:
    """Apply the proposed text to prompt_overrides.json and mark as approved."""
    proposals = _load_proposals()
    found = next((p for p in proposals if p["id"] == proposal_id), None)
    if not found:
        return f"Proposition `{proposal_id}` introuvable."
    if found["status"] != "pending":
        return f"Proposition `{proposal_id}` est déjà **{found['status']}**."

    # Write override
    overrides = _load_overrides()
    overrides[found["prompt_name"]] = found["proposed_text"]
    _save_overrides(overrides)

    # Mark approved
    found["status"] = "approved"
    found["approved_at"] = datetime.now(timezone.utc).isoformat()
    _save_proposals(proposals)

    # Full knowledge-gap reset for this topic:
    # 1. counter hash   — so refine_prompt threshold is not immediately re-crossed
    # 2. sorted set     — remove all entries for this topic so it no longer appears in LACUNES
    # 3. cooldown key   — prevent re-flagging for 30 days after approval
    topic_slug = re.sub(r"\s+", "_", found.get("topic", "").lower())[:40]
    if topic_slug:
        r = get_redis()
        # hdel is unconditional — never swallowed by a broad except.
        # A failed hdel would leave count intact and allow immediate re-crossing of threshold.
        r.hdel(_GAP_COUNTS_KEY, topic_slug)
        r.setex(f"jarvis:self:gap_cooldown:{topic_slug}", 30 * 86400, "1")
        # Remove sorted-set entries one by one; skip malformed JSON silently so
        # a single corrupt entry doesn't abort the whole cleanup.
        try:
            all_entries = r.zrange(_KNOWLEDGE_GAPS_KEY, 0, -1)
        except Exception as exc:
            logger.warning("gap cleanup: could not read knowledge_gaps set: %s", exc)
            all_entries = []
        for e in all_entries:
            try:
                e_slug = re.sub(r"\s+", "_", json.loads(e).get("topic", "").lower())[
                    :40
                ]
                if e_slug == topic_slug:
                    r.zrem(_KNOWLEDGE_GAPS_KEY, e)
            except Exception:
                # Malformed entry — zrem by raw value as fallback
                try:
                    r.zrem(_KNOWLEDGE_GAPS_KEY, e)
                except Exception:
                    pass

    # Invalidate prompts in-memory cache so the new text is returned immediately
    # (clears both the mtime sentinel and the cached dict — belt & suspenders)
    try:
        import prompts as _pm

        _pm._override_mtime = -1.0
        _pm._override_cache = {}
    except Exception:
        pass

    logger.info("Proposal %s approved: %s updated", proposal_id, found["prompt_name"])
    return (
        f"✓ Proposition `{proposal_id}` approuvée.\n"
        f"Le prompt **{found['prompt_name']}** est maintenant actif — aucun redémarrage nécessaire."
    )


def reject_proposal(proposal_id: str) -> str:
    """Mark a proposal as rejected."""
    proposals = _load_proposals()
    found = next((p for p in proposals if p["id"] == proposal_id), None)
    if not found:
        return f"Proposition `{proposal_id}` introuvable."
    if found["status"] != "pending":
        return f"Proposition `{proposal_id}` est déjà **{found['status']}**."

    found["status"] = "rejected"
    found["rejected_at"] = datetime.now(timezone.utc).isoformat()
    _save_proposals(proposals)

    logger.info("Proposal %s rejected", proposal_id)
    return f"✗ Proposition `{proposal_id}` rejetée."


def _notify_proposal(user_code: str, proposal: dict) -> None:
    """Send an email notification with the proposal diff."""
    to = USER_EMAILS.get(user_code, "")
    if not to or not is_google_available(user_code):
        logger.warning(
            "_notify_proposal: skipped for %s (no email or Google unavailable)", user_code
        )
        return

    try:
        _send_proposal_email(user_code, to, proposal)
    except Exception as exc:
        logger.error("_notify_proposal: unexpected error for %s: %s", user_code, exc)


def _send_proposal_email(user_code: str, to: str, proposal: dict) -> None:
    """Inner send — separated so _notify_proposal can wrap it in try/except."""
    pid = proposal["id"]
    name = proposal["prompt_name"]
    rationale = proposal["rationale"]
    current_text = proposal["current_text"]
    proposed_text = proposal["proposed_text"]

    # ── Unified diff (plain text) ──────────────────────────────────────────
    diff_lines = list(
        difflib.unified_diff(
            current_text.splitlines(),
            proposed_text.splitlines(),
            fromfile="actuel",
            tofile="proposé",
            lineterm="",
            n=5,
        )
    )
    diff_plain = "\n".join(diff_lines) if diff_lines else "(aucune différence détectée)"

    # ── Unified diff (HTML colorisé) ───────────────────────────────────────
    def _colorize_diff_html(lines: list[str]) -> str:
        parts = []
        for line in lines:
            escaped = _html.escape(line)
            if line.startswith("+++") or line.startswith("---"):
                parts.append(
                    f"<span style='color:#555;font-weight:bold'>{escaped}</span>"
                )
            elif line.startswith("+"):
                parts.append(
                    f"<span style='background:#d4edda;color:#155724'>{escaped}</span>"
                )
            elif line.startswith("-"):
                parts.append(
                    f"<span style='background:#f8d7da;color:#721c24'>{escaped}</span>"
                )
            elif line.startswith("@@"):
                parts.append(
                    f"<span style='color:#0d6efd;font-weight:bold'>{escaped}</span>"
                )
            else:
                parts.append(escaped)
        return "\n".join(parts)

    diff_html = (
        _colorize_diff_html(diff_lines)
        if diff_lines
        else "<em>(aucune différence détectée)</em>"
    )

    text = (
        f"Jarvis a identifié une opportunité d'amélioration du prompt « {name} ».\n\n"
        f"Raison : {rationale}\n\n"
        f"── DIFFÉRENCES ──\n{diff_plain}\n\n"
        f"── TEXTE ACTUEL (complet) ──\n{current_text}\n\n"
        f"── TEXTE PROPOSÉ (complet) ──\n{proposed_text}\n\n"
        f"Pour approuver : dis à Jarvis « accepte la proposition {pid} »\n"
        f"Pour rejeter  : dis à Jarvis « rejette la proposition {pid} »"
    )
    html = (
        f"<p>Jarvis a identifié une opportunité d'amélioration du prompt <strong>{name}</strong>.</p>"
        f"<p><strong>Raison :</strong> {_html.escape(rationale)}</p>"
        f"<h3>Différences</h3>"
        f"<pre style='background:#f8f9fa;padding:10px;font-size:12px;white-space:pre-wrap;border:1px solid #dee2e6;border-radius:4px'>{diff_html}</pre>"
        f"<h3>Texte actuel</h3>"
        f"<pre style='background:#f5f5f5;padding:10px;font-size:12px;white-space:pre-wrap'>{_html.escape(current_text)}</pre>"
        f"<h3>Texte proposé</h3>"
        f"<pre style='background:#e8f5e9;padding:10px;font-size:12px;white-space:pre-wrap'>{_html.escape(proposed_text)}</pre>"
        f"<p>Pour approuver : dis à Jarvis <strong>« accepte la proposition {pid} »</strong><br>"
        f"Pour rejeter : dis à Jarvis <strong>« rejette la proposition {pid} »</strong></p>"
        f"<p><em>— Jarvis</em></p>"
    )
    success = send_gmail_message(
        to=to,
        subject=f"Jarvis — Proposition de prompt #{pid} ({name})",
        html_body=html,
        text_body=text,
        user_code=user_code,
    )
    if not success:
        logger.warning(
            "_notify_proposal: email not sent for %s (Gmail unavailable?)", user_code
        )


def _action_refine_prompt(params: dict) -> str:
    """
    Call the reasoning model to propose an improved version of a prompt.
    Stores the proposal in prompt_proposals.json and notifies by email.
    Runs synchronously (called via asyncio.to_thread from run_reflection).
    """
    prompt_name = params.get("prompt_name", "").strip()
    topic = params.get("topic", "").strip()
    context_str = params.get("context", "").strip()
    user_code = params.get("user_code", "").strip()

    if not prompt_name or not topic:
        return "refine_prompt: missing prompt_name or topic"
    if user_code and user_code not in USER_CODES:
        return f"refine_prompt: unknown user_code {user_code!r}"

    current_text = get_prompt(prompt_name)
    if not current_text:
        return f"refine_prompt: unknown prompt {prompt_name!r}"

    # Guard: no duplicate pending proposal for the same prompt (data integrity, not a cooldown)
    existing = [p for p in list_pending_proposals() if p["prompt_name"] == prompt_name]
    if existing:
        return f"refine_prompt: proposal already pending for {prompt_name} (id={existing[0]['id']})"

    max_budget = PROMPT_TOKEN_BUDGETS.get(prompt_name, 600)
    current_token_count = len(current_text) // 4  # approximation : 1 token ≈ 4 chars

    refine_prompt_text = get_prompt("REFINE_PROMPT_USER").format(
        prompt_name=prompt_name,
        topic=topic,
        context=context_str or "aucun contexte supplémentaire",
        current_text=current_text[:6000],
        current_token_count=current_token_count,
        max_token_budget=max_budget,
    )

    try:
        content = call_llm(
            [
                {"role": "system", "content": get_prompt("REFINE_PROMPT_SYSTEM")},
                {"role": "user", "content": refine_prompt_text},
            ],
            model=REASONING_MODEL,
            api_url=REASONING_API_URL,
            api_key=REASONING_API_KEY,
            temperature=DEFAULT_TEMP,
            max_tokens=MAX_TOKENS_REASONING,
            thinking_budget=THINKING_BUDGET_DEEP,
            json_response=True,
            no_think=False,
            timeout=llm_timeout(MAX_TOKENS_REASONING),
        )
        result = extract_llm_json(content)
    except Exception as exc:
        logger.error("refine_prompt: LLM call failed: %s", exc)
        return f"refine_prompt: LLM call failed ({type(exc).__name__})"

    raw_proposed = result.get("proposed_text")
    rationale = result.get("rationale", "").strip()

    # LLM explicitly decided no change is needed — not an error
    if raw_proposed is None:
        logger.info(
            "refine_prompt: no modification needed for %s — %s", prompt_name, rationale
        )
        return f"refine_prompt: no modification needed for {prompt_name} ({rationale})"

    proposed_text = raw_proposed.strip() if isinstance(raw_proposed, str) else ""
    if not proposed_text:
        return "refine_prompt: LLM returned empty proposed_text"

    # Guard: format-string safety — detect unescaped JSON braces in proposed_text.
    # JSON literals like {"key":"..."} must be escaped as {{"key":"..."}} in format templates.
    # An unescaped {word} that isn't a known placeholder would crash str.format() with KeyError.
    _original_placeholders = set(re.findall(r"\{(\w+)\}", current_text))
    _proposed_new = (
        set(re.findall(r"\{(\w+)\}", proposed_text)) - _original_placeholders
    )
    if _proposed_new:
        logger.warning(
            "refine_prompt: proposed text for %s contains unescaped braces: %s — rejecting",
            prompt_name,
            _proposed_new,
        )
        return (
            f"refine_prompt: proposed text contains unescaped brace placeholders {_proposed_new} "
            f"that would break str.format(). JSON object literals must use {{{{ }}}} escaping. "
            f"Proposal discarded."
        )

    # Guard: reject if proposed text exceeds the token budget — retry once with explicit feedback
    proposed_token_count = len(proposed_text) // 4
    if proposed_token_count > max_budget:
        logger.warning(
            "refine_prompt: proposed text for %s is ~%d tokens (budget=%d) — retrying with feedback",
            prompt_name,
            proposed_token_count,
            max_budget,
        )
        retry_messages = [
            {"role": "system", "content": get_prompt("REFINE_PROMPT_SYSTEM")},
            {"role": "user", "content": refine_prompt_text},
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": (
                    f"Ton proposed_text fait ~{proposed_token_count} tokens mais le budget maximum "
                    f"est {max_budget} tokens. Tu dois le raccourcir. "
                    f"Retourne uniquement le JSON avec le proposed_text raccourci."
                ),
            },
        ]
        try:
            content = call_llm(
                retry_messages,
                model=REASONING_MODEL,
                api_url=REASONING_API_URL,
                api_key=REASONING_API_KEY,
                temperature=DEFAULT_TEMP,
                max_tokens=MAX_TOKENS_REASONING,
                thinking_budget=THINKING_BUDGET_DEEP,
                json_response=True,
                no_think=False,
                timeout=llm_timeout(MAX_TOKENS_REASONING),
            )
            result = extract_llm_json(content)
            proposed_text = result.get("proposed_text", "").strip()
            rationale = result.get("rationale", rationale).strip()
        except Exception as exc:
            logger.error("refine_prompt: retry failed: %s", exc)
            return f"refine_prompt: proposed text too long and retry failed ({type(exc).__name__})"

        proposed_token_count = len(proposed_text) // 4
        if not proposed_text or proposed_token_count > max_budget:
            logger.warning(
                "refine_prompt: retry still too long (%d tokens) — proposal rejected",
                proposed_token_count,
            )
            return (
                f"refine_prompt: proposed text still too long after retry "
                f"(~{proposed_token_count} tokens, budget={max_budget}) — proposal rejected"
            )
        logger.info("refine_prompt: retry succeeded (%d tokens)", proposed_token_count)

    proposal = {
        "id": uuid.uuid4().hex[:8],
        "prompt_name": prompt_name,
        "topic": topic,
        "current_text": current_text,
        "proposed_text": proposed_text,
        "rationale": rationale,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    proposals = _load_proposals()
    proposals.append(proposal)
    _save_proposals(proposals)

    # Notify admins only — prompt changes are a system-level action
    for _code in USER_ADMINS:
        _notify_proposal(_code, proposal)

    logger.info(
        "refine_prompt: proposal %s created for %s (topic: %s)",
        proposal["id"],
        prompt_name,
        topic,
    )
    return f"proposal {proposal['id']} created for {prompt_name}"


def handle_proposal_command(message: str, user_code: str) -> str | None:
    """
    Detect and execute proposal management commands from a chat message.
    Returns a formatted response string, or None if the message is not a proposal command.
    Called by main.py before the full LLM pipeline when use_self=True.
    """
    msg = message.strip().lower()

    # ── List pending proposals ──
    if any(
        kw in msg
        for kw in (
            "montre les propositions",
            "liste les propositions",
            "propositions en attente",
            "show proposals",
            "list proposals",
            "quelles propositions",
        )
    ):
        proposals = list_pending_proposals()
        if not proposals:
            return "Aucune proposition de prompt en attente."
        lines = [f"**{len(proposals)} proposition(s) en attente :**\n"]
        for p in proposals:
            lines.append(
                f"- `{p['id']}` — **{p['prompt_name']}** : {p['rationale'][:100]}"
            )
        lines.append(
            "\nDis « accepte la proposition [id] » ou « rejette la proposition [id] »."
        )
        return "\n".join(lines)

    # ── Approve ──
    m = re.search(r"(accepte?|approu?ve?)\s+la\s+proposition\s+([a-f0-9]{6,8})\b", msg)
    if m:
        if user_code not in USER_ADMINS:
            return "⛔ Seul un administrateur peut approuver une proposition de prompt."
        return approve_proposal(m.group(2))

    # ── Approve sans ID ──
    if re.search(r"\b(accepte?|approu?ve?)\b", msg) and "proposition" in msg:
        proposals = list_pending_proposals()
        if not proposals:
            return "Aucune proposition de prompt en attente."
        lines = ["ID manquant. Propositions en attente :"]
        for p in proposals:
            lines.append(
                f"- `{p['id']}` — **{p['prompt_name']}** : {p['rationale'][:80]}"
            )
        lines.append("\nDis « accepte la proposition [id] ».")
        return "\n".join(lines)

    # ── Reject ──
    m = re.search(
        r"(rejette?|refu?se?|reject)\s+la\s+proposition\s+([a-f0-9]{6,8})\b", msg
    )
    if m:
        if user_code not in USER_ADMINS:
            return "⛔ Seul un administrateur peut rejeter une proposition de prompt."
        return reject_proposal(m.group(2))

    # ── Reject sans ID ──
    if re.search(r"\b(rejette?|refu?se?|reject)\b", msg) and "proposition" in msg:
        proposals = list_pending_proposals()
        if not proposals:
            return "Aucune proposition de prompt en attente."
        lines = ["ID manquant. Propositions en attente :"]
        for p in proposals:
            lines.append(
                f"- `{p['id']}` — **{p['prompt_name']}** : {p['rationale'][:80]}"
            )
        lines.append("\nDis « rejette la proposition [id] ».")
        return "\n".join(lines)

    # ── Show specific proposal ──
    m = re.search(
        r"(montre?|show|détail)\s+(la\s+proposition\s+)?([a-f0-9]{6,8})\b", msg
    )
    if m:
        pid = m.group(3)
        proposals = _load_proposals()
        found = next((p for p in proposals if p["id"] == pid), None)
        if not found:
            return f"Proposition `{pid}` introuvable."
        import difflib as _difflib

        cur = found["current_text"]
        prop = found["proposed_text"]
        diff_lines = list(
            _difflib.unified_diff(
                cur.splitlines(),
                prop.splitlines(),
                fromfile="actuel",
                tofile="proposé",
                lineterm="",
                n=3,
            )
        )
        diff_block = "\n".join(diff_lines) if diff_lines else "(aucune différence)"
        return (
            f"**Proposition `{pid}` — {found['prompt_name']}** ({found['status']})\n\n"
            f"**Raison :** {found['rationale']}\n\n"
            f"**Diff :**\n```diff\n{diff_block}\n```\n\n"
            f"**Texte actuel :**\n```\n{cur}\n```\n\n"
            f"**Texte proposé :**\n```\n{prop}\n```"
        )

    return None

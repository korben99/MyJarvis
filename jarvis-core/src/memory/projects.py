"""Projets & tâches de l'utilisateur (liste Redis JSON) : lecture, persistance avec
purge des projets terminés, résolution de nom (exacte + fuzzy), application des
événements structurés issus de l'analyzer.

Isolé du profil et de l'épisodique parce que ces sections s'appellent mutuellement
autour des projets — les regrouper ici casse le cycle d'import.
"""

import re
from datetime import datetime, timezone

from config import DONE_PROJECT_TTL_DAYS
from helpers import get_logger, redis_get_json, redis_set_json

logger = get_logger("jarvis-memory")


def get_user_projects(user_code: str) -> list:
    """Return the user's project list from Redis."""
    return redis_get_json(f"user:{user_code}:projects", default=[])


def update_user_projects(user_code: str, projects: list):
    """Persist the project list.
    Done projects older than DONE_PROJECT_TTL_DAYS are dropped.
    Schema: {name, status, first_mentioned, last_update, description?, updates?}
    updates: [{date: "YYYY-MM-DD", summary: "..."}], capped at 20 FIFO.
    """
    cutoff = datetime.now(timezone.utc).timestamp() - DONE_PROJECT_TTL_DAYS * 86400
    result = []
    for p in projects:
        if p.get("status") == "done" and p.get("last_update"):
            try:
                if datetime.fromisoformat(p["last_update"]).timestamp() < cutoff:
                    continue
            except (ValueError, TypeError):
                pass
        entry = {
            "name": p["name"],
            "status": p.get("status", "in_progress"),
            "first_mentioned": p.get("first_mentioned"),
            "last_update": p.get("last_update"),
        }
        if p.get("description"):
            entry["description"] = p["description"]
        if p.get("updates"):
            entry["updates"] = p["updates"][-20:]
        # due_at MUST stay in this whitelist: the entry is rebuilt field by field on every
        # write, so any field omitted here is silently dropped at the next nightly merge.
        if p.get("due_at"):
            entry["due_at"] = p["due_at"]
        result.append(entry)
    redis_set_json(f"user:{user_code}:projects", result)


def get_project_detail(user_code: str, project_name: str) -> dict | None:
    """Return the full project dict (with updates timeline) by fuzzy name match."""
    projects = get_user_projects(user_code)
    project_map = {p["name"]: p for p in projects}
    if project_name in project_map:
        return project_map[project_name]
    resolved = _fuzzy_project_name(project_name, project_map, threshold=0.4)
    return project_map.get(resolved) if resolved else None


def get_project_timeline_text(project: dict) -> str:
    """Format a project's timeline for prompt injection."""
    status_fr = {
        "done": "terminé",
        "in_progress": "en cours",
        "active": "en cours",
    }.get(project.get("status", ""), project.get("status", "en cours"))
    lines = [f"Projet : {project['name']} ({status_fr})"]
    updates = project.get("updates") or []
    if updates:
        lines.append("Historique :")
        for u in updates:
            d = u.get("date", "?")
            s = u.get("summary", "")
            if s:
                lines.append(f"  {d} : {s}")
    elif project.get("description"):
        lines.append(f"Description : {project['description']}")
    return "\n".join(lines)


def _normalize_project_name(name: str) -> str:
    """Normalize a project name to a consistent space-separated form.

    Replaces slug-style hyphens (between word chars) with spaces so that
    LLM-generated kebab-case names ("installation-attelage-bmw") and natural
    language names ("installation attelage bmw") are stored identically.
    Em-dashes (—) used as title separators are preserved.
    """
    normalized = re.sub(r"(?<=\w)-(?=\w)", " ", name)
    return re.sub(r" {2,}", " ", normalized).strip()


def _fuzzy_project_name(
    name: str, project_map: dict, threshold: float = 0.6
) -> str | None:
    """Return the best-matching project name from project_map, or None.

    Scoring uses word overlap with two metrics (max is taken):
    - General: overlap / max(|A|, |B|)  — Jaccard-like
    - Subset : overlap / min(|A|, |B|)  — catches subset names ("Jarvis" → "Jarvis v9")

    Tokens are split on spaces, hyphens, and em-dashes so that slugified names
    and natural-language names score identically.

    When two projects tie, in_progress beats done — avoids re-opening a closed
    project when an active one with similar words exists.
    """
    tokenize = lambda s: set(re.split(r"[\s\-—]+", s.lower()))
    words_new = tokenize(name)
    best_match: str | None = None
    best_score = 0.0
    for existing_name, existing_proj in project_map.items():
        words_ex = tokenize(existing_name)
        overlap = len(words_new & words_ex)
        if overlap == 0:
            continue
        general = overlap / max(len(words_new), len(words_ex))
        subset = overlap / min(len(words_new), len(words_ex))
        score = max(general, subset)
        if score < threshold:
            continue
        is_active = existing_proj.get("status") != "done"
        best_is_active = (
            project_map.get(best_match, {}).get("status") != "done"
            if best_match
            else False
        )
        if score > best_score or (
            score == best_score and is_active and not best_is_active
        ):
            best_score = score
            best_match = existing_name
    return best_match


def apply_project_updates(user_code: str, project_events: list[dict]):
    """Apply structured project events from the analyzer.

    Each event is a dict: {name, action, summary, [rename_to]}
      action  : "create" | "update" | "done" | "rename"
      summary : 1-sentence description of what happened (appended to updates timeline)
      rename_to : new name (rename only)
    """
    projects = get_user_projects(user_code)
    now = datetime.now(timezone.utc).isoformat()
    today = now[:10]  # YYYY-MM-DD

    project_map: dict[str, dict] = {p["name"]: p for p in projects}

    for event in project_events:
        if not isinstance(event, dict):
            continue
        action = event.get("action", "").strip()
        name = _normalize_project_name(event.get("name", "").strip())
        summary = (event.get("summary") or "").strip()
        due = (event.get("due") or "").strip()
        if not name or action not in ("create", "update", "done", "rename"):
            continue

        # Exact match first, then fuzzy — prevents name drift duplicates
        resolved = (
            name
            if name in project_map
            else (_fuzzy_project_name(name, project_map) or name)
        )

        if action == "create":
            if resolved not in project_map:
                soft_match = _fuzzy_project_name(name, project_map, threshold=0.4)
                if soft_match:
                    resolved = soft_match
                    project_map[resolved]["last_update"] = now
                    logger.debug(
                        "Project create: '%s' soft-matched to '%s' — skipping create",
                        name,
                        soft_match,
                    )
                else:
                    project_map[resolved] = {
                        "name": resolved,
                        "status": "in_progress",
                        "first_mentioned": now,
                        "last_update": now,
                        "updates": [],
                    }
            else:
                project_map[resolved]["last_update"] = now

        elif action == "update":
            if resolved not in project_map:
                project_map[resolved] = {
                    "name": resolved,
                    "status": "in_progress",
                    "first_mentioned": now,
                    "last_update": now,
                    "updates": [],
                }
            else:
                project_map[resolved]["last_update"] = now

        elif action == "done":
            if resolved in project_map:
                project_map[resolved]["status"] = "done"
                project_map[resolved]["last_update"] = now
            else:
                logger.warning(
                    "Project done: '%s' not found — possible LLM confabulation (raw='%s')",
                    resolved,
                    name,
                )

        elif action == "rename":
            old_raw = _normalize_project_name(event.get("name", "").strip())
            new_name = _normalize_project_name(event.get("rename_to", "").strip())
            if not new_name:
                continue
            old_resolved = (
                old_raw
                if old_raw in project_map
                else (_fuzzy_project_name(old_raw, project_map) or old_raw)
            )
            if old_resolved in project_map:
                entry = project_map.pop(old_resolved)
                entry["name"] = new_name
                entry["last_update"] = now
                project_map[new_name] = entry
            else:
                logger.warning(
                    "Project rename: '%s' not found — possible LLM confabulation (raw='%s', rename_to='%s')",
                    old_resolved,
                    old_raw,
                    new_name,
                )
            continue  # rename has no summary entry

        # Due date (tasks): absolute ISO date only. A relative date ("jeudi", "dans 2
        # semaines") would be resolved against the wrong day, so it is rejected rather
        # than guessed. Logged either way — this is the signal to watch when observing
        # whether the extractor dates tasks correctly.
        if due and action in ("create", "update") and resolved in project_map:
            try:
                datetime.fromisoformat(due)
            except (ValueError, TypeError):
                logger.warning(
                    "Task due date ignored (not absolute ISO): %s / '%s' → %r",
                    user_code, resolved, due,
                )
            else:
                project_map[resolved]["due_at"] = due
                logger.info(
                    "Task due date set: %s / '%s' → %s", user_code, resolved, due
                )

        # Append summary to updates timeline (all actions except rename)
        if summary and resolved in project_map:
            proj = project_map[resolved]
            proj.setdefault("updates", [])
            proj["updates"].append({"date": today, "summary": summary})
            proj["description"] = summary  # compact context always shows latest

    update_user_projects(user_code, list(project_map.values()))

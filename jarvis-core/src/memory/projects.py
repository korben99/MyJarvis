"""Projets & tâches de l'utilisateur (liste Redis JSON) : lecture, persistance avec
purge des projets terminés, résolution de nom (exacte, sinon arbitrée par le modèle
principal), application des
événements structurés issus de l'analyzer.

Isolé du profil et de l'épisodique parce que ces sections s'appellent mutuellement
autour des projets — les regrouper ici casse le cycle d'import.
"""

import difflib
import re
from datetime import datetime, timezone

from config import (
    DONE_PROJECT_TTL_DAYS,
    MAX_TOKENS_TINY,
    PRIMARY_API_KEY,
    PRIMARY_API_URL,
    PRIMARY_MODEL,
    STALE_PROJECT_DAYS,
    llm_timeout,
)
from helpers import (
    _FR_STOPWORDS,
    call_llm_bg,
    extract_llm_json,
    get_logger,
    normalize_key,
    redis_get_json,
    redis_set_json,
    rel_time_fr,
)
from prompts import PROJECT_ARBITRATION_LABELS, get_prompt

logger = get_logger("jarvis-memory")

# Seuil de similarité caractère pour rapprocher deux noms sans token commun (variante de
# casse, faute de frappe, pluriel). Volontairement strict : un rapprochement trop lâche
# fusionne des projets distincts, et un rattachement abusif perd la tâche en silence, alors
# qu'un doublon se voit et se corrige.
_SEUIL_SIMILARITE_CARACTERE = 0.85

_MAX_CANDIDATS_ARBITRAGE = 3

_MOTS_VIDES = {normalize_key(m) for m in _FR_STOPWORDS} | {
    "the", "and", "for", "with", "from", "into",
}


def get_user_projects(user_code: str) -> list:
    """Return the user's project list from Redis."""
    return redis_get_json(f"user:{user_code}:projects", default=[])


def projets_actifs(projects: list | None) -> list:
    """Les projets non clôturés. Filtre unique, partagé par tous les points d'injection.

    `analyzer` en a un autre, délibérément : il conserve aussi les projets clos depuis
    moins de 90 jours, pour que le modèle évite d'en recréer un déjà terminé.
    """
    return [
        p for p in (projects or []) if isinstance(p, dict) and p.get("status") != "done"
    ]


def projets_clos(projects: list | None) -> list:
    """Les projets clôturés, du plus récemment terminé au plus ancien.

    `update_user_projects` purge au-delà de DONE_PROJECT_TTL_DAYS : figurer ici signifie
    donc « clos récemment ». L'ordre compte parce que la liste sert à dater des clés de
    profil, et qu'une clôture ancienne n'en explique plus aucune.
    """
    clos = [
        p for p in (projects or []) if isinstance(p, dict) and p.get("status") == "done"
    ]
    return sorted(clos, key=lambda p: p.get("last_update") or "", reverse=True)


def fmt_projets(projects: list | None) -> str:
    """Rendu d'une liste de projets pour les blocs injectés — conversation et réflexion.

    Tâches et projets partagent une liste : c'est l'échéance qui les distingue, donc le
    modèle lit une « échéance » ou n'en lit pas, rien à classer de lui-même.
    Le dernier mouvement est rendu parce qu'aucune purge ne vise les projets ouverts —
    seule leur ancienneté distingue un sujet vivant d'un sujet abandonné, et la clôture
    appartient à l'utilisateur.
    """
    lignes = []
    for p in projects or []:
        ligne = f"- {p.get('name', 'sans nom')}"
        if p.get("due_at"):
            ligne += f" (échéance : {p['due_at'][:10]})"
        try:
            horodatage = datetime.fromisoformat(p["last_update"]).timestamp()
            ligne += f" [{rel_time_fr(horodatage)}]"
        except (KeyError, TypeError, ValueError):
            pass
        lignes.append(ligne)
    return "\n".join(lignes)


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


def cloturer_projets_dormants(user_code: str, seuil_jours: int = STALE_PROJECT_DAYS) -> int:
    """Clôt (status `done`) les projets in_progress sans mise à jour depuis `seuil_jours`.

    `last_update` est laissé tel quel, et non ramené à maintenant : la purge des projets
    clos (DONE_PROJECT_TTL_DAYS) doit compter à partir de la dernière activité réelle, pas
    de la clôture, sinon un projet oublié depuis des mois repartirait pour 180 jours. Une
    ligne d'update trace que la clôture est automatique et non confirmée par l'utilisateur.
    Rend le nombre de projets clos.
    """
    projects = get_user_projects(user_code)
    if not projects:
        return 0
    cutoff = datetime.now(timezone.utc).timestamp() - seuil_jours * 86400
    today = datetime.now(timezone.utc).isoformat()[:10]
    n = 0
    for p in projects:
        if p.get("status") != "in_progress" or not p.get("last_update"):
            continue
        try:
            if datetime.fromisoformat(p["last_update"]).timestamp() >= cutoff:
                continue
        except (ValueError, TypeError):
            continue
        p["status"] = "done"
        p.setdefault("updates", []).append(
            {
                "date": today,
                "summary": f"Clôture automatique : sans activité depuis {seuil_jours} jours.",
            }
        )
        n += 1
    if n:
        update_user_projects(user_code, projects)
        logger.info("Projets dormants clôturés pour %s : %d", user_code, n)
    return n


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

    Replaces slug-style hyphens and underscores (between word chars) with spaces so that
    LLM-generated kebab-case and snake_case names ("installation-attelage-bmw",
    "installation_attelage_bmw") and natural language names ("installation attelage bmw")
    are stored identically. Em-dashes (—) used as title separators are preserved.

    Sans le souligné, un nom slugifié entre en base tel quel et ne se rapproche plus
    d'aucun autre : les jetons du découpage flou se font sur les espaces.
    """
    normalized = re.sub(r"(?<=\w)[-_](?=\w)", " ", name)
    return re.sub(r" {2,}", " ", normalized).strip()


def _score_nom_projet(name: str, existing_name: str) -> float:
    """Proximité de deux noms de projet, entre 0 et 1 (voir `_fuzzy_project_name`)."""
    tokenize = lambda s: set(re.split(r"[\s\-—]+", s.lower()))
    words_new = tokenize(name)
    words_ex = tokenize(existing_name)
    overlap = len(words_new & words_ex)
    # Le recouvrement de mots ne voit pas une variante À L'INTÉRIEUR d'un mot : deux
    # noms mono-mot comme « ClaimSentry » et « claimssentry » n'ont aucun token commun
    # et marquent 0. La similarité caractère rattrape ce cas, sous un seuil séparé et
    # strict pour ne pas fusionner des projets réellement distincts.
    char_ratio = difflib.SequenceMatcher(
        None, name.lower(), existing_name.lower()
    ).ratio()
    if overlap == 0 and char_ratio < _SEUIL_SIMILARITE_CARACTERE:
        return 0.0
    general = overlap / max(len(words_new), len(words_ex)) if overlap else 0.0
    subset = (
        1.0 if overlap and overlap == min(len(words_new), len(words_ex)) else 0.0
    )
    return max(
        general, subset, char_ratio if char_ratio >= _SEUIL_SIMILARITE_CARACTERE else 0.0
    )


def _fuzzy_project_name(
    name: str, project_map: dict, threshold: float = 0.6
) -> str | None:
    """Return the best-matching project name from project_map, or None.

    Scoring uses word overlap with two metrics (max is taken):
    - General: overlap / max(|A|, |B|)  — Jaccard-like
    - Subset : 1 when the shorter name is fully contained in the longer one, else 0
               ("Jarvis" → "Jarvis v9", "attelage bmw" → "installation attelage bmw")

    L'inclusion doit être COMPLÈTE : notée en proportion, deux noms de deux mots partageant
    un seul mot marquent 0,5, et « vente X » absorbe « vente Y » alors que rien ne les
    rapproche. Les noms de projets faisant deux à quatre mots, ce cas est le plus courant.

    Tokens are split on spaces, hyphens, and em-dashes so that slugified names
    and natural-language names score identically.

    When two projects tie, in_progress beats done — avoids re-opening a closed
    project when an active one with similar words exists.
    """
    best_match: str | None = None
    best_score = 0.0
    for existing_name, existing_proj in project_map.items():
        score = _score_nom_projet(name, existing_name)
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


def _mots_significatifs(nom: str) -> set[str]:
    return {
        m for m in normalize_key(nom).split("_")
        if len(m) >= 3 and m not in _MOTS_VIDES
    }


def _candidat_plausible(name: str, existing_name: str) -> bool:
    """Un mot porteur en commun, ou une variante orthographique du nom entier.

    Un mot vide partagé (« de », « pour ») ne suffit pas : presque tout nom nouveau en
    partage un avec un projet existant, et chaque candidat coûte un appel LLM.
    """
    if _mots_significatifs(name) & _mots_significatifs(existing_name):
        return True
    ratio = difflib.SequenceMatcher(None, name.lower(), existing_name.lower()).ratio()
    return ratio >= _SEUIL_SIMILARITE_CARACTERE


def _rattachement_sur(name: str, project_map: dict) -> str | None:
    """Le projet que `name` désigne sans ambiguïté possible, sans appel LLM.

    Deux cas seulement, chacun exigé UNIQUE :
    - même nom à la casse, aux accents et aux séparateurs près ;
    - un nom d'un seul mot, variante orthographique d'un autre nom d'un seul mot, aux
      mêmes nombres (« claimssentry » / « ClaimSentry »).
    La similarité caractère ne vaut pas preuve sur un nom de plusieurs mots : « vente bmw
    m4 » et « vente bmw m5 » dépassent le seuil en ne différant que par ce qui les
    distingue. Même raison pour les nombres, qui séparent les versions d'un projet.
    """
    cle = normalize_key(name)
    egaux = [nom for nom in project_map if normalize_key(nom) == cle]
    if len(egaux) == 1:
        return egaux[0]
    if "_" in cle:
        return None
    nombres = re.findall(r"\d+", cle)
    variantes = [
        nom for nom in project_map
        if "_" not in (cle_ex := normalize_key(nom))
        and re.findall(r"\d+", cle_ex) == nombres
        and difflib.SequenceMatcher(None, cle, cle_ex).ratio() >= _SEUIL_SIMILARITE_CARACTERE
    ]
    return variantes[0] if len(variantes) == 1 else None


def _arbitrer_nom_projet(event: dict, name: str, project_map: dict) -> str:
    """Rattache un nom inexact à un projet connu, sur décision du modèle principal.

    Le score de `_score_nom_projet` ne sert qu'à présélectionner : il ne lit que les mots
    du nom, et deux noms qui ne diffèrent que par leur mot distinctif (« vente X » et
    « vente Y ») partagent assez de mots pour passer tout seuil utile. Un embedding n'y
    voit pas plus clair : il mesure le sujet, pas l'identité du dossier. Seul un modèle
    qui lit l'événement et la description des candidats peut trancher.

    Rend le nom du projet choisi, ou `name` inchangé si aucun ne convient. Une réponse
    inexploitable vaut « aucun » : un doublon se voit et se corrige, un rattachement
    abusif clôt ou réécrit le mauvais projet en silence.
    """
    sur = _rattachement_sur(name, project_map)
    if sur is not None:
        return sur

    # À score égal, l'actif passe devant le clos : des versions closes d'un même nom
    # (« X v7 », « X v8 ») occuperaient sinon toutes les places et masqueraient l'actif.
    candidats = sorted(
        (
            (_score_nom_projet(name, nom), proj.get("status") != "done", nom)
            for nom, proj in project_map.items()
            if _candidat_plausible(name, nom)
        ),
        reverse=True,
    )[:_MAX_CANDIDATS_ARBITRAGE]
    if not candidats:
        return name

    libelles = PROJECT_ARBITRATION_LABELS
    lignes = []
    for i, (_, actif, nom) in enumerate(candidats, 1):
        statut = libelles["actif"] if actif else libelles["clos"]
        description = project_map[nom].get("description") or libelles["vide"]
        lignes.append(f"{i}. {nom} ({statut}) — {description}")
    prompt = get_prompt("PROJECT_ARBITRATION_PROMPT").format(
        nom=name,
        action=event.get("action", ""),
        resume=(event.get("summary") or "").strip() or libelles["aucun"],
        contexte=(event.get("_contexte") or "").strip() or libelles["aucun"],
        candidats="\n".join(lignes),
    )
    try:
        # Température nulle : c'est un choix dans une liste fermée, pas une rédaction, et
        # deux passes sur le même événement doivent rendre le même rattachement.
        content = call_llm_bg(
            [{"role": "user", "content": prompt}],
            model=PRIMARY_MODEL,
            api_url=PRIMARY_API_URL,
            api_key=PRIMARY_API_KEY,
            temperature=0.0,
            max_tokens=MAX_TOKENS_TINY,
            json_response=True,
            no_think=True,
            timeout=llm_timeout(MAX_TOKENS_TINY),
        )
        choix = int(extract_llm_json(content).get("choix", 0))
    except Exception as exc:
        logger.warning(
            "Arbitrage projet '%s' : %s — traité comme nouveau projet",
            name, type(exc).__name__,
        )
        return name

    noms = [nom for _, _, nom in candidats]
    if 1 <= choix <= len(candidats):
        retenu = noms[choix - 1]
        logger.info(
            "Arbitrage projet : '%s' (%s) → '%s' parmi %s",
            name, event.get("action"), retenu, noms,
        )
        return retenu
    logger.info(
        "Arbitrage projet : '%s' (%s) → aucun parmi %s",
        name, event.get("action"), noms,
    )
    return name


def _resoudre_noms_projets(user_code: str, project_events: list[dict]) -> dict[str, str]:
    """Nom brut → nom résolu, pour chaque nom d'événement absent de la liste.

    Calculé AVANT la lecture qui précède l'écriture : l'arbitrage coûte un appel LLM par
    nom inexact, et un projet écrit pendant ce temps par un autre chemin serait écrasé si
    la liste avait été lue avant.

    La liste de référence reçoit au fil de l'eau les projets que les événements précédents
    du lot créent ou renomment : l'analyzer applique d'un bloc les événements de plusieurs
    sessions, et un projet ouvert par la première doit être candidat pour la suivante.
    Mêmes filtres que la boucle d'application, pour ne pas arbitrer un événement qu'elle
    écartera.
    """
    project_map = {p["name"]: p for p in get_user_projects(user_code)}
    resolus: dict[str, str] = {}
    for event in project_events:
        if not isinstance(event, dict):
            continue
        action = (event.get("action") or "").strip()
        name = _normalize_project_name((event.get("name") or "").strip())
        new_name = _normalize_project_name((event.get("rename_to") or "").strip())
        if not name or action not in ("create", "update", "done", "rename"):
            continue
        if action == "rename" and not new_name:
            continue
        if name in project_map:
            resolved = name
        else:
            if name not in resolus:
                resolus[name] = _arbitrer_nom_projet(event, name, project_map)
            resolved = resolus[name]
        if action in ("create", "update") and resolved not in project_map:
            project_map[resolved] = {
                "name": resolved,
                "status": "in_progress",
                "description": (event.get("summary") or "").strip(),
            }
        elif action == "rename" and resolved in project_map:
            project_map[new_name] = {**project_map.pop(resolved), "name": new_name}
    return resolus


def apply_project_updates(user_code: str, project_events: list[dict]):
    """Apply structured project events from the analyzer.

    Each event is a dict: {name, action, summary, [rename_to]}
      action  : "create" | "update" | "done" | "rename"
      summary : 1-sentence description of what happened (appended to updates timeline)
      rename_to : new name (rename only)
    """
    resolus = _resoudre_noms_projets(user_code, project_events)
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

        resolved = name if name in project_map else resolus.get(name, name)

        if action in ("create", "update"):
            if resolved not in project_map:
                project_map[resolved] = {
                    "name": resolved,
                    "status": "in_progress",
                    "first_mentioned": now,
                    "last_update": now,
                    "updates": [],
                }
            else:
                # Une activité sur un projet clos le rouvre : sinon l'événement s'ajoute
                # à un projet que `projets_actifs` écarte de tout contexte, puis que la
                # purge des projets terminés emporte. La clôture automatique des dormants
                # rend ce cas courant : un chantier repris après soixante jours est clos.
                if project_map[resolved].get("status") == "done":
                    project_map[resolved]["status"] = "in_progress"
                    logger.info(
                        "Projet rouvert : %s / '%s' (%s)", user_code, resolved, action
                    )
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
                old_raw if old_raw in project_map else resolus.get(old_raw, old_raw)
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

"""Outils de la boucle agentique — Phase 1 : lecture du monde, écriture confinée.

Aucun outil n'exécute quoi que ce soit ici. Surf, RAG, lecture de fichiers, écriture dans
le workspace : le pire cas est un fichier inutile dans un dossier jetable. Le shell et la
délégation de code arrivent en Phase 2 et 3, une fois la fiabilité du modèle sur des
enchaînements longs mesurée sur ce périmètre-là.

Deux règles de conception, toutes deux dictées par le contexte d'un 35B local :

  Peu d'outils. Sept. Chaque outil supplémentaire est une occasion de se tromper de
  choix, et le coût se paie à CHAQUE pas puisque les schémas sont rendus en tête de prompt.

  Toute sortie est tronquée. Une page web ou un fichier de logs entier ferait exploser un
  contexte qui est déjà réinjecté intégralement à chaque pas.
"""

import json
import os

from config import (
    AGENT_DOCS_MIN_SCORE,
    AGENT_MAX_TOOL_OUTPUT,
    AGENT_PAGE_MAX_CHARS,
    AGENT_WRITE_MAX_CHARS,
)
from helpers import get_logger

from .sandbox import SandboxError, ensure_parent, relative, resolve

logger = get_logger("jarvis-agent")

# Nom réservé : traité par la boucle, jamais dispatché ici.
FINISH = "finish"


def _truncate(text: str, limit: int = 0) -> str:
    """Tronque en le DISANT. Un modèle qui ignore qu'il lit un extrait conclut sur un vide."""
    limit = limit or AGENT_MAX_TOOL_OUTPUT
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[…tronqué — {len(text) - limit} caractères restants]"


# ── Schémas (format OpenAI, rendus par le template Qwen3.6) ───────────────

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Recherche sur le web. Renvoie titre, extrait et URL. Les extraits servent "
                "à REPÉRER les bonnes sources, jamais à rédiger : ils sont courts, souvent "
                "sans date et parfois trompeurs. Ouvre ensuite la source avec fetch_url."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "La requête de recherche."},
                    "max_results": {"type": "integer", "description": "Défaut 5, max 10."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Lit une page web en entier. C'est ICI que tu prends tes dates, tes chiffres "
                "et tes citations — un extrait de web_search ne suffit jamais pour ça. "
                "Passage OBLIGÉ avant de rédiger une synthèse."
            ),
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "URL complète (http/https)."}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Cherche dans les documents PERSONNELS de l'utilisateur : contrats, "
                "factures, rapports internes, notes. Ne contient AUCUNE connaissance "
                "générale — pour un sujet externe (entreprise, acteur, technologie), "
                "c'est web_search qu'il faut."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Ce que tu cherches."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "threat_intel",
            "description": (
                "Renseignement sur un groupe d'attaquants (ransomware, extorsion), à partir "
                "des agrégateurs de sites de fuite darknet. À utiliser AVANT le web pour "
                "tout ce qui touche un groupe nommé : le web de surface recopie la presse, "
                "ces sources-là remontent aux publications des groupes eux-mêmes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "group": {"type": "string", "description": "Nom du groupe, ex 'lockbit3'."},
                    "kind": {
                        "type": "string",
                        "description": (
                            "search = retrouver le nom exact/alias (COMMENCE par là) · "
                            "profile = sites .onion connus et notes · "
                            "victims = victimes publiées · recent = actualité tous groupes · "
                            "certfr = avis et alertes du CERT-FR, à utiliser pour toute "
                            "cible française et pour les acteurs absents des autres sources"
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "Liste le contenu d'un dossier. Par défaut ton espace de travail.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Chemin. Défaut '.'."}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Lit un fichier texte. Ton espace de travail est accessible, ainsi que le "
                "code source de Jarvis en LECTURE SEULE."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Chemin du fichier."},
                    "offset": {"type": "integer", "description": "Première ligne (1 par défaut)."},
                    "limit": {"type": "integer", "description": "Nombre de lignes, défaut 200."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Écrit un fichier dans ton espace de travail. C'est ainsi que tu produis "
                "tes livrables : notes, rapports, scripts. "
                f"MAXIMUM {AGENT_WRITE_MAX_CHARS} caractères par appel — au-delà, ta sortie "
                "est coupée et le tour est perdu. Pour un document plus long, découpe-le : "
                "premier appel sans append, suivants avec append=true."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Chemin relatif, ex 'rapport.md'."},
                    "content": {
                        "type": "string",
                        "description": f"Contenu à écrire, {AGENT_WRITE_MAX_CHARS} caractères max.",
                    },
                    "append": {
                        "type": "boolean",
                        "description": "true = ajoute à la fin du fichier. Défaut false (écrase).",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": FINISH,
            "description": (
                "Termine la tâche. À appeler UNIQUEMENT quand l'objectif est atteint, avec "
                "un résumé de ce que tu as fait et la liste des fichiers produits."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Réponse à l'objectif, en français, adressée à l'utilisateur.",
                    },
                    "deliverables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Chemins des fichiers produits, relatifs au workspace.",
                    },
                },
                "required": ["summary"],
            },
        },
    },
]

TOOL_NAMES = frozenset(t["function"]["name"] for t in TOOL_SCHEMAS)


# ── Implémentations ───────────────────────────────────────────────────────


async def _web_search(task: dict, args: dict) -> str:
    from web_search import search_web

    query = (args.get("query") or "").strip()
    if not query:
        return "Erreur : query vide."
    max_results = min(int(args.get("max_results") or 5), 10)
    results = await search_web(query, max_results)
    if not results:
        return "Aucun résultat."
    lines = []
    for r in results:
        title = r.get("title") or "(sans titre)"
        url = r.get("url") or ""
        body = (r.get("body") or "").strip()
        date = f" ({r['date']})" if r.get("date") else ""
        lines.append(f"### {title}{date}\n{url}\n{body}")
    return _truncate("\n\n".join(lines))


async def _fetch_url(task: dict, args: dict) -> str:
    from web_search import _fetch_page_text

    url = (args.get("url") or "").strip()
    if not url.startswith("http"):
        return "Erreur : URL invalide (http/https attendu)."
    text = await _fetch_page_text(url, AGENT_PAGE_MAX_CHARS)
    if not text:
        return f"Page vide ou inaccessible : {url}"
    return _truncate(text)


async def _search_docs(task: dict, args: dict) -> str:
    from rag import search_documents

    query = (args.get("query") or "").strip()
    if not query:
        return "Erreur : query vide."
    chunks = await search_documents(query)

    # Plancher PROPRE à l'agent, plus strict que celui du chat. Le RAG garantit de rendre
    # des extraits dès qu'un document a été adopté (repli à score_threshold=0.0 dans
    # rag.py) : très bien pour une conversation, désastreux ici. L'agent ne peut pas
    # reconnaître du hors-sujet, il le prend pour de la matière et diverge.
    kept = [c for c in chunks if c.get("score", 0.0) >= AGENT_DOCS_MIN_SCORE]
    if not kept:
        best = max((c.get("score", 0.0) for c in chunks), default=0.0)
        return (
            f"Rien de pertinent dans la base documentaire pour « {query} » "
            f"(meilleur score {best:.2f}, seuil {AGENT_DOCS_MIN_SCORE:.2f}). "
            "Cette base contient les documents PERSONNELS de l'utilisateur — contrats, "
            "factures, rapports internes — pas de la connaissance générale. "
            "Pour un sujet externe, utilise web_search."
        )
    lines = [
        f"### {c.get('source', '?')} (score {c.get('score', 0.0):.2f})\n{c.get('text', '')}"
        for c in kept
    ]
    return _truncate("\n\n".join(lines))


async def _threat_intel(task: dict, args: dict) -> str:
    from . import cti

    group = (args.get("group") or "").strip()
    kind = (args.get("kind") or ("search" if group else "recent")).strip().lower()

    if kind == "recent":
        return _truncate(await cti.recent_activity())
    if kind == "certfr":
        return _truncate(await cti.cert_fr(group))
    if not group:
        return "Erreur : `group` requis pour kind=search|profile|victims."
    if kind == "search":
        return _truncate(await cti.list_groups(group))
    if kind == "profile":
        return _truncate(await cti.group_profile(group))
    if kind == "victims":
        return _truncate(await cti.group_victims(group))
    return f"kind inconnu : {kind}. Valeurs acceptées : search, profile, victims, recent."


async def _list_dir(task: dict, args: dict) -> str:
    path = resolve(task["id"], args.get("path") or ".", write=False)
    if not os.path.isdir(path):
        return f"Pas un dossier : {args.get('path')}"
    entries = sorted(os.listdir(path))
    if not entries:
        return "(dossier vide)"
    lines = []
    for name in entries[:200]:
        full = os.path.join(path, name)
        if os.path.isdir(full):
            lines.append(f"{name}/")
        else:
            lines.append(f"{name}  ({os.path.getsize(full)} o)")
    suffix = f"\n[…{len(entries) - 200} entrées de plus]" if len(entries) > 200 else ""
    return f"{relative(task['id'], path)} :\n" + "\n".join(lines) + suffix


async def _read_file(task: dict, args: dict) -> str:
    path = resolve(task["id"], args.get("path") or "", write=False)
    if not os.path.isfile(path):
        return f"Fichier introuvable : {args.get('path')}"
    offset = max(int(args.get("offset") or 1), 1)
    limit = min(int(args.get("limit") or 200), 800)
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as exc:
        return f"Lecture impossible : {exc}"
    selected = lines[offset - 1 : offset - 1 + limit]
    if not selected:
        return f"Rien à lire à partir de la ligne {offset} (le fichier en compte {len(lines)})."
    # Numérotation : l'agent doit pouvoir demander la suite sans recompter.
    body = "".join(f"{offset + i}\t{line}" for i, line in enumerate(selected))
    tail = ""
    if offset - 1 + limit < len(lines):
        tail = f"\n[…{len(lines) - (offset - 1 + limit)} lignes suivantes — relis avec offset={offset + limit}]"
    return _truncate(body) + tail


async def _write_file(task: dict, args: dict) -> str:
    rel = args.get("path") or ""
    content = args.get("content")
    if content is None:
        return "Erreur : content manquant."
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, indent=2)
    path = resolve(task["id"], rel, write=True)
    ensure_parent(path)
    append = bool(args.get("append"))
    try:
        with open(path, "a" if append else "w", encoding="utf-8") as f:
            f.write(content)
        total = os.path.getsize(path)
    except OSError as exc:
        return f"Écriture impossible : {exc}"
    verb = "Ajouté à" if append else "Écrit"
    # On rappelle le plafond à chaque écriture : c'est au moment où il vient d'écrire que
    # le modèle décide s'il continue en append ou s'il conclut.
    return (
        f"{verb} : {relative(task['id'], path)} — {len(content)} caractères écrits, "
        f"{total} octets au total. Plafond par appel : {AGENT_WRITE_MAX_CHARS} caractères."
    )


_DISPATCH = {
    "web_search": _web_search,
    "fetch_url": _fetch_url,
    "search_docs": _search_docs,
    "threat_intel": _threat_intel,
    "list_dir": _list_dir,
    "read_file": _read_file,
    "write_file": _write_file,
}


async def execute_tool(task: dict, name: str, args: dict) -> str:
    """Exécute un outil et renvoie TOUJOURS une chaîne destinée au modèle.

    Aucune exception ne remonte : un outil qui échoue doit donner à l'agent de quoi
    corriger son tir au pas suivant, pas tuer la tâche. Seule l'annulation et les budgets,
    décidés par la boucle, arrêtent une tâche.
    """
    fn = _DISPATCH.get(name)
    if fn is None:
        return (
            f"Outil inconnu : {name}. Outils disponibles : "
            f"{', '.join(sorted(TOOL_NAMES))}."
        )
    try:
        return await fn(task, args)
    except SandboxError as exc:
        return f"Refusé : {exc}"
    except Exception as exc:
        logger.warning("agent: outil %s en échec (%s: %s)", name, type(exc).__name__, exc)
        return f"L'outil {name} a échoué : {type(exc).__name__}: {exc}"

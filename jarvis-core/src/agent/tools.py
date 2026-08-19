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
    AGENT_SHELL_ENABLED,
    AGENT_SHELL_TIMEOUT,
    AGENT_MAX_TOOL_OUTPUT,
    AGENT_PAGE_MAX_CHARS,
    AGENT_READ_MAX_CHARS,
    AGENT_WRITE_MAX_CHARS,
)
from helpers import get_logger

from .sandbox import SandboxError, ensure_parent, relative, resolve

logger = get_logger("jarvis-agent")

# Fichiers de service de la boucle, présents dans chaque workspace. Masqués à l'agent :
# au run du 19/08/2026 il a listé son dossier puis lu son PROPRE transcript au pas 2,
# dépensant un tour à relire sa propre trace. Ils ne lui apprennent rien qu'il n'ait déjà
# dans son contexte, et messages.json en est une copie intégrale.
_FICHIERS_INTERNES = frozenset({"transcript.jsonl", "messages.json", "messages.json.tmp"})

# Noms réservés, traités à part par la boucle.
FINISH = "finish"   # jamais dispatché ici — c'est la sortie de la boucle
PLAN = "plan"       # seul outil autorisé EN PLUS d'une action dans le même tour


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
                "Recherche web : titres, extraits, URL. Les extraits servent à REPÉRER "
                "les bonnes sources, jamais à rédiger. Ouvre-les ensuite avec fetch_url."
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
                "Lit une page web en entier. C'est ici que tu prends dates, chiffres et "
                "citations. Passage obligé avant de rédiger."
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
                "Documents PERSONNELS de l'utilisateur : contrats, factures, notes. "
                "Aucune connaissance générale — pour un sujet externe, c'est web_search."
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
                "Groupes d'attaquants, via les agrégateurs de sites de fuite darknet. "
                "À tenter AVANT le web sur tout groupe nommé."
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
            "name": "plan",
            "description": (
                "Ton plan de travail, ton tout premier appel. `done` coche une étape faite, "
                "`steps` remplace le plan quand la réalité le dément. Reposer le même plan "
                "sans `done` ne fait rien."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Les étapes, dans l'ordre. 3 à 6, courtes et vérifiables.",
                    },
                    "done": {
                        "type": "integer",
                        "description": "Numéro de l'étape à marquer comme faite (1 = la première).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": (
                "Exécute une commande shell dans ton espace de travail, en bac à sable : "
                "écriture limitée à ton workspace, AUCUN accès réseau, secrets illisibles. "
                "Pour compter, filtrer, chercher dans des fichiers, lancer un script. "
                "Rien d'interactif : la commande doit se terminer seule. "
                "MACHINE macOS, OUTILS BSD — pas les mêmes que sous Linux : pas de "
                "`grep -P` (utilise `grep -E`), `sed -i` exige un argument (`sed -i ''`), "
                "pas de `timeout`, `date` et `stat` ont une autre syntaxe. "
                "Au moindre doute, passe par `python3` : il est présent et portable."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "La commande, ex \"wc -l *.md\"."},
                    "timeout": {
                        "type": "integer",
                        "description": f"Délai en secondes. Défaut {AGENT_SHELL_TIMEOUT:.0f}.",
                    },
                },
                "required": ["cmd"],
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
                "Lit un fichier : ton workspace, plus le code source de Jarvis en lecture "
                "seule. Un gros fichier arrive en morceaux, le résultat indique la suite."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Chemin du fichier."},
                    "offset": {"type": "integer", "description": "Première ligne (1 par défaut)."},
                    "limit": {
                        "type": "integer",
                        "description": (
                            "Nombre de lignes max. Par défaut, autant que le budget le permet "
                            "— ne le fixe que pour une lecture ciblée."
                        ),
                    },
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
                "Écrit dans ton workspace — c'est ainsi que tu produis tes livrables. "
                f"Maximum {AGENT_WRITE_MAX_CHARS} caractères par appel, au-delà ta sortie est "
                "coupée. Document plus long : découpe-le, append=true à partir du second."
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
                        "description": (
                            "true = AJOUTE à la fin. À utiliser dès que le fichier existe "
                            "déjà : sans ce drapeau, tu remplaces tout son contenu."
                        ),
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": (
                            "true = remplacer délibérément un fichier par un contenu plus "
                            "court. Sans ce drapeau, un tel remplacement est refusé."
                        ),
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
                "Termine la tâche, uniquement quand l'objectif est atteint."
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

# Le shell n'est PAS déclaré au modèle tant que la capacité est éteinte : un outil annoncé
# puis refusé lui fait perdre des tours à réessayer.
if not AGENT_SHELL_ENABLED:
    TOOL_SCHEMAS = [t for t in TOOL_SCHEMAS if t["function"]["name"] != "shell"]

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
    task.setdefault("sources_seen", [])
    if url not in task["sources_seen"]:
        task["sources_seen"].append(url)
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
    # Mémorisé au même titre qu'une URL ou un fichier lu : un article sourcé sur la base
    # documentaire cite un nom de document, pas une URL — sans ça, _has_sources refusait
    # le finish d'un livrable pourtant correctement sourcé.
    task.setdefault("sources_seen", [])
    for c in kept:
        src = c.get("source")
        if src and src not in task["sources_seen"]:
            task["sources_seen"].append(src)

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


def _same_step(a: str, b: str) -> bool:
    """Deux intitulés désignent-ils la même étape, à une reformulation près ?

    Le modèle réécrit ses étapes en replanifiant — « Recherche sources » devient
    « recherche des sources ». Une comparaison caractère par caractère échoue sur un mot
    inséré, et l'avancement acquis est perdu.

    On s'appuie sur keyword_overlap_score (helpers/text.py), qui compte les mots de
    contenu partagés hors stopwords français : c'est l'outil déjà utilisé ailleurs dans
    Jarvis pour rapprocher deux formulations. Seuil bas assumé — rapprocher deux étapes
    voisines à tort ne coûte qu'une case cochée un peu tôt, les séparer efface du travail
    réellement fait.
    """
    from helpers import keyword_overlap_score

    shared = keyword_overlap_score(a, b)
    if shared == 0:
        return False
    # Rapporté au PLUS COURT des deux intitulés : « Analyse » et « Analyse des sources
    # trouvées » ne partagent qu'un mot, mais c'est tout ce que le premier contient.
    # keyword_overlap_score(x, x) donne le nombre de mots de contenu de x.
    shortest = min(keyword_overlap_score(a, a), keyword_overlap_score(b, b))
    return shortest > 0 and shared >= shortest * 0.6


def render_plan(task: dict) -> str:
    """Rend le plan courant, tel qu'il est réaffiché sous chaque résultat d'outil.

    C'est ce réaffichage qui fait tout le travail : le plan est une DONNÉE stable, pas du
    raisonnement réinjecté. Il ne peut donc ni être confondu avec une sortie attendue du
    modèle, ni être relu comme un ordre frais à ré-exécuter — les deux pannes du
    19/08/2026. Le modèle avance dedans en marquant ses étapes, et sait toujours où il en
    est sans avoir à le redéduire.
    """
    steps = task.get("plan") or []
    if not steps:
        return ""
    current = next((i for i, s in enumerate(steps) if not s.get("done")), len(steps))
    lines = []
    for i, step in enumerate(steps):
        mark = "x" if step.get("done") else ("→" if i == current else " ")
        lines.append(f"  [{mark}] {i + 1}. {step.get('text', '')}")
    position = f"étape {current + 1}/{len(steps)}" if current < len(steps) else "toutes faites"
    out = f"\n\nPlan ({position}) :\n" + "\n".join(lines)

    # Relance quand rien n'a été coché depuis un moment : le modèle avance dans le travail
    # mais oublie de le marquer, et son plan cesse alors de refléter où il en est.
    stalled = task.get("steps", 0) - task.get("plan_marked_at", 0)
    if current < len(steps) and stalled >= 3:
        out += (
            f"\n  (aucune étape cochée depuis {stalled} pas — si l'étape {current + 1} "
            f"est faite, joins plan(done={current + 1}) à ton prochain appel)"
        )
    return out


async def _plan(task: dict, args: dict) -> str:
    from . import store

    steps = args.get("steps")
    done = args.get("done")

    if isinstance(steps, str):
        steps = [steps]
    unchanged = False
    if steps:
        unchanged = [str(x).strip() for x in steps] == [
            s["text"] for s in (task.get("plan") or [])
        ]
        # Une replanification REPORTE l'avancement déjà acquis, au lieu de le remettre à
        # zéro. Le modèle replanifie en listant ce qu'il lui reste à faire, éventuellement
        # en reprenant des étapes au même intitulé : effacer leur état, c'était décocher
        # derrière lui puis lui reprocher de ne pas cocher (observé le 19/08/2026 — plan
        # final à 1 étape sur 3 alors que le travail était fait).
        previously_done = [s["text"] for s in (task.get("plan") or []) if s.get("done")]
        task["plan"] = [
            {
                "text": str(s).strip(),
                "done": any(_same_step(str(s), old) for old in previously_done),
            }
            for s in steps
            if str(s).strip()
        ][:8]

    if done is not None:
        try:
            index = int(done) - 1
        except (TypeError, ValueError):
            return "Erreur : `done` doit être un numéro d'étape (1 = la première)."
        current = task.get("plan") or []
        if not 0 <= index < len(current):
            return f"Erreur : l'étape {done} n'existe pas (le plan en compte {len(current)})."
        # Re-cocher une étape déjà faite ne fait rien avancer : même tour perdu que reposer
        # un plan inchangé, et le modèle y revenait autant (19/08/2026).
        if current[index]["done"] and unchanged:
            done = None
        else:
            current[index]["done"] = True

    if not task.get("plan"):
        return "Erreur : donne `steps` pour poser un plan."

    # Reposer le MÊME plan sans rien cocher ne fait rien avancer et consomme un pas. Le
    # modèle le fait en réaction à la relance de marquage, en renvoyant `steps` au lieu de
    # `done` — observé le 19/08/2026 : 4 appels identiques d'affilée, un tiers du budget.
    if unchanged and done is None:
        current = next((i for i, s in enumerate(task["plan"]) if not s.get("done")), None)
        hint = (
            f" Pour marquer l'étape en cours comme faite, rappelle plan avec done={current + 1} "
            f"— et joins-le à une VRAIE action, pas seul."
            if current is not None else ""
        )
        return ("Plan inchangé, rien n'a bougé." + hint) + render_plan(task)

    # Sert la relance de render_plan : on repère le décrochage entre le travail réel et
    # son marquage. `steps` est le compteur de pas de la tâche, tenu par la boucle.
    task["plan_marked_at"] = task.get("steps", 0)
    store.save_task(task)
    return "Plan à jour." + render_plan(task)


async def _shell(task: dict, args: dict) -> str:
    from . import shell as sh

    sortie = await sh.executer(task, args.get("cmd") or "", float(args.get("timeout") or 0))
    restantes = sh.commandes_restantes(task)
    if restantes <= 5:
        sortie += f"\n\n[{restantes} commande(s) restante(s) pour cette tâche]"
    return _truncate(sortie)


async def _list_dir(task: dict, args: dict) -> str:
    path = resolve(task["id"], args.get("path") or ".", write=False)
    if not os.path.isdir(path):
        return f"Pas un dossier : {args.get('path')}"
    entries = sorted(e for e in os.listdir(path) if e not in _FICHIERS_INTERNES)
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
    if os.path.basename(path) in _FICHIERS_INTERNES:
        return (
            f"{os.path.basename(path)} est un fichier de service de la boucle, pas une "
            "source : il ne contient que la trace de ce que tu as déjà fait."
        )
    if not os.path.isfile(path):
        return f"Fichier introuvable : {args.get('path')}"
    offset = max(int(args.get("offset") or 1), 1)
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as exc:
        return f"Lecture impossible : {exc}"
    if offset > len(lines):
        return f"Rien à lire à partir de la ligne {offset} (le fichier en compte {len(lines)})."

    # Le budget est en CARACTÈRES, pas en lignes. Un plafond de 200 lignes rendait 9 000
    # caractères là où 15 000 étaient permis, et obligeait à paginer un fichier de 525
    # lignes en trois appels — que le modèle a préféré rejouer à l'identique (19/08/2026).
    # `limit` reste disponible pour une lecture ciblée, mais ne borne plus par défaut.
    budget = AGENT_READ_MAX_CHARS
    limit = int(args.get("limit") or 0)
    kept, used = [], 0
    for i, line in enumerate(lines[offset - 1:]):
        if limit and i >= limit:
            break
        numbered = f"{offset + i}\t{line}"
        if kept and used + len(numbered) > budget:
            break
        kept.append(numbered)
        used += len(numbered)

    task.setdefault("sources_seen", [])
    if path not in task["sources_seen"]:
        task["sources_seen"].append(path)

    body = "".join(kept)
    next_offset = offset + len(kept)
    if next_offset > len(lines):
        return body

    # L'avertissement est répété EN TÊTE : placé au seul pied d'un bloc de 15 000 à 32 000
    # caractères de code, il est noyé — mesuré le 19/08/2026, ignoré quatre fois de suite.
    remaining = len(lines) - next_offset + 1
    warning = (
        f"[LECTURE PARTIELLE — lignes {offset} à {next_offset - 1} sur {len(lines)}. "
        f"Il en reste {remaining}. Pour la suite : read_file avec offset={next_offset}. "
        f"Redemander ces mêmes lignes ne rendra rien de neuf.]"
    )
    return f"{warning}\n\n{body}\n{warning}"


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

    # ── Garde anti-écrasement ────────────────────────────────────────────────
    # `append` par défaut à false : une écriture sans ce drapeau REMPLACE le fichier. Le
    # 19/08/2026, un rapport bâti en cinq pas (11 ko) a été réduit à sa seule section
    # « Sources » (726 o) parce que le modèle, invité à compléter ses sources, a réécrit
    # sans append. Cinq pas de travail détruits par un booléen par défaut.
    #
    # On ne bloque QUE le cas accidentel : le fichier existe, on ne lui ajoute rien, et le
    # nouveau contenu est nettement plus court que l'ancien. Une réécriture légitime
    # (correction, réorganisation) produit un texte comparable ou plus long ; si elle est
    # vraiment plus courte, `overwrite` la débloque en un mot.
    if not append and not bool(args.get("overwrite")) and os.path.exists(path):
        try:
            ancien = os.path.getsize(path)
        except OSError:
            ancien = 0
        if ancien > 0 and len(content) < ancien * 0.7:
            return (
                f"Écriture REFUSÉE : {relative(task['id'], path)} contient déjà {ancien} "
                f"octets et tu n'en écris que {len(content)} — tu allais effacer ton propre "
                "travail. Pour AJOUTER à la fin, utilise append=true. Pour remplacer "
                "délibérément par une version plus courte, utilise overwrite=true."
            )

    # Un ajout qui démarre sans saut de ligne colle au contenu précédent et casse le
    # markdown — mesuré le 19/08/2026 : « [5] https://…html## Contexte juridique ». Le
    # modèle raisonne par blocs et ne pense pas à la jointure ; on la pose ici.
    if append and content and not content.startswith("\n") and os.path.exists(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            f.seek(max(os.path.getsize(path) - 1, 0))
            if f.read(1) not in ("\n", ""):
                content = "\n\n" + content
    try:
        with open(path, "a" if append else "w", encoding="utf-8") as f:
            f.write(content)
        total = os.path.getsize(path)
    except OSError as exc:
        return f"Écriture impossible : {exc}"
    verb = "Ajouté à" if append else "Écrit"
    # On rend la FIN du fichier, pas seulement sa taille. Le modèle rédige par morceaux et
    # ne se souvient pas de ce qu'il a déjà posé : au run du 19/08/2026 il a réécrit deux
    # sections déjà présentes, dupliquant un paragraphe entier dans l'article final.
    # Lui montrer sa dernière phrase lui dit où reprendre.
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            fin = f.read()[-220:]
    except OSError:
        fin = ""
    return (
        f"{verb} : {relative(task['id'], path)} — {len(content)} caractères écrits, "
        f"{total} octets au total. Plafond par appel : {AGENT_WRITE_MAX_CHARS} caractères.\n"
        f"Fin actuelle du fichier — reprends APRÈS, ne la réécris pas :\n…{fin}"
    )


_DISPATCH = {
    "web_search": _web_search,
    "fetch_url": _fetch_url,
    "plan": _plan,
    "search_docs": _search_docs,
    "shell": _shell,
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

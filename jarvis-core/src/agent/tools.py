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

import ast
import json
import os
import sys

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
# a l'usage il a listé son dossier puis lu son PROPRE transcript au pas 2,
# dépensant un tour à relire sa propre trace. Ils ne lui apprennent rien qu'il n'ait déjà
# dans son contexte, et messages.json en est une copie intégrale.
_FICHIERS_INTERNES = frozenset({"transcript.jsonl", "messages.json", "messages.json.tmp"})

# `repo_ref` est la copie vierge que la mesure compare au travail de l'agent : la montrer
# l'invite à lire un arbre qui n'est pas le sien — et rendre des constats situés dedans.
# Masquée, la racine du workspace ne porte plus que `repo`, donc plus rien à choisir.
#
# `autocode` et `agent` sont le cycle qui produit cette revue et la boucle qui l'exécute.
# Ils changent d'un run à l'autre pendant qu'on les met au point : un constat qui les vise
# porte sur un état déjà périmé quand on le lit. TEMPORAIRE — à retirer quand la mécanique
# sera stabilisée, `agent/` portant par ailleurs du code de production qui mérite relecture.
_DOSSIERS_MASQUES = frozenset({"repo_ref", "autocode", "agent"})

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
                            "Nombre de lignes max. NE LE FIXE PAS pour découvrir un fichier : "
                            "sans lui, une lecture rend jusqu'à ~800 lignes d'un coup. Une "
                            "limite basse coûte un tour par tranche. Réserve-le à la relecture "
                            "d'un passage précis, avec offset."
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


# ── Jeu d'outils par origine ──────────────────────────────────────────────
# Le filtrage à l'import ne suffit plus dès qu'une deuxième origine existe : une tâche que
# Jarvis se donne n'a pas les droits d'une tâche qu'un humain lui confie. L'écart doit être
# porté par la TÂCHE, sans quoi activer une capacité pour l'une l'activerait pour l'autre.
#
# `verify` n'entre dans aucun jeu par défaut : il n'a de sens que sur un worktree, que seule
# l'origine `autocode` prépare.
VERIFY = "verify"
GREP = "grep"
APPELANTS = "appelants"

# Recherche littérale, jamais une expression régulière. Une regex fournie par le modèle
# s'exécute sur deux cents fichiers et peut s'emballer sans qu'on sache l'interrompre ;
# « où est défini ceci » se répond par une sous-chaîne, et c'est la question qu'il pose.
_GREP_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": GREP,
        "description": (
            "Cherche un TEXTE EXACT dans les fichiers du dépôt et rend les lignes qui le "
            "contiennent, avec leur fichier et leur numéro. C'est par là qu'on trouve où "
            "quelque chose est défini ou utilisé — puis read_file avec offset=<ligne>."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Le texte à chercher, littéralement. Ex : 'def est_protege'.",
                },
                "path": {
                    "type": "string",
                    "description": "Sous-dossier où chercher. Par défaut : tout le dépôt.",
                },
            },
            "required": ["pattern"],
        },
    },
}

_APPELANTS_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": APPELANTS,
        "description": (
            "Rend TOUS les endroits du dépôt qui importent un module ou appellent une "
            "fonction, avec fichier et numéro de ligne. Répond à « qui se sert de ça ? » — "
            "ce qu'un fichier ne dit jamais de lui-même : du code qu'aucun appelant "
            "n'atteint est inerte, le même code appelé sur chaque requête ne l'est pas. "
            "Suit les imports indirects que grep manque (from m import f, puis f(…))."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "nom": {
                    "type": "string",
                    "description": (
                        "Un module ou une fonction, sans chemin ni extension. "
                        "Ex : 'vitals', 'est_protege'."
                    ),
                },
                "path": {
                    "type": "string",
                    "description": "Sous-dossier où chercher. Par défaut : tout le dépôt.",
                },
            },
            "required": ["nom"],
        },
    },
}

_VERIFY_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": VERIFY,
        "description": (
            "Vérifie l'état du dépôt dans repo/ : compilation, pyflakes, et la suite de "
            "tests unitaires. Sans argument. À appeler avant finish, et autant de fois "
            "que nécessaire pendant que tu corriges."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

# Pas de web ni de RAG pour une tâche de code : elle n'en a pas besoin, et chaque schéma
# est rendu en tête de prompt à CHAQUE pas — un outil inutile se paie quarante fois et
# reste une occasion de se tromper de choix.
_OUTILS_AUTOCODE = frozenset(
    {PLAN, GREP, APPELANTS, "list_dir", "read_file", "write_file", VERIFY, FINISH}
)

# Ajoutés avant finish : l'ordre des schémas est l'ordre de lecture du modèle, et
# « chercher », « remonter les appelants », « vérifier » se lisent avant « terminer ».
_SCHEMAS_AUTOCODE = (_GREP_SCHEMA, _APPELANTS_SCHEMA, _VERIFY_SCHEMA)


def schemas_for(task: dict) -> list[dict]:
    """Schémas d'outils visibles par cette tâche, selon son origine."""
    if task.get("origin") != "autocode":
        return TOOL_SCHEMAS
    retenus: list[dict] = []
    for schema in TOOL_SCHEMAS:
        nom = schema["function"]["name"]
        if nom == FINISH:
            retenus.extend(_SCHEMAS_AUTOCODE)
        if nom in _OUTILS_AUTOCODE:
            retenus.append(schema)
    return retenus


def names_for(task: dict) -> frozenset[str]:
    return frozenset(t["function"]["name"] for t in schemas_for(task))


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
. Le modèle avance dedans en marquant ses étapes, et sait toujours où il en
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
        # derrière lui puis lui reprocher de ne pas cocher (observé — plan
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
        # un plan inchangé, et le modèle y revenait autant.
        if current[index]["done"] and unchanged:
            done = None
        else:
            current[index]["done"] = True

    if not task.get("plan"):
        return "Erreur : donne `steps` pour poser un plan."

    # Reposer le MÊME plan sans rien cocher ne fait rien avancer et consomme un pas. Le
    # modèle le fait en réaction à la relance de marquage, en renvoyant `steps` au lieu de
    # `done` — observé : 4 appels identiques d'affilée, un tiers du budget.
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


# Large devant les ~6 s mesurées (compilation + pyflakes + 107 tests) : le délai borne une
# suite partie en boucle, il n'arbitre pas la durée nominale.
_VERIFY_TIMEOUT = 180.0


def commande_verification() -> str:
    """Les trois contrôles, dans l'ordre du moins au plus cher.

    `sys.executable` et non un chemin codé en dur : c'est l'interpréteur qui fait tourner
    Jarvis, donc celui de son venv, sur n'importe quelle installation.
    """
    py = sys.executable
    return (
        "cd repo || exit 1\n"
        "echo '── py_compile ──'\n"
        f"{py} -m compileall -q jarvis-core/src || echo 'ÉCHEC compilation'\n"
        "echo '── pyflakes ──'\n"
        f"{py} -m pyflakes jarvis-core/src/*.py jarvis-core/src/*/*.py\n"
        "echo '── pytest (unitaires) ──'\n"
        f"{py} -m pytest jarvis-core/tests/ -m 'not integration' -q\n"
    )


# Bornes de la recherche. Rendre trois cents lignes remplirait le contexte, ce que cet
# outil existe précisément pour éviter : on cherche un point d'entrée, pas un inventaire.
_GREP_MAX_RESULTATS = 60
_GREP_LIGNE_MAX = 200
# Un même fichier peut appeler cent fois la même fonction sans rien apprendre de plus :
# ce qu'on cherche, c'est QUELS fichiers touchent le symbole, pas combien de fois.
_APPELANTS_PAR_FICHIER = 8

# Profondeur maximale d'un couloir traversé d'un seul appel. Assez pour `.` → `repo` →
# `jarvis-core` → `src`, sans risquer de dérouler une arborescence entière.
_COULOIR_MAX_PALIERS = 5

_GREP_FICHIERS_MAX = 2000
# `repo_ref` est la copie vierge de HEAD que la mesure compare au travail de l'agent. La
# parcourir rend chaque occurrence en double et attribue au dépôt des lignes qui viennent
# de l'arbre de référence.
_GREP_IGNORES = frozenset(
    {".git", "__pycache__", "node_modules", ".pytest_cache", "venv"} | _DOSSIERS_MASQUES
)
_GREP_EXTENSIONS = (".py", ".md", ".jinja", ".json", ".toml", ".cfg", ".txt", ".sh", ".yml")


async def _grep(task: dict, args: dict) -> str:
    """Cherche un texte littéral et rend `fichier:ligne: contenu`.

    Écrit en Python plutôt que délégué à `grep` : le motif vient du modèle, et le passer
    à un shell reviendrait à composer une commande avec son entrée. Ici il n'est qu'une
    sous-chaîne comparée en mémoire — aucune commande, aucune échappatoire.

    C'est l'outil qui manquait à une tâche de code. Sans lui, chercher où quelque chose est
    défini se fait en déroulant les dossiers un `list_dir` à la fois : mesuré sur le
    deuxième cycle réel, sept listages, une divagation hors cible, puis une tentative
    d'écrire un script de recherche — qu'aucun outil n'aurait pu exécuter.
    """
    motif = (args.get("pattern") or "").strip()
    if not motif:
        return "Erreur : pattern vide."

    cible = resolve(task["id"], args.get("path") or ".", write=False)

    trouves, scannes, tronque = [], 0, False
    # Un chemin de FICHIER cherche dans ce fichier, pas dans son dossier. Remonter au
    # parent rendrait des occurrences venues d'ailleurs, que le modèle attribuerait au
    # fichier qu'il avait nommé.
    parcours = (
        [(os.path.dirname(cible), [], [os.path.basename(cible)])]
        if os.path.isfile(cible)
        else os.walk(cible)
    )
    for dossier, sous_dossiers, fichiers in parcours:
        sous_dossiers[:] = [d for d in sous_dossiers if d not in _GREP_IGNORES]
        for nom in sorted(fichiers):
            if not nom.endswith(_GREP_EXTENSIONS) and not os.path.isfile(cible):
                continue
            scannes += 1
            if scannes > _GREP_FICHIERS_MAX:
                tronque = True
                break
            chemin = os.path.join(dossier, nom)
            try:
                with open(chemin, encoding="utf-8", errors="replace") as f:
                    for numero, ligne in enumerate(f, 1):
                        if motif in ligne:
                            trouves.append(
                                f"{relative(task['id'], chemin)}:{numero}: "
                                f"{ligne.strip()[:_GREP_LIGNE_MAX]}"
                            )
                            if len(trouves) >= _GREP_MAX_RESULTATS:
                                tronque = True
                                break
            except OSError:
                continue
            if tronque:
                break
        if tronque:
            break

    if not trouves:
        # Le modèle échappe ses crochets par réflexe d'expression régulière — mesuré :
        # `etape\[` cherché tel quel, zéro résultat. Plutôt que d'accepter les regex (et
        # leur risque d'emballement sur deux cents fichiers), on le lui dit : la recherche
        # est littérale, et voici le motif dépouillé de ses échappements.
        nu = motif.replace("\\", "")
        indice = (
            f" La recherche est LITTÉRALE, pas une expression régulière — réessaie avec "
            f"{nu!r}."
            if nu != motif else ""
        )
        return (
            f"Aucune occurrence de {motif!r} dans {relative(task['id'], cible)} "
            f"({scannes} fichier(s) parcouru(s)). Vérifie l'orthographe, ou cherche plus "
            f"court.{indice}"
        )
    entete = f"{len(trouves)} occurrence(s) de {motif!r}"
    if tronque:
        entete += f" (arrêté à {_GREP_MAX_RESULTATS} — affine ta recherche)"
    return _truncate(entete + " :\n" + "\n".join(trouves))


def _references_du_fichier(
    chemin: str, nom: str
) -> tuple[list[tuple[int, str]], int, set[str]]:
    """Références à `nom` : (ligne, forme), la ligne de définition, et les noms importés.

    Les noms importés DEPUIS `nom` sont rendus à part parce qu'un module n'est presque
    jamais appelé sous son propre nom : `from m import f` puis `f(…)` est la forme
    courante, et s'arrêter à la ligne d'import ferait conclure que personne ne s'en sert.

    Passe par l'AST et non par le texte parce qu'un import et son usage ne partagent
    aucune chaîne : `from m import f` puis `f(...)` est la forme la plus courante, et
    une recherche littérale sur `m` ne la voit pas.
    """
    try:
        with open(chemin, encoding="utf-8", errors="replace") as f:
            arbre = ast.parse(f.read())
    except (OSError, SyntaxError, ValueError):
        return [], 0, set()

    refs: list[tuple[int, str]] = []
    definit = 0
    importes: set[str] = set()

    for noeud in ast.walk(arbre):
        if isinstance(noeud, ast.Import):
            for alias in noeud.names:
                if alias.name.split(".")[0] == nom or alias.name == nom:
                    refs.append((noeud.lineno, f"import {alias.name}"))
        elif isinstance(noeud, ast.ImportFrom):
            module = noeud.module or ""
            if module.split(".")[-1] == nom or module == nom:
                noms = [a.name for a in noeud.names]
                importes.update(noms)
                refs.append((noeud.lineno, f"from {module} import {', '.join(noms)}"))
            else:
                for alias in noeud.names:
                    if alias.name == nom:
                        refs.append((noeud.lineno, f"from {module} import {alias.name}"))
        elif isinstance(noeud, ast.Call):
            cible = noeud.func
            if isinstance(cible, ast.Name) and cible.id == nom:
                refs.append((noeud.lineno, f"{nom}(…)"))
            elif isinstance(cible, ast.Attribute) and cible.attr == nom:
                porteur = getattr(cible.value, "id", "…")
                refs.append((noeud.lineno, f"{porteur}.{nom}(…)"))
        elif isinstance(noeud, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if noeud.name == nom:
                definit = noeud.lineno

    return sorted(set(refs)), definit, importes


async def _appelants(task: dict, args: dict) -> str:
    """Qui importe un module ou appelle une fonction, dans tout le dépôt.

    Un fichier ne dit pas ce qu'il fait au système : un module dangereux qu'aucun appelant
    ne touche est inerte, le même module appelé sur chaque requête ne l'est pas. Cette
    différence ne se lit jamais dans le fichier lui-même.

    Rend les références au format `fichier:ligne: forme`, comme `grep` : c'est ce format
    qui permet ensuite de rouvrir l'endroit exact avec `read_file`.
    """
    nom = (args.get("nom") or "").strip()
    if not nom:
        return "Erreur : nom vide. Donne un module (ex: 'vitals') ou une fonction."
    # Un chemin pointé désigne un module : seul son dernier segment est un identifiant.
    nom = os.path.basename(nom).removesuffix(".py").split(".")[-1]
    if not nom.isidentifier():
        return f"Erreur : {nom!r} n'est pas un nom de module ni de fonction."

    racine = resolve(task["id"], args.get("path") or ".", write=False)

    fichiers_py: list[str] = []
    for dossier, sous_dossiers, fichiers in os.walk(racine):
        sous_dossiers[:] = [d for d in sous_dossiers if d not in _GREP_IGNORES]
        fichiers_py += [
            os.path.join(dossier, f) for f in sorted(fichiers) if f.endswith(".py")
        ]
    fichiers_py = fichiers_py[:_GREP_FICHIERS_MAX]

    def _passe(cible: str) -> tuple[list[str], list[str], set[str]]:
        trouves, defs, importes = [], [], set()
        for chemin in fichiers_py:
            refs, definit, depuis = _references_du_fichier(chemin, cible)
            relatif = relative(task["id"], chemin)
            if definit:
                defs.append(f"{relatif}:{definit}: définit {cible}")
            importes |= depuis
            for ligne, forme in refs[:_APPELANTS_PAR_FICHIER]:
                trouves.append(f"{relatif}:{ligne}: {forme}")
        return trouves, defs, importes

    appelants, definitions, importes = _passe(nom)

    # Un module n'est presque jamais appelé sous son propre nom : s'arrêter aux lignes
    # d'import ferait conclure que personne ne s'en sert alors que les symboles importés
    # sont, eux, appelés ailleurs. On remonte donc jusqu'à l'usage réel.
    usages: list[str] = []
    for symbole in sorted(importes):
        if symbole == nom or not symbole.isidentifier():
            continue
        trouves, _, _ = _passe(symbole)
        usages += [t for t in trouves if t.endswith(f"{symbole}(…)")]

    if not appelants and not usages:
        constat = (
            f"Aucun fichier n'importe ni n'appelle {nom!r} ({len(fichiers_py)} fichier(s) "
            f"Python parcouru(s))."
        )
        if definitions:
            constat += (
                "\n" + "\n".join(definitions)
                + f"\n\n{nom} est donc défini mais jamais atteint depuis le reste du code."
            )
        return constat

    entete = f"{len(appelants)} référence(s) à {nom!r} hors définition"
    corps = definitions + appelants
    if usages:
        entete += f", et {len(usages)} appel(s) de ce qu'il exporte"
        corps += ["", "Appels des symboles importés depuis ce module :"] + usages
    return _truncate(f"{entete} :\n" + "\n".join(corps))


async def _verify(task: dict, args: dict) -> str:
    """Compile, lint et suite unitaire sur le worktree de la tâche.

    Passe par le MÊME bac à sable que le shell, et c'est la raison d'être de ce détour :
    l'outil exécute du code que l'agent vient d'écrire. Le lancer en direct reviendrait à
    lui donner l'exécution arbitraire sous le compte de l'utilisateur, par le simple fait
    d'écrire un fichier de test — précisément ce que seatbelt existe pour empêcher.

    Ce qui change par rapport à `shell`, ce n'est donc pas le confinement : c'est QUI
    compose la commande. Ici, personne — elle est fixe.
    """
    from . import shell as sh

    sortie = await sh.executer(task, commande_verification(), _VERIFY_TIMEOUT)
    return _truncate(sortie)


async def _shell(task: dict, args: dict) -> str:
    from . import shell as sh

    sortie = await sh.executer(task, args.get("cmd") or "", float(args.get("timeout") or 0))
    restantes = sh.commandes_restantes(task)
    if restantes <= 5:
        sortie += f"\n\n[{restantes} commande(s) restante(s) pour cette tâche]"
    return _truncate(sortie)


def _rendre_dossier(task: dict, path: str) -> tuple[str, list[str]]:
    """Le contenu d'un dossier, et la liste de ses sous-dossiers."""
    entries = sorted(
        e for e in os.listdir(path)
        if e not in _FICHIERS_INTERNES and e not in _DOSSIERS_MASQUES
    )
    if not entries:
        return f"{relative(task['id'], path)} : (dossier vide)", []
    lines, sous = [], []
    for name in entries[:200]:
        full = os.path.join(path, name)
        if os.path.isdir(full):
            lines.append(f"{name}/")
            sous.append(full)
        else:
            lines.append(f"{name}  ({os.path.getsize(full)} o)")
    suffix = f"\n[…{len(entries) - 200} entrées de plus]" if len(entries) > 200 else ""
    return f"{relative(task['id'], path)} :\n" + "\n".join(lines) + suffix, sous


async def _list_dir(task: dict, args: dict) -> str:
    """Contenu d'un dossier, en traversant les couloirs.

    Un dossier qui ne contient qu'un sous-dossier n'apprend rien : il indique seulement où
    aller. La racine d'un workspace rend `repo/`, qui rend `jarvis-core/`, qui rend `src/`
    — trois allers-retours pour atteindre le premier dossier qui porte quelque chose, et
    trois résultats de cinquante octets occupant trois pas sur quarante. On descend donc
    tant qu'il n'y a rien d'autre à voir, en rendant chaque palier traversé.
    """
    path = resolve(task["id"], args.get("path") or ".", write=False)
    if not os.path.isdir(path):
        return f"Pas un dossier : {args.get('path')}"

    rendus = []
    vus: set[str] = set()
    for _ in range(_COULOIR_MAX_PALIERS):
        rendu, sous = _rendre_dossier(task, path)
        rendus.append(rendu)
        vus.add(path)
        # Un seul sous-dossier ET aucun fichier : le palier ne porte que le chemin.
        if len(sous) != 1 or rendu.count("\n") != 1 or sous[0] in vus:
            break
        path = sous[0]
    return "\n\n".join(rendus)


# Plafond de la carte. Les modules réels du dépôt en comptent moins de cinquante ; au-delà
# c'est un fichier généré ou une agglomération, et lister tout n'aiderait plus personne.
_CARTE_MAX_ENTREES = 120


_MARQUE_DOCSTRING = '"""[docstring retiré]"""'


def _sans_docstrings(source: str) -> str:
    """Vide les docstrings en conservant la numérotation des lignes.

    Un docstring est ce qu'un module dit de lui-même, et cela se lit comme un fait alors
    que rien ne le garantit : deux revues ont conclu sur la foi d'un docstring — l'une en
    tenant un mécanisme pour justifié parce qu'il s'y déclarait tel, l'autre en prenant
    pour le comportement courant une phrase qui décrivait le comportement écarté. Le code,
    lui, ne se décrit pas : il fait.

    Les lignes sont blanchies et non retirées, sinon les numéros glissent — or ce sont eux
    que la citation porte et que le relecteur humain ouvre. Les commentaires `#` restent :
    ils portent les invariants qu'on ne doit pas « corriger », et se lisent au contact du
    code qu'ils commentent.

    Rend la source inchangée si elle ne s'analyse pas — un fichier en cours d'édition se
    lit tel quel plutôt que pas du tout.
    """
    try:
        arbre = ast.parse(source)
    except (SyntaxError, ValueError):
        return source

    portees = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    plages: list[tuple[int, int]] = []
    for noeud in ast.walk(arbre):
        if not isinstance(noeud, portees):
            continue
        corps = getattr(noeud, "body", None)
        if not corps:
            continue
        tete = corps[0]
        if (isinstance(tete, ast.Expr) and isinstance(tete.value, ast.Constant)
                and isinstance(tete.value.value, str)):
            plages.append((tete.lineno, tete.end_lineno or tete.lineno))

    if not plages:
        return source

    lignes = source.splitlines(keepends=True)
    for debut, fin in plages:
        if debut > len(lignes):
            continue
        brute = lignes[debut - 1]
        indentation = brute[:len(brute) - len(brute.lstrip())]
        lignes[debut - 1] = f"{indentation}{_MARQUE_DOCSTRING}\n"
        for i in range(debut, min(fin, len(lignes))):
            lignes[i] = "\n"
    return "".join(lignes)


def _carte_du_module(source: str, nb_lignes: int) -> str | None:
    """Plan d'un module trop gros pour une lecture : définitions et leurs lignes.

    Rend None si le fichier tient dans le budget, ou s'il ne s'analyse pas — dans les deux
    cas la lecture ordinaire fait mieux.
    """
    if len(source) <= AGENT_READ_MAX_CHARS:
        return None
    try:
        arbre = ast.parse(source)
    except SyntaxError:
        return None

    entrees = []
    for noeud in arbre.body:
        if isinstance(noeud, ast.ClassDef):
            entrees.append(f"{noeud.lineno:5}  class {noeud.name}")
            entrees.extend(
                f"{m.lineno:5}      {m.name}()"
                for m in noeud.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
        elif isinstance(noeud, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefixe = "async def" if isinstance(noeud, ast.AsyncFunctionDef) else "def"
            entrees.append(f"{noeud.lineno:5}  {prefixe} {noeud.name}")
    if not entrees:
        return None

    # La carte est bornée elle aussi : un module de milliers de définitions en produirait
    # une aussi lourde que le fichier, ce qui réintroduirait le problème qu'elle corrige.
    reste = 0
    if len(entrees) > _CARTE_MAX_ENTREES:
        reste = len(entrees) - _CARTE_MAX_ENTREES
        entrees = entrees[:_CARTE_MAX_ENTREES]

    pied = (
        f"\n… et {reste} définition(s) de plus. Ce module est trop gros pour être lu "
        f"d'un bloc : vise ce que tu cherches."
        if reste else ""
    )
    return (
        f"[CARTE DU MODULE — {nb_lignes} lignes, trop long pour une lecture entière.\n"
        f"Ci-dessous ses définitions et leur ligne. Relis avec offset=<ligne> pour ouvrir "
        f"ce qui t'intéresse.\n"
        f"Lire le fichier en entier remplirait ton contexte et ferait disparaître ce que "
        f"tu as déjà lu.]\n\n" + "\n".join(entrees) + pied
    )


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

    # Une revue de code lit le code. Le docstring est ce qu'un module dit de lui-même, et
    # il a déjà emporté deux conclusions : la numérotation est conservée, seul le contenu
    # part. Réservé à l'origine `autocode` — une tâche humaine lit des sources dont la
    # prose EST le contenu.
    if task.get("origin") == "autocode" and path.endswith(".py"):
        lines = _sans_docstrings("".join(lines)).splitlines(keepends=True)

    if offset > len(lines):
        return f"Rien à lire à partir de la ligne {offset} (le fichier en compte {len(lines)})."

    # Un fichier qui ne tient pas en une lecture rend sa CARTE, pas ses 32 000 premiers
    # caractères. Mesuré sur le premier cycle d'autocoding réel : l'agent a ouvert un
    # module de 76 ko hors de sa cible, ce qui a rempli les trois quarts du contexte et
    # provoqué l'élision du fichier qu'il devait réellement examiner — qu'il a donc relu,
    # évinçant autre chose à son tour. Cinq relectures du même fichier en quatorze pas.
    #
    # La pagination n'était pas la cause : le fichier utile tenait en une lecture. La cause
    # est qu'un gros fichier coûte le contexte entier alors qu'on y cherche un appel précis.
    # La carte coûte ~2 ko et mène directement à l'offset utile.
    if offset == 1 and not int(args.get("limit") or 0) and path.endswith(".py"):
        if (carte := _carte_du_module("".join(lines), len(lines))):
            return carte

    # Le budget est en CARACTÈRES, pas en lignes. Un plafond de 200 lignes rendait 9 000
    # caractères là où 15 000 étaient permis, et obligeait à paginer un fichier de 525
    # lignes en trois appels — que le modèle a préféré rejouer à l'identique.
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
    # caractères de code, il est noyé — mesuré, ignoré quatre fois de suite.
    remaining = len(lines) - next_offset + 1

    # Deux coupures possibles, et elles n'appellent pas la même suite. Quand c'est `limit`
    # qui a coupé alors que le reste tenait dans le budget, renvoyer l'offset suivant
    # entérine une pagination inutile : le modèle fixe une limite basse par réflexe, puis
    # paie un tour par tranche — un module de cinq cents lignes lu soixante par soixante
    # consomme dix tours sur quarante pour un contenu qui venait d'un seul bloc.
    reste = sum(len(f"{i}\t{ligne}") for i, ligne in enumerate(lines[offset - 1:], offset))
    if limit and reste <= budget:
        warning = (
            f"[COUPÉ PAR TON limit={limit} — lignes {offset} à {next_offset - 1} sur "
            f"{len(lines)}. Le fichier tient ENTIER dans une seule lecture : relance "
            f"read_file sur ce chemin sans `limit`, au lieu de paginer.]"
        )
    else:
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
    # un rapport bâti en cinq pas (11 ko) a été réduit à sa seule section
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
    # markdown — mesuré : « [5] https://…html## Contexte juridique ». Le
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
    # ne se souvient pas de ce qu'il a déjà posé : a l'usage il a réécrit deux
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
    VERIFY: _verify,
    GREP: _grep,
    APPELANTS: _appelants,
}


async def execute_tool(task: dict, name: str, args: dict) -> str:
    """Exécute un outil et renvoie TOUJOURS une chaîne destinée au modèle.

    Aucune exception ne remonte : un outil qui échoue doit donner à l'agent de quoi
    corriger son tir au pas suivant, pas tuer la tâche. Seule l'annulation et les budgets,
    décidés par la boucle, arrêtent une tâche.
    """
    # Le jeu d'outils est appliqué ICI et pas seulement à la déclaration : des schémas non
    # rendus au modèle ne sont pas une barrière — rien n'empêche une génération d'inventer
    # un nom d'outil réel mais hors de son périmètre. Le message nomme les outils de CETTE
    # tâche, jamais le catalogue complet, qui enverrait l'agent réessayer l'interdit.
    disponibles = names_for(task)
    fn = _DISPATCH.get(name) if name in disponibles else None
    if fn is None:
        return (
            f"Outil inconnu : {name}. Outils disponibles : "
            f"{', '.join(sorted(disponibles))}."
        )
    try:
        return await fn(task, args)
    except SandboxError as exc:
        return f"Refusé : {exc}"
    except Exception as exc:
        logger.warning("agent: outil %s en échec (%s: %s)", name, type(exc).__name__, exc)
        return f"L'outil {name} a échoué : {type(exc).__name__}: {exc}"

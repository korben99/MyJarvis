"""Phase 5 — le rapport. Un appel LLM qui explique, et qui ne juge pas.

Le verdict est calculé par `mesure.verdict()` avant que ce module soit appelé. Le modèle
le reçoit comme une donnée, au même titre que le diff : il rédige de quoi relire, il ne
rattrape rien. C'est ce qui empêche une prose convaincante de présenter un patch cassé
comme un succès.

Le rapport reste lisible si l'appel échoue : les mesures, elles, sont déjà là.
"""

from config import (
    AUTOCODE_MAX_DIFF_LINES,
    DEFAULT_TEMP,
    MAX_TOKENS_THINK_MEDIUM,
    REASONING_API_KEY,
    REASONING_API_URL,
    REASONING_MODEL,
    THINKING_BUDGET_MEDIUM,
    llm_timeout,
)
from helpers import call_llm_async_bg, extract_llm_json, get_logger
from prompts import get_prompt

from . import mesure as mod_mesure
from . import store

logger = get_logger("jarvis-autocode")

# Plafond du diff soumis au modèle. Un patch qui dépasse est de toute façon déjà rejeté
# mécaniquement : inutile de payer le contexte pour le faire commenter.
_DIFF_MAX_CARS = 24000

_PARCOURS = {
    mod_mesure.CORRIGE: "rouge sur HEAD, vert avec le correctif — un patch à relire",
    mod_mesure.REPRODUIT: "rouge sur HEAD, et encore rouge — défaut prouvé, pas corrigé",
    mod_mesure.GARDE: (
        "aucun défaut, la propriété tient — un test de garde à relire, "
        "la suite ne l'avait pas"
    ),
    mod_mesure.SIGNALE: (
        "aucun test, mais des constats sourcés — une revue à lire, pas un patch"
    ),
    mod_mesure.RIEN_TROUVE: "rien n'a pu être étayé — aucun livrable",
    mod_mesure.REJETE: "irrecevable — rien à appliquer",
}


def _parcours(verdict: str) -> str:
    """Chaque verdict dit aussi s'il y a quelque chose à relire : c'est ce que le
    relecteur cherche en tête de rapport."""
    return _PARCOURS.get(verdict, verdict)


def _fmt_test(m: dict) -> str:
    """Ce que le test neuf a démontré, avec la cause quand il n'a rien démontré."""
    if not m.get("tests_ajoutes"):
        return "aucun test ajouté"
    avant, apres = m.get("avant"), m.get("apres")
    if avant == mod_mesure.ERREUR:
        if apres == mod_mesure.PASSE:
            # À signaler explicitement : le relecteur doit vérifier que le symbole ajouté
            # a un sens, puisque le test ne pouvait pas tourner avant.
            return (
                "ne démarrait pas sur HEAD (symbole absent), passe avec le correctif — "
                "vérifie que ce qui a été ajouté répond bien au constat"
            )
        return "échoue sur une erreur (import, collecte ou fixture), pas sur une assertion"
    if avant == mod_mesure.PASSE:
        if mod_mesure.garde_proposee(m):
            return (
                "passe sur HEAD — aucun défaut, et le test devient un garde que la suite "
                "n'avait pas"
            )
        return "passe déjà sur HEAD — il décrit ce qui marchait"
    if avant is None:
        return "n'a pas pu être éprouvé contre HEAD"
    return (
        "échoue sur HEAD (assertion), "
        + ("passe avec le correctif" if apres == mod_mesure.PASSE else "échoue encore")
    )


async def rediger(constat: dict, m: dict, verdict: str, motifs: list[str],
                  diff: str, resume_agent: str) -> dict:
    """{resume, risques, angles_morts}. Jamais d'exception : champs vides en cas d'échec."""
    vide = {"resume": "", "risques": [], "angles_morts": []}

    soumis = diff[:_DIFF_MAX_CARS]
    if len(diff) > _DIFF_MAX_CARS:
        soumis += f"\n[…tronqué — {len(diff) - _DIFF_MAX_CARS} caractères restants]"

    prompt = get_prompt("AUTOCODE_VERDICT_USER").format(
        constat_id=constat["id"],
        constat=constat["constat"],
        raison=constat.get("raison", ""),
        verdict=verdict,
        motifs="; ".join(motifs) or "aucun",
        parcours=_parcours(verdict),
        test=_fmt_test(m),
        fichiers=", ".join(m["fichiers"]) or "aucun",
        lignes_source=m["lignes_source"],
        suite="verte" if m["suite_ok"] else "rouge",
        pyflakes_avant=m["pyflakes_avant"],
        pyflakes_apres=m["pyflakes_apres"],
        resume_agent=resume_agent or "(rien)",
        diff=soumis,
    )

    try:
        contenu = await call_llm_async_bg(
            [
                {"role": "system", "content": get_prompt("AUTOCODE_VERDICT_SYSTEM")},
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
        brut = extract_llm_json(contenu)
    except Exception as exc:
        logger.warning("autocode: rédaction en échec (%s)", type(exc).__name__)
        return vide

    return {
        "resume": str(brut.get("resume") or ""),
        "risques": [str(x) for x in (brut.get("risques") or []) if x],
        "angles_morts": [str(x) for x in (brut.get("angles_morts") or []) if x],
    }


def _liste(titre: str, items: list[str]) -> str:
    if not items:
        return ""
    return f"\n## {titre}\n\n" + "\n".join(f"- {x}" for x in items) + "\n"


def rendre_rapport(constat: dict, m: dict, verdict: str, motifs: list[str],
                   redaction: dict, task: dict, dossier: str = "autocode/<ce dossier>") -> str:
    """RAPPORT.md — ce que l'administrateur lit avant d'ouvrir le diff.

    Les mesures viennent AVANT la prose : c'est le verdict calculé qui décide, et il doit
    se lire sans avoir à faire confiance à ce qui suit.
    """
    entete = [
        f"# {constat['id']} — {constat['constat']}",
        "",
        f"**Verdict : {verdict}** — {_parcours(verdict)}",
        "",
        "| Mesure | Valeur |",
        "|---|---|",
        f"| Test ajouté | {_fmt_test(m)} |",
        f"| Suite unitaire | {'verte' if m['suite_ok'] else 'rouge'}"
        + (" (sans le test neuf : "
           + {True: 'verte', False: 'rouge', None: 'non mesurée'}[m.get('suite_hors_test_ok')]
           + ")" if m.get("tests_ajoutes") and m.get("apres") != mod_mesure.PASSE else "")
        + " |",
        f"| Compilation | {'ok' if m['compile_ok'] else 'ÉCHEC'} |",
        f"| pyflakes | {m['pyflakes_avant']} → {m['pyflakes_apres']} |",
        f"| Lignes de correctif | {m['lignes_source']} / {AUTOCODE_MAX_DIFF_LINES} |",
        f"| Fichiers touchés | {', '.join(m['fichiers']) or 'aucun'} |",
        f"| Pas consommés | {task.get('steps', 0)} |",
        *([
            f"| Constats écrits | {len(m['constats_ecrits'])}, "
            f"{len(m['citations'])} situé(s) par fichier:ligne |"
        ] if m.get("constats_ecrits") else []),
        "",
        "## Motifs du verdict",
        "",
        "\n".join(f"- {x}" for x in motifs) or "- aucun",
        "",
        "## Pourquoi ce constat cette nuit",
        "",
        constat.get("raison") or "(non dit)",
        "",
        f"**Angle annoncé :** {constat.get('angle') or '(non dit)'}",
        f"**Origine du constat :** {constat.get('origine', '?')}",
        "",
    ]

    corps = ""
    if redaction.get("resume"):
        corps += f"## Ce que fait le patch\n\n{redaction['resume']}\n"
    corps += _liste("Risques", redaction.get("risques", []))
    corps += _liste("Angles morts", redaction.get("angles_morts", []))

    # Un rapport sans livrable ne propose pas de l'appliquer : `rejeté` et `rien trouvé`
    # ne laissent rien sur l'étagère, et offrir un `git apply` sur un patch vide envoyait
    # le relecteur chercher ce qui n'existe pas.
    if verdict in (mod_mesure.REJETE, mod_mesure.RIEN_TROUVE):
        pied = [
            "",
            "## Rien à appliquer",
            "",
            "Ce cycle ne laisse pas de patch. Le constat est remis en sommeil, et le "
            "cycle suivant repartira sur un autre.",
        ]
    elif verdict == mod_mesure.SIGNALE:
        # Une revue ne s'applique pas : elle se lit. Proposer `git apply` ici enverrait le
        # relecteur vers un diff vide, le livrable étant le fichier de constats.
        pied = [
            "",
            "## Lire les constats",
            "",
            "```bash",
            f"cat {dossier}/{mod_mesure.FICHIER_REVUE}",
            "```",
            "",
            "Chaque constat porte son `fichier:ligne` : ouvre-le pour trancher, rien ne "
            "vérifie la citation à ta place. Réponds « accepte le patch "
            f"{constat['id']} » une fois lu, ou « rejette le patch {constat['id']} » si "
            "rien ne mérite d'être retenu.",
        ]
    else:
        pied = [
            "",
            "## Appliquer",
            "",
            "```bash",
            f"cd /opt/jarvis && git apply --check {dossier}/3-patch.diff",
            "```",
            "",
            "Rien n'a été appliqué. Réponds « accepte le patch "
            f"{constat['id']} » ou « rejette le patch {constat['id']} » dans le chat pour "
            "que le cycle suivant en tienne compte.",
        ]
    return "\n".join(entete) + corps + "\n".join(pied) + "\n"


def entree_journal(constat: dict, m: dict, verdict: str, task: dict) -> dict:
    """Ce qui est consigné, et relu par le choix du run suivant.

    Volontairement maigre : un id, un verdict, une raison. Le détail vit dans le dossier —
    le charger dans un prompt à chaque nuit le noierait.
    """
    return {
        "constat_id": constat["id"],
        "constat": constat["constat"][:160],
        "origine": constat.get("origine", "?"),
        "verdict": verdict,
        "raison": constat.get("raison", ""),
        "task_id": task["id"],
        "lignes_source": m["lignes_source"],
        "at": store.now_iso(),
    }

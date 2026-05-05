"""
test_thinking_budget.py — Vérifie que <budget_remaining> contraint réellement le thinking.

Usage :
  cd /opt/jarvis
  source venv/bin/activate
  python scripts/test_thinking_budget.py [--budget N] [--rounds N]

Ce que mesure le script :
  - Tokens réels dans le bloc <think>…</think>
  - Tokens de réponse visible
  - Total généré
  - Respect du budget : thinking_tokens <= budget + marge_10%

Deux rounds consécutifs par défaut : budget=600 puis budget=0 (libre).
Permet de voir si la contrainte est réellement respectée.
"""

import argparse
import asyncio
import os
import sys
import time

# ── path bootstrap ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jarvis-core", "src"))

# ── force LLM_LOCAL before any jarvis import ───────────────────────────────
os.environ.setdefault("LLM_LOCAL", "yes")

from config import PRIMARY_MODEL, THINKING_BUDGET_TOKENS  # noqa: E402
from llm_local import _build_prompt, _load_model  # noqa: E402

try:
    from mlx_lm import generate as mlx_generate
except ImportError:
    print("ERROR: mlx_lm not found — run inside the Jarvis venv")
    sys.exit(1)


PROMPT_SIMPLE = [
    {
        "role": "system",
        "content": "Tu es Jarvis, un assistant IA. Réponds en JSON uniquement.",
    },
    {
        "role": "user",
        "content": (
            "Analyse cette phrase et retourne un JSON avec les clés : "
            '"sujet", "verbe", "objet", "sentiment" (positif/neutre/négatif).\n\n'
            'Phrase : "J\'adore programmer en Python, c\'est vraiment élégant."'
        ),
    },
]

# Prompt complexe : analyse multi-sessions similaire à l'analyzer Jarvis
PROMPT_COMPLEX = [
    {
        "role": "system",
        "content": "Tu es un assistant d'analyse. Réponds en JSON uniquement, sans markdown.",
    },
    {
        "role": "user",
        "content": """\
Analyse la conversation suivante et extrait les informations clés.

Utilisateur : J'ai finalement décidé de quitter mon poste chez Accenture après 7 ans. \
Je rejoins une startup FinTech à Lyon comme CTO. Le salaire est moins bon mais c'est \
exactement ce que je voulais faire. Ma femme est inquiète pour la stabilité financière \
mais elle me soutient. On a un crédit immobilier qui court encore 18 ans.
Jarvis : C'est un grand changement ! Qu'est-ce qui a motivé cette décision maintenant ?
Utilisateur : Le rachat par Capgemini a tout changé. La culture s'est dégradée, \
les projets sont moins intéressants. Et cette startup développe une plateforme de \
trading algorithmique pour les PME, exactement dans mes cordes.
Jarvis : Passionnant. Tu as négocié des stock-options ?
Utilisateur : Oui, 0.8% du capital avec un vesting sur 4 ans. Si ça marche, \
ça compense largement la baisse de salaire.

Retourne ce JSON :
{
  "topics": ["liste", "de", "sujets"],
  "mood": "happy|neutral|stressed|curious|focused|frustrated|tired",
  "user_facts": [{"key": "clé_profil", "value": "valeur"}],
  "projects": ["liste de projets mentionnés"],
  "memory_summary": "résumé en 1-2 phrases de ce qui mérite d'être retenu",
  "should_remember": true
}""",
    },
]

PROMPT_MESSAGES = PROMPT_SIMPLE  # défaut


def count_think_tokens(raw: str, tokenizer) -> tuple[int, int]:
    """
    Retourne (think_tokens, response_tokens) depuis le raw output.

    Le prompt se termine déjà par '<think>\\n' (ou '...<budget_remaining>N</budget_remaining>\\n'),
    donc la génération commence DANS le bloc think. Le raw output ne contient PAS de '<think>'
    ouvrant — il commence directement par le contenu du raisonnement.

    Formats possibles :
      "raisonnement...\\n</think>\\n\\nréponse"   → normal
      "</think>\\n\\nréponse"                      → think immédiatement fermé (budget=0 ou skip)
      "raisonnement..."                             → budget épuisé avant </think>
    """
    if "</think>" in raw:
        think_part = raw.split("</think>", 1)[0]
        response_part = raw.split("</think>", 1)[1].lstrip("\n")
    else:
        # Pas de </think> → tout est du thinking (budget épuisé avant fermeture)
        think_part = raw
        response_part = ""

    think_tokens = len(tokenizer.encode(think_part))
    resp_tokens  = len(tokenizer.encode(response_part))
    return think_tokens, resp_tokens


def run_one(budget: int, max_tokens: int, model_path: str) -> dict:
    """Lance une inférence complète et retourne les métriques."""
    model, tokenizer = _load_model(model_path)
    prompt = _build_prompt(PROMPT_MESSAGES, tokenizer, model_path, no_think=False, thinking_budget=budget)

    # Show what was injected at the end of the prompt
    tail = prompt[-150:]
    budget_tag_present = "<budget_remaining>" in tail

    print(f"\n{'─'*60}")
    print(f"  thinking_budget = {budget}  |  max_tokens = {max_tokens}")
    print(f"  <budget_remaining> in prompt tail : {budget_tag_present}")
    if budget_tag_present:
        # Extract the tag to show the injected value
        import re
        m = re.search(r"<budget_remaining>(\d+)</budget_remaining>", tail)
        if m:
            print(f"  injected value : {m.group(1)}")
    print(f"{'─'*60}")

    t0 = time.time()
    raw_output = mlx_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    elapsed = time.time() - t0

    think_tok, resp_tok = count_think_tokens(raw_output, tokenizer)
    total_tok = think_tok + resp_tok
    budget_respected = think_tok <= budget * 1.10 if budget > 0 else True  # 10% marge

    print(f"  Elapsed          : {elapsed:.1f}s")
    print(f"  Think tokens     : {think_tok}")
    print(f"  Response tokens  : {resp_tok}")
    print(f"  Total generated  : {total_tok}")
    if budget > 0:
        ratio = think_tok / budget if budget else 0
        status = "✓ RESPECTÉ" if budget_respected else "✗ DÉPASSÉ"
        print(f"  Budget ratio     : {think_tok}/{budget} = {ratio:.1%}  →  {status}")
    else:
        print(f"  Budget ratio     : libre (budget=0)")

    # Dump raw (first 500 chars) pour vérification
    print(f"\n  --- Raw output (500 chars) ---")
    print(f"  {repr(raw_output[:500])}")

    print(f"\n  --- Réponse visible ---")
    if "</think>" in raw_output:
        visible = raw_output.split("</think>", 1)[1].strip()
    else:
        visible = raw_output.strip()
    print(f"  {visible[:300]}")

    return {
        "budget": budget,
        "max_tokens": max_tokens,
        "think_tokens": think_tok,
        "resp_tokens": resp_tok,
        "total_tokens": total_tok,
        "budget_respected": budget_respected,
        "elapsed": elapsed,
        "budget_tag_injected": budget_tag_present,
    }


def main():
    parser = argparse.ArgumentParser(description="Test thinking budget on Qwen3.6")
    parser.add_argument("--budgets", type=int, nargs="+", default=[600],
                        help="Budget values to test (default: 600). Ex: --budgets 100 300 600 0")
    parser.add_argument("--max-tokens", type=int, default=2000,
                        help="max_tokens for all runs (default: 2000)")
    parser.add_argument("--model", type=str, default=PRIMARY_MODEL,
                        help=f"Model path (default: {PRIMARY_MODEL})")
    parser.add_argument("--complex", action="store_true",
                        help="Use complex analyzer-like prompt instead of simple JSON")
    args = parser.parse_args()

    global PROMPT_MESSAGES
    if args.complex:
        PROMPT_MESSAGES = PROMPT_COMPLEX
        prompt_label = "COMPLEXE (analyzer-like)"
    else:
        PROMPT_MESSAGES = PROMPT_SIMPLE
        prompt_label = "SIMPLE (JSON extraction)"

    print(f"\n{'='*60}")
    print(f"  Thinking Budget Test — {args.model.split('/')[-1]}")
    print(f"  THINKING_BUDGET_TOKENS (env) = {THINKING_BUDGET_TOKENS}")
    print(f"  Prompt type  = {prompt_label}")
    print(f"  max_tokens   = {args.max_tokens}")
    print(f"  Budgets testés : {args.budgets}")
    print(f"{'='*60}")

    results = []
    for b in args.budgets:
        r = run_one(b, args.max_tokens, args.model)
        results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("  RÉSUMÉ COMPARATIF")
    print(f"{'='*60}")
    print(f"  {'Budget':>10}  {'Think tok':>10}  {'Resp tok':>10}  {'Total':>7}  {'Respecté':>10}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*7}  {'─'*10}")
    for r in results:
        label = str(r["budget"]) if r["budget"] > 0 else "libre"
        respected = "✓" if r["budget_respected"] else "✗" if r["budget"] > 0 else "—"
        print(f"  {label:>10}  {r['think_tokens']:>10}  {r['resp_tokens']:>10}  {r['total_tokens']:>7}  {respected:>10}")

    # Comparaison vs run libre (budget=0) si présent
    free_runs = [r for r in results if r["budget"] == 0]
    constrained_runs = [r for r in results if r["budget"] > 0]
    if free_runs and constrained_runs:
        free_think = free_runs[0]["think_tokens"]
        print(f"\n  Référence libre  : {free_think} tokens de thinking")
        for r in constrained_runs:
            delta = free_think - r["think_tokens"]
            pct = delta / max(free_think, 1) * 100
            status = "✓ réduit" if delta > 0 else ("= identique" if delta == 0 else "✗ augmenté")
            print(f"  budget={r['budget']:>5} : {r['think_tokens']:>4} tok thinking  →  {status} de {delta:+d} tok ({pct:.0f}%)")

    print()


if __name__ == "__main__":
    main()

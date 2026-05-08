"""
test_prune.py — Reproduit exactement l'appel _action_prune_self_memory pour diagnostiquer
               pourquoi le modèle génère "[0]" au lieu de penser + répondre en JSON.

Usage :
  cd /opt/jarvis
  source venv/bin/activate
  python scripts/test_prune.py [--rounds N] [--no-think]

Ce que mesure le script :
  - Tokens de thinking générés (entre <think> et </think>)
  - Présence d'un </think> dans la réponse brute
  - JSON parseable ou non
  - Répartition sur N rounds pour voir si le comportement est systématique
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jarvis-core", "src"))
os.environ.setdefault("LLM_LOCAL", "yes")
os.environ.setdefault("LLM_DEBUG_PROMPTS", "no")  # on gère nous-mêmes le logging

# ── Garde GPU : refuse de lancer si Jarvis tourne déjà ────────────────────
# Charger le modèle une 2e fois pendant qu'il est déjà en RAM = ~42 GB peak
# + deux noyaux Metal simultanés → kernel panic sur Mac Mini M4 Pro.
def _check_jarvis_not_running() -> None:
    try:
        out = subprocess.check_output(["pgrep", "-f", "jarvis"], text=True)
        pids = out.strip().splitlines()
        own_pid = str(os.getpid())
        others = [p for p in pids if p != own_pid]
        if others:
            print(f"ERREUR : Jarvis tourne déjà (pids {others}).")
            print("Arrête jarvis-core avant de lancer ce test (risque de crash GPU/RAM).")
            print("  docker compose -f /opt/jarvis/docker-compose.yml stop jarvis-core")
            sys.exit(1)
    except subprocess.CalledProcessError:
        pass  # pgrep retourne exit 1 si aucun process trouvé → OK

_check_jarvis_not_running()

from config import REASONING_MODEL, THINKING_BUDGET_TOKENS  # noqa: E402
from llm_local import _build_prompt, _load_model, _model_profile  # noqa: E402

try:
    from mlx_lm import stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
except ImportError:
    print("ERROR: mlx_lm not found — run inside the Jarvis venv")
    sys.exit(1)


# ── Données de test (structure réelle de jarvis-self.json) ─────────────────

SELF_MEMORY_PATH = "/opt/jarvis/jarvis-core/JarvisData/jarvis-self.json"

def _load_self_memory() -> dict:
    with open(SELF_MEMORY_PATH) as f:
        return json.load(f)


def _fmt(items: list) -> str:
    """Reproduction exacte de _action_prune_self_memory._fmt (version actuelle — boguée)."""
    if not items:
        return "  (vide)"
    lines = []
    for i, item in enumerate(items):
        text = item.get("text", str(item)) if isinstance(item, dict) else str(item)
        date = (
            f" ({item['date']})"
            if isinstance(item, dict) and "date" in item
            else ""
        )
        lines.append(f"  [{i}] {text}{date}")
    return "\n".join(lines)


def _fmt_fixed(items: list, text_key: str = "text") -> str:
    """Version corrigée : utilise la bonne clé pour extraire le texte principal."""
    if not items:
        return "  (vide)"
    lines = []
    for i, item in enumerate(items):
        if isinstance(item, dict):
            # Essaie les clés dans l'ordre : key explicite → note → opinion → text → repr
            text = item.get(text_key) or item.get("note") or item.get("opinion") or item.get("text") or str(item)
            date = item.get("date") or item.get("created") or ""
            date_str = f" ({date})" if date else ""
        else:
            text = str(item)
            date_str = ""
        lines.append(f"  [{i}] {text}{date_str}")
    return "\n".join(lines)


SYSTEM_PROMPT = (
    "Tu es Jarvis. Tu examines ta propre mémoire personnelle pour identifier les entrées obsolètes,\n"
    "redondantes ou sans valeur durable, afin de garder uniquement ce qui est réellement utile.\n"
    "Retourne du JSON valide uniquement."
)

USER_TEMPLATE = """\
Examine ces listes de ta mémoire personnelle et identifie les entrées à supprimer.

SELF_NOTES :
{self_notes}

OPINIONS :
{opinions}

LEARNINGS :
{learnings}

Critères de suppression :
- Redondances : même idée formulée à plusieurs reprises (garder la plus précise)
- Banalités génériques sans valeur spécifique (ex: "je dois être plus attentif")
- Entrées dépassées ou contredites par des plus récentes
- Apprentissages évidents qui n'apportent rien d'actionnable

Critères de conservation (prioritaires) :
- Entrées actionables, spécifiques et datées
- Opinions fortes qui influencent le comportement de Jarvis
- Apprentissages issus d'échecs concrets

Contraintes absolues :
- Ne supprime jamais plus de 30% d'une liste en un seul passage (arrondi inférieur)
- Ne supprime pas d'entrée si la liste n'a qu'un seul élément
- Conserve toujours les entrées récentes (< 14 jours) sauf doublon évident
- En cas de doute sur la valeur d'une entrée : conserve-la

JSON uniquement :
{{"to_delete": {{"self_notes": [indices...], "opinions": [indices...], "learnings": [indices...]}}}}"""


def run_test(model_path: str, messages: list, no_think: bool, max_tokens: int, label: str) -> dict:
    """Appel brut stream_generate, retourne les stats."""
    model, tokenizer = _load_model(model_path)
    prompt = _build_prompt(messages, tokenizer, model_path, no_think=no_think, thinking_budget=0)
    profile = _model_profile(model_path)

    prompt_tokens = len(tokenizer.encode(prompt))
    print(f"\n{'─'*70}")
    print(f"[{label}] no_think={no_think} | max_tokens={max_tokens} | prompt_tokens={prompt_tokens}")
    # Show last 200 chars of prompt to verify think-block suffix
    print(f"  prompt tail: {repr(prompt[-120:])}")

    sampler = make_sampler(
        temp=profile.temp_nothink if no_think else profile.temp_think,
        top_p=profile.top_p_nothink if no_think else profile.top_p_think,
        top_k=profile.top_k,
        min_p=profile.min_p,
    )
    logits_procs = make_logits_processors(
        repetition_penalty=profile.repetition_penalty,
        repetition_context_size=profile.repetition_context_size,
        frequency_penalty=profile.frequency_penalty,
        frequency_context_size=profile.repetition_context_size,
        presence_penalty=profile.presence_penalty,
        presence_context_size=profile.repetition_context_size,
    )

    t0 = time.time()
    raw = ""
    tok_count = 0
    for chunk in stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens,
        sampler=sampler, logits_processors=logits_procs,
    ):
        if chunk.text:
            raw += chunk.text
            tok_count += 1
            # Early abort after 20 tokens if no thinking — something is wrong
            if tok_count == 20 and "</think>" not in raw and not no_think:
                print(f"  [!] First 20 tokens without </think>: {repr(raw)}")

    elapsed = time.time() - t0

    # Stats
    has_end_think = "</think>" in raw
    think_part = ""
    answer_part = raw
    if has_end_think:
        think_part = raw.split("</think>", 1)[0]
        answer_part = raw.split("</think>", 1)[1].strip()
    think_tokens = len(tokenizer.encode(think_part)) if think_part else 0

    try:
        parsed = json.loads(answer_part.strip()) if answer_part.strip() else None
        json_ok = isinstance(parsed, dict) and "to_delete" in parsed
    except Exception:
        json_ok = False
        parsed = None

    print(f"  elapsed: {elapsed:.1f}s | resp_tokens: {tok_count}")
    print(f"  has_</think>: {has_end_think} | think_tokens: {think_tokens}")
    print(f"  json_ok: {json_ok}")
    print(f"  raw (first 300): {repr(raw[:300])}")
    if has_end_think and not json_ok:
        print(f"  answer_part: {repr(answer_part[:300])}")
    if json_ok:
        print(f"  to_delete: {parsed['to_delete']}")

    return {
        "label": label,
        "no_think": no_think,
        "prompt_tokens": prompt_tokens,
        "resp_tokens": tok_count,
        "has_end_think": has_end_think,
        "think_tokens": think_tokens,
        "json_ok": json_ok,
        "raw_prefix": raw[:80],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--no-think", action="store_true")
    args = parser.parse_args()

    data = _load_self_memory()
    self_notes = data.get("self_notes", [])
    opinions = data.get("opinions", [])
    learnings = data.get("learnings", [])

    print(f"Memory loaded: {len(self_notes)} self_notes, {len(opinions)} opinions, {len(learnings)} learnings")

    # ── Construit les deux versions du prompt (bugged vs fixed) ────────────
    user_bugged = USER_TEMPLATE.format(
        self_notes=_fmt(self_notes),
        opinions=_fmt(opinions),
        learnings=_fmt(learnings),
    )
    user_fixed = USER_TEMPLATE.format(
        self_notes=_fmt_fixed(self_notes, "note"),
        opinions=_fmt_fixed(opinions, "opinion"),
        learnings=_fmt_fixed(learnings, "text"),
    )

    msgs_bugged = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_bugged},
    ]
    msgs_fixed = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_fixed},
    ]

    tok_diff = len(user_bugged) - len(user_fixed)
    print(f"\nPrompt diff (bugged vs fixed): {tok_diff:+d} chars")
    print(f"  bugged self_notes sample: {repr(_fmt(self_notes[:1]))[:120]}")
    print(f"  fixed  self_notes sample: {repr(_fmt_fixed(self_notes[:1], 'note'))[:120]}")

    max_tokens = THINKING_BUDGET_TOKENS + 2000
    results = []

    for i in range(args.rounds):
        no_think = args.no_think
        # Round 1+: alternate bugged/fixed to compare
        use_fixed = (i % 2 == 1)
        label = f"round {i+1} {'[FIXED fmt]' if use_fixed else '[BUGGED fmt]'}"
        msgs = msgs_fixed if use_fixed else msgs_bugged
        r = run_test(REASONING_MODEL, msgs, no_think=no_think, max_tokens=max_tokens, label=label)
        results.append(r)

    print(f"\n{'='*70}")
    print("SUMMARY")
    for r in results:
        ok = "✅" if r["json_ok"] else "❌"
        think = f"think={r['think_tokens']}tok" if r["has_end_think"] else "NO_</think>"
        print(f"  {ok} {r['label']} | {think} | resp={r['resp_tokens']}tok | raw={repr(r['raw_prefix'])}")


if __name__ == "__main__":
    main()

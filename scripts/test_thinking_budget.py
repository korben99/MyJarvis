"""
test_thinking_budget.py — Valide ThinkingBudgetProcessor + speculative decoding.

Deux modes de test (--scenario) :
  conversational  Prompt complexe sans contrainte JSON (défaut)
  prune           Prompt exact de _action_prune_self_memory avec jarvis-self.json réel
                  → cas critique : JSON requis, thinking verbeux, anciennement bugué

Ce que mesure le script :
  - Tokens de thinking générés avant </think>
  - Que le processor force bien </think> au bon budget
  - JSON valide ou non (mode prune)
  - Comparaison baseline vs budgets configurables
  - Comparaison tok/sec baseline vs speculative decoding (si --draft-model fourni)

Usage :
  cd /opt/jarvis && source venv/bin/activate
  python scripts/test_thinking_budget.py [--scenario prune] [--budgets 1024,2048] [--no-baseline]

  # Test speculative decoding (nécessite un modèle draft Qwen3 compatible) :
  python scripts/test_thinking_budget.py --draft-model /opt/jarvis/models/hub/Qwen3-0.6B-MLX-4bit

Jarvis DOIT être arrêté avant de lancer :
  jarvis-stop
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jarvis-core", "src"))
os.environ.setdefault("LLM_LOCAL", "yes")
os.environ.setdefault("LLM_DEBUG_PROMPTS", "no")


# ── Garde GPU ─────────────────────────────────────────────────────────────
def _check_jarvis_not_running() -> None:
    try:
        out = subprocess.check_output(["pgrep", "-f", "jarvis"], text=True)
        pids = [p for p in out.strip().splitlines() if p != str(os.getpid())]
        if pids:
            print(f"ERREUR : Jarvis tourne déjà (pids {pids}).")
            print("Arrête jarvis-core avant de lancer ce test.")
            print("  jarvis-stop")
            sys.exit(1)
    except subprocess.CalledProcessError:
        pass


_check_jarvis_not_running()

from config import REASONING_MODEL, THINKING_BUDGET_TOKENS  # noqa: E402
from helpers import extract_llm_json  # noqa: E402
from llm_local import (  # noqa: E402
    ThinkingBudgetProcessor,
    _build_prompt,
    _load_model,
    _model_profile,
)
from prompts import get_prompt  # noqa: E402

try:
    from mlx_lm import stream_generate
    from mlx_lm.models.cache import ArraysCache, make_prompt_cache
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
except ImportError:
    print("ERROR: mlx_lm not found — run inside le venv Jarvis")
    sys.exit(1)

# Qwen3.5 hybrid models (GatedDeltaNet layers) store recurrent state in ArraysCache.
# ArraysCache.is_trimmable() returns False by default — this blocks speculative decoding.
# Patch: recurrent state has no sequence-length offset; trim is a no-op (returns n to
# satisfy the protocol). Quality impact: <1-token drift per rejected draft, negligible.
if not getattr(ArraysCache, "_trim_patched", False):
    ArraysCache.is_trimmable = lambda self: True
    ArraysCache.trim = lambda self, n: n
    ArraysCache._trim_patched = True

SELF_MEMORY_PATH = "/opt/jarvis/jarvis-core/JarvisData/jarvis-self.json"


# ── Construction des messages selon le scénario ────────────────────────────

def _build_conversational_messages() -> list[dict]:
    """Prompt complexe conversationnel — pas de JSON requis."""
    return [
        {
            "role": "system",
            "content": "Tu es Jarvis, assistant IA. Réponds en français, de façon directe et concise.",
        },
        {
            "role": "user",
            "content": (
                "Explique-moi en détail les différences architecturales entre un modèle de langage "
                "dense et un modèle MoE (Mixture of Experts), leurs avantages et inconvénients "
                "respectifs pour l'inférence locale sur Apple Silicon, et donne une recommandation "
                "concrète pour un usage comme assistant personnel."
            ),
        },
    ]


def _fmt(items: list, text_key: str = "text") -> str:
    """Reproduction exacte du _fmt de _action_prune_self_memory."""
    if not items:
        return "  (vide)"
    lines = []
    for i, item in enumerate(items):
        if isinstance(item, dict):
            text = (
                item.get(text_key)
                or item.get("text")
                or item.get("note")
                or item.get("opinion")
                or str(item)
            )
            date = item.get("date") or item.get("created") or ""
            date_str = f" ({date[:10]})" if date else ""
        else:
            text = str(item)
            date_str = ""
        lines.append(f"  [{i}] {text}{date_str}")
    return "\n".join(lines)


def _build_prune_messages() -> tuple[list[dict], int, int, int]:
    """Prompt exact de _action_prune_self_memory avec les données réelles."""
    with open(SELF_MEMORY_PATH) as f:
        data = json.load(f)

    self_notes = data.get("self_notes", [])
    opinions = data.get("opinions", [])
    learnings = data.get("learnings", [])

    user_prompt = get_prompt("PRUNE_SELF_MEMORY_USER").format(
        self_notes=_fmt(self_notes, "note"),
        opinions=_fmt(opinions, "opinion"),
        learnings=_fmt(learnings, "text"),
    )

    messages = [
        {"role": "system", "content": get_prompt("PRUNE_SELF_MEMORY_SYSTEM")},
        {"role": "user", "content": user_prompt},
    ]
    return messages, len(self_notes), len(opinions), len(learnings)


# ── Runner générique ───────────────────────────────────────────────────────

def run_once(
    model,
    tokenizer,
    profile,
    messages: list[dict],
    budget_proc,
    max_tokens: int,
    label: str,
    expect_json: bool = False,
    draft_model=None,
    num_draft_tokens: int = 3,
) -> dict:
    prompt = _build_prompt(
        messages, tokenizer, REASONING_MODEL, no_think=False, thinking_budget=0
    )
    prompt_tokens = len(tokenizer.encode(prompt))

    base_procs = make_logits_processors(
        repetition_penalty=profile.repetition_penalty,
        repetition_context_size=profile.repetition_context_size,
        frequency_penalty=profile.frequency_penalty,
        frequency_context_size=profile.repetition_context_size,
        presence_penalty=profile.presence_penalty,
        presence_context_size=profile.repetition_context_size,
    )
    procs = ([budget_proc] + list(base_procs)) if budget_proc is not None else base_procs

    sampler = make_sampler(
        temp=profile.temp_think,
        top_p=profile.top_p_think,
        top_k=profile.top_k,
        min_p=profile.min_p,
    )

    if draft_model is not None:
        # speculative_generate_step requires RotatingKVCache (trimmable); the default
        # ArraysCache created internally is not — so we pre-build the combined cache.
        _kv_size = max_tokens + 2048
        _spec_cache = make_prompt_cache(model, max_kv_size=_kv_size) + make_prompt_cache(draft_model, max_kv_size=_kv_size)
        spec_kwargs = {"draft_model": draft_model, "num_draft_tokens": num_draft_tokens, "prompt_cache": _spec_cache}
    else:
        spec_kwargs = {}

    print(f"\n{'─' * 70}")
    print(f"[{label}]  max_tokens={max_tokens}  prompt_tokens={prompt_tokens}")
    if budget_proc is not None:
        print(f"  processor : budget={budget_proc.budget}  end_think_id={budget_proc._end_think_id}")
    if draft_model is not None:
        print(f"  speculative: num_draft_tokens={num_draft_tokens}")

    t0 = time.time()
    raw = ""
    tok_count = 0
    for chunk in stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens,
        sampler=sampler, logits_processors=procs,
        **spec_kwargs,
    ):
        if chunk.text:
            raw += chunk.text
            tok_count += 1

    elapsed = time.time() - t0
    tok_per_sec = tok_count / elapsed if elapsed > 0 else 0

    has_end_think = "</think>" in raw
    think_part = ""
    answer_part = raw

    if has_end_think:
        think_part = raw.split("</think>", 1)[0]
        if "<think>" in think_part:
            think_part = think_part.split("<think>", 1)[1]
        answer_part = raw.split("</think>", 1)[1].strip()

    think_tokens = len(tokenizer.encode(think_part)) if think_part else 0
    answer_tokens = len(tokenizer.encode(answer_part)) if answer_part else 0

    print(f"  elapsed: {elapsed:.1f}s | total_tokens: {tok_count}")
    print(f"  has_</think>: {has_end_think} | think_tokens: {think_tokens} | answer_tokens: {answer_tokens}")

    json_ok = None
    if budget_proc is not None and has_end_think:
        ok = think_tokens <= budget_proc.budget + 10
        status = (
            f"✅ DANS LE BUDGET ({think_tokens} ≤ {budget_proc.budget})"
            if ok
            else f"❌ DÉPASSEMENT ({think_tokens} > {budget_proc.budget})"
        )
        print(f"  budget check: {status}")

    if expect_json:
        try:
            parsed = extract_llm_json(answer_part)
            json_ok = isinstance(parsed, dict) and "to_delete" in parsed
            if json_ok:
                td = parsed["to_delete"]
                print(
                    f"  JSON ✅  to_delete: notes={td.get('self_notes',[])} "
                    f"opinions={td.get('opinions',[])} learnings={td.get('learnings',[])}"
                )
            else:
                print(f"  JSON ❌  clé 'to_delete' absente — parsed={str(parsed)[:100]}")
        except Exception as exc:
            json_ok = False
            print(f"  JSON ❌  parse error: {exc}")
            print(f"  answer_part: {repr(answer_part[:200])}")
    else:
        if answer_part:
            preview = answer_part[:200].replace("\n", " ")
            print(f"  réponse: {preview}{'...' if len(answer_part) > 200 else ''}")

    if not has_end_think:
        print("  ⚠️  </think> jamais généré — thinking a épuisé tout le budget")
        print(f"  raw tail: {repr(raw[-120:])}")

    return {
        "label": label,
        "has_end_think": has_end_think,
        "think_tokens": think_tokens,
        "answer_tokens": answer_tokens,
        "total_tokens": tok_count,
        "elapsed": elapsed,
        "tok_per_sec": tok_per_sec,
        "budget": budget_proc.budget if budget_proc is not None else None,
        "json_ok": json_ok,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=["conversational", "prune"],
        default="conversational",
        help="conversational (défaut) | prune (prompt exact _action_prune_self_memory)",
    )
    parser.add_argument(
        "--budgets",
        default=str(THINKING_BUDGET_TOKENS),
        help=f"Budgets à tester, séparés par virgule (défaut: {THINKING_BUDGET_TOKENS})",
    )
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument(
        "--draft-model",
        default="",
        metavar="PATH",
        help="Path to Qwen3 draft model for speculative decoding (e.g. /opt/jarvis/models/hub/Qwen3-0.6B-MLX-4bit)",
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=3,
        help="Draft tokens per speculative step (défaut: 3)",
    )
    args = parser.parse_args()

    budgets = [int(b.strip()) for b in args.budgets.split(",")]
    expect_json = args.scenario == "prune"

    print(f"Modèle    : {REASONING_MODEL}")
    print(f"Scénario  : {args.scenario}  |  budgets : {budgets}  |  kill switch : {args.max_tokens}")
    if args.draft_model:
        print(f"Draft     : {args.draft_model}  |  num_draft_tokens={args.num_draft_tokens}")

    if args.scenario == "prune":
        messages, n_notes, n_opinions, n_learnings = _build_prune_messages()
        print(f"Mémoire   : {n_notes} self_notes  {n_opinions} opinions  {n_learnings} learnings")
    else:
        messages = _build_conversational_messages()

    model, tokenizer = _load_model(REASONING_MODEL)
    profile = _model_profile(REASONING_MODEL)

    draft_model = None
    if args.draft_model:
        print(f"\nChargement draft model : {args.draft_model}")
        draft_model, _ = _load_model(args.draft_model)

    vocab = tokenizer.get_vocab()
    print(
        f"Token IDs : <think>={vocab.get('<think>', 'NOT FOUND')}  "
        f"</think>={vocab.get('</think>', 'NOT FOUND')}"
    )

    results = []

    if not args.no_baseline:
        r = run_once(
            model, tokenizer, profile, messages, None,
            args.max_tokens, "BASELINE (sans processor)", expect_json,
        )
        results.append(r)

        if draft_model is not None:
            r = run_once(
                model, tokenizer, profile, messages, None,
                args.max_tokens, "BASELINE+speculative", expect_json,
                draft_model=draft_model, num_draft_tokens=args.num_draft_tokens,
            )
            results.append(r)

    for budget in budgets:
        proc = ThinkingBudgetProcessor(tokenizer, budget)
        r = run_once(
            model, tokenizer, profile, messages, proc,
            args.max_tokens, f"budget={budget} tok", expect_json,
        )
        results.append(r)

        if draft_model is not None:
            r = run_once(
                model, tokenizer, profile, messages, proc,
                args.max_tokens, f"budget={budget}+speculative", expect_json,
                draft_model=draft_model, num_draft_tokens=args.num_draft_tokens,
            )
            results.append(r)

    print(f"\n{'=' * 70}")
    print("RÉSUMÉ")
    for r in results:
        think_ok = "✅" if r["has_end_think"] else "❌"
        think_str = f"think={r['think_tokens']}tok" if r["has_end_think"] else "NO_</think>"
        budget_str = f"budget={r['budget']}" if r["budget"] else "no_proc"
        json_str = f"  json={'✅' if r['json_ok'] else '❌'}" if r["json_ok"] is not None else ""
        speed_str = f"{r['tok_per_sec']:.1f}tok/s"
        print(
            f"  {think_ok} {r['label']:<32s} | {think_str:<20s} | "
            f"answer={r['answer_tokens']}tok | {r['elapsed']:.0f}s | {speed_str} | {budget_str}{json_str}"
        )


if __name__ == "__main__":
    main()

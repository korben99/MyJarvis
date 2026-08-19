"""
test_lru_cache.py — Valide LRUPromptCache dans llm_local.py

Simule une conversation de 6 échanges et mesure à chaque tour :
  • tokens du prompt déjà cachés (prefix hit) vs tokens restants à traiter
  • hit-rate LRU progressif (doit augmenter à chaque tour)
  • temps de génération total
  • cohérence des réponses (le modèle a bien le contexte)
  • taille du LRU en mémoire

Turn 1 → toujours LRU miss : système KV construit
Turn 2 → hit partiel : préfixe sys+turn1 réutilisé
Turn 3+  → hit croissant : contexte accumulé réutilisé

Usage :
  cd /opt/jarvis && source venv/bin/activate
  python scripts/test_lru_cache.py
  python scripts/test_lru_cache.py --no-think    # sans thinking → meilleur prefix sharing
  python scripts/test_lru_cache.py --model router

Jarvis doit être arrêté avant : jarvis-stop
"""

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jarvis-core", "src"))

# Env vars AVANT import du module (ils sont lus au niveau module)
os.environ.setdefault("LLM_LOCAL", "yes")
os.environ.setdefault("LLM_DEBUG_PROMPTS", "no")
os.environ.setdefault("QUANT_KV", "yes")
os.environ.setdefault("QUANT_KV_BITS", "4")
os.environ.setdefault("LRU_KV_SIZE", "8")   # plus de slots pour le test
os.environ.setdefault("LRU_KV_GB", "4.0")
os.environ.setdefault("USE_THINKING_BUDGET_PROCESSOR", "yes")


# ── Garde GPU ─────────────────────────────────────────────────────────────
def _check_jarvis_not_running() -> None:
    # Pattern précis (uvicorn main:app) — "pgrep -f jarvis" matchait n'importe quel
    # process dont la ligne de commande contient /opt/jarvis (shells, éditeurs…).
    try:
        out = subprocess.check_output(["pgrep", "-f", "uvicorn main:app"], text=True)
        pids = [p for p in out.strip().splitlines() if p != str(os.getpid())]
        if pids:
            print(f"ERREUR : Jarvis tourne (pids {pids}).")
            print("Arrête-le avant : jarvis-stop")
            sys.exit(1)
    except subprocess.CalledProcessError:
        pass


_check_jarvis_not_running()

import llm_local as lru_mod  # noqa: E402
from llm_local import (  # noqa: E402
    LRU_KV_MAX_BYTES,
    LRU_KV_MAX_SIZE,
    _generate_sync,
    _get_lru,
    _load_model,
    _lru_get_cache,
    _lru_insert,
    _model_profile,
    _prompt_token_ids,
)
from config import PRIMARY_MODEL, REASONING_MODEL, ROUTER_MODEL, THINKING_BUDGET_COMPACT  # noqa: E402

# ── Conversation test ──────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Tu es Jarvis, assistant IA personnel de Sébastien. "
    "Tu réponds en français, de manière concise et précise. "
    "Tu mémorises le contexte de la conversation et fais référence aux échanges précédents."
)

# 6 échanges progressifs sur un sujet cohérent — le modèle doit pouvoir
# référencer les tours précédents (test du contexte transmis via LRU)
EXCHANGES = [
    "Explique-moi en deux phrases ce qu'est un cache LRU.",
    "Quel avantage concret cela donne pour l'inférence IA par rapport à reconstruire le KV cache à chaque fois ?",
    "Dans un modèle Qwen3.6 MoE, combien de couches ont un KV cache standard ? Rappelle aussi ce que tu m'as dit au premier échange.",
    "Quelle différence de mémoire y a-t-il entre 4-bit et 6-bit KV quantization pour ce même modèle ?",
    "Comment le LRU identifie-t-il le préfixe commun entre deux prompts consécutifs ?",
    "Fais-moi un résumé de nos 5 échanges précédents en 3 points.",
]


# ── Monkey-patch : capture stats sans overhead ────────────────────────────

_stats: dict = {}
_orig_get = lru_mod._lru_get_cache
_orig_insert = lru_mod._lru_insert


def _patched_get(model_path, model, tokenizer, sys_content, prompt_token_ids):
    cache, remaining = _orig_get(model_path, model, tokenizer, sys_content, prompt_token_ids)
    _stats["prompt_len"] = len(prompt_token_ids)
    _stats["remaining_len"] = len(remaining)
    _stats["hit"] = cache is not None
    _stats["cached_len"] = len(prompt_token_ids) - len(remaining) if cache is not None else 0
    return cache, remaining


def _patched_insert(model_path, model, prompt_token_ids, raw_output, cache, tokenizer):
    _orig_insert(model_path, model, prompt_token_ids, raw_output, cache, tokenizer)
    lru = _get_lru(model_path)
    _stats["lru_entries"] = len(lru)
    _stats["lru_mb"] = lru.nbytes / 1024**2


lru_mod._lru_get_cache = _patched_get
lru_mod._lru_insert = _patched_insert


# ── Runner ─────────────────────────────────────────────────────────────────

def run_turn(
    model_path: str,
    messages: list[dict],
    no_think: bool,
    max_tokens: int,
    thinking_budget: int,
    turn_idx: int,
) -> tuple[str, dict]:
    """Génère une réponse et affiche les stats LRU. Retourne la réponse pour continuer la conv."""
    _stats.clear()

    t0 = time.time()
    response = _generate_sync(
        model_path, messages,
        temperature=0.7,
        max_tokens=max_tokens,
        no_think=no_think,
        thinking_budget=thinking_budget,
    )
    elapsed = time.time() - t0

    prompt_len   = _stats.get("prompt_len", 0)
    remaining_len = _stats.get("remaining_len", 0)
    cached_len   = _stats.get("cached_len", 0)
    hit          = _stats.get("hit", False)
    lru_entries  = _stats.get("lru_entries", "?")
    lru_mb       = _stats.get("lru_mb", 0.0)

    hit_pct = cached_len * 100 // prompt_len if prompt_len else 0
    hit_label = "✅ HIT " if (hit and cached_len > 0) else "❌ MISS"

    BAR_W = 32
    filled = cached_len * BAR_W // prompt_len if prompt_len else 0
    bar = "█" * filled + "░" * (BAR_W - filled)

    print(f"\n{'─' * 72}")
    print(f"  Tour {turn_idx + 1}  [{hit_label}]  {elapsed:.1f}s")
    print(f"  Prompt  : {prompt_len:5d} tok total  │  cachés={cached_len:5d} ({hit_pct:3d}%)  restants={remaining_len:5d}")
    print(f"  LRU     : [{bar}] {lru_entries} entrées, {lru_mb:.1f} MB")

    preview = response[:200].replace("\n", " ")
    print(f"  Réponse : {preview}{'…' if len(response) > 200 else ''}")

    return response, {
        "turn": turn_idx + 1,
        "elapsed": elapsed,
        "prompt_len": prompt_len,
        "cached_len": cached_len,
        "remaining_len": remaining_len,
        "hit": hit,
        "hit_pct": hit_pct,
        "lru_entries": lru_entries,
        "lru_mb": lru_mb,
    }


# ── Assertions de validation ───────────────────────────────────────────────

def validate_results(results: list[dict], no_think: bool, model_path: str = "") -> None:
    from config import is_qwen3_hybrid
    print(f"\n{'=' * 72}")
    print("VALIDATION")
    errors = []

    # Turn 1 : système KV construit sur miss → hit_pct = system_tokens / prompt_total
    # Peut être élevé (ex. 64%) si le prompt est court — c'est normal, pas une erreur.
    r0 = results[0]
    label = "système KV préchargé" if r0["hit"] else "LRU miss (système KV construit)"
    print(f"  ✅ Turn 1 : {r0['cached_len']} tok cachés ({r0['hit_pct']}%) — {label}")

    # À partir du tour 2, hit avec tokens économisés
    for r in results[1:]:
        if not r["hit"]:
            errors.append(f"Turn {r['turn']}: pas de hit LRU — cache non utilisé")
        elif r["cached_len"] == 0:
            errors.append(f"Turn {r['turn']}: hit=True mais cached_len=0")
        else:
            print(f"  ✅ Turn {r['turn']} : {r['cached_len']} tok cachés ({r['hit_pct']}%)")

    # Tokens absolus cachés doivent croître à chaque tour (invariant clé du LRU de session)
    # Note : le hit-rate (%) peut baisser temporairement si le prompt grandit plus vite
    # que le préfixe commun (normal au tour 2 où le dénominateur saute).
    counts = [r["cached_len"] for r in results]
    if all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1)):
        print(f"  ✅ Tokens cachés croissants : {counts}")
    else:
        errors.append(f"Tokens cachés non croissants : {counts} — LRU ne progresse pas correctement")

    # Hit-rate doit être non-décroissant à partir du tour 3
    # (au tour 2 le dénominateur saute, ce qui peut faire baisser le %)
    # Générations hybrides (3.5/3.6/3.8) : cache multi-tours désactivé par design
    # (ArraysCache non trimmable — voir l'en-tête de llm/local.py). Seul le préfixe
    # système est réutilisé, donc le hit-rate % décroît mécaniquement quand le prompt
    # grandit — pas une erreur.
    if is_qwen3_hybrid(model_path):
        print("  ⏭️  Hit-rate % : check sauté (Qwen3 hybride — multi-tours désactivé par "
              "design, préfixe système seul réutilisé)")
    elif len(results) >= 3:
        pcts_3plus = [r["hit_pct"] for r in results[2:]]
        if all(pcts_3plus[i] <= pcts_3plus[i + 1] + 5 for i in range(len(pcts_3plus) - 1)):
            print(f"  ✅ Hit-rate non-décroissant dès tour 3 : {pcts_3plus}")
        else:
            errors.append(f"Hit-rate décroissant après tour 3 : {pcts_3plus}")

    # LRU doit contenir des entrées à la fin
    last = results[-1]
    if last["lru_entries"] == 0:
        errors.append("LRU vide après la conversation — insert_cache ne fonctionne pas")
    else:
        print(f"  ✅ LRU contient {last['lru_entries']} entrée(s) après la conversation")

    if errors:
        print(f"\n  ❌ {len(errors)} erreur(s) :")
        for e in errors:
            print(f"     • {e}")
    else:
        print(f"\n  ✅ TOUS LES CHECKS PASSÉS")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["primary", "reasoning", "router"],
        default="primary",
        help="primary=PRIMARY_MODEL (défaut)  reasoning=REASONING_MODEL  router=ROUTER_MODEL",
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="Désactive le thinking (plus rapide, meilleur prefix sharing — recommandé pour 1er test)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=0,
        help="Max tokens par réponse (0 = auto : thinking_budget + 800)",
    )
    parser.add_argument(
        "--thinking-budget", type=int, default=THINKING_BUDGET_COMPACT,
        help=f"Budget thinking en tokens, ignoré si --no-think (défaut: {THINKING_BUDGET_COMPACT})",
    )
    args = parser.parse_args()

    model_map = {"primary": PRIMARY_MODEL, "reasoning": REASONING_MODEL, "router": ROUTER_MODEL}
    model_path = model_map[args.model]
    no_think = args.no_think or (args.model == "router")
    thinking_budget = 0 if no_think else args.thinking_budget
    max_tokens = args.max_tokens if args.max_tokens > 0 else (800 if no_think else thinking_budget + 800)

    print(f"{'=' * 72}")
    print(f"TEST LRU CACHE — llm_local.py")
    print(f"{'=' * 72}")
    print(f"Modèle         : {model_path.split('/')[-1]}")
    print(f"no_think       : {no_think}")
    print(f"thinking_budget: {thinking_budget} tok  (ignored if no_think)")
    print(f"max_tokens     : {max_tokens} tok")
    print(f"LRU conf       : max_entries={LRU_KV_MAX_SIZE}  max_bytes={LRU_KV_MAX_BYTES/1024**2:.0f} MB")
    print(f"Échanges       : {len(EXCHANGES)}")
    print(f"{'=' * 72}")

    print(f"\nChargement du modèle…")
    t_load = time.time()
    model, tokenizer = _load_model(model_path)
    print(f"Modèle chargé en {time.time() - t_load:.1f}s\n")

    # Warmup JIT sans LRU (1 appel avec message court)
    print("Warmup JIT…")
    _generate_sync(
        model_path,
        [{"role": "system", "content": "Tu es un assistant."}, {"role": "user", "content": "Bonjour"}],
        temperature=0.0, max_tokens=10, no_think=True,
    )
    # Réinitialiser le LRU pour démarrer le test proprement (sans entrée du warmup)
    from mlx_lm.models.cache import LRUPromptCache
    lru_mod._lru_caches[model_path] = LRUPromptCache(
        max_size=LRU_KV_MAX_SIZE, max_bytes=LRU_KV_MAX_BYTES
    )
    print("LRU réinitialisé.\n")

    # ── Conversation ──────────────────────────────────────────────────────
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    results = []

    for i, user_msg in enumerate(EXCHANGES):
        print(f"\nQ{i+1}: {user_msg}")
        messages.append({"role": "user", "content": user_msg})

        response, stats = run_turn(model_path, messages, no_think, max_tokens, thinking_budget, i)
        results.append(stats)

        # Ajouter la réponse au contexte pour les tours suivants
        messages.append({"role": "assistant", "content": response})

    # ── Résumé final ──────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("RÉSUMÉ")
    print(f"{'─' * 72}")
    print(f"  {'Tour':<6} {'Temps':>7} {'Prompt':>8} {'Cachés':>8} {'%Cache':>7} {'LRU':>10}")
    print(f"{'─' * 72}")
    for r in results:
        bar = "█" * (r["hit_pct"] // 5) + "░" * (20 - r["hit_pct"] // 5)
        hit_mark = "✅" if r["hit"] else "❌"
        print(
            f"  {hit_mark} {r['turn']:<4}  {r['elapsed']:>6.1f}s"
            f"  {r['prompt_len']:>7} tok"
            f"  {r['cached_len']:>7} tok"
            f"  {r['hit_pct']:>6}%"
            f"  {bar}"
        )
    print(f"{'─' * 72}")
    total_time = sum(r["elapsed"] for r in results)
    total_saved = sum(r["cached_len"] for r in results)
    total_prompt = sum(r["prompt_len"] for r in results)
    print(f"  Total   {total_time:>6.1f}s   {total_prompt:>7} tok   {total_saved:>7} tok économisés")

    validate_results(results, no_think, model_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
curate_router_dataset.py
------------------------
Re-labels routing_samples.jsonl using Qwen3.6-35B as oracle, then writes
train/valid splits in mlx_lm.lora chat format.

Stop Jarvis before running (needs exclusive GPU access).

Usage:
    cd /opt/jarvis && source venv/bin/activate
    python scripts/curate_router_dataset.py
    python scripts/curate_router_dataset.py --limit 20 --dry-run
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────

BASE_DIR = Path("/opt/jarvis")
DATA_DIR = BASE_DIR / "RouterData"
MODEL_PATH = (
    BASE_DIR
    / "models/hub/models--majentik--Qwen3.6-35B-A3B-RotorQuant-MLX-5bit"
    / "snapshots/3718ead51200ff17b013873be5ed43fd05fda462"
)
SAMPLES_FILE = DATA_DIR / "routing_samples.jsonl"
CHECKPOINT_FILE = DATA_DIR / "curate_checkpoint.json"
CURATED_FILE = DATA_DIR / "routing_samples_curated.jsonl"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VALID_FILE = DATA_DIR / "valid.jsonl"

VALID_SPLIT = 0.10
RANDOM_SEED = 42

# ── Router schema ─────────────────────────────────────────────────────────

SCHEMA_KEYS = frozenset({
    "intents", "weather_location", "gmail_query",
    "calendar_days", "rag_query", "project_name", "use_reasoning",
})
ALLOWED_INTENTS = frozenset({
    "memory", "rag", "web", "weather", "gmail",
    "calendar", "self", "briefing", "portfolio",
})

ROUTER_SYSTEM = """\
Tu es un routeur JSON. Ton seul rôle : analyser l'intention du message et produire un JSON de routage. Tu ne réponds JAMAIS au message. Tu n'expliques JAMAIS. Tu ne résumes JAMAIS le message. Tu produis uniquement du JSON.

Schéma exact — 7 clés, ni plus ni moins :
{"intents":[...],"weather_location":null,"gmail_query":null,"calendar_days":null,"rag_query":null,"project_name":null,"use_reasoning":false}

Valeurs autorisées pour intents : "memory" "rag" "web" "weather" "gmail" "calendar" "self"
Clés autorisées : intents, weather_location, gmail_query, calendar_days, rag_query, project_name, use_reasoning
Toute autre clé est INTERDITE.

memory   → conversation, aide, explication, rappelle (par défaut)
rag      → documents de l'utilisateur →  rag_query=3-5 mots-clés (null si "rag" absent des intents)
web      → news, recherches, infos (si URL http(s) dans le message → memory seulement)
weather  → météo  →  weather_location=ville ou null
gmail    → emails  →  gmail_query=syntaxe Gmail
calendar → agenda  →  calendar_days=1-90
self     → état interne de Jarvis

Règle stricte : chaque champ ne doit être renseigné que si l'intent correspondant est présent. rag_query=null si "rag" absent. gmail_query=null si "gmail" absent. weather_location=null si "weather" absent.

use_reasoning=true pour réaliser un diagnostic, calcul multi-étapes, conseil médical/fiscal/juridique/mathématique ou physique avancé

"C'est quoi mon planning pour les deux prochaines semaines ?"
{"intents":["calendar"],"weather_location":null,"gmail_query":null,"calendar_days":14,"rag_query":null,"project_name":null,"use_reasoning":false}

"Est-ce que j'ai reçu des mails de la banque cette semaine ?"
{"intents":["gmail"],"weather_location":null,"gmail_query":"from:comptable newer_than:7d","calendar_days":null,"rag_query":null,"project_name":null,"use_reasoning":false}

"Il fait quel temps à Bordeaux ce week-end ? On pense partir samedi."
{"intents":["weather"],"weather_location":"Bordeaux","gmail_query":null,"calendar_days":null,"rag_query":null,"project_name":null,"use_reasoning":false}

"C'est quoi le cours du Bitcoin ?"
{"intents":["web"],"weather_location":null,"gmail_query":null,"calendar_days":null,"rag_query":null,"project_name":null,"use_reasoning":false}

"Tu peux retrouver mon document sur le brevet mixture-of-expert ?"
{"intents":["rag"],"weather_location":null,"gmail_query":null,"calendar_days":null,"rag_query":"brevet mixture-of-expert","project_name":null,"use_reasoning":false}

"Montre-moi mon planning de demain et vérifie mes mails non lus."
{"intents":["calendar","gmail"],"weather_location":null,"gmail_query":"is:unread is:important","calendar_days":2,"rag_query":null,"project_name":null,"use_reasoning":false}

"Retrouve dans mes docs ce que j'ai noté sur le RGPD et donne-moi aussi les dernières actualités réglementaires."
{"intents":["rag","web"],"weather_location":null,"gmail_query":null,"calendar_days":null,"rag_query":"RGPD réglementation","project_name":null,"use_reasoning":false}

"Où on en est sur le projet attelage BMW ? On avance ?"
{"intents":["memory"],"weather_location":null,"gmail_query":null,"calendar_days":null,"rag_query":null,"project_name":"attelage BMW","use_reasoning":false}

"Question qui n'a rien à voir — tu sais à quelle vitesse montent les ascenseurs dans les grands hôtels ?"
{"intents":["memory"],"weather_location":null,"gmail_query":null,"calendar_days":null,"rag_query":null,"project_name":null,"use_reasoning":false}

"Mon script Python plante aléatoirement en prod mais jamais en local."
{"intents":["memory"],"weather_location":null,"gmail_query":null,"calendar_days":null,"rag_query":null,"project_name":null,"use_reasoning":true}

"C'est quoi tes dernières réflexions Jarvis ?"
{"intents":["self"],"weather_location":null,"gmail_query":null,"calendar_days":null,"rag_query":null,"project_name":null,"use_reasoning":false}
"""

# ── JSON helpers ──────────────────────────────────────────────────────────

def _first_json(text: str) -> dict | None:
    """Extract first complete JSON object from text."""
    start = -1
    stack = []
    in_string = False
    escape_next = False

    for i, c in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if c == "\\" and in_string:
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if start == -1:
            if c == "{":
                start = i
                stack.append(c)
            continue
        if c in "{[":
            stack.append(c)
        elif c in "}]":
            if not stack:
                continue
            last = stack.pop()
            if (last == "{" and c != "}") or (last == "[" and c != "]"):
                start = -1
                stack.clear()
                continue
            if not stack:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    start = -1
    return None


def _normalize(raw: dict) -> dict | None:
    """
    Validate and normalize a raw router response dict.
    Returns a clean 7-key dict or None if unrecoverable.
    """
    # Intents
    intents = raw.get("intents", [])
    if not isinstance(intents, list):
        return None
    intents = [i for i in intents if i in ALLOWED_INTENTS]
    if not intents:
        intents = ["memory"]

    # Conditional fields — only populated when their intent is present
    weather_location = raw.get("weather_location") or None
    if "weather" not in intents:
        weather_location = None

    gmail_query = raw.get("gmail_query") or None
    if "gmail" not in intents:
        gmail_query = None

    cal_raw = raw.get("calendar_days")
    if "calendar" not in intents or cal_raw is None:
        calendar_days = None
    else:
        try:
            calendar_days = max(1, min(int(cal_raw), 90))
        except (ValueError, TypeError):
            calendar_days = None

    rag_query = raw.get("rag_query") or None
    if "rag" not in intents:
        rag_query = None
    elif rag_query and len(rag_query) > 80:
        # Reject rag_query that looks like echoed user text
        return None

    project_name = raw.get("project_name") or None
    use_reasoning = bool(raw.get("use_reasoning", False))

    return {
        "intents": intents,
        "weather_location": weather_location,
        "gmail_query": gmail_query,
        "calendar_days": calendar_days,
        "rag_query": rag_query,
        "project_name": project_name,
        "use_reasoning": use_reasoning,
    }


# ── MLX inference ─────────────────────────────────────────────────────────

def load_model(model_path: str):
    from mlx_lm import load as mlx_load
    print(f"Loading model from {model_path} …", flush=True)
    t0 = time.time()
    model, tokenizer = mlx_load(model_path)
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)
    return model, tokenizer


def _build_prompt_qwen36_nothink(messages: list[dict], tokenizer) -> str:
    """Build no-think prompt for Qwen3.6 and append '{' to force JSON start."""
    base_kw = {"tokenize": False, "add_generation_prompt": True}
    try:
        prompt = tokenizer.apply_chat_template(
            messages, **base_kw, enable_thinking=False, thinking_budget=0
        )
    except TypeError:
        try:
            prompt = tokenizer.apply_chat_template(
                messages, **base_kw, enable_thinking=False
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, **base_kw)

    # Enforce closed think block
    if prompt.endswith("<think>\n") or prompt.rstrip().endswith("<think>"):
        prompt = prompt.rstrip("\n") + "\n</think>\n\n"
    elif "</think>" not in prompt[-30:]:
        prompt = prompt.rstrip("\n") + "\n<think>\n\n</think>\n\n"

    # Force JSON start
    return prompt.rstrip("\n") + "{"


def run_oracle(message: str, model, tokenizer, max_tokens: int = 120) -> dict | None:
    """Ask Qwen3.6 to produce the correct routing JSON for `message`."""
    from mlx_lm import generate as mlx_generate

    messages = [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": f"<message>{message[:400]}</message>"},
    ]
    prompt = _build_prompt_qwen36_nothink(messages, tokenizer)
    raw = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                       verbose=False)

    # Strip think block if present (defensive)
    if "</think>" in raw:
        raw = raw.rsplit("</think>", 1)[-1]

    # Restore the "{" we prepended
    raw = "{" + raw

    parsed = _first_json(raw)
    if parsed is None:
        return None
    return _normalize(parsed)


# ── Checkpoint helpers ────────────────────────────────────────────────────

def load_checkpoint() -> set[str]:
    if not CHECKPOINT_FILE.exists():
        return set()
    with open(CHECKPOINT_FILE) as f:
        return set(json.load(f).get("done", []))


def save_checkpoint(done: set[str]) -> None:
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"done": sorted(done)}, f)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Run inference but don't write output files")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N samples (0 = all)")
    args = parser.parse_args()

    # Load samples
    samples = [json.loads(l) for l in open(SAMPLES_FILE)]
    if args.limit:
        samples = samples[:args.limit]
    print(f"Samples to process: {len(samples)}", flush=True)

    # Resume from checkpoint
    done_ids = load_checkpoint()
    pending = [s for s in samples if s["id"] not in done_ids]
    print(f"Already done: {len(done_ids)} — Remaining: {len(pending)}", flush=True)

    # Load model (skip in dry-run only if nothing pending)
    model, tokenizer = None, None
    if pending:
        model, tokenizer = load_model(str(MODEL_PATH))

    # Process
    curated: list[dict] = []
    failed: list[str] = []
    t_start = time.time()

    # Re-load already-processed results if resuming
    if CURATED_FILE.exists() and done_ids:
        with open(CURATED_FILE) as f:
            curated = [json.loads(l) for l in f if l.strip()]

    for i, sample in enumerate(pending):
        msg = sample["message"]
        sid = sample["id"]

        t0 = time.time()
        result = run_oracle(msg, model, tokenizer)
        elapsed = time.time() - t0

        if result is None:
            print(f"[{i+1}/{len(pending)}] FAIL  ({elapsed:.1f}s) — {msg[:60]!r}", flush=True)
            failed.append(sid)
            # Fallback: keep original routing (memory only) as a safe default
            result = {
                "intents": ["memory"], "weather_location": None,
                "gmail_query": None, "calendar_days": None,
                "rag_query": None, "project_name": None, "use_reasoning": False,
            }

        curated_sample = {
            "id": sid,
            "ts": sample["ts"],
            "message": msg,
            "routing": result,
            "model": str(MODEL_PATH).split("/")[-3],
            "ok": True,
        }
        curated.append(curated_sample)
        done_ids.add(sid)

        eta = (time.time() - t_start) / (i + 1) * (len(pending) - i - 1)
        print(
            f"[{i+1}/{len(pending)}] {elapsed:.1f}s  ETA {eta/60:.1f}min  "
            f"{result['intents']}  reasoning={result['use_reasoning']}  "
            f"— {msg[:55]!r}",
            flush=True,
        )

        # Save checkpoint every 10 samples
        if not args.dry_run and (i + 1) % 10 == 0:
            save_checkpoint(done_ids)
            with open(CURATED_FILE, "w") as f:
                for s in curated:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Final save
    if not args.dry_run:
        save_checkpoint(done_ids)
        with open(CURATED_FILE, "w") as f:
            for s in curated:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"\nCurated dataset saved → {CURATED_FILE} ({len(curated)} samples)", flush=True)

    # ── Build train/valid splits ──────────────────────────────────────────
    if not curated:
        print("No curated samples — nothing to write.", flush=True)
        return

    random.seed(RANDOM_SEED)
    shuffled = curated.copy()
    random.shuffle(shuffled)
    n_valid = max(1, int(len(shuffled) * VALID_SPLIT))
    valid_set = shuffled[:n_valid]
    train_set = shuffled[n_valid:]

    def to_lora_entry(sample: dict) -> dict:
        routing = sample["routing"]
        assistant_json = json.dumps(routing, ensure_ascii=False, separators=(",", ":"))
        return {
            "messages": [
                {"role": "system", "content": ROUTER_SYSTEM},
                {"role": "user", "content": f"<message>{sample['message'][:400]}</message>"},
                {"role": "assistant", "content": assistant_json},
            ]
        }

    if not args.dry_run:
        with open(TRAIN_FILE, "w") as f:
            for s in train_set:
                f.write(json.dumps(to_lora_entry(s), ensure_ascii=False) + "\n")
        with open(VALID_FILE, "w") as f:
            for s in valid_set:
                f.write(json.dumps(to_lora_entry(s), ensure_ascii=False) + "\n")

    # ── Summary ───────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Total processed : {len(pending)}")
    print(f"Failed (fallback): {len(failed)}")
    print(f"Total curated   : {len(curated)}")
    print(f"Train set       : {len(train_set)}")
    print(f"Valid set       : {len(valid_set)}")
    print(f"Elapsed         : {total_time/60:.1f} min")
    if not args.dry_run:
        print(f"Train → {TRAIN_FILE}")
        print(f"Valid → {VALID_FILE}")
    else:
        print("(dry-run: no files written)")

    # Intent distribution in curated set
    from collections import Counter
    intent_counts: Counter = Counter()
    for s in curated:
        for intent in s["routing"]["intents"]:
            intent_counts[intent] += 1
    print("\nIntent distribution (curated):")
    for k, v in intent_counts.most_common():
        print(f"  {k:12s}: {v:4d}  ({v*100//len(curated)}%)")

    reasoning_count = sum(1 for s in curated if s["routing"]["use_reasoning"])
    print(f"\nuse_reasoning=true: {reasoning_count}/{len(curated)}")


if __name__ == "__main__":
    main()

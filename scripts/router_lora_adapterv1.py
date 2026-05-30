#!/usr/bin/env python3
"""
router_lora_adapterv1.py
-------------------------
Étape 1 : génère train.jsonl / valid.jsonl depuis routing_samples_curated.jsonl
Étape 2 : fine-tune Qwen2.5-1.5B-Instruct (bf16) avec LoRA sur la tâche de routage
Étape 3 : affiche les commandes fuse + quantize 4-bit pour déploiement

Usage (Jarvis arrêté) :
    cd /opt/jarvis && source venv/bin/activate
    python scripts/router_lora_adapterv1.py
    python scripts/router_lora_adapterv1.py --dry-run   # prépare les données sans entraîner

mlx_lm.fuse \
--model /opt/jarvis/models/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89 \
--adapter-path /opt/jarvis/RouterData/adapters/qwen25-1.5b-router-v1 \
--save-path /opt/jarvis/RouterData/adapters/qwen25-1.5b-router-v1/fused_bf16

mlx_lm.convert \
--hf-path /opt/jarvis/RouterData/adapters/qwen25-1.5b-router-v1/fused_bf16 \
-q --q-bits 4 \
--mlx-path /opt/jarvis/models/hub/Qwen2.5-1.5B-router-v1-4bit

"""

import argparse
import json
import random
import subprocess
import sys
import textwrap
from pathlib import Path

import yaml

# ── Chemins ───────────────────────────────────────────────────────────────

BASE_DIR = Path("/opt/jarvis")
DATA_DIR = BASE_DIR / "RouterData"
ADAPTER_DIR = DATA_DIR / "adapters" / "qwen25-1.5b-router-v1"

MODEL_PATH = (
    BASE_DIR
    / "models/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16"
    / "snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89"
)

CURATED_FILE = DATA_DIR / "routing_samples_curated.jsonl"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VALID_FILE = DATA_DIR / "valid.jsonl"
CONFIG_FILE = ADAPTER_DIR / "lora_config.yaml"

VALID_SPLIT = 0.10
RANDOM_SEED = 42
MAX_MSG_CHARS = 400  # doit correspondre au message[:400] de llm_router.py

# ── System prompt (doit rester en sync avec prompts.py::ROUTER_SYSTEM) ────

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

# ── Étape 1 : génération des splits ──────────────────────────────────────


def build_splits() -> tuple[int, int]:
    """
    Lit routing_samples_curated.jsonl, applique les mêmes transformations
    qu'à l'inférence (truncature 400 chars), écrit train.jsonl + valid.jsonl.
    Retourne (n_train, n_valid).
    """
    samples = [json.loads(l) for l in open(CURATED_FILE)]
    # Filtre les messages vides
    samples = [s for s in samples if s["message"].strip()]
    print(f"Curated samples (non-vides) : {len(samples)}")

    random.seed(RANDOM_SEED)
    random.shuffle(samples)
    n_valid = max(1, int(len(samples) * VALID_SPLIT))
    valid_set = samples[:n_valid]
    train_set = samples[n_valid:]

    def to_entry(s: dict) -> dict:
        routing = s["routing"]
        # Compact JSON, clés dans l'ordre du schéma
        label = json.dumps(
            {
                "intents": routing["intents"],
                "weather_location": routing.get("weather_location"),
                "gmail_query": routing.get("gmail_query"),
                "calendar_days": routing.get("calendar_days"),
                "rag_query": routing.get("rag_query"),
                "project_name": routing.get("project_name"),
                "use_reasoning": bool(routing.get("use_reasoning", False)),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return {
            "messages": [
                {"role": "system", "content": ROUTER_SYSTEM},
                {
                    "role": "user",
                    "content": f"<message>{s['message'][:MAX_MSG_CHARS]}</message>",
                },
                {"role": "assistant", "content": label},
            ]
        }

    with open(TRAIN_FILE, "w") as f:
        for s in train_set:
            f.write(json.dumps(to_entry(s), ensure_ascii=False) + "\n")
    with open(VALID_FILE, "w") as f:
        for s in valid_set:
            f.write(json.dumps(to_entry(s), ensure_ascii=False) + "\n")

    print(f"Train : {len(train_set)} exemples → {TRAIN_FILE}")
    print(f"Valid : {len(valid_set)} exemples → {VALID_FILE}")
    return len(train_set), len(valid_set)


# ── Étape 2 : config LoRA ─────────────────────────────────────────────────


def write_config(n_train: int) -> None:
    """
    Écrit lora_config.yaml dans ADAPTER_DIR.
    Paramètres choisis pour une tâche de classification JSON sur ~400 exemples.
    """
    # 5 époques environ : floor(n_train / batch_size) * 5
    steps_per_epoch = n_train // 4
    iters = steps_per_epoch * 5

    config = {
        # Modèle
        "model": str(MODEL_PATH),
        "train": True,
        "data": str(DATA_DIR),
        # Format
        "mask_prompt": True,  # loss uniquement sur le JSON assistant
        "max_seq_length": 2048,  # system prompt ~1340 tok + msg ~100 tok + label ~40 tok
        # LoRA
        "fine_tune_type": "lora",
        "num_layers": 8,  # 8 couches sur 28 suffisent pour une tâche de routage
        "lora_parameters": {
            "rank": 8,
            "dropout": 0.05,
            "scale": 20.0,  # alpha = scale = 20 → scaling factor 20/8 = 2.5
        },
        # Entraînement
        "batch_size": 4,
        "iters": iters,
        "learning_rate": 2e-4,
        "optimizer": "adamw",
        # Évaluation & sauvegarde
        "val_batches": -1,  # toute la validation à chaque eval
        "steps_per_report": 10,
        "steps_per_eval": max(20, steps_per_epoch // 2),
        "save_every": steps_per_epoch,
        # Sorties
        "adapter_path": str(ADAPTER_DIR),
    }

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"\nConfig LoRA :")
    print(f"  iters          : {iters}  (~5 époques)")
    print(f"  num_layers     : {config['num_layers']}")
    print(
        f"  lora rank/scale: {config['lora_parameters']['rank']} / {config['lora_parameters']['scale']}"
    )
    print(f"  batch_size     : {config['batch_size']}")
    print(f"  learning_rate  : {config['learning_rate']}")
    print(f"  max_seq_length : {config['max_seq_length']}")
    print(f"  Config → {CONFIG_FILE}")


# ── Étape 3 : lancement ───────────────────────────────────────────────────


def run_training() -> None:
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--config",
        str(CONFIG_FILE),
    ]
    print(f"\nLancement : {' '.join(cmd)}\n{'=' * 60}\n")
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    if result.returncode != 0:
        print(f"\nEntraînement terminé avec code {result.returncode}.", file=sys.stderr)
        sys.exit(result.returncode)


def print_next_steps() -> None:
    fused_path = ADAPTER_DIR / "fused_bf16"
    quant_path = BASE_DIR / "models/hub/Qwen2.5-1.5B-router-v1-4bit"
    print(
        textwrap.dedent(f"""
    {"=" * 60}
    Adaptateur LoRA sauvegardé dans :
      {ADAPTER_DIR}/adapters.safetensors

    Étapes suivantes — fusion + quantization 4-bit :

    1. Fusionner l'adaptateur dans le modèle bf16 :
       mlx_lm.fuse \\
         --model {MODEL_PATH} \\
         --adapter-path {ADAPTER_DIR} \\
         --save-path {fused_path}

    2. Quantizer en 4-bit pour Jarvis :
       mlx_lm.convert \\
         --hf-path {fused_path} \\
         -q --q-bits 4 \\
         --mlx-path {quant_path}

    3. Mettre à jour .env :
       ROUTER_MODEL_LOCAL={quant_path}
    {"=" * 60}
    """)
    )


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prépare données + config sans lancer l'entraînement",
    )
    args = parser.parse_args()

    if not CURATED_FILE.exists():
        print(f"Fichier manquant : {CURATED_FILE}", file=sys.stderr)
        print("Lancez d'abord curate_router_dataset.py", file=sys.stderr)
        sys.exit(1)

    if not MODEL_PATH.exists():
        print(f"Modèle introuvable : {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    print("=== Étape 1 : génération des splits ===")
    n_train, n_valid = build_splits()

    print("\n=== Étape 2 : configuration LoRA ===")
    write_config(n_train)

    if args.dry_run:
        print("\n[dry-run] Données et config prêtes — entraînement non lancé.")
        print_next_steps()
        return

    print("\n=== Étape 3 : entraînement ===")
    run_training()

    print_next_steps()


if __name__ == "__main__":
    main()

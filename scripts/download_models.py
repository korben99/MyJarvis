import os
import shutil

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, snapshot_download

# Charge le .env
load_dotenv()

# ---- CONFIG ----
# Les modèles à télécharger sont ceux réellement configurés pour le mode local
# (mêmes variables et mêmes defaults que jarvis-core/src/config.py) — modifier
# .env pour changer de modèle, pas ce fichier.
MODELS = [
    m
    for m in {
        os.getenv("ROUTER_MODEL_LOCAL", "mlx-community/Qwen2.5-1.5B-Instruct-4bit"),
        os.getenv("PRIMARY_MODEL_LOCAL", "spicyneuron/Qwen3.6-35B-A3B-MLX-5.4bit"),
        os.getenv("REASONING_MODEL_LOCAL", "")
        or os.getenv("PRIMARY_MODEL_LOCAL", "spicyneuron/Qwen3.6-35B-A3B-MLX-5.4bit"),
        os.getenv(
            "VISION_MODEL_LOCAL", "lmstudio-community/Qwen3-VL-8B-Instruct-MLX-5bit"
        ),
    }
    if m and not m.startswith("/")  # skip local filesystem paths (e.g. a custom
    # fused/quantized router built via scripts/router_lora_adapterv1.py — not an
    # HF repo id, nothing to download)
]

# Fichiers de template à télécharger séparément (indépendants du cache HF).
# local_path doit correspondre à QWEN36_NINJA_TEMPLATE dans config.py.
# Uniquement nécessaire si le primary/reasoning model est un Qwen3.6 "ninja template" :
# le fichier est lié à ce tokenizer précis, il ne se transpose pas à une autre
# génération. En bascule vers Qwen3.6 → Qwen3.8, ce bloc se désactive tout seul et
# config.is_qwen36() cesse de matcher — le template livré avec le modèle est utilisé.
TEMPLATES = (
    [
        {
            "repo_id": "spicyneuron/Qwen3.6-35B-A3B-MLX-5.4bit",
            "filename": "chat_template.optional.jinja",
            "local_path": os.getenv(
                "QWEN36_NINJA_TEMPLATE",
                "/opt/jarvis/models/templates/qwen36_ninja.jinja",
            ),
        },
    ]
    if "qwen3.6" in "".join(MODELS).lower()
    else []
)

# Récup variables
HF_HOME = os.getenv("HF_HOME", os.path.expanduser("~/models"))
HF_TOKEN = os.getenv("HF_TOKEN")

# Applique environnement
os.environ["HF_HOME"] = HF_HOME
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

HF_HUB_CACHE = os.path.join(HF_HOME, "hub")


def model_exists(model_name):
    model_id = model_name.replace("/", "--")
    path = os.path.join(HF_HUB_CACHE, f"models--{model_id}")
    if not os.path.exists(path):
        return False
    # Vérifie qu'il n'y a pas de blobs incomplets (download interrompu)
    blobs_dir = os.path.join(path, "blobs")
    if os.path.exists(blobs_dir):
        for f in os.listdir(blobs_dir):
            if f.endswith(".incomplete"):
                return False
    # Vérifie qu'il y a au least un fichier de poids dans les snapshots
    snapshots_dir = os.path.join(path, "snapshots")
    if os.path.exists(snapshots_dir):
        for rev in os.listdir(snapshots_dir):
            rev_path = os.path.join(snapshots_dir, rev)
            if any(
                f.endswith(".safetensors")
                or f.endswith(".npz")
                or f == "model.safetensors.index.json"
                for f in os.listdir(rev_path)
            ):
                # model.safetensors.index.json without the actual shards = incomplete
                files = set(os.listdir(rev_path))
                if "model.safetensors.index.json" in files and not any(
                    f.endswith(".safetensors") for f in files
                ):
                    return False
                return True
    return False


def download_model(model_name):
    print(f"[DOWNLOAD] {model_name}")
    # Pas de local_dir — on laisse HF utiliser HF_HUB_CACHE (structure models--org--name)
    # Ce format est celui que mlx_lm.load() et model_exists() attendent.
    snapshot_download(
        repo_id=model_name,
        cache_dir=HF_HUB_CACHE,
        revision="main",
        token=HF_TOKEN,
    )


def download_template(entry: dict) -> None:
    """Télécharge un fichier template depuis HF et le copie au chemin local."""
    local_path = entry["local_path"]
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if os.path.isfile(local_path):
        print(f"[SKIP] Template déjà présent : {local_path}")
        return

    print(f"[TEMPLATE] {entry['repo_id']} / {entry['filename']} → {local_path}")
    cached = hf_hub_download(
        repo_id=entry["repo_id"],
        filename=entry["filename"],
        cache_dir=HF_HUB_CACHE,
        token=HF_TOKEN,
    )
    shutil.copy2(cached, local_path)
    print(f"[TEMPLATE] OK → {local_path}")


def main():
    print(f"[HF_HOME] {HF_HOME}")
    print(f"[TOKEN] {'OK' if HF_TOKEN else 'MISSING'}")
    print(f"[MODELS] {', '.join(MODELS) if MODELS else '(none — all local paths)'}")

    for model in MODELS:
        if model_exists(model):
            print(f"[SKIP] Déjà présent : {model}")
        else:
            download_model(model)

    for entry in TEMPLATES:
        download_template(entry)


if __name__ == "__main__":
    main()

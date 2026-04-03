import os

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

# Charge le .env
load_dotenv()

# ---- CONFIG ----
MODELS = [
    "inferencerlabs/Qwen3.5-35B-A3B-MLX-5.5bit",
    "mlx-community/Qwen2.5-3B-Instruct-8bit",
    "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
]

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
            if any(f.endswith(".safetensors") or f.endswith(".npz") or f == "model.safetensors.index.json"
                   for f in os.listdir(rev_path)):
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


def main():
    print(f"[HF_HOME] {HF_HOME}")
    print(f"[TOKEN] {'OK' if HF_TOKEN else 'MISSING'}")

    for model in MODELS:
        if model_exists(model):
            print(f"[SKIP] Déjà présent : {model}")
        else:
            download_model(model)


if __name__ == "__main__":
    main()

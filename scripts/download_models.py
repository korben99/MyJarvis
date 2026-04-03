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
    return os.path.exists(path)


def download_model(model_name):
    print(f"[DOWNLOAD] {model_name}")
    snapshot_download(
        repo_id=model_name,
        resume_download=True,
        token=HF_TOKEN,  # explicite = plus robuste
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

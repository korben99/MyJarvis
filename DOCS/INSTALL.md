# Installation


## Prerequisites

- **macOS on Apple Silicon** — `jarvis-core/src/helpers.py` unconditionally imports `llm_local.py`, which imports `mlx` at module level. This is required even in cloud-API mode (`LLM_LOCAL=no`); Jarvis does not currently run on Linux/Windows/Intel Mac.
- Python 3.13 (`brew install python@3.13`)
- Docker or OrbStack (for Qdrant, Redis, Open WebUI)
- `grype` (`brew install grype`) — optional, for the daily CVE scan (`cve.py`). The SBOM generator (`cyclonedx-bom`) ships in `requirements.txt`; without `grype`, the scan is simply skipped.
- Google OAuth credentials (for Gmail / Calendar) — optional
- Cloud API key (OpenAI or compatible) — only if you set `LLM_LOCAL=no`; the default is fully local, no key required

## Quick install

```bash
git clone <repo> /opt/jarvis
cd /opt/jarvis
./install.sh
```

`install.sh` is idempotent (safe to re-run after a `git pull`) and gets you all the way to "just fill in `.env`":
- checks prerequisites (macOS/arm64, Python 3.13, Docker)
- creates the venv and installs `requirements.txt`
- creates every gitignored runtime directory (`RAGData/*` — including `RAGData/Trade/` —, `logs/`, `keys/`, `models/`, `jarvis-core/JarvisData/`)
- copies `.env.example` → `.env` and `DOCS/examples/users_list.example.json` → `jarvis-core/JarvisData/users_list.json` (never overwrites existing files)
- installs the `com.jarvis.api` launchd service from `DOCS/examples/com.jarvis.api.plist.template` and adds the `jarvis-start`/`jarvis-stop`/`jarvis-reload` aliases to your shell rc

What's left, by hand:

## 1. Configure `.env`

Full local mode is the default (`LLM_LOCAL=yes`) — no API key required. Pick your models by uncommenting/editing the `*_MODEL_LOCAL` lines (defaults work out of the box), or set `HF_TOKEN` if a chosen model is gated. Prefer a cloud API instead? Set `LLM_LOCAL=no` and `OPENAI_API_KEY`. See the Variables section below for everything else.

## 2. Fill in your user list

Edit `jarvis-core/JarvisData/users_list.json` (created by `install.sh`, gitignored — holds personal data): one entry per user, `code` is that user's API access secret — generate a random string per user, don't ship the example values.

## 3. Download local models (only if `LLM_LOCAL=yes`, the default)

```bash
source venv/bin/activate
python scripts/download_models.py   # downloads whatever *_MODEL_LOCAL points to in .env
```

Models are stored in `HF_HOME` (default `/opt/jarvis/models`). The script skips models already present and detects interrupted downloads via `.incomplete` blobs.

## 4. Start all services

```bash
jarvis-start
```

(alias posé par `install.sh` ; équivaut à `scripts/jarvis-launchd.sh start`, idempotent —
safe to replay if the service is already running. Docker is started by `scripts/jarvis-entrypoint.sh`.)

This starts:
- `docker compose up -d` — Qdrant, Redis, Open WebUI (port 3000)
- `uvicorn main:app` — Jarvis API on port 8000, running **natively** (not in Docker) for direct Metal GPU access via MLX

For an always-on setup use the launchd service instead (`jarvis-start`, see below).

## 5. Verify

```bash
curl http://localhost:8000/status
```

## 6. Index documents (optional)

Place documents in `RAGData/` subdirectories (`personal/`, `work/`, `documents/`, `company/`, `reflexions/`), then run:

```bash
./venv/bin/python scripts/uploadrag.py --dry-run   # lists what would be uploaded
./venv/bin/python scripts/uploadrag.py             # indexe réellement
```

L'indexation tourne aussi automatiquement via launchd (`com.moi.uploadrag`, 23:10).
It requires `ENABLE_API_KEYS` **enabled in the Open WebUI admin panel**: it is a
`PersistentConfig`, so the `docker-compose.yml` variable only seeds the very first
initialisation and is ignored afterwards. Symptom when disabled:
`403 — Use of API key is not enabled in the environment`.

## 7. Import trading portfolio (optional)

Export your Boursorama positions as CSV (*Mes comptes → Exporter*) and drop the file in `RAGData/Trade/`. Jarvis imports it automatically on the next hourly tick, or immediately on restart.

## macOS launchd Service

The Jarvis API runs as a launchd agent (`com.jarvis.api`) on macOS. Useful commands:

The plist and the `jarvis-start`/`jarvis-stop`/`jarvis-reload` aliases below are installed automatically by `./install.sh` (template: `DOCS/examples/com.jarvis.api.plist.template`, aliases: `DOCS/examples/jarvis-aliases.sh`).

```bash
# Logs en live
tail -f /opt/jarvis/logs/jarvis-service.log
```

| Alias | Commande |
|-------|---------|
| `jarvis-start` | `launchctl bootstrap` — starts the service |
| `jarvis-stop` | `launchctl bootout` — arrêt propre (launchd ne relance pas) |
| `jarvis-reload` | `bootout` + `bootstrap` — stop + redémarre immédiatement |

## Common Commands

```bash
# Restart Jarvis API after a code change (the server does not hot-reload)
jarvis-restart

# Stream API logs
tail -f /opt/jarvis/logs/jarvis-api.log

# Restart only infra (Redis, Qdrant, Open WebUI)
docker compose restart

# Stop infra
docker compose down

# Re-download / resume a failed model download
python scripts/download_models.py

# Sauvegarde (écrit un reçu local → exemplaires_etat=2 ; clé USB montée requise)
./scripts/backup-jarvis.sh

# Sauvegarde PUIS mises à jour Docker + venv, sous fenêtre de maintenance (aucun mv de venv)
./scripts/backup-jarvis.sh updates

# On-demand CVE scan (otherwise daily at 04:30) — counts only what is fixable
/opt/jarvis/venv/bin/python -c "import sys;sys.path.insert(0,'jarvis-core/src');import cve;print(cve.scan()['cve_critiques'],'critiques corrigeables')"

# Reclasser/neutraliser un incident (buffer Redis + self.json) — ex. un artefact de maintenance
./scripts/reclassify-incident.py                      # liste
./scripts/reclassify-incident.py --set 0 maintenance  # reclassify (drops out of the risk scalar)

# Ad-hoc maintenance window: incidents in the next N minutes tagged "maintenance"
curl -X POST "http://localhost:8000/self/maintenance?minutes=120"
```

---


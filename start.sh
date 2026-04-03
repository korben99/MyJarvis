#!/usr/bin/env bash
set -e
cd /opt/jarvis

# Kill uvicorn si déjà en cours
pkill -f "uvicorn main:app" 2>/dev/null && echo "uvicorn précédent tué" || true
sleep 1

# 1. Infrastructure Docker (Redis + Qdrant + Open WebUI)
docker compose up -d

# 2. jarvis-api natif (MLX se charge au démarrage, ~30 s premier lancement)
mkdir -p /opt/jarvis/logs
source /opt/jarvis/venv/bin/activate
cd /opt/jarvis/jarvis-core/src
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 \
    --log-level info &
echo "jarvis-api démarré PID=$!"
echo "Logs : tail -f /opt/jarvis/logs/jarvis-api.log"

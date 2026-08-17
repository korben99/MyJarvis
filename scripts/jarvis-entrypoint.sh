#!/usr/bin/env bash
# Point d'entrée du service — CE que launchd exécute (ProgramArguments du plist).
# À ne pas confondre avec scripts/jarvis-launchd.sh, qui PILOTE launchd (start/stop/restart).
# PAS de & et un `exec uvicorn` final : launchd doit rester parent direct du process pour
# que KeepAlive fonctionne et que son SIGTERM d'arrêt atteigne uvicorn (shutdown gracieux).

# Garde : si le port 8000 est déjà occupé, un Jarvis tourne déjà.
# On sort proprement (exit 0) pour éviter la boucle KeepAlive.
if lsof -iTCP:8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') jarvis-service: port 8000 already in use — exiting cleanly (no restart)" >&2
    exit 0
fi

# Attendre qu'OrbStack/Docker soit prêt (max 60s après le boot)
for i in $(seq 1 30); do
    timeout 5 docker info >/dev/null 2>&1 && break
    sleep 2
done

# Démarrer l'infra Docker
cd /opt/jarvis
docker compose up -d

# Démarrer uvicorn (exec = launchd devient parent direct → KeepAlive fonctionne)
source /opt/jarvis/venv/bin/activate
cd /opt/jarvis/jarvis-core/src

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 \
    --loop uvloop --http httptools \
    --log-level info

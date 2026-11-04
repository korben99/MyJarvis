#!/usr/bin/env bash
# Point d'entrée du service — CE que launchd exécute (ProgramArguments du plist).
# À ne pas confondre avec scripts/jarvis-launchd.sh, qui PILOTE launchd (start/stop/restart).
# PAS de & et un `exec uvicorn` final : launchd doit rester parent direct du process pour
# que KeepAlive fonctionne et que son SIGTERM d'arrêt atteigne uvicorn (shutdown gracieux).

# Racine déduite de la position du script, jamais en dur : le plist est rendu depuis un
# template paramétré (__JARVIS_HOME__), donc launchd peut nous lancer depuis n'importe quel
# chemin d'installation. Un /opt/jarvis en dur ici démarrait l'infra et uvicorn d'une AUTRE
# installation que celle que le plist désigne.
JARVIS_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Garde : si le port 8000 est déjà occupé, un Jarvis tourne déjà.
# On sort proprement (exit 0) pour éviter la boucle KeepAlive.
if lsof -iTCP:8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') jarvis-service: port 8000 already in use — exiting cleanly (no restart)" >&2
    exit 0
fi

# Plafond de mémoire GPU unifiée, lu dans .env et jamais en dur : la valeur dépend de la
# RAM de la machine et de ce qu'il faut y laisser au système et aux conteneurs. Sous le
# plafond que macOS applique par défaut, un prompt d'agent de plusieurs dizaines de
# milliers de jetons fait échouer l'allocation Metal en cours de prefill — les poids du
# modèle et le cache de préfixes occupent déjà l'essentiel du budget.
# Le réglage est volatile, il retombe à 0 à chaque démarrage de la machine : d'où sa place
# ici plutôt que dans une procédure d'installation.
# `sudo -n` n'invite jamais à saisir un mot de passe, launchd n'ayant pas de terminal.
# Sans règle sudoers, le réglage est donc sauté et journalisé ; le service démarre.
LIMITE_METAL_MB="$(
    sed -n 's/^[[:space:]]*METAL_WIRED_LIMIT_MB[[:space:]]*=[[:space:]]*\([0-9]\{1,\}\).*/\1/p' \
        "$JARVIS_HOME/.env" 2>/dev/null | tail -1
)"
if [ -n "$LIMITE_METAL_MB" ] &&
   [ "$(sysctl -n iogpu.wired_limit_mb 2>/dev/null)" != "$LIMITE_METAL_MB" ]; then
    if sudo -n sysctl -w "iogpu.wired_limit_mb=$LIMITE_METAL_MB" >/dev/null 2>&1; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') jarvis-service: iogpu.wired_limit_mb=$LIMITE_METAL_MB appliqué" >&2
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') jarvis-service: iogpu.wired_limit_mb laissé au défaut système — sudo sans mot de passe indisponible" >&2
    fi
fi

# Attendre qu'OrbStack/Docker soit prêt (max 60s après le boot)
for i in $(seq 1 30); do
    timeout 5 docker info >/dev/null 2>&1 && break
    sleep 2
done

# Démarrer l'infra Docker
cd "$JARVIS_HOME"
docker compose up -d

# Démarrer uvicorn (exec = launchd devient parent direct → KeepAlive fonctionne)
source "$JARVIS_HOME/venv/bin/activate"
cd "$JARVIS_HOME/jarvis-core/src"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 \
    --loop uvloop --http httptools \
    --log-level info

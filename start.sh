#!/usr/bin/env bash
set -e

cd /opt/jarvis

PID_FILE="/tmp/jarvis.pid"

echo "Stopping existing Jarvis..."

if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat $PID_FILE)
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "Killing PID $OLD_PID"
        kill $OLD_PID || true
        sleep 2
        kill -9 $OLD_PID 2>/dev/null || true
    fi
    rm -f $PID_FILE
fi

# Kill sécurité (au cas où)
pkill -f "uvicorn main:app" 2>/dev/null || true

echo "Starting infra..."
docker compose up -d

echo "Starting Jarvis..."

mkdir -p /opt/jarvis/logs
source /opt/jarvis/venv/bin/activate
cd /opt/jarvis/jarvis-core/src

uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 \
    --log-level info &

PID=$!
echo $PID > $PID_FILE

echo "Jarvis démarré PID=$PID"

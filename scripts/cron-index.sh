#!/bin/bash
# Cron job: index new documents every 15 minutes
# Logs to /opt/jarvis/logs/cron-index.log

LOG="/opt/jarvis/logs/cron-index.log"
VENV="/opt/jarvis/venv/bin/python3"
SCRIPT="/opt/jarvis/scripts/upload-to-openwebui.py"
LOCK="/tmp/jarvis-indexing.lock"

# Prevent concurrent runs
if [ -f "$LOCK" ]; then
    exit 0
fi
touch "$LOCK"

echo "$(date '+%Y-%m-%d %H:%M:%S') — Starting index check..." >> "$LOG"
$VENV $SCRIPT >> "$LOG" 2>&1
echo "$(date '+%Y-%m-%d %H:%M:%S') — Done." >> "$LOG"
echo "" >> "$LOG"

rm -f "$LOCK"

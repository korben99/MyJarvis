#!/usr/bin/env bash
# Sauvegarde Jarvis — exclut RAGData, modèles (sauf LoRA routeur), collections RAG Qdrant
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
JARVIS_DIR="/opt/jarvis"
USB_MOUNT="/Volumes/NO NAME"
BACKUP_ROOT="${BACKUP_ROOT:-$USB_MOUNT/jarvis_backups}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="$BACKUP_ROOT/$TIMESTAMP"
QDRANT_URL="http://localhost:6333"
QDRANT_MEMORY_COLLECTION="jarvis_memory"
LOG="$BACKUP_DIR/backup.log"
KEEP_LAST=5

# Mode : "backup" (défaut) ou "updates" (sauvegarde PUIS mises à jour Docker + venv).
#   ./backup-jarvis.sh            → sauvegarde seule
#   ./backup-jarvis.sh updates    → sauvegarde, puis pull Docker + pip upgrade, sous maintenance
MODE="${1:-backup}"

# ── Vérification clé USB ──────────────────────────────────────────────────────
if [ ! -d "$USB_MOUNT" ]; then
  echo "[ERROR] Clé USB non montée : $USB_MOUNT" >&2
  echo "        Branchez la clé et réessayez." >&2
  exit 1
fi

# Espace disponible (en Mo)
USB_FREE_MB=$(df -m "$USB_MOUNT" | awk 'NR==2 {print $4}')
log_pre() { echo "[$(date +%H:%M:%S)] $*"; }
log_pre "Clé USB détectée — espace libre : ${USB_FREE_MB} Mo"
if [ "$USB_FREE_MB" -lt 500 ]; then
  echo "[ERROR] Espace insuffisant sur la clé USB (${USB_FREE_MB} Mo < 500 Mo)" >&2
  exit 1
fi

# Détection FAT32 (limite fichier 4 Go)
USB_FS=$(diskutil info "$USB_MOUNT" 2>/dev/null | awk '/File System Personality/ {print $NF}' || true)
IS_FAT32=false
[[ "$USB_FS" == *"FAT"* || "$USB_FS" == *"MS-DOS"* ]] && IS_FAT32=true

# ── Helpers ───────────────────────────────────────────────────────────────────
mkdir -p "$BACKUP_DIR"
log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
fail() { echo "[ERROR] $*" | tee -a "$LOG" >&2; exit 1; }

# Fenêtre de maintenance : pendant l'intervention, Jarvis tague les incidents "maintenance"
# (pas de peur, pas de trauma). Posée en direct dans Redis pour éviter d'importer le stack Python.
maint_window() {
  local sec="${1:-1800}"
  docker exec jarvis-redis redis-cli SET jarvis:maintenance '{"reason":"backup-updates"}' EX "$sec" \
    >/dev/null 2>&1 && log "Fenêtre de maintenance posée (${sec}s)" || log "[WARN] maintenance non posée"
}

log "=== Sauvegarde Jarvis — $TIMESTAMP ==="
log "Destination : $BACKUP_DIR"
log "Système de fichiers USB : ${USB_FS:-inconnu} (FAT32=$IS_FAT32)"

# ── 1. Code source Jarvis ─────────────────────────────────────────────────────
log "--- Code source ---"
CODE_DEST="$BACKUP_DIR/code"
mkdir -p "$CODE_DEST"

rsync -a --stats \
  --exclude="models/"            \
  --exclude="RAGData/"           \
  --exclude="venv/"              \
  --exclude="logs/"              \
  --exclude="__pycache__/"       \
  --exclude="*.pyc"              \
  --exclude=".git/"              \
  --exclude="jarvis-core/JarvisData/model_cache/" \
  "$JARVIS_DIR/" "$CODE_DEST/" 2>>"$LOG"

log "Code copié."

# ── 2. Qdrant — snapshot collection jarvis_memory uniquement ──────────────────
log "--- Qdrant : snapshot $QDRANT_MEMORY_COLLECTION ---"
QDRANT_DEST="$BACKUP_DIR/qdrant"
mkdir -p "$QDRANT_DEST"

SNAPSHOT_RESP=$(curl -sf -X POST "$QDRANT_URL/collections/$QDRANT_MEMORY_COLLECTION/snapshots" \
  -H "Content-Type: application/json") \
  || fail "Impossible de créer le snapshot Qdrant (Qdrant accessible ?)"

SNAPSHOT_NAME=$(echo "$SNAPSHOT_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['name'])")
log "Snapshot créé : $SNAPSHOT_NAME"

curl -sf "$QDRANT_URL/collections/$QDRANT_MEMORY_COLLECTION/snapshots/$SNAPSHOT_NAME" \
  -o "$QDRANT_DEST/${QDRANT_MEMORY_COLLECTION}_${SNAPSHOT_NAME}" \
  || fail "Impossible de télécharger le snapshot Qdrant"

curl -sf -X DELETE "$QDRANT_URL/collections/$QDRANT_MEMORY_COLLECTION/snapshots/$SNAPSHOT_NAME" \
  -H "Content-Type: application/json" >> "$LOG" 2>&1 || true

log "Snapshot Qdrant sauvegardé."

# ── 3. Redis ──────────────────────────────────────────────────────────────────
log "--- Redis ---"
REDIS_DEST="$BACKUP_DIR/redis"
mkdir -p "$REDIS_DEST"

docker exec jarvis-redis redis-cli BGSAVE >> "$LOG" 2>&1 || fail "redis-cli BGSAVE échoué"

for i in $(seq 1 30); do
  BGSAVE_STATUS=$(docker exec jarvis-redis redis-cli INFO persistence 2>/dev/null \
    | grep "rdb_bgsave_in_progress" | tr -d '\r' | cut -d: -f2)
  [ "$BGSAVE_STATUS" = "0" ] && break
  sleep 1
done

docker cp jarvis-redis:/data/dump.rdb "$REDIS_DEST/dump.rdb" \
  || fail "Impossible de copier dump.rdb depuis Redis"

log "Redis sauvegardé."

# ── 4. OpenWebUI — base SQLite uniquement (sans uploads RAG ni cache) ─────────
log "--- OpenWebUI (webui.db) ---"
WEBUI_DEST="$BACKUP_DIR/openwebui"
mkdir -p "$WEBUI_DEST"

if docker exec jarvis-webui which sqlite3 > /dev/null 2>&1; then
  docker exec jarvis-webui sqlite3 /app/backend/data/webui.db \
    ".backup /tmp/webui_backup.db" >> "$LOG" 2>&1 \
    && docker cp jarvis-webui:/tmp/webui_backup.db "$WEBUI_DEST/webui.db" \
    && docker exec jarvis-webui rm -f /tmp/webui_backup.db \
    || fail "Backup SQLite (sqlite3) échoué"
else
  log "sqlite3 absent dans le container — copie directe des fichiers db"
  docker cp jarvis-webui:/app/backend/data/webui.db     "$WEBUI_DEST/webui.db"     2>>"$LOG" || true
  docker cp jarvis-webui:/app/backend/data/webui.db-wal "$WEBUI_DEST/webui.db-wal" 2>>"$LOG" || true
  docker cp jarvis-webui:/app/backend/data/webui.db-shm "$WEBUI_DEST/webui.db-shm" 2>>"$LOG" || true
fi

log "OpenWebUI sauvegardé."

# ── 5. Compression ────────────────────────────────────────────────────────────
log "--- Compression ---"

# Dossier temporaire local (évite d'écrire un gros tar en streaming sur FAT32)
TMP_ARCHIVE=$(mktemp -t jarvis_backup_XXXXXX.tar.gz)
trap 'rm -f "$TMP_ARCHIVE"' EXIT

tar -czf "$TMP_ARCHIVE" -C "$BACKUP_ROOT" "$TIMESTAMP" 2>>"$LOG" \
  || fail "Compression échouée"

ARCHIVE_SIZE_BYTES=$(stat -f%z "$TMP_ARCHIVE" 2>/dev/null || stat -c%s "$TMP_ARCHIVE")
log "Taille archive : $(( ARCHIVE_SIZE_BYTES / 1024 / 1024 )) Mo"

# ── 6. Copie vers la clé USB (split si FAT32 et > 3,9 Go) ────────────────────
FINAL_ARCHIVE="$BACKUP_ROOT/jarvis_backup_${TIMESTAMP}.tar.gz"

if $IS_FAT32 && [ "$ARCHIVE_SIZE_BYTES" -gt 4188282880 ]; then
  log "FAT32 détecté et archive > 4 Go — découpage en volumes de 3,9 Go"
  SPLIT_PREFIX="$BACKUP_ROOT/jarvis_backup_${TIMESTAMP}.tar.gz.part"
  split -b 3900m "$TMP_ARCHIVE" "$SPLIT_PREFIX"
  ARCHIVE_DISPLAY="${SPLIT_PREFIX}*"
  log "Archive découpée : $(ls "${SPLIT_PREFIX}"* | wc -l) parties"
  # Créer un fichier de restauration pour rappeler la commande cat
  echo "cat ${SPLIT_PREFIX}* > jarvis_backup_${TIMESTAMP}.tar.gz && tar -xzf jarvis_backup_${TIMESTAMP}.tar.gz" \
    > "$BACKUP_ROOT/restore_${TIMESTAMP}.sh"
  chmod +x "$BACKUP_ROOT/restore_${TIMESTAMP}.sh"
else
  cp "$TMP_ARCHIVE" "$FINAL_ARCHIVE"
  ARCHIVE_DISPLAY="$FINAL_ARCHIVE"
  ARCHIVE_HUMAN=$(du -sh "$FINAL_ARCHIVE" | cut -f1)
  log "Archive finale : $FINAL_ARCHIVE ($ARCHIVE_HUMAN)"
fi

# Supprimer le dossier temporaire
rm -rf "$BACKUP_DIR"
log "Dossier temporaire supprimé."

# ── 7. Rotation — garder les N dernières sauvegardes ─────────────────────────
log "--- Rotation (${KEEP_LAST} dernières) ---"
ls -t "$BACKUP_ROOT"/jarvis_backup_*.tar.gz 2>/dev/null | tail -n +"$(( KEEP_LAST + 1 ))" | while read -r OLD; do
  log "Suppression : $(basename "$OLD")"
  rm -f "$OLD"
done
# Nettoyer aussi les éventuels anciens fichiers split et scripts restore
ls -t "$BACKUP_ROOT"/jarvis_backup_*.tar.gz.partaa 2>/dev/null | tail -n +"$(( KEEP_LAST + 1 ))" | while read -r OLD_PART; do
  PREFIX="${OLD_PART%.partaa}"
  log "Suppression parties : $(basename "$PREFIX").part*"
  rm -f "${PREFIX}".part*
  RESTORE="${BACKUP_ROOT}/restore_$(basename "$PREFIX" | sed 's/jarvis_backup_//' | sed 's/.tar.gz//')".sh
  rm -f "$RESTORE"
done

# ── 8. Reçu local ─────────────────────────────────────────────────────────────
# Seule trace locale d'une sauvegarde réussie : l'archive part sur la clé USB, qui est
# ensuite débranchée. vitals.py lit ce reçu pour connaître l'âge de la dernière sauvegarde.
RECEIPT="$JARVIS_DIR/jarvis-core/JarvisData/backup_receipt.json"
printf '{"completed_at": %s, "iso": "%s", "archive": "%s", "size_mb": %s}\n' \
  "$(date +%s)" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$(basename "$ARCHIVE_DISPLAY")" \
  "$(( ARCHIVE_SIZE_BYTES / 1024 / 1024 ))" \
  > "$RECEIPT"
log "Reçu écrit : $RECEIPT"

log "=== Sauvegarde terminée ==="
log "Archive : $ARCHIVE_DISPLAY"

# ── 9. Mises à jour (optionnel : "backup-jarvis.sh updates") ─────────────────
# Toujours APRÈS la sauvegarde : on ne touche à rien tant qu'une copie n'est pas sécurisée.
if [ "$MODE" = "updates" ]; then
  log "=== Mises à jour ==="
  maint_window 1800   # couvre pull/recreate Docker + upgrade venv + redémarrage Jarvis

  log "--- Docker : pull + recreate (volumes préservés) ---"
  ( cd "$JARVIS_DIR" && docker compose pull && docker compose up -d ) 2>&1 \
    | while IFS= read -r l; do log "  $l"; done \
    || log "[WARN] mise à jour Docker incomplète — vérifier manuellement"
  docker image prune -f >/dev/null 2>&1 && log "  images obsolètes purgées" || true

  log "--- Python (venv) : pip install -r requirements.txt --upgrade (en place, pas de mv) ---"
  /opt/jarvis/venv/bin/python -m pip install -r "$JARVIS_DIR/requirements.txt" --upgrade --quiet \
    && log "  venv à jour" || log "[WARN] échec pip — vérifier manuellement"

  log "Rappel : un bump de l'interpréteur Python (brew) = rebuild venv manuel (voir README)."
  log "Rappel : redémarrer Jarvis pour charger le venv à jour — la fenêtre de maintenance couvre le redémarrage."
  log "=== Mises à jour terminées ==="
fi

echo "$ARCHIVE_DISPLAY"

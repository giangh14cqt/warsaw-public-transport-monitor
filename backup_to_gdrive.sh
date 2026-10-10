#!/usr/bin/env bash
# ==============================================================================
# Warsaw Transit Telemetry - Daily Google Drive Backup Script
# Synchronizes append-only Parquet partitions to Google Drive via rclone.
# ==============================================================================

set -euo pipefail

# Determine script directory (project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Automatically load .env if present
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Configuration (can be overridden via environment variables or .env)
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"
DEST_FOLDER="${DEST_FOLDER:-WarsawDelayTelemetry/data/raw}"
SOURCE_DIR="${SOURCE_DIR:-$SCRIPT_DIR/data/raw}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
LOG_FILE="$LOG_DIR/backup.log"

mkdir -p "$LOG_DIR"

echo "==================================================================" >> "$LOG_FILE"
echo "[$(date -u '+%Y-%m-%d %H:%M:%SZ')] Starting Google Drive backup..." >> "$LOG_FILE"
echo "Source: $SOURCE_DIR" >> "$LOG_FILE"
echo "Destination: $RCLONE_REMOTE:$DEST_FOLDER" >> "$LOG_FILE"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "[$(date -u '+%Y-%m-%d %H:%M:%SZ')] WARNING: Source directory '$SOURCE_DIR' does not exist yet. Nothing to backup." >> "$LOG_FILE"
    exit 0
fi

# Run rclone copy (only uploads new or modified .parquet files; never deletes)
rclone copy "$SOURCE_DIR" "$RCLONE_REMOTE:$DEST_FOLDER" \
    --fast-list \
    --transfers=4 \
    --checkers=8 \
    --log-file="$LOG_FILE" \
    --log-level=INFO

echo "[$(date -u '+%Y-%m-%d %H:%M:%SZ')] Backup completed successfully." >> "$LOG_FILE"
echo "==================================================================" >> "$LOG_FILE"

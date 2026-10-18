#!/usr/bin/env bash
# ==============================================================================
# Warsaw Transit Historian - 6-Hourly Google Drive Backup Script
# Synchronizes raw Bronze responses, Silver trajectory fixes, and Gold stop events
# to Google Drive via rclone (every 6h at :30).
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

# Configuration (overridable via environment variables or .env)
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"
HISTORIAN_DEST_FOLDER="${HISTORIAN_DEST_FOLDER:-${HISTORIAN_BACKUP_DEST:-WarsawDelayTelemetry/data/historian}}"
HISTORIAN_SOURCE_DIR="${HISTORIAN_SOURCE_DIR:-$SCRIPT_DIR/data/historian}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
LOG_FILE="$LOG_DIR/backup_historian.log"

mkdir -p "$LOG_DIR"

# Determine full rclone target
if [[ "$HISTORIAN_DEST_FOLDER" == *":"* ]]; then
    FULL_TARGET="$HISTORIAN_DEST_FOLDER"
else
    FULL_TARGET="$RCLONE_REMOTE:$HISTORIAN_DEST_FOLDER"
fi

echo "==================================================================" >> "$LOG_FILE"
echo "[$(date -u '+%Y-%m-%d %H:%M:%SZ')] Starting Historian Google Drive backup..." >> "$LOG_FILE"
echo "Source: $HISTORIAN_SOURCE_DIR" >> "$LOG_FILE"
echo "Destination: $FULL_TARGET" >> "$LOG_FILE"

if [ ! -d "$HISTORIAN_SOURCE_DIR" ]; then
    echo "[$(date -u '+%Y-%m-%d %H:%M:%SZ')] WARNING: Source directory '$HISTORIAN_SOURCE_DIR' does not exist yet. Nothing to backup." >> "$LOG_FILE"
    exit 0
fi

# Run rclone copy (uploads new or modified files; never deletes)
rclone copy "$HISTORIAN_SOURCE_DIR" "$FULL_TARGET" \
    --fast-list \
    --transfers=4 \
    --checkers=8 \
    --log-file="$LOG_FILE" \
    --log-level=INFO

echo "[$(date -u '+%Y-%m-%d %H:%M:%SZ')] Historian backup completed successfully." >> "$LOG_FILE"
echo "==================================================================" >> "$LOG_FILE"

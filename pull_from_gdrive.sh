#!/usr/bin/env bash
# ==============================================================================
# Warsaw Transit Telemetry - Pull Parquet Partitions from Google Drive
# Synchronizes raw telemetry partitions from Google Drive down to local data/raw/
# for local analytics, ML feature engineering, and xAI evaluations.
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
REMOTE_FOLDER="${REMOTE_FOLDER:-${DEST_FOLDER:-WarsawDelayTelemetry/data/raw}}"
TARGET_DIR="${TARGET_DIR:-$SCRIPT_DIR/data/raw}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
LOG_FILE="$LOG_DIR/pull_gdrive.log"

mkdir -p "$LOG_DIR"
mkdir -p "$TARGET_DIR"

echo "==================================================================" | tee -a "$LOG_FILE"
echo "[$(date -u '+%Y-%m-%d %H:%M:%SZ')] Pulling telemetry partitions from Google Drive..." | tee -a "$LOG_FILE"
echo "Remote Source:      $RCLONE_REMOTE:$REMOTE_FOLDER" | tee -a "$LOG_FILE"
echo "Local Destination:  $TARGET_DIR" | tee -a "$LOG_FILE"
echo "Log File:           $LOG_FILE" | tee -a "$LOG_FILE"

# Check if rclone is installed
if ! command -v rclone &> /dev/null; then
    echo "ERROR: 'rclone' is not installed or not in PATH." | tee -a "$LOG_FILE"
    echo "Install via Homebrew on macOS: brew install rclone" | tee -a "$LOG_FILE"
    exit 1
fi

# Run rclone copy (downloads only new/updated partitions; preserves local files)
rclone copy "$RCLONE_REMOTE:$REMOTE_FOLDER" "$TARGET_DIR" \
    --fast-list \
    --transfers=8 \
    --checkers=16 \
    --update \
    --stats=5s \
    --log-file="$LOG_FILE" \
    --log-level=INFO \
    --stats-log-level=NOTICE \
    -P

echo "[$(date -u '+%Y-%m-%d %H:%M:%SZ')] Synchronization complete." | tee -a "$LOG_FILE"

# Print quick local inventory summary if duckdb is available
if [ -d "$TARGET_DIR" ]; then
    PARTITION_COUNT=$(find "$TARGET_DIR" -type f -name "*.parquet" 2>/dev/null | wc -l | tr -d ' ')
    echo "Local Parquet Partitions Available: $PARTITION_COUNT files" | tee -a "$LOG_FILE"
fi

echo "==================================================================" | tee -a "$LOG_FILE"

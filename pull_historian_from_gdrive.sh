#!/usr/bin/env bash
# ==============================================================================
# Warsaw Transit Historian - Pull Partitions from Google Drive
# Synchronizes Bronze, Silver, and Gold Historian datasets from Google Drive
# down to local workstation data/historian/ for offline modeling and xAI.
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
REMOTE_FOLDER="${HISTORIAN_DEST_FOLDER:-${HISTORIAN_BACKUP_DEST:-WarsawDelayTelemetry/data/historian}}"
TARGET_DIR="${HISTORIAN_TARGET_DIR:-$SCRIPT_DIR/data/historian}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
LOG_FILE="$LOG_DIR/pull_historian_gdrive.log"

mkdir -p "$LOG_DIR"
mkdir -p "$TARGET_DIR"

# Determine full remote source
if [[ "$REMOTE_FOLDER" == *":"* ]]; then
    FULL_SOURCE="$REMOTE_FOLDER"
else
    FULL_SOURCE="$RCLONE_REMOTE:$REMOTE_FOLDER"
fi

echo "==================================================================" | tee -a "$LOG_FILE"
echo "[$(date -u '+%Y-%m-%d %H:%M:%SZ')] Pulling Historian datasets from Google Drive..." | tee -a "$LOG_FILE"
echo "Remote Source:      $FULL_SOURCE" | tee -a "$LOG_FILE"
echo "Local Destination:  $TARGET_DIR" | tee -a "$LOG_FILE"
echo "Log File:           $LOG_FILE" | tee -a "$LOG_FILE"

# Check if rclone is installed
if ! command -v rclone &> /dev/null; then
    echo "ERROR: 'rclone' is not installed or not in PATH." | tee -a "$LOG_FILE"
    echo "Install via Homebrew on macOS: brew install rclone" | tee -a "$LOG_FILE"
    exit 1
fi

# Optional flags processing
EXTRA_RCLONE_FLAGS=()
for arg in "$@"; do
    case "$arg" in
        --no-bronze)
            EXTRA_RCLONE_FLAGS+=(--exclude "bronze/**")
            echo "Option: Excluded Bronze raw telemetry (--no-bronze)." | tee -a "$LOG_FILE"
            ;;
        --compacted-only)
            EXTRA_RCLONE_FLAGS+=(--exclude "bronze/**/hour=*/**")
            echo "Option: Excluded loose tick files, pulling only hourly bundles (--compacted-only)." | tee -a "$LOG_FILE"
            ;;
        *)
            EXTRA_RCLONE_FLAGS+=("$arg")
            ;;
    esac
done

# Run rclone copy (downloads only new/updated files; preserves local files)
# Excludes active temporary files and extracted GTFS text/duckdb files
rclone copy "$FULL_SOURCE" "$TARGET_DIR" \
    --fast-list \
    --transfers=8 \
    --checkers=16 \
    --update \
    --stats=5s \
    --exclude "*.tmp" \
    --exclude "gtfs/*/**" \
    ${EXTRA_RCLONE_FLAGS[@]+"${EXTRA_RCLONE_FLAGS[@]}"} \
    --log-file="$LOG_FILE" \
    --log-level=INFO \
    --stats-log-level=NOTICE \
    -P

echo "[$(date -u '+%Y-%m-%d %H:%M:%SZ')] Synchronization complete." | tee -a "$LOG_FILE"
echo "==================================================================" | tee -a "$LOG_FILE"

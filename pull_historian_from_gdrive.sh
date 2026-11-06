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
# Default behavior ("Good Case"): skip raw bronze/source=*/** (.tar.zst / .gz) and pull
# only daily partitioned bronze_parquet/, bronze/_ticks/, silver/, gold/, and gtfs/*.zip.
# Pass --with-raw-bronze (optionally with --date YYYY-MM-DD) for cold backup recovery.
EXTRA_RCLONE_FLAGS=()
PULL_RAW_BRONZE=0
TARGET_DATE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-bronze)
            PULL_RAW_BRONZE=0
            shift
            ;;
        --with-raw-bronze|--compacted-only)
            PULL_RAW_BRONZE=1
            shift
            ;;
        --date)
            if [[ -n "${2:-}" ]]; then
                TARGET_DATE="$2"
                shift 2
            else
                echo "ERROR: --date requires a YYYY-MM-DD argument." | tee -a "$LOG_FILE"
                exit 1
            fi
            ;;
        --date=*)
            TARGET_DATE="${1#*=}"
            shift
            ;;
        *)
            EXTRA_RCLONE_FLAGS+=("$1")
            shift
            ;;
    esac
done

if [[ -n "$TARGET_DATE" ]]; then
    if [[ "$PULL_RAW_BRONZE" -eq 1 ]]; then
        echo "Option: Recovery Mode — pulling cold raw Bronze (.tar.zst) + daily Parquet for date=$TARGET_DATE." | tee -a "$LOG_FILE"
        EXTRA_RCLONE_FLAGS+=(
            --include "/bronze/source=*/date=${TARGET_DATE}/*.tar.zst"
            --include "/bronze_parquet/date=${TARGET_DATE}/**"
            --include "/bronze/_ticks/date=${TARGET_DATE}/**"
            --include "/silver/date=${TARGET_DATE}/**"
            --include "/gold/stop_events/date=${TARGET_DATE}/**"
        )
    else
        echo "Option: Fast Mode — pulling daily Parquet partitions for date=$TARGET_DATE." | tee -a "$LOG_FILE"
        EXTRA_RCLONE_FLAGS+=(
            --include "/bronze_parquet/date=${TARGET_DATE}/**"
            --include "/bronze/_ticks/date=${TARGET_DATE}/**"
            --include "/silver/date=${TARGET_DATE}/**"
            --include "/gold/stop_events/date=${TARGET_DATE}/**"
        )
    fi
else
    if [[ "$PULL_RAW_BRONZE" -eq 1 ]]; then
        EXTRA_RCLONE_FLAGS+=(--exclude "bronze/**/hour=*/**")
        echo "Option: Recovery Mode (--with-raw-bronze) — pulling cold hourly .tar.zst archives in addition to daily Parquet." | tee -a "$LOG_FILE"
    else
        EXTRA_RCLONE_FLAGS+=(--exclude "bronze/source=*/**")
        echo "Option: Fast Parquet Mode (default) — skipping cold raw bronze/source=*/** archives (use --with-raw-bronze to pull raw .tar.zst backup)." | tee -a "$LOG_FILE"
    fi
fi

# Run rclone copy (downloads only new/updated files; preserves local files)
# Excludes active temporary files and extracted GTFS text/duckdb files
if [[ -n "$TARGET_DATE" ]]; then
    rclone copy "$FULL_SOURCE" "$TARGET_DIR" \
        --fast-list \
        --transfers=8 \
        --checkers=16 \
        --update \
        --stats=5s \
        ${EXTRA_RCLONE_FLAGS[@]+"${EXTRA_RCLONE_FLAGS[@]}"} \
        --log-file="$LOG_FILE" \
        --log-level=INFO \
        --stats-log-level=NOTICE \
        -P
else
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
fi

echo "[$(date -u '+%Y-%m-%d %H:%M:%SZ')] Synchronization complete." | tee -a "$LOG_FILE"
echo "==================================================================" | tee -a "$LOG_FILE"

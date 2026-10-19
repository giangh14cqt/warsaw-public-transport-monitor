#!/usr/bin/env python3
"""
One-Time Migration Script: Bronze Loose Ticks to Hourly Bundles.
Compacts all historical loose tick_*.gz files into hour=HH.tar.zst bundles,
verifies payload SHA256 integrity, and optionally cleans up Google Drive.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env if present
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    try:
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() not in os.environ:
                        os.environ[k.strip()] = v.strip()
    except Exception:
        pass

from src.historian.compact import HourlyBronzeCompactor
from src.historian.config import paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bronze_migration")


def get_rclone_target() -> str:
    remote = os.environ.get("RCLONE_REMOTE", "gdrive")
    folder = os.environ.get("HISTORIAN_DEST_FOLDER", os.environ.get("HISTORIAN_BACKUP_DEST", "WarsawDelayTelemetry/data/historian"))
    if ":" in folder:
        return folder
    return f"{remote}:{folder}"


def run_command(cmd: list[str]) -> bool:
    logger.info(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        logger.error(f"Command failed with exit code {res.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Migrate Historian Bronze loose tick files into hourly tar.zst bundles.")
    parser.add_argument("--bronze-dir", type=Path, default=paths.bronze_dir, help="Path to bronze directory")
    parser.add_argument("--compression-level", type=int, default=15, help="Zstandard level (default 15)")
    parser.add_argument("--include-current-hour", action="store_true", help="Force compaction of the current hour")
    parser.add_argument("--keep-loose-files", action="store_true", help="Keep loose tick files locally")
    parser.add_argument("--clean-remote-drive", action="store_true", help="Sync bundles to Google Drive and delete loose files on Drive")
    parser.add_argument("--dedupe-only", action="store_true", help="Only run rclone dedupe on Google Drive")
    args = parser.parse_args()

    remote_target = get_rclone_target()

    if args.dedupe_only:
        logger.info(f"Deduplicating Google Drive target: {remote_target}")
        run_command(["rclone", "dedupe", "--dedupe-mode", "largest", remote_target])
        return

    logger.info("=== STEP 1: Local Bronze Hourly Compaction ===")
    compactor = HourlyBronzeCompactor(
        bronze_dir=args.bronze_dir,
        compression_level=args.compression_level,
    )

    results = compactor.compact_all(
        only_past_hours=not args.include_current_hour,
        delete_source=not args.keep_loose_files,
    )

    compacted = [r for r in results if r.get("status") == "COMPACTED"]
    logger.info(f"Local compaction finished. Compacted {len(compacted)} hour directories.")

    if not args.clean_remote_drive:
        logger.info("To synchronize and clean Google Drive, run this script with --clean-remote-drive.")
        return

    logger.info("=== STEP 2: Sync New Bundles to Google Drive ===")
    # 2a. Sync bundles to remote
    sync_ok = run_command([
        "rclone", "copy",
        str(args.bronze_dir),
        f"{remote_target}/bronze",
        "--fast-list",
        "--transfers=8",
        "--checkers=16",
        "--include", "**/hour=*.tar.zst",
        "-P",
    ])
    if not sync_ok:
        logger.error("Failed to sync bundles to remote. Aborting remote deletion.")
        return

    logger.info("=== STEP 3: Deduplicate Google Drive Remote ===")
    run_command(["rclone", "dedupe", "--dedupe-mode", "largest", remote_target])

    logger.info("=== STEP 4: Remove Loose Files from Google Drive ===")
    # Delete loose tick files on remote
    del_ok = run_command([
        "rclone", "delete",
        f"{remote_target}/bronze",
        "--include", "**/hour=*/*.gz",
        "-P",
    ])
    if del_ok:
        logger.info("Pruning empty remote directories...")
        run_command(["rclone", "rmdirs", f"{remote_target}/bronze", "--leave-root"])

    logger.info("=== STEP 5: Clean Up Remote GTFS (Keep zip & versions only) ===")
    run_command([
        "rclone", "delete",
        f"{remote_target}/gtfs",
        "--exclude", "*.zip",
        "--exclude", "*.parquet",
        "-P",
    ])
    run_command(["rclone", "rmdirs", f"{remote_target}/gtfs", "--leave-root"])

    logger.info("=== MIGRATION COMPLETE ===")
    logger.info(f"Target {remote_target} is now clean, compacted, and deduplicated.")


if __name__ == "__main__":
    main()

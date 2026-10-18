"""
Warsaw Observed-Delay Historian: Batch Reconstruction Pipeline CLI (Track B).
Transforms raw Bronze files into Silver trajectories and Gold stop arrival/departure events.
Usage:
    python -m src.historian.reconstruct --date 2026-10-18
    python -m src.historian.reconstruct --date today
"""

import argparse
import datetime
import logging
import sys

from src.historian.gold import GoldStopEventReconstructor
from src.historian.silver import SilverTrajectoryProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("historian_reconstruct")


def run_reconstruction(date_str: str) -> bool:
    """Runs Silver trajectory parsing and Gold stop event reconstruction for a single date."""
    if date_str == "today":
        date_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

    logger.info(f"=== Starting Historian Track B Reconstruction for {date_str} ===")

    # 1. Bronze -> Silver (Cleaned trajectories & GTFS trip matching)
    silver_proc = SilverTrajectoryProcessor()
    silver_file = silver_proc.process_date(date_str)
    if not silver_file:
        logger.error(f"Silver trajectory processing failed or produced no records for {date_str}.")
        return False

    # 2. Silver -> Gold (Stop arrival events, sub-10s interpolation, delays)
    gold_proc = GoldStopEventReconstructor()
    gold_file = gold_proc.process_date(date_str)
    if not gold_file:
        logger.error(f"Gold stop event reconstruction failed or produced no records for {date_str}.")
        return False

    logger.info(f"=== Track B Reconstruction Successfully Completed for {date_str} ===")
    logger.info(f"  Silver Parquet : {silver_file}")
    logger.info(f"  Gold Parquet   : {gold_file}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warsaw Observed-Delay Reconstruction CLI (Track B)")
    parser.add_argument(
        "--date",
        type=str,
        default="today",
        help="Date to reconstruct in YYYY-MM-DD format, or 'today' (default: today)",
    )
    args = parser.parse_args()

    success = run_reconstruction(args.date)
    sys.exit(0 if success else 1)

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


def run_reconstruction(
    date_str: str,
    require_complete_trips: bool = True,
    force: bool = False,
) -> bool:
    """Runs Silver trajectory parsing and Gold stop event reconstruction for a single date."""
    if date_str == "today":
        date_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

    logger.info(f"=== Starting Historian Track B Reconstruction for {date_str} (force={force}) ===")

    # 1. Bronze -> Silver (Cleaned trajectories & GTFS trip matching)
    silver_proc = SilverTrajectoryProcessor()
    silver_file = silver_proc.process_date(date_str, force=force)
    if not silver_file:
        logger.error(f"Silver trajectory processing failed or produced no records for {date_str}.")
        return False

    # 2. Silver -> Gold (Stop arrival events, sub-10s interpolation, delays)
    gold_proc = GoldStopEventReconstructor(require_complete_trips=require_complete_trips)
    gold_file = gold_proc.process_date(date_str, force=force)
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
        help="Date to reconstruct in YYYY-MM-DD format, comma-separated list, 'all', or 'today' (default: today)",
    )
    parser.add_argument(
        "--include-incomplete-trips",
        action="store_true",
        default=False,
        help="Include partial trips that do not span from starting stop to terminal stop (default: False)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-extraction of Silver trajectories from Bronze files even if Silver partition exists",
    )
    args = parser.parse_args()

    from src.historian.config import paths

    target_dates = []
    if args.date == "all":
        found_dates = set()
        for src in ["um_bus", "um_tram"]:
            src_dir = paths.bronze_dir / f"source={src}"
            if src_dir.exists():
                for p in src_dir.glob("date=*"):
                    found_dates.add(p.name.split("=")[-1])
        target_dates = sorted(found_dates)
    elif "," in args.date:
        target_dates = [d.strip() for d in args.date.split(",") if d.strip()]
    else:
        target_dates = [args.date]

    all_success = True
    for d in target_dates:
        success = run_reconstruction(
            d,
            require_complete_trips=not args.include_incomplete_trips,
            force=args.force,
        )
        if not success:
            all_success = False

    sys.exit(0 if all_success else 1)

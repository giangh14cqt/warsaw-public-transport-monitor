#!/usr/bin/env python3
"""
Phase 2: Production Continuous Polling Daemon for Warsaw Transit Delay Ingestion.
Polls https://cdn.zbiorkom.live/gtfs-rt/warsaw.pb at 30s intervals.
Deduplicates records on (trip_id, stop_id, stop_sequence, observed_at).
Appends to DuckDB (data/delays.duckdb) and logs periodic health metrics.
"""

import os
import sys
import time
import signal
import logging
from datetime import datetime

from src.parser import (
    WarsawTransitParser,
    CriticalBlockerError,
    TransientHttpError,
)
from src.db import DelayDatabase, DEFAULT_DB_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("collector")

POLL_INTERVAL_SECONDS = 30
HEALTH_REPORT_INTERVAL_SECONDS = 600  # 10 minutes


class TransitCollector:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.running = False
        self.parser = WarsawTransitParser(data_dir="data")
        self.db = DelayDatabase(db_path=self.db_path)
        self.total_inserted = 0
        self.last_health_report = time.time()
        self._setup_signals()

    def _setup_signals(self):
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info(f"Received signal {sig_name} ({signum}). Initiating graceful shutdown...")
        self.running = False

    def log_health_metrics(self):
        stats = self.db.get_summary_stats()
        avg_delay_str = f"{stats['avg_delay_sec']:.1f}s" if stats['avg_delay_sec'] is not None else "N/A"
        min_delay_str = f"{stats['min_delay_sec']}s" if stats['min_delay_sec'] is not None else "N/A"
        max_delay_str = f"{stats['max_delay_sec']}s" if stats['max_delay_sec'] is not None else "N/A"
        size_kb = stats['db_size_bytes'] / 1024

        logger.info("=" * 60)
        logger.info("PERIODIC HEALTH METRICS (10-minute heartbeat)")
        logger.info(f"  • Total DB Rows:       {stats['total_records']:,}")
        logger.info(f"  • Active Trips:        {stats['active_trips']:,}")
        logger.info(f"  • Active Vehicles:     {stats['active_vehicles']:,}")
        logger.info(f"  • Active Routes:       {stats['active_routes']:,}")
        logger.info(f"  • Average Delay:       {avg_delay_str}")
        logger.info(f"  • Delay Range:         [{min_delay_str} .. {max_delay_str}]")
        logger.info(f"  • Storage Size:        {size_kb:,.1f} KB ({stats['db_size_bytes']:,} bytes)")
        logger.info(f"  • Session Inserted:    {self.total_inserted:,} rows")
        logger.info("=" * 60)
        self.last_health_report = time.time()

    def run(self):
        self.running = True
        logger.info(f"Starting Warsaw Transit Delay Collector daemon (Interval: {POLL_INTERVAL_SECONDS}s)...")
        logger.info(f"Target Database: {self.db_path}")

        cycle_count = 0
        while self.running:
            cycle_start = time.time()
            cycle_count += 1

            try:
                # 1. Fetch GTFS-RT feed snapshot
                feed_bytes = self.parser.fetch_feed_bytes()

                # 2. Parse protobuf and normalize delays
                records, stats = self.parser.parse_snapshot(feed_bytes)

                # 3. Deduplicate and insert into DuckDB
                new_inserted = self.db.insert_delays(records)
                self.total_inserted += new_inserted

                logger.info(
                    f"Cycle #{cycle_count}: Parsed {stats['total_trip_updates']} trip updates "
                    f"({stats['records_with_delay']} with delay) | "
                    f"New rows inserted: {new_inserted} (Total session: {self.total_inserted:,})"
                )

            except CriticalBlockerError as e:
                logger.critical(f"[CRITICAL BLOCKER]: Endpoint response unexpected. Human review required. Error: {e}")
                self.running = False
                break

            except TransientHttpError as e:
                logger.error(f"Transient HTTP failure in poll cycle: {e}")

            except Exception as e:
                logger.exception(f"Unexpected error in collection cycle: {e}")

            # 4. Check periodic health report (every 10 minutes)
            if time.time() - self.last_health_report >= HEALTH_REPORT_INTERVAL_SECONDS:
                self.log_health_metrics()

            # 5. Sleep respecting poll interval and checking running flag
            elapsed = time.time() - cycle_start
            sleep_duration = max(0.0, POLL_INTERVAL_SECONDS - elapsed)
            
            # Sleep in small slices to be responsive to signals
            sleep_until = time.time() + sleep_duration
            while self.running and time.time() < sleep_until:
                time.sleep(0.5)

        logger.info("Collector daemon stopped. Performing cleanup...")
        self.log_health_metrics()
        self.parser.close()
        self.db.close()
        logger.info("Graceful shutdown complete.")


if __name__ == "__main__":
    collector = TransitCollector()
    collector.run()

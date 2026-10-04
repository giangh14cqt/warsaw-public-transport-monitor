#!/usr/bin/env python3
"""
Step 2: Continuous Production Daemon for Warsaw Transit Delay Ingestion.
Polls https://cdn.zbiorkom.live/gtfs-rt/warsaw.pb at 30-second intervals.
Deduplicates on (trip_id, stop_sequence, rt_arrival_time).
Appends to partitioned Parquet files under data/raw/year=YYYY/month=MM/day=DD/.
Buffers in-memory and flushes every 5 minutes or 5,000 records.
Logs health metrics every 15 minutes to logs/collector_health.log.
Graceful shutdown on SIGINT / SIGTERM flushes all remaining records.
"""

import os
import sys
import time
import signal
import logging
import datetime
from typing import Optional

from logging.handlers import RotatingFileHandler

from src.gtfs_manager import GTFSManager
from src.fetcher import (
    GTFSRTFetcher,
    CriticalSchemaMismatchError,
    RateLimitCircuitBreakerError,
    NetworkOutageError,
    TransientHttpError,
)
from src.storage import ParquetPartitionStorage, DuckDBCatalog

# Configure dual logging: console and rotating file (max 20MB, 5 backups)
os.makedirs("logs", exist_ok=True)
health_logger = logging.getLogger("collector_health")
health_logger.setLevel(logging.INFO)
file_handler = RotatingFileHandler("logs/collector_health.log", maxBytes=20 * 1024 * 1024, backupCount=5, mode="a")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
health_logger.addHandler(file_handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("collector_daemon")

# Automatically load .env if present
if os.path.exists(".env"):
    try:
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() not in os.environ:
                        os.environ[k.strip()] = v.strip()
    except Exception:
        pass

POLL_INTERVAL_SECONDS = 30
HEALTH_REPORT_INTERVAL_SECONDS = 900  # 15 minutes
WEEKLY_CHECK_INTERVAL_SECONDS = 86400  # Daily check for weekly schedule update
HEARTBEAT_URL = os.getenv("HEARTBEAT_URL", "").strip()


class WarsawCollectorDaemon:
    """Production daemon collecting continuous delay telemetry."""

    def __init__(
        self,
        raw_dir: str = "data/raw",
        gtfs_dir: str = "data/gtfs",
        poll_interval: int = POLL_INTERVAL_SECONDS,
        heartbeat_url: str = HEARTBEAT_URL
    ):
        self.poll_interval = poll_interval
        self.heartbeat_url = heartbeat_url
        self.running = False

        self.gtfs_mgr = GTFSManager(gtfs_base_dir=gtfs_dir)
        self.fetcher = GTFSRTFetcher()
        self.storage = ParquetPartitionStorage(base_dir=raw_dir)
        self.catalog = DuckDBCatalog(raw_base_dir=raw_dir)

        self.consecutive_successful_cycles = 0
        self.total_cycles = 0
        self.records_since_last_health = 0
        self.last_health_time = time.time()
        self.last_gtfs_check = time.time()
        self.last_active_vehicles = 0

        self._setup_signal_handlers()

    def _send_heartbeat(self, status: str = "ok", message: str = ""):
        """Send dead man's switch ping to Healthchecks.io / webhook service."""
        if not self.heartbeat_url:
            return

        try:
            import requests
            url = self.heartbeat_url.rstrip("/")
            if status == "start":
                requests.get(f"{url}/start", timeout=5)
            elif status == "fail":
                requests.post(f"{url}/fail", data=message.encode("utf-8"), timeout=5)
            else:
                # Regular OK heartbeat with telemetry payload
                requests.post(url, data=message.encode("utf-8"), timeout=10)
        except Exception as e:
            logger.debug(f"Heartbeat ping to {self.heartbeat_url} omitted: {e}")

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info(f"Signal {sig_name} received. Initiating graceful daemon shutdown...")
        self.running = False

    def log_health_metrics(self):
        now = time.time()
        elapsed_minutes = max(0.1, (now - self.last_health_time) / 60.0)
        ingest_rate_per_min = self.records_since_last_health / elapsed_minutes
        day_storage_bytes = self.storage.get_current_day_storage_bytes()
        day_storage_mb = day_storage_bytes / (1024 * 1024)

        report = (
            f"[HEALTH REPORT] "
            f"Active Vehicles: {self.last_active_vehicles} | "
            f"Ingestion Rate: {ingest_rate_per_min:.1f} rec/min | "
            f"Current Day Parquet Size: {day_storage_mb:.2f} MB ({day_storage_bytes:,} bytes) | "
            f"Consecutive Success Cycles: {self.consecutive_successful_cycles} | "
            f"Total Records Flushed: {self.storage.get_total_records_flushed():,}"
        )
        logger.info(report)
        health_logger.info(report)
        self._send_heartbeat("ok", report)

        # Reset window counters
        self.records_since_last_health = 0
        self.last_health_time = now

    def run(self):
        self.running = True
        logger.info("Initializing Warsaw Transit Ingestion Daemon (Multi-Month Production Engine)...")
        self._send_heartbeat("start", "Warsaw collector daemon started")

        # Initial GTFS Sync
        self.gtfs_mgr.sync_weekly_gtfs()

        while self.running:
            cycle_start = time.time()
            self.total_cycles += 1

            # Check weekly schedule freshness once per day
            if time.time() - self.last_gtfs_check >= WEEKLY_CHECK_INTERVAL_SECONDS:
                try:
                    self.gtfs_mgr.sync_weekly_gtfs()
                except Exception as e:
                    logger.warning(f"Failed periodic GTFS sync check: {e}")
                self.last_gtfs_check = time.time()

            try:
                # 1. Fetch live GTFS-RT feed
                feed_bytes = self.fetcher.fetch_feed()

                # 2. Parse and normalize
                records, stats = self.fetcher.parse_feed(feed_bytes, self.gtfs_mgr)
                self.last_active_vehicles = stats["active_vehicles"]

                # 3. Deduplicate and buffer
                new_buffered = self.storage.filter_and_buffer(records)
                self.records_since_last_health += new_buffered
                self.consecutive_successful_cycles += 1

                logger.info(
                    f"Cycle #{self.total_cycles}: {stats['total_trip_updates']} trip updates "
                    f"({stats['records_with_delay']} with delay) | "
                    f"New state deltas buffered: {new_buffered} | "
                    f"Active vehicles: {stats['active_vehicles']}"
                )

            except CriticalSchemaMismatchError as e:
                # Unrecoverable protobuf break
                logger.critical(f"[CRITICAL BLOCKER]: {e}")
                health_logger.critical(f"[CRITICAL BLOCKER]: {e}")
                self._send_heartbeat("fail", f"CriticalSchemaMismatchError: {e}")
                self.running = False
                break

            except RateLimitCircuitBreakerError as e:
                # HTTP 429 persisted after retries: Pause 5 minutes
                pause_sec = 300
                logger.error(f"[CIRCUIT BREAKER]: Rate limit 429 persisted. Pausing daemon for {pause_sec}s. Retry-After: {e.retry_after}")
                health_logger.error(f"[CIRCUIT BREAKER]: Rate limit 429. Pausing {pause_sec}s. Retry-After: {e.retry_after}")
                self.consecutive_successful_cycles = 0
                time.sleep(pause_sec)

            except NetworkOutageError as e:
                # Network outage > 15 minutes: Pause 5 minutes before retrying
                pause_sec = 300
                logger.error(f"[NETWORK OUTAGE]: {e}. Pausing for {pause_sec}s...")
                health_logger.error(f"[NETWORK OUTAGE]: {e}. Pausing for {pause_sec}s...")
                self.consecutive_successful_cycles = 0
                time.sleep(pause_sec)

            except TransientHttpError as e:
                logger.warning(f"Transient HTTP error in cycle #{self.total_cycles}: {e}")
                self.consecutive_successful_cycles = 0

            except Exception as e:
                logger.exception(f"Unexpected error in cycle #{self.total_cycles}: {e}")
                self.consecutive_successful_cycles = 0

            # Periodic state map pruning to prevent memory leak
            if self.total_cycles % 100 == 0:
                self.storage.prune_old_state()

            # Health metrics heartbeat check (every 15 minutes)
            if time.time() - self.last_health_time >= HEALTH_REPORT_INTERVAL_SECONDS:
                self.log_health_metrics()

            # Sleep to maintain 30s interval
            elapsed = time.time() - cycle_start
            sleep_duration = max(0.0, self.poll_interval - elapsed)
            sleep_until = time.time() + sleep_duration
            while self.running and time.time() < sleep_until:
                time.sleep(0.5)

        logger.info("Daemon shutdown sequence active. Flushing in-memory buffer to Parquet...")
        flushed_file = self.storage.flush()
        if flushed_file:
            logger.info(f"Final shutdown buffer flushed to: {flushed_file}")
        self.log_health_metrics()
        self.gtfs_mgr.close()
        logger.info("Collector daemon shutdown cleanly.")


if __name__ == "__main__":
    daemon = WarsawCollectorDaemon()
    daemon.run()

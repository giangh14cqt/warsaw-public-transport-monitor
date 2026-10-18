#!/usr/bin/env python3
"""
Warsaw Observed-Delay Historian: High-Frequency Ingestion Daemon (Track A).
Polls dane.um.warszawa.pl (buses + trams) and zbiorkom GTFS-RT on a strict
10-second wall-clock grid (:00, :10, :20, :30, :40, :50).
Archives verbatim compressed responses into Bronze storage and logs tick metrics.
"""

import argparse
import concurrent.futures
import datetime
import logging
import os
import signal
import sys
import time
from logging.handlers import RotatingFileHandler
from typing import Dict

# Ensure project root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Automatically load .env if present
if os.path.exists(".env"):
    try:
        with open(".env", "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() not in os.environ:
                        os.environ[k.strip()] = v.strip()
    except Exception:
        pass

from src.historian.bronze import BronzeStorageWriter  # noqa: E402
from src.historian.clock import generate_aligned_ticks  # noqa: E402
from src.historian.config import config  # noqa: E402
from src.historian.sources.um_api import WarsawCityApiClient  # noqa: E402
from src.historian.sources.zbiorkom import ZbiorkomClient  # noqa: E402
from src.historian.static_gtfs import HistorianGTFSVersioner  # noqa: E402

# Configure dual logging: console + rotating log file
os.makedirs("logs", exist_ok=True)
health_logger = logging.getLogger("historian_health")
health_logger.setLevel(logging.INFO)
file_handler = RotatingFileHandler(
    "logs/historian_health.log", maxBytes=20 * 1024 * 1024, backupCount=5, mode="a"
)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)
health_logger.addHandler(file_handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("historian_daemon")


class WarsawHistorianDaemon:
    """Production 10-second ingestion daemon for Warsaw transit telemetry."""

    def __init__(self, max_ticks: int = 0):
        self.max_ticks = max_ticks
        self.running = False
        self.total_ticks = 0
        self.successful_ticks = 0
        self.missed_ticks = 0

        self.writer = BronzeStorageWriter()
        self.city_client = WarsawCityApiClient()
        self.zbk_client = ZbiorkomClient()
        self.gtfs_versioner = HistorianGTFSVersioner()

        self.last_health_time = time.time()
        self.last_gtfs_sync_time = 0.0

        # In-flight guard to prevent thread accumulation if a source hangs
        self.in_flight: Dict[str, bool] = {"um_bus": False, "um_tram": False, "zbk_rt": False}
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="historian_worker")

        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info(f"Signal {sig_name} received. Initiating graceful Historian shutdown...")
        self.running = False

    def _send_heartbeat(self, status: str = "ok", message: str = ""):
        """Send dead man's switch ping to Healthchecks.io if configured."""
        if not config.heartbeat_url:
            return
        try:
            import requests

            url = config.heartbeat_url.rstrip("/")
            if status == "start":
                resp = requests.get(f"{url}/start", timeout=5)
                logger.info(f"Sent 'start' heartbeat ping to {url}/start (HTTP {resp.status_code})")
            elif status == "fail":
                requests.post(f"{url}/fail", data=message.encode("utf-8"), timeout=5)
                logger.warning(f"Sent 'fail' alert ping to {url}/fail")
            else:
                requests.post(url, data=message.encode("utf-8"), timeout=10)
        except Exception as e:
            logger.warning(f"Heartbeat ping to {config.heartbeat_url} omitted/failed: {e}")

    def log_health_metrics(self):
        """Logs periodic health statistics."""
        now = time.time()
        success_pct = (self.successful_ticks / max(1, self.total_ticks)) * 100

        # Calculate today's bronze storage size
        today_bronze_bytes = 0
        try:
            today_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
            for sub in ("source=um_bus", "source=um_tram", "source=zbk_rt"):
                day_path = self.writer.base_dir / sub / f"date={today_str}"
                if day_path.exists():
                    today_bronze_bytes += sum(f.stat().st_size for f in day_path.glob("**/*") if f.is_file())
        except Exception:
            pass

        today_mb = today_bronze_bytes / (1024 * 1024)

        report = (
            f"[HISTORIAN HEALTH] Ticks Total: {self.total_ticks} | "
            f"Success: {self.successful_ticks} ({success_pct:.1f}%) | "
            f"Missed: {self.missed_ticks} | "
            f"Today Bronze Size: {today_mb:.2f} MB ({today_bronze_bytes:,} bytes)"
        )
        logger.info(report)
        health_logger.info(report)
        self._send_heartbeat("ok", report)
        self.last_health_time = now

    def _fetch_and_archive_city(self, source_name: str, vehicle_type: int, tick_ts: int):
        self.in_flight[source_name] = True
        try:
            res = self.city_client.fetch_vehicles(vehicle_type=vehicle_type)
            raw_bytes = res.get("raw_bytes", b"")
            sha = ""
            if raw_bytes and res.get("status") == "OK":
                _, sha = self.writer.write_payload(source_name, tick_ts, raw_bytes, "json")

            self.writer.record_tick(
                tick_ts=tick_ts,
                source=source_name,
                status=res.get("status", "UNKNOWN"),
                http_status=res.get("http_status", 0),
                latency_ms=res.get("latency_ms", 0.0),
                bytes_count=len(raw_bytes),
                payload_sha256=sha,
                n_vehicles=len(res.get("records", [])),
                server_date=res.get("server_date"),
                clock_offset_s=res.get("clock_offset_s"),
                error_message=res.get("error"),
            )
            return res
        except Exception as e:
            logger.exception(f"Unexpected error fetching {source_name}: {e}")
            self.writer.record_tick(tick_ts=tick_ts, source=source_name, status="ERROR", error_message=str(e))
            return None
        finally:
            self.in_flight[source_name] = False

    def _fetch_and_archive_zbk(self, tick_ts: int):
        self.in_flight["zbk_rt"] = True
        try:
            res = self.zbk_client.fetch_feed()
            raw_bytes = res.get("raw_bytes", b"")
            sha = ""
            if raw_bytes and res.get("status") == "OK":
                _, sha = self.writer.write_payload("zbk_rt", tick_ts, raw_bytes, "pb")

            self.writer.record_tick(
                tick_ts=tick_ts,
                source="zbk_rt",
                status=res.get("status", "UNKNOWN"),
                http_status=res.get("http_status", 0),
                latency_ms=res.get("latency_ms", 0.0),
                bytes_count=len(raw_bytes),
                payload_sha256=sha,
                n_vehicles=res.get("n_vehicles", 0),
                server_date=res.get("server_date"),
                clock_offset_s=res.get("clock_offset_s"),
                error_message=res.get("error"),
            )
            return res
        except Exception as e:
            logger.exception(f"Unexpected error fetching zbk_rt: {e}")
            self.writer.record_tick(tick_ts=tick_ts, source="zbk_rt", status="ERROR", error_message=str(e))
            return None
        finally:
            self.in_flight["zbk_rt"] = False

    def execute_tick(self, scheduled_ts: float, actual_ts: float, is_missed: bool):
        tick_int = int(scheduled_ts)
        self.total_ticks += 1

        if is_missed:
            self.missed_ticks += 1
            logger.warning(f"Tick {tick_int} missed due to scheduler lag ({actual_ts - scheduled_ts:.3f}s)")
            self.writer.record_tick(tick_ts=tick_int, source="system", status="MISSED_TICK")
            return

        futures = {}

        # 1. City Buses
        if not self.in_flight["um_bus"]:
            futures[self.pool.submit(self._fetch_and_archive_city, "um_bus", 1, tick_int)] = "um_bus"
        else:
            self.writer.record_tick(tick_ts=tick_int, source="um_bus", status="SKIPPED_OVERLAP")

        # 2. City Trams
        if not self.in_flight["um_tram"]:
            futures[self.pool.submit(self._fetch_and_archive_city, "um_tram", 2, tick_int)] = "um_tram"
        else:
            self.writer.record_tick(tick_ts=tick_int, source="um_tram", status="SKIPPED_OVERLAP")

        # 3. zbiorkom GTFS-RT
        if not self.in_flight["zbk_rt"]:
            futures[self.pool.submit(self._fetch_and_archive_zbk, tick_int)] = "zbk_rt"
        else:
            self.writer.record_tick(tick_ts=tick_int, source="zbk_rt", status="SKIPPED_OVERLAP")

        # Wait for all sources to complete within deadline
        results = {}
        for future in concurrent.futures.as_completed(futures, timeout=config.request_deadline_seconds):
            source = futures[future]
            try:
                results[source] = future.result()
            except Exception as e:
                logger.error(f"Worker exception for {source}: {e}")

        # Flush tick logs periodically or every 6 ticks (1 minute)
        if self.total_ticks % 6 == 0:
            self.writer.flush_ticks()

        # Log cycle summary
        b_count = len(results.get("um_bus", {}).get("records", [])) if "um_bus" in results and results["um_bus"] else 0
        t_count = len(results.get("um_tram", {}).get("records", [])) if "um_tram" in results and results["um_tram"] else 0
        z_count = results.get("zbk_rt", {}).get("n_vehicles", 0) if "zbk_rt" in results and results["zbk_rt"] else 0

        self.successful_ticks += 1
        logger.info(
            f"Tick #{self.total_ticks} ({time.strftime('%H:%M:%S', time.gmtime(tick_int))} UTC): "
            f"Buses: {b_count:,} | Trams: {t_count:,} | zbiorkom VP: {z_count:,}"
        )

    def run(self):
        self.running = True
        logger.info("Initializing Warsaw Observed-Delay Historian (10-Second Native Ingestion)...")
        logger.info(f"City API Target: {config.um_api_url}")
        logger.info(f"Storage Base Directory: {config.paths.base_dir}")
        logger.info(
            f"Heartbeat Target: {config.heartbeat_url if config.heartbeat_url else 'Disabled (HISTORIAN_HEARTBEAT_URL not set)'}"
        )
        self._send_heartbeat("start", "Warsaw Historian daemon started")

        # Initial static GTFS sync check
        try:
            self.gtfs_versioner.sync_daily_gtfs()
            self.last_gtfs_sync_time = time.time()
        except Exception as e:
            logger.warning(f"Could not perform initial static GTFS sync: {e}")

        tick_generator = generate_aligned_ticks(
            period_seconds=config.poll_period_seconds,
            max_lag_seconds=config.max_start_lag_seconds,
        )

        while self.running:
            if self.max_ticks and self.total_ticks >= self.max_ticks:
                logger.info(f"Reached configured max_ticks ({self.max_ticks}). Exiting run loop.")
                break

            # Check daily static GTFS sync once per day
            if time.time() - self.last_gtfs_sync_time >= 86400:
                try:
                    self.gtfs_versioner.sync_daily_gtfs()
                except Exception as e:
                    logger.warning(f"Periodic GTFS sync check failed: {e}")
                self.last_gtfs_sync_time = time.time()

            # Wait for next wall-clock aligned tick
            sched_ts, actual_ts, is_missed = next(tick_generator)
            self.execute_tick(sched_ts, actual_ts, is_missed)

            # Health reporting every 15 minutes
            if time.time() - self.last_health_time >= config.health_interval_seconds:
                self.log_health_metrics()

        logger.info("Shutdown sequence active. Flushing buffered audit logs to Parquet...")
        flushed_file = self.writer.flush_ticks()
        if flushed_file:
            logger.info(f"Final tick audit log written to: {flushed_file}")
        self.pool.shutdown(wait=True)
        self.log_health_metrics()
        logger.info("Historian daemon shutdown cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warsaw Observed-Delay Historian Daemon")
    parser.add_argument("--max-ticks", type=int, default=0, help="Stop after N ticks (useful for testing)")
    args = parser.parse_args()

    daemon = WarsawHistorianDaemon(max_ticks=args.max_ticks)
    daemon.run()

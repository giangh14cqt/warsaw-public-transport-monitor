"""
Resilient GTFS-RT Protobuf fetcher and parser for Warsaw public transit.
Implements exponential backoff, circuit-breakers, and schema normalization.
"""

import time
import logging
import datetime
from typing import Dict, List, Tuple, Optional, Any

import requests
from google.transit import gtfs_realtime_pb2
from google.protobuf.message import DecodeError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from src.gtfs_manager import GTFSManager

logger = logging.getLogger(__name__)

FEED_URL = "https://cdn.zbiorkom.live/gtfs-rt/warsaw.pb"

def get_warsaw_tz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("Europe/Warsaw")
    except Exception:
        return datetime.timezone(datetime.timedelta(hours=2))

WARSAW_TZ = get_warsaw_tz()


class CriticalSchemaMismatchError(Exception):
    """Raised when Protobuf DecodeError occurs, indicating endpoint schema change."""
    pass


class RateLimitCircuitBreakerError(Exception):
    """Raised when HTTP 429 persists after retries, requiring a pause."""
    def __init__(self, message: str, retry_after: Optional[str] = None):
        super().__init__(message)
        self.retry_after = retry_after


class NetworkOutageError(Exception):
    """Raised when network/server failures persist beyond 15 minutes."""
    pass


class TransientHttpError(Exception):
    """Temporary network or server errors eligible for retry."""
    pass


class GTFSRTFetcher:
    """Fetches and parses GTFS-RT protobuf feeds with circuit breakers."""

    def __init__(self, feed_url: str = FEED_URL):
        self.feed_url = feed_url
        self.last_successful_fetch = time.time()
        self.consecutive_failures = 0
        self.first_failure_time: Optional[float] = None

    @retry(
        retry=retry_if_exception_type((TransientHttpError, requests.RequestException)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=3, min=3, max=24),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _fetch_bytes_with_retry(self) -> bytes:
        """Fetch raw bytes with tenacity exponential backoff (3s, 6s, 12s, 24s)."""
        try:
            resp = requests.get(
                self.feed_url,
                timeout=12,
                headers={"User-Agent": "WarsawDelayTelemetry/2.0"}
            )
        except (requests.ConnectionError, requests.Timeout) as e:
            logger.warning(f"Connection issue fetching {self.feed_url}: {e}")
            raise TransientHttpError(f"Transient connection error: {e}") from e

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After", "unknown")
            logger.warning(f"HTTP 429 Rate Limit encountered. Retry-After: {retry_after}")
            raise RateLimitCircuitBreakerError(
                f"Rate limit 429 encountered from {self.feed_url}",
                retry_after=retry_after
            )

        if resp.status_code in (500, 502, 503, 504):
            logger.warning(f"HTTP {resp.status_code} server error from {self.feed_url}. Retrying...")
            raise TransientHttpError(f"Server error {resp.status_code}")

        if resp.status_code != 200:
            raise TransientHttpError(f"Unexpected status code {resp.status_code}")

        return resp.content

    def fetch_feed(self) -> bytes:
        """
        Public fetch method tracking consecutive network outages and rate limits.
        """
        try:
            content = self._fetch_bytes_with_retry()
            self.last_successful_fetch = time.time()
            self.consecutive_failures = 0
            self.first_failure_time = None
            return content
        except RateLimitCircuitBreakerError as e:
            logger.error(f"[CIRCUIT BREAKER]: {e}. Response header Retry-After: {e.retry_after}")
            raise
        except Exception as e:
            now = time.time()
            self.consecutive_failures += 1
            if self.first_failure_time is None:
                self.first_failure_time = now

            outage_duration = now - self.first_failure_time
            if outage_duration >= 900:  # 15 minutes
                msg = f"[ALERT]: Long network outage or endpoint unreachable ({outage_duration:.0f}s). Pausing."
                logger.error(msg)
                raise NetworkOutageError(msg) from e
            raise

    def parse_time_str_to_seconds(self, time_str: str) -> Optional[int]:
        if not time_str or not isinstance(time_str, str):
            return None
        parts = time_str.split(":")
        if len(parts) != 3:
            return None
        try:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        except ValueError:
            return None

    def parse_feed(
        self,
        feed_bytes: bytes,
        gtfs_mgr: GTFSManager
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Deserializes Protobuf FeedMessage and normalizes delay records.
        """
        feed = gtfs_realtime_pb2.FeedMessage()
        try:
            feed.ParseFromString(feed_bytes)
        except DecodeError as e:
            msg = "[CRITICAL]: Protobuf schema mismatch. Check for zbiorkom API update."
            logger.critical(msg)
            raise CriticalSchemaMismatchError(msg) from e

        feed_timestamp_int = int(feed.header.timestamp) if feed.header.HasField("timestamp") else int(time.time())
        feed_dt = datetime.datetime.fromtimestamp(feed_timestamp_int, WARSAW_TZ)
        feed_midnight = feed_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        feed_midnight_ts = int(feed_midnight.timestamp())
        ingest_time_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Step 1: Collect raw updates and unique trip IDs
        raw_items = []
        trip_ids = set()
        active_vehicles = set()

        for entity in feed.entity:
            if not entity.HasField("trip_update"):
                continue

            tu = entity.trip_update
            trip_id = tu.trip.trip_id if tu.HasField("trip") else ""
            if trip_id:
                trip_ids.add(trip_id)

            vehicle_id = tu.vehicle.id if (tu.HasField("vehicle") and tu.vehicle.HasField("id")) else ""
            if vehicle_id:
                active_vehicles.add(vehicle_id)

            route_id_pb = tu.trip.route_id if (tu.HasField("trip") and tu.trip.HasField("route_id") and tu.trip.route_id) else None
            start_date_pb = tu.trip.start_date if (tu.HasField("trip") and tu.trip.HasField("start_date") and tu.trip.start_date) else None
            start_time_pb = tu.trip.start_time if (tu.HasField("trip") and tu.trip.HasField("start_time") and tu.trip.start_time) else None

            for stu in tu.stop_time_update:
                stop_seq = int(stu.stop_sequence) if stu.HasField("stop_sequence") else None
                stop_id_pb = stu.stop_id if (stu.HasField("stop_id") and stu.stop_id) else None

                rt_arr_ts = int(stu.arrival.time) if (stu.HasField("arrival") and stu.arrival.HasField("time")) else None
                direct_arr_delay = int(stu.arrival.delay) if (stu.HasField("arrival") and stu.arrival.HasField("delay")) else None

                rt_dep_ts = int(stu.departure.time) if (stu.HasField("departure") and stu.departure.HasField("time")) else None
                direct_dep_delay = int(stu.departure.delay) if (stu.HasField("departure") and stu.departure.HasField("delay")) else None

                raw_items.append({
                    "trip_id": trip_id,
                    "vehicle_id": vehicle_id,
                    "route_id": route_id_pb,
                    "start_date": start_date_pb,
                    "start_time": start_time_pb,
                    "stop_id": stop_id_pb,
                    "stop_sequence": stop_seq,
                    "rt_arr_ts": rt_arr_ts,
                    "direct_arr_delay": direct_arr_delay,
                    "rt_dep_ts": rt_dep_ts,
                    "direct_dep_delay": direct_dep_delay,
                })

        # Step 2: Batch query static schedule for these trip_ids
        sched_mapping = gtfs_mgr.lookup_schedule_batch(list(trip_ids))

        # Step 3: Build normalized records conforming to required schema
        records: List[Dict[str, Any]] = []
        routes_seen = set()

        for item in raw_items:
            trip_id = item["trip_id"]
            stop_seq = item["stop_sequence"]
            sched_info = sched_mapping.get((trip_id, stop_seq)) if (trip_id and stop_seq is not None) else None

            route_id = item["route_id"] or (sched_info.get("route_id") if sched_info else None)
            stop_id = item["stop_id"] or (sched_info.get("stop_id") if sched_info else None)
            if route_id:
                routes_seen.add(route_id)

            # Start date default to feed date in YYYYMMDD if not provided
            start_date = item["start_date"] or feed_dt.strftime("%Y%m%d")
            start_time = item["start_time"]

            # Compute Arrival Delay
            arr_delay_sec = item["direct_arr_delay"]
            if arr_delay_sec is None and item["rt_arr_ts"] and sched_info and sched_info.get("arrival_time"):
                sched_sec = self.parse_time_str_to_seconds(sched_info["arrival_time"])
                if sched_sec is not None:
                    sched_arr_ts = feed_midnight_ts + sched_sec
                    arr_delay_sec = int(item["rt_arr_ts"] - sched_arr_ts)
                    if not start_time:
                        start_time = sched_info["arrival_time"]

            # Compute Departure Delay
            dep_delay_sec = item["direct_dep_delay"]
            if dep_delay_sec is None and item["rt_dep_ts"] and sched_info and sched_info.get("departure_time"):
                sched_sec = self.parse_time_str_to_seconds(sched_info["departure_time"])
                if sched_sec is not None:
                    sched_dep_ts = feed_midnight_ts + sched_sec
                    dep_delay_sec = int(item["rt_dep_ts"] - sched_dep_ts)

            record = {
                "feed_timestamp": feed_timestamp_int,
                "trip_id": trip_id,
                "route_id": route_id,
                "start_date": start_date,
                "start_time": start_time,
                "vehicle_id": item["vehicle_id"],
                "stop_sequence": stop_seq,
                "stop_id": stop_id,
                "arrival_delay_seconds": arr_delay_sec,
                "departure_delay_seconds": dep_delay_sec,
                "rt_arrival_time": item["rt_arr_ts"],
                "rt_departure_time": item["rt_dep_ts"],
                "record_ingested_at": ingest_time_utc
            }
            records.append(record)

        stats = {
            "feed_timestamp": feed_timestamp_int,
            "feed_datetime": feed_dt.isoformat(),
            "total_entities": len(feed.entity),
            "total_trip_updates": sum(1 for e in feed.entity if e.HasField("trip_update")),
            "total_stop_records": len(records),
            "records_with_delay": sum(1 for r in records if r["arrival_delay_seconds"] is not None),
            "active_vehicles": len(active_vehicles),
            "active_routes": len(routes_seen),
        }
        return records, stats

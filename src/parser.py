import os
import sys
import time
import zipfile
import logging
import datetime
from typing import Dict, List, Optional, Tuple, Any

import requests
import duckdb
import pandas as pd
from google.transit import gtfs_realtime_pb2
from google.protobuf.message import DecodeError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

FEED_URL = "https://cdn.zbiorkom.live/gtfs-rt/warsaw.pb"
STATIC_GTFS_URL = "https://cdn.zbiorkom.live/gtfs/warsaw.zip"
HISTORICAL_API_URL = "https://api.zbiorkom.live/api6/warsaw/dispatches"

def get_warsaw_tz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("Europe/Warsaw")
    except Exception:
        return datetime.timezone(datetime.timedelta(hours=2))

WARSAW_TZ = get_warsaw_tz()


class CriticalBlockerError(Exception):
    """Raised when an unrecoverable response or protocol error is encountered."""
    pass


class RateLimitExceededError(Exception):
    """Raised when rate limiting (429) persists beyond allowed retries."""
    def __init__(self, message: str, headers: Dict[str, str]):
        super().__init__(message)
        self.headers = headers


class TransientHttpError(Exception):
    """Temporary network or server errors eligible for retry."""
    pass


class WarsawTransitParser:
    """
    Resilient parser for Warsaw GTFS-RT feed and static schedule mapping.
    Handles network retries, blocker escalation, schema verification, and delay calculation.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.static_gtfs_dir = os.path.join(data_dir, "gtfs_static")
        self.static_db_path = os.path.join(data_dir, "static_gtfs.duckdb")
        os.makedirs(self.static_gtfs_dir, exist_ok=True)
        self._static_conn: Optional[duckdb.DuckDBPyConnection] = None

    @retry(
        retry=retry_if_exception_type((TransientHttpError, requests.ConnectionError, requests.Timeout)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=16),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def fetch_feed_bytes(self, url: str = FEED_URL, timeout: int = 15) -> bytes:
        """
        Fetch protobuf bytes with exponential backoff and circuit-breaker escalation.
        """
        try:
            response = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "WarsawTransitIngestion/1.0"}
            )
        except (requests.ConnectionError, requests.Timeout) as e:
            logger.warning(f"Connection issue fetching {url}: {e}")
            raise TransientHttpError(f"Transient connection error: {e}") from e

        # Blocker Escalation check: 401 / 403
        if response.status_code in (401, 403):
            msg = f"[CRITICAL BLOCKER]: Endpoint response unexpected ({response.status_code}). Human review required."
            logger.critical(msg)
            raise CriticalBlockerError(msg)

        # Rate Limit check: 429
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "unknown")
            logger.warning(f"HTTP 429 Rate Limit encountered. Retry-After: {retry_after}")
            raise TransientHttpError(f"HTTP 429 Rate Limit (Retry-After: {retry_after})")

        # Server Errors: 500, 502, 503, 504
        if response.status_code in (500, 502, 503, 504):
            logger.warning(f"HTTP {response.status_code} encountered. Retrying...")
            raise TransientHttpError(f"HTTP Server Error: {response.status_code}")

        if response.status_code != 200:
            msg = f"[CRITICAL BLOCKER]: Endpoint response unexpected ({response.status_code}). Human review required."
            logger.critical(msg)
            raise CriticalBlockerError(msg)

        return response.content

    def ensure_static_gtfs(self) -> None:
        """
        Ensure static GTFS schedule is indexed in DuckDB for ultra-fast schedule lookups.
        """
        if self._static_conn is not None:
            return

        trips_file = os.path.join(self.static_gtfs_dir, "trips.txt")
        stop_times_file = os.path.join(self.static_gtfs_dir, "stop_times.txt")

        if not (os.path.exists(trips_file) and os.path.exists(stop_times_file)):
            zip_path = os.path.join(self.data_dir, "warsaw_static.zip")
            if not os.path.exists(zip_path):
                logger.info(f"Downloading static GTFS schedule from {STATIC_GTFS_URL}...")
                resp = requests.get(STATIC_GTFS_URL, timeout=60)
                if resp.status_code != 200:
                    raise RuntimeError(f"Failed to download static GTFS: HTTP {resp.status_code}")
                with open(zip_path, "wb") as f:
                    f.write(resp.content)

            logger.info("Extracting trips.txt, routes.txt, stop_times.txt...")
            with zipfile.ZipFile(zip_path, "r") as z:
                for fname in ["trips.txt", "routes.txt", "stop_times.txt"]:
                    z.extract(fname, self.static_gtfs_dir)

        # Connect or create static duckdb index
        self._static_conn = duckdb.connect(self.static_db_path)
        tables = [t[0] for t in self._static_conn.execute("SHOW TABLES").fetchall()]
        if "trips" not in tables or "stop_times" not in tables:
            logger.info("Building DuckDB indexes on static GTFS...")
            self._static_conn.execute(f"""
                CREATE TABLE trips AS SELECT trip_id, route_id, brigade FROM read_csv('{trips_file}', header=true, all_varchar=true);
                CREATE TABLE stop_times AS SELECT trip_id, stop_id, CAST(stop_sequence AS INTEGER) AS stop_sequence, arrival_time, departure_time FROM read_csv('{stop_times_file}', header=true, all_varchar=true);
                CREATE INDEX idx_trips ON trips(trip_id);
                CREATE INDEX idx_stop_times ON stop_times(trip_id, stop_sequence);
            """)
            logger.info("Static GTFS indexing complete.")

    def parse_time_str_to_seconds(self, time_str: str) -> Optional[int]:
        """Convert HH:MM:SS (possibly > 24:00:00) into seconds from midnight."""
        if not time_str or not isinstance(time_str, str):
            return None
        parts = time_str.split(":")
        if len(parts) != 3:
            return None
        try:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        except ValueError:
            return None

    def parse_snapshot(self, feed_bytes: bytes) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Parse raw GTFS-RT protobuf bytes into normalized delay records.
        """
        feed = gtfs_realtime_pb2.FeedMessage()
        try:
            feed.ParseFromString(feed_bytes)
        except DecodeError as e:
            msg = f"[CRITICAL BLOCKER]: Endpoint response unexpected. Human review required. (Protobuf DecodeError: {e})"
            logger.critical(msg)
            raise CriticalBlockerError(msg) from e

        feed_timestamp_int = feed.header.timestamp if feed.header.HasField("timestamp") else int(time.time())
        feed_dt = datetime.datetime.fromtimestamp(feed_timestamp_int, WARSAW_TZ)
        feed_midnight = feed_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        feed_midnight_ts = int(feed_midnight.timestamp())

        # Ensure static schedule tables are ready in DuckDB
        self.ensure_static_gtfs()

        # Step 1: Collect all active trip_ids from the feed
        raw_updates = []
        feed_trip_ids = set()
        vehicles_seen = set()

        for entity in feed.entity:
            if not entity.HasField("trip_update"):
                continue

            tu = entity.trip_update
            trip_id = tu.trip.trip_id if tu.HasField("trip") else ""
            if trip_id:
                feed_trip_ids.add(trip_id)

            vehicle_id = tu.vehicle.id if (tu.HasField("vehicle") and tu.vehicle.HasField("id")) else ""
            if vehicle_id:
                vehicles_seen.add(vehicle_id)

            obs_timestamp = tu.timestamp if tu.HasField("timestamp") else feed_timestamp_int
            obs_dt = datetime.datetime.fromtimestamp(obs_timestamp, WARSAW_TZ)

            route_id_pb = tu.trip.route_id if (tu.HasField("trip") and tu.trip.HasField("route_id") and tu.trip.route_id) else None
            sched_rel = "SCHEDULED" if not tu.trip.HasField("schedule_relationship") else str(tu.trip.schedule_relationship)

            for stu in tu.stop_time_update:
                stop_seq = stu.stop_sequence if stu.HasField("stop_sequence") else None
                stop_id_pb = stu.stop_id if (stu.HasField("stop_id") and stu.stop_id) else None
                rt_arr_ts = stu.arrival.time if (stu.HasField("arrival") and stu.arrival.HasField("time")) else None
                direct_arr_delay = int(stu.arrival.delay) if (stu.HasField("arrival") and stu.arrival.HasField("delay")) else None
                rt_dep_ts = stu.departure.time if (stu.HasField("departure") and stu.departure.HasField("time")) else None
                direct_dep_delay = int(stu.departure.delay) if (stu.HasField("departure") and stu.departure.HasField("delay")) else None

                raw_updates.append({
                    "trip_id": trip_id,
                    "vehicle_id": vehicle_id,
                    "route_id": route_id_pb,
                    "stop_id": stop_id_pb,
                    "stop_sequence": stop_seq,
                    "rt_arr_ts": rt_arr_ts,
                    "direct_arr_delay": direct_arr_delay,
                    "rt_dep_ts": rt_dep_ts,
                    "direct_dep_delay": direct_dep_delay,
                    "schedule_relationship": sched_rel,
                    "obs_dt": obs_dt
                })

        # Step 2: Batch query static schedule for only the active trip_ids
        schedule_cache = {}
        if feed_trip_ids:
            # Query active trips and stop_times
            self._static_conn.register("active_trips_df", pd.DataFrame({"trip_id": list(feed_trip_ids)}))
            rows = self._static_conn.execute("""
                SELECT 
                    s.trip_id,
                    s.stop_sequence,
                    t.route_id,
                    t.brigade,
                    s.stop_id,
                    s.arrival_time,
                    s.departure_time
                FROM active_trips_df a
                JOIN trips t ON a.trip_id = t.trip_id
                JOIN stop_times s ON a.trip_id = s.trip_id
            """).fetchall()
            self._static_conn.unregister("active_trips_df")

            for r in rows:
                # key is (trip_id, stop_sequence)
                schedule_cache[(r[0], r[1])] = {
                    "route_id": r[2],
                    "brigade": r[3],
                    "stop_id": r[4],
                    "arrival_time": r[5],
                    "departure_time": r[6]
                }

        # Step 3: Normalize records and calculate delay deltas
        records: List[Dict[str, Any]] = []
        routes_seen = set()

        for u in raw_updates:
            trip_id = u["trip_id"]
            stop_seq = u["stop_sequence"]
            sched_info = schedule_cache.get((trip_id, stop_seq)) if (trip_id and stop_seq is not None) else None

            route_id = u["route_id"] or (sched_info.get("route_id") if sched_info else None)
            stop_id = u["stop_id"] or (sched_info.get("stop_id") if sched_info else None)
            brigade = sched_info.get("brigade") if sched_info else None

            if route_id:
                routes_seen.add(route_id)

            # Compute arrival delay
            arr_delay_sec = u["direct_arr_delay"]
            sched_arr_dt = None
            if sched_info and sched_info.get("arrival_time"):
                sched_sec = self.parse_time_str_to_seconds(sched_info["arrival_time"])
                if sched_sec is not None:
                    sched_arr_ts = feed_midnight_ts + sched_sec
                    sched_arr_dt = datetime.datetime.fromtimestamp(sched_arr_ts, WARSAW_TZ)
                    if arr_delay_sec is None and u["rt_arr_ts"]:
                        arr_delay_sec = int(u["rt_arr_ts"] - sched_arr_ts)

            # Compute departure delay
            dep_delay_sec = u["direct_dep_delay"]
            sched_dep_dt = None
            if sched_info and sched_info.get("departure_time"):
                sched_sec = self.parse_time_str_to_seconds(sched_info["departure_time"])
                if sched_sec is not None:
                    sched_dep_ts = feed_midnight_ts + sched_sec
                    sched_dep_dt = datetime.datetime.fromtimestamp(sched_dep_ts, WARSAW_TZ)
                    if dep_delay_sec is None and u["rt_dep_ts"]:
                        dep_delay_sec = int(u["rt_dep_ts"] - sched_dep_ts)

            rt_arr_dt = datetime.datetime.fromtimestamp(u["rt_arr_ts"], WARSAW_TZ) if u["rt_arr_ts"] else None
            rt_dep_dt = datetime.datetime.fromtimestamp(u["rt_dep_ts"], WARSAW_TZ) if u["rt_dep_ts"] else None

            records.append({
                "feed_timestamp": feed_dt,
                "trip_id": trip_id,
                "route_id": route_id,
                "brigade": brigade,
                "vehicle_id": u["vehicle_id"],
                "stop_id": stop_id,
                "stop_sequence": stop_seq,
                "scheduled_arrival": sched_arr_dt,
                "rt_arrival": rt_arr_dt,
                "arrival_delay_seconds": arr_delay_sec,
                "scheduled_departure": sched_dep_dt,
                "rt_departure": rt_dep_dt,
                "departure_delay_seconds": dep_delay_sec,
                "schedule_relationship": u["schedule_relationship"],
                "observed_at": u["obs_dt"]
            })

        stats = {
            "feed_timestamp": feed_dt.isoformat(),
            "total_entities": len(feed.entity),
            "total_trip_updates": sum(1 for e in feed.entity if e.HasField("trip_update")),
            "total_stop_records": len(records),
            "records_with_delay": sum(1 for r in records if r["arrival_delay_seconds"] is not None),
            "unique_vehicles": len(vehicles_seen),
            "unique_routes": len(routes_seen),
        }
        return records, stats

    def fetch_historical_executions(self, date_str: str, route_id: str, timeout: int = 10) -> List[Dict[str, Any]]:
        """
        Query historical trip executions from zbiorkom API (v6 dispatches endpoint).
        Used for fallback and ground truth validation.
        """
        url = f"{HISTORICAL_API_URL}?date={date_str}&route={route_id}"
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            logger.warning(f"Historical API returned HTTP {resp.status_code} for route {route_id}")
            return []

        data = resp.json()
        results = []
        if isinstance(data, list):
            for row in data:
                if len(row) >= 15:
                    results.append({
                        "trip_id": row[0],
                        "route_id": str(row[1]),
                        "brigade": str(row[2]),
                        "vehicle_id": str(row[3]),
                        "start_stop_id": row[4],
                        "start_stop_name": row[5],
                        "end_stop_id": row[6],
                        "end_stop_name": row[7],
                        "start_delay_seconds": row[10],
                        "end_delay_seconds": row[13],
                        "total_stops": row[14],
                    })
        return results

    def close(self):
        if self._static_conn:
            self._static_conn.close()
            self._static_conn = None

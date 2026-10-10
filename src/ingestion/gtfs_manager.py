"""
Automated weekly static GTFS schedule archiver and indexer for Warsaw public transit.
Stores snapshots under data/gtfs/gtfs_{YYYY_MM_DD}.zip and unzips under data/gtfs/{YYYY_MM_DD}/.
Maintains DuckDB-backed fast lookup indices to enrich real-time feeds.
"""

import os
import glob
import zipfile
import logging
import datetime
from typing import Dict, List, Optional, Tuple, Any

import requests
import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

STATIC_GTFS_URL = "https://cdn.zbiorkom.live/gtfs/warsaw.zip"


class GTFSManager:
    """Manages downloading, archiving, and querying static GTFS schedules."""

    def __init__(self, gtfs_base_dir: str = "data/gtfs"):
        self.gtfs_base_dir = gtfs_base_dir
        os.makedirs(self.gtfs_base_dir, exist_ok=True)
        self._current_date_str: Optional[str] = None
        self._current_unpack_dir: Optional[str] = None
        self._db_conn: Optional[duckdb.DuckDBPyConnection] = None

    def get_latest_existing_snapshot(self) -> Optional[Tuple[str, str]]:
        """
        Check for existing archives in data/gtfs/.
        Returns (date_str, unpack_dir) if a valid archive from the past 7 days exists, else None.
        """
        zip_files = sorted(glob.glob(os.path.join(self.gtfs_base_dir, "gtfs_*.zip")), reverse=True)
        for zf in zip_files:
            basename = os.path.basename(zf)  # gtfs_YYYY_MM_DD.zip
            date_part = basename.replace("gtfs_", "").replace(".zip", "")
            try:
                dt = datetime.datetime.strptime(date_part, "%Y_%m_%d")
                now = datetime.datetime.now()
                # If archive is within 7 days, consider it fresh
                if (now - dt).days < 7:
                    unpack_dir = os.path.join(self.gtfs_base_dir, date_part)
                    if os.path.exists(os.path.join(unpack_dir, "trips.txt")) and \
                       os.path.exists(os.path.join(unpack_dir, "stop_times.txt")):
                        return date_part, unpack_dir
            except ValueError:
                continue
        return None

    def sync_weekly_gtfs(self, force: bool = False) -> str:
        """
        Ensure this week's GTFS archive is downloaded and unpacked.
        Returns the unpack directory path.
        """
        if not force:
            existing = self.get_latest_existing_snapshot()
            if existing:
                self._current_date_str, self._current_unpack_dir = existing
                logger.info(f"Using existing fresh static GTFS snapshot: {self._current_unpack_dir}")
                self._init_duckdb_index()
                return self._current_unpack_dir

        today_str = datetime.datetime.now().strftime("%Y_%m_%d")
        zip_dest = os.path.join(self.gtfs_base_dir, f"gtfs_{today_str}.zip")
        unpack_dir = os.path.join(self.gtfs_base_dir, today_str)
        os.makedirs(unpack_dir, exist_ok=True)

        logger.info(f"Downloading static GTFS schedule from {STATIC_GTFS_URL} to {zip_dest}...")
        resp = requests.get(STATIC_GTFS_URL, timeout=90, headers={"User-Agent": "WarsawDelayTelemetry/1.0"})
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to download static GTFS: HTTP {resp.status_code}")

        with open(zip_dest, "wb") as f:
            f.write(resp.content)
        logger.info(f"Saved {len(resp.content):,} bytes to {zip_dest}")

        logger.info(f"Extracting schedule files to {unpack_dir}...")
        with zipfile.ZipFile(zip_dest, "r") as z:
            z.extractall(unpack_dir)

        self._current_date_str = today_str
        self._current_unpack_dir = unpack_dir
        self._init_duckdb_index()
        return unpack_dir

    def _init_duckdb_index(self):
        """Index trips and stop_times in an embedded DuckDB file within the snapshot folder."""
        if not self._current_unpack_dir:
            return

        if self._db_conn:
            try:
                self._db_conn.close()
            except Exception:
                pass
            self._db_conn = None

        db_path = os.path.join(self._current_unpack_dir, "schedule_index.duckdb")
        trips_csv = os.path.join(self._current_unpack_dir, "trips.txt")
        stop_times_csv = os.path.join(self._current_unpack_dir, "stop_times.txt")

        # Check if tables already exist
        needs_build = True
        if os.path.exists(db_path):
            try:
                test_conn = duckdb.connect(db_path, read_only=True)
                tables = [t[0] for t in test_conn.execute("SHOW TABLES").fetchall()]
                test_conn.close()
                if "trips" in tables and "stop_times" in tables:
                    needs_build = False
            except Exception:
                needs_build = True

        if needs_build:
            logger.info(f"Building DuckDB index at {db_path}...")
            write_conn = duckdb.connect(db_path, read_only=False)
            write_conn.execute(f"""
                CREATE TABLE trips AS 
                SELECT trip_id, route_id, brigade, service_id 
                FROM read_csv('{trips_csv}', header=true, all_varchar=true);

                CREATE TABLE stop_times AS 
                SELECT 
                    trip_id, 
                    stop_id, 
                    CAST(stop_sequence AS INTEGER) AS stop_sequence, 
                    arrival_time, 
                    departure_time 
                FROM read_csv('{stop_times_csv}', header=true, all_varchar=true);

                CREATE INDEX idx_trips ON trips(trip_id);
                CREATE INDEX idx_stop_times ON stop_times(trip_id, stop_sequence);
            """)
            write_conn.close()
            logger.info("Schedule indexing complete.")

        self._db_conn = duckdb.connect(db_path, read_only=True)

    def lookup_schedule_batch(self, trip_ids: List[str]) -> Dict[Tuple[str, int], Dict[str, Any]]:
        """
        Batch query static schedule for active trip IDs.
        Returns a dictionary mapping (trip_id, stop_sequence) -> details.
        """
        if not self._db_conn or not trip_ids:
            return {}

        self._db_conn.register("active_trips_df", pd.DataFrame({"trip_id": list(set(trip_ids))}))
        rows = self._db_conn.execute("""
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
        self._db_conn.unregister("active_trips_df")

        cache = {}
        for r in rows:
            cache[(r[0], r[1])] = {
                "route_id": r[2],
                "brigade": r[3],
                "stop_id": r[4],
                "arrival_time": r[5],
                "departure_time": r[6]
            }
        return cache

    def close(self):
        if self._db_conn:
            self._db_conn.close()
            self._db_conn = None

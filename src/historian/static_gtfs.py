"""
Daily Content-Addressed Static GTFS Schedule Versioner for Warsaw Historian.
Downloads and indexes static schedules under data/historian/gtfs/<sha12>/,
tracking timetable snapshots in versions.parquet for exact retrospective joins.
"""

import datetime
import hashlib
import logging
import zipfile
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from src.historian.config import config, paths

logger = logging.getLogger(__name__)

VERSIONS_SCHEMA = pa.schema(
    [
        ("sha256", pa.string()),
        ("sha_prefix", pa.string()),
        ("fetched_at_utc", pa.string()),
        ("zip_bytes", pa.int64()),
        ("unpack_dir", pa.string()),
    ]
)


class HistorianGTFSVersioner:
    """Manages daily content-addressed static GTFS schedule archives and DuckDB indices."""

    def __init__(self, gtfs_dir: Optional[Path] = None):
        self.gtfs_dir = gtfs_dir or paths.gtfs_dir
        self.gtfs_dir.mkdir(parents=True, exist_ok=True)
        self.versions_file = self.gtfs_dir / "versions.parquet"
        self._current_unpack_dir: Optional[Path] = None

    def sync_daily_gtfs(self, force: bool = False) -> Path:
        """
        Fetches today's static GTFS schedule, hashes it with SHA-256,
        and unzips into data/historian/gtfs/<sha12>/.
        Reuses existing version if unchanged or fetched within the last 24 hours.
        """
        # If versions table exists and not forcing, check freshness
        if not force and self.versions_file.exists():
            try:
                df = pq.read_table(self.versions_file).to_pandas()
                if not df.empty:
                    latest = df.iloc[-1]
                    fetched_dt = datetime.datetime.fromisoformat(latest["fetched_at_utc"])
                    age_hours = (datetime.datetime.now(datetime.timezone.utc) - fetched_dt).total_seconds() / 3600
                    unpack_path = Path(latest["unpack_dir"])
                    if age_hours < 24.0 and (unpack_path / "trips.txt").exists():
                        logger.info(f"Using fresh static GTFS snapshot ({age_hours:.1f}h old): {unpack_path}")
                        self._current_unpack_dir = unpack_path
                        self._build_duckdb_index(unpack_path)
                        return unpack_path
            except Exception as e:
                logger.warning(f"Could not inspect static GTFS versions table: {e}")

        logger.info(f"Downloading static GTFS schedule from {config.zbk_static_gtfs_url}...")
        resp = requests.get(
            config.zbk_static_gtfs_url,
            timeout=90,
            headers={"User-Agent": "WarsawDelayTelemetry/2.0"},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to download static GTFS: HTTP {resp.status_code}")

        zip_bytes = resp.content
        sha256_full = hashlib.sha256(zip_bytes).hexdigest()
        sha_prefix = sha256_full[:12]
        unpack_dir = self.gtfs_dir / sha_prefix
        unpack_dir.mkdir(parents=True, exist_ok=True)

        zip_archive_path = self.gtfs_dir / f"gtfs_{sha_prefix}.zip"
        with open(zip_archive_path, "wb") as f:
            f.write(zip_bytes)

        logger.info(f"Extracting schedule files to {unpack_dir}...")
        with zipfile.ZipFile(zip_archive_path, "r") as z:
            z.extractall(unpack_dir)

        # Build DuckDB index for fast schedule queries
        self._build_duckdb_index(unpack_dir)

        # Record in versions table
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        new_row = pd.DataFrame(
            [
                {
                    "sha256": sha256_full,
                    "sha_prefix": sha_prefix,
                    "fetched_at_utc": now_iso,
                    "zip_bytes": len(zip_bytes),
                    "unpack_dir": str(unpack_dir),
                }
            ]
        )
        new_table = pa.Table.from_pandas(new_row, schema=VERSIONS_SCHEMA, preserve_index=False)

        if self.versions_file.exists():
            try:
                existing = pq.read_table(self.versions_file, schema=VERSIONS_SCHEMA)
                # Avoid duplicate rows for same sha
                existing_df = existing.to_pandas()
                if sha256_full not in existing_df["sha256"].values:
                    combined = pa.concat_tables([existing, new_table])
                else:
                    combined = existing
            except Exception:
                combined = new_table
        else:
            combined = new_table

        tmp_file = self.versions_file.with_suffix(".parquet.tmp")
        pq.write_table(combined, tmp_file)
        tmp_file.replace(self.versions_file)

        self._current_unpack_dir = unpack_dir
        logger.info(f"Static GTFS snapshot indexed successfully: {unpack_dir} (SHA: {sha_prefix})")
        return unpack_dir

    def _build_duckdb_index(self, unpack_dir: Path) -> None:
        """Indexes trips, stop_times, stops, and shapes in DuckDB for sub-millisecond timetable lookups."""
        db_path = unpack_dir / "schedule_index.duckdb"
        trips_csv = unpack_dir / "trips.txt"
        stop_times_csv = unpack_dir / "stop_times.txt"
        stops_csv = unpack_dir / "stops.txt"
        shapes_csv = unpack_dir / "shapes.txt"

        if db_path.exists():
            return

        if not trips_csv.exists() or not stop_times_csv.exists():
            logger.warning(f"Incomplete GTFS files in {unpack_dir}; skipping DuckDB indexing.")
            return

        logger.info(f"Building DuckDB index at {db_path}...")
        con = duckdb.connect(str(db_path))
        con.execute(f"""
            CREATE TABLE trips AS
            SELECT trip_id, route_id, brigade, service_id, shape_id, direction_id
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

        if stops_csv.exists():
            con.execute(f"""
                CREATE TABLE stops AS
                SELECT
                    stop_id,
                    stop_name,
                    CAST(stop_lat AS DOUBLE) AS stop_lat,
                    CAST(stop_lon AS DOUBLE) AS stop_lon
                FROM read_csv('{stops_csv}', header=true, all_varchar=true);

                CREATE INDEX idx_stops ON stops(stop_id);
            """)

        if shapes_csv.exists():
            con.execute(f"""
                CREATE TABLE shapes AS
                SELECT
                    shape_id,
                    CAST(shape_pt_lat AS DOUBLE) AS shape_pt_lat,
                    CAST(shape_pt_lon AS DOUBLE) AS shape_pt_lon,
                    CAST(shape_pt_sequence AS INTEGER) AS shape_pt_sequence
                FROM read_csv('{shapes_csv}', header=true, all_varchar=true);

                CREATE INDEX idx_shapes ON shapes(shape_id, shape_pt_sequence);
            """)

        con.close()
        logger.info("Schedule indexing complete.")

    def get_current_unpack_dir(self) -> Optional[Path]:
        return self._current_unpack_dir

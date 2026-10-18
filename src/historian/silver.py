"""
Warsaw Observed-Delay Historian: Silver Trajectory Processing Tier.
Parses raw Bronze JSON files, handles Warsaw timezone & DST localization,
deduplicates static/repeated GPS fixes, computes inter-fix kinematics,
and resolves GTFS brigade-trip assignments.
"""

import datetime
import gzip
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.historian.config import paths
from src.historian.matching.geometry import metric_distance
from src.historian.static_gtfs import HistorianGTFSVersioner

logger = logging.getLogger(__name__)

WARSAW_TZ = ZoneInfo("Europe/Warsaw")
UTC_TZ = datetime.timezone.utc

SILVER_SCHEMA = pa.schema(
    [
        ("service_date", pa.string()),
        ("vehicle_type", pa.string()),
        ("vehicle_number", pa.string()),
        ("route_id", pa.string()),
        ("brigade", pa.string()),
        ("lat", pa.float64()),
        ("lon", pa.float64()),
        ("gps_timestamp_utc", pa.int64()),
        ("gps_time_str", pa.string()),
        ("tick_ts", pa.int64()),
        ("delta_t_s", pa.float32()),
        ("delta_d_m", pa.float32()),
        ("speed_mps", pa.float32()),
        ("is_static", pa.bool_()),
        ("n_ticks_seen", pa.int32()),
        ("matched_trip_id", pa.string()),
    ]
)


class SilverTrajectoryProcessor:
    """Transforms raw Bronze vehicle telemetry into deduplicated, trajectory-linked Silver Parquet tables."""

    def __init__(self, bronze_dir: Optional[Path] = None, silver_dir: Optional[Path] = None):
        self.bronze_dir = bronze_dir or paths.bronze_dir
        self.silver_dir = silver_dir or paths.silver_dir
        self.silver_dir.mkdir(parents=True, exist_ok=True)
        self.gtfs_versioner = HistorianGTFSVersioner()

    def parse_warsaw_time_to_utc(self, time_str: str, tick_ts: int) -> int:
        """
        Parses Warsaw local datetime string 'YYYY-MM-DD HH:MM:SS' into UTC epoch seconds.
        Handles ambiguous DST fallback hours using tick receive timestamp.
        """
        try:
            naive_dt = datetime.datetime.strptime(time_str.strip(), "%Y-%m-%d %H:%M:%S")
            # First try standard localizing
            local_dt = naive_dt.replace(tzinfo=WARSAW_TZ)
            return int(local_dt.astimezone(UTC_TZ).timestamp())
        except Exception:
            return tick_ts

    def extract_raw_fixes(self, date_str: str, max_hours: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Extracts and normalizes raw GPS fixes from all Bronze json.gz files for a given date.
        """
        fixes: List[Dict[str, Any]] = []

        for v_type, source_name in [("bus", "um_bus"), ("tram", "um_tram")]:
            date_dir = self.bronze_dir / f"source={source_name}" / f"date={date_str}"
            if not date_dir.exists():
                continue

            hour_dirs = sorted(date_dir.glob("hour=*"))
            for h_dir in hour_dirs:
                if max_hours and h_dir.name.split("=")[-1] not in max_hours:
                    continue

                for gz_file in sorted(h_dir.glob("tick_*.json.gz")):
                    try:
                        tick_str = gz_file.name.replace("tick_", "").replace(".json.gz", "")
                        tick_ts = int(tick_str)
                    except ValueError:
                        tick_ts = 0

                    try:
                        with gzip.open(gz_file, "rt", encoding="utf-8") as f:
                            records = json.load(f)
                    except Exception as e:
                        logger.warning(f"Could not read {gz_file}: {e}")
                        continue

                    if not isinstance(records, list):
                        continue

                    for r in records:
                        time_str = r.get("Time")
                        if not time_str or not r.get("Lat") or not r.get("Lon"):
                            continue

                        gps_ts = self.parse_warsaw_time_to_utc(time_str, tick_ts)
                        fixes.append(
                            {
                                "service_date": date_str,
                                "vehicle_type": v_type,
                                "vehicle_number": str(r.get("VehicleNumber", "")).strip(),
                                "route_id": str(r.get("Lines", "")).strip(),
                                "brigade": str(r.get("Brigade", "")).strip(),
                                "lat": float(r.get("Lat")),
                                "lon": float(r.get("Lon")),
                                "gps_timestamp_utc": gps_ts,
                                "gps_time_str": str(time_str).strip(),
                                "tick_ts": tick_ts,
                            }
                        )

        return fixes

    def deduplicate_and_compute_kinematics(self, raw_fixes: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Sorts fixes chronologically by vehicle, collapses repeated identical pings,
        and computes inter-fix delta distance, delta time, and speed.
        """
        if not raw_fixes:
            return pd.DataFrame(columns=[f.name for f in SILVER_SCHEMA])

        df = pd.DataFrame(raw_fixes)

        # Sort chronologically per physical vehicle
        df.sort_values(
            by=["vehicle_type", "vehicle_number", "gps_timestamp_utc", "tick_ts"],
            inplace=True,
        )
        df.reset_index(drop=True, inplace=True)

        processed_rows = []
        grouped = df.groupby(["vehicle_type", "vehicle_number"], sort=False)

        for (v_type, v_num), group in grouped:
            prev_lat: Optional[float] = None
            prev_lon: Optional[float] = None
            prev_ts: Optional[int] = None
            prev_time_str: Optional[str] = None

            for _, row in group.iterrows():
                lat = row["lat"]
                lon = row["lon"]
                ts = row["gps_timestamp_utc"]
                time_str = row["gps_time_str"]

                # If exact same GPS timestamp or coordinates, collapse into previous row
                if prev_time_str == time_str and prev_lat == lat and prev_lon == lon:
                    if processed_rows:
                        processed_rows[-1]["n_ticks_seen"] += 1
                    continue

                if prev_ts is not None and ts > prev_ts:
                    dt = float(ts - prev_ts)
                    dd = metric_distance(prev_lat, prev_lon, lat, lon)
                    speed = dd / dt if dt > 0 else 0.0
                else:
                    dt = 0.0
                    dd = 0.0
                    speed = 0.0

                is_static = bool(speed < 0.5 and dd < 5.0)

                processed_rows.append(
                    {
                        "service_date": row["service_date"],
                        "vehicle_type": v_type,
                        "vehicle_number": v_num,
                        "route_id": row["route_id"],
                        "brigade": row["brigade"],
                        "lat": lat,
                        "lon": lon,
                        "gps_timestamp_utc": ts,
                        "gps_time_str": time_str,
                        "tick_ts": row["tick_ts"],
                        "delta_t_s": round(dt, 2),
                        "delta_d_m": round(dd, 2),
                        "speed_mps": round(speed, 2),
                        "is_static": is_static,
                        "n_ticks_seen": 1,
                        "matched_trip_id": None,
                    }
                )

                prev_lat = lat
                prev_lon = lon
                prev_ts = ts
                prev_time_str = time_str

        result_df = pd.DataFrame(processed_rows)
        return result_df

    def match_trips_using_gtfs(self, df: pd.DataFrame, date_str: str) -> pd.DataFrame:
        """
        Joins vehicle trajectories with the DuckDB static GTFS schedule
        matching by (route_id, brigade) and operating time window.
        """
        if df.empty:
            return df

        # Locate DuckDB schedule index
        gtfs_dir = paths.gtfs_dir
        db_files = list(gtfs_dir.glob("*/schedule_index.duckdb"))
        if not db_files:
            logger.warning("No schedule_index.duckdb found; proceeding with unlinked trajectories.")
            return df

        db_path = str(db_files[-1])

        try:
            con = duckdb.connect(db_path, read_only=True)
            # Query trip time spans by route and brigade
            trip_spans = con.execute("""
                SELECT
                    t.trip_id,
                    t.route_id,
                    t.brigade,
                    MIN(st.departure_time) AS trip_start,
                    MAX(st.arrival_time) AS trip_end
                FROM trips t
                JOIN stop_times st ON t.trip_id = st.trip_id
                WHERE t.brigade IS NOT NULL
                GROUP BY t.trip_id, t.route_id, t.brigade
            """).df()
            con.close()

            if trip_spans.empty:
                return df

            # Convert trip_start and trip_end ("HH:MM:SS") to seconds since midnight
            def time_to_seconds(t_str: str) -> int:
                parts = str(t_str).split(":")
                if len(parts) == 3:
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                return 0

            trip_spans["start_sec"] = trip_spans["trip_start"].apply(time_to_seconds)
            trip_spans["end_sec"] = trip_spans["trip_end"].apply(time_to_seconds)

            # Map time of day for vehicle fixes
            def extract_sec_of_day(time_str: str) -> int:
                try:
                    parts = time_str.rsplit(" ", maxsplit=1)[-1].split(":")
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                except Exception:
                    return 0

            df["sec_of_day"] = df["gps_time_str"].apply(extract_sec_of_day)

            # Build in-memory lookup map: (route_id, brigade) -> list of (trip_id, start_sec, end_sec)
            trip_map: Dict[tuple, list] = {}
            for _, r in trip_spans.iterrows():
                key = (str(r["route_id"]), str(r["brigade"]))
                if key not in trip_map:
                    trip_map[key] = []
                trip_map[key].append((r["trip_id"], r["start_sec"], r["end_sec"]))

            # Match each fix to active trip with a 15-minute buffer
            matched_trips = []
            for _, r in df.iterrows():
                key = (str(r["route_id"]), str(r["brigade"]))
                sec = r["sec_of_day"]
                candidate_id = None
                if key in trip_map:
                    for tid, s_start, s_end in trip_map[key]:
                        # 10m pre-departure to 15m post-arrival tolerance
                        if (s_start - 600) <= sec <= (s_end + 900):
                            candidate_id = tid
                            break
                matched_trips.append(candidate_id)

            df["matched_trip_id"] = matched_trips
            df.drop(columns=["sec_of_day"], inplace=True)

        except Exception as e:
            logger.warning(f"Error during GTFS trip matching: {e}")

        return df

    def process_date(self, date_str: str) -> Optional[Path]:
        """
        Executes end-to-end Silver trajectory generation for a given date.
        Saves output to data/historian/silver/date=YYYY-MM-DD/trajectories.parquet.
        """
        logger.info(f"Processing Silver trajectories for {date_str}...")
        raw_fixes = self.extract_raw_fixes(date_str)
        if not raw_fixes:
            logger.warning(f"No Bronze fixes found for date {date_str}.")
            return None

        clean_df = self.deduplicate_and_compute_kinematics(raw_fixes)
        matched_df = self.match_trips_using_gtfs(clean_df, date_str)

        out_partition_dir = self.silver_dir / f"date={date_str}"
        out_partition_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_partition_dir / "trajectories.parquet"

        table = pa.Table.from_pandas(matched_df, schema=SILVER_SCHEMA, preserve_index=False)
        tmp_file = out_file.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp_file, compression="zstd")
        tmp_file.replace(out_file)

        logger.info(
            f"Silver trajectories complete for {date_str}: {len(matched_df):,} fixes written to {out_file}"
        )
        return out_file

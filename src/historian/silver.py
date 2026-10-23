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
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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


def _parse_warsaw_time_to_utc(time_str: str, tick_ts: int) -> int:
    """Parses Warsaw local datetime string 'YYYY-MM-DD HH:MM:SS' into UTC epoch seconds."""
    try:
        naive_dt = datetime.datetime.strptime(time_str.strip(), "%Y-%m-%d %H:%M:%S")
        local_dt = naive_dt.replace(tzinfo=WARSAW_TZ)
        return int(local_dt.astimezone(UTC_TZ).timestamp())
    except Exception:
        return tick_ts


def _extract_single_hour_fixes(task_args: Tuple[str, str, str, str, str]) -> List[Dict[str, Any]]:
    """Worker function: extracts and normalizes raw GPS fixes from a single hour archive/partition."""
    base_dir_str, source_name, v_type, date_str, hour_str = task_args
    base_dir = Path(base_dir_str)
    from src.historian.bronze import iter_bronze_ticks

    hour_fixes: List[Dict[str, Any]] = []
    for tick_ts, raw_bytes, filename in iter_bronze_ticks(
        base_dir=base_dir,
        source=source_name,
        date_str=date_str,
        max_hours=[hour_str],
    ):
        try:
            records = json.loads(raw_bytes.decode("utf-8"))
        except Exception:
            continue

        if not isinstance(records, list):
            continue

        for r in records:
            time_str = r.get("Time")
            lat = r.get("Lat")
            lon = r.get("Lon")
            if not time_str or not lat or not lon:
                continue

            gps_ts = _parse_warsaw_time_to_utc(time_str, tick_ts)
            hour_fixes.append(
                {
                    "service_date": date_str,
                    "vehicle_type": v_type,
                    "vehicle_number": str(r.get("VehicleNumber", "")).strip(),
                    "route_id": str(r.get("Lines", "")).strip(),
                    "brigade": str(r.get("Brigade", "")).strip(),
                    "lat": float(lat),
                    "lon": float(lon),
                    "gps_timestamp_utc": gps_ts,
                    "gps_time_str": str(time_str).strip(),
                    "tick_ts": tick_ts,
                }
            )
    return hour_fixes


class SilverTrajectoryProcessor:
    """Transforms raw Bronze vehicle telemetry into deduplicated, trajectory-linked Silver Parquet tables."""

    def __init__(self, bronze_dir: Optional[Path] = None, silver_dir: Optional[Path] = None):
        self.bronze_dir = bronze_dir or paths.bronze_dir
        self.silver_dir = silver_dir or paths.silver_dir
        self.silver_dir.mkdir(parents=True, exist_ok=True)
        self.gtfs_versioner = HistorianGTFSVersioner()

    def parse_warsaw_time_to_utc(self, time_str: str, tick_ts: int) -> int:
        """Parses Warsaw local datetime string 'YYYY-MM-DD HH:MM:SS' into UTC epoch seconds."""
        return _parse_warsaw_time_to_utc(time_str, tick_ts)

    def extract_raw_fixes(
        self,
        date_str: str,
        max_hours: Optional[List[str]] = None,
        max_workers: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extracts and normalizes raw GPS fixes from all Bronze json.gz/tar.zst files for a given date.
        Uses ProcessPoolExecutor to decompress and parse hourly archives in parallel across CPU cores.
        """
        tasks: List[Tuple[str, str, str, str, str]] = []

        for v_type, source_name in [("bus", "um_bus"), ("tram", "um_tram")]:
            date_dir = self.bronze_dir / f"source={source_name}" / f"date={date_str}"
            if not date_dir.exists():
                continue

            # Discover all hours present
            hours_set = set()
            for item in date_dir.iterdir():
                name = item.name
                if name.startswith("hour=") and not name.endswith(".tmp"):
                    hour_part = name.split(".")[0].split("=")[-1]
                    hours_set.add(hour_part)

            for hour_str in sorted(hours_set):
                if max_hours and hour_str not in max_hours:
                    continue
                tasks.append((str(self.bronze_dir), source_name, v_type, date_str, hour_str))

        if not tasks:
            logger.warning(f"No Bronze fixes found for date {date_str}.")
            return []

        num_cores = os.cpu_count() or 4
        workers = min(len(tasks), max_workers or num_cores)

        if workers <= 1:
            logger.info(f"Extracting raw fixes sequentially across {len(tasks)} hour partitions...")
            fixes: List[Dict[str, Any]] = []
            for t in tasks:
                fixes.extend(_extract_single_hour_fixes(t))
            return fixes

        logger.info(
            f"Extracting raw fixes in parallel across {len(tasks)} hour partitions using {workers} worker processes..."
        )
        fixes = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for hour_fixes in executor.map(_extract_single_hour_fixes, tasks):
                fixes.extend(hour_fixes)

        logger.info(f"Parallel Bronze extraction complete for {date_str}: {len(fixes):,} raw fixes gathered.")
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

            for row in group.itertuples(index=False):
                lat = row.lat
                lon = row.lon
                ts = row.gps_timestamp_utc
                time_str = row.gps_time_str

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
                        "service_date": row.service_date,
                        "vehicle_type": v_type,
                        "vehicle_number": v_num,
                        "route_id": row.route_id,
                        "brigade": row.brigade,
                        "lat": lat,
                        "lon": lon,
                        "gps_timestamp_utc": ts,
                        "gps_time_str": time_str,
                        "tick_ts": row.tick_ts,
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

        # Locate DuckDB schedule index for the specific service date
        db_path = self.gtfs_versioner.get_active_duckdb(date_str)
        if not db_path or not Path(db_path).exists():
            logger.warning(f"No schedule_index.duckdb found for {date_str}; proceeding with unlinked trajectories.")
            return df

        try:
            con = duckdb.connect(str(db_path), read_only=True)
            tables = set(t[0] for t in con.execute("SHOW TABLES").fetchall())

            target_date_compact = date_str.replace("-", "")
            dt_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            day_of_week = dt_obj.strftime("%A").lower()

            # Filter trips by service_id active on date_str
            active_clause = ""
            if "calendar_dates" in tables and "calendar" in tables:
                active_clause = f"""
                    WITH active_services AS (
                        SELECT service_id
                        FROM calendar
                        WHERE {day_of_week} = 1
                          AND start_date <= '{target_date_compact}'
                          AND end_date >= '{target_date_compact}'
                          AND service_id NOT IN (
                              SELECT service_id FROM calendar_dates WHERE date = '{target_date_compact}' AND exception_type = 2
                          )
                        UNION
                        SELECT service_id
                        FROM calendar_dates
                        WHERE date = '{target_date_compact}' AND exception_type = 1
                    )
                """
            elif "calendar_dates" in tables:
                active_clause = f"""
                    WITH active_services AS (
                        SELECT DISTINCT service_id
                        FROM calendar_dates
                        WHERE date = '{target_date_compact}' AND exception_type = 1
                    )
                """
            elif "calendar" in tables:
                active_clause = f"""
                    WITH active_services AS (
                        SELECT DISTINCT service_id
                        FROM calendar
                        WHERE {day_of_week} = 1
                          AND start_date <= '{target_date_compact}'
                          AND end_date >= '{target_date_compact}'
                    )
                """

            if active_clause:
                query = f"""
                    {active_clause}
                    SELECT
                        t.trip_id,
                        t.route_id,
                        t.brigade,
                        MIN(st.departure_time) AS trip_start,
                        MAX(st.arrival_time) AS trip_end
                    FROM trips t
                    JOIN stop_times st ON t.trip_id = st.trip_id
                    JOIN active_services act ON t.service_id = act.service_id
                    WHERE t.brigade IS NOT NULL
                    GROUP BY t.trip_id, t.route_id, t.brigade
                    ORDER BY trip_start
                """
            else:
                query = """
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
                    ORDER BY trip_start
                """

            trip_spans = con.execute(query).df()

            # Graceful fallback: if calendar produced 0 trips (e.g., date out of range), use unfiltered trips
            if trip_spans.empty and active_clause:
                logger.warning(f"No active services matched for {date_str} in GTFS calendar; falling back to unfiltered trips.")
                fallback_query = """
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
                    ORDER BY trip_start
                """
                trip_spans = con.execute(fallback_query).df()

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
            for r in trip_spans[["route_id", "brigade", "trip_id", "start_sec", "end_sec"]].itertuples(index=False):
                key = (str(r.route_id), str(r.brigade))
                if key not in trip_map:
                    trip_map[key] = []
                trip_map[key].append((r.trip_id, r.start_sec, r.end_sec))

            # Match each fix to active trip with a 45-minute buffer (preventing severe delay censoring),
            # supporting GTFS 24:00+ midnight schedule conventions.
            matched_trips = []
            for r in df[["route_id", "brigade", "sec_of_day"]].itertuples(index=False):
                key = (str(r.route_id), str(r.brigade))
                sec = r.sec_of_day
                candidate_id = None
                if key in trip_map:
                    valid_candidates = []
                    # Evaluate both standard sec and next-day midnight continuation (sec + 86400) for late-night trips
                    candidate_secs = [sec, sec + 86400] if sec < 18000 else [sec]
                    for tid, s_start, s_end in trip_map[key]:
                        for eff_sec in candidate_secs:
                            # 15m pre-departure to 45m post-arrival tolerance
                            if (s_start - 900) <= eff_sec <= (s_end + 2700):
                                if s_start <= eff_sec <= s_end:
                                    dist = 0
                                elif eff_sec < s_start:
                                    dist = s_start - eff_sec
                                else:
                                    dist = eff_sec - s_end
                                valid_candidates.append((dist, tid))
                    if valid_candidates:
                        valid_candidates.sort(key=lambda x: x[0])
                        candidate_id = valid_candidates[0][1]

                matched_trips.append(candidate_id)

            df["matched_trip_id"] = matched_trips
            df.drop(columns=["sec_of_day"], inplace=True)

        except Exception as e:
            logger.warning(f"Error during GTFS trip matching: {e}")

        return df

    def process_date(self, date_str: str, force: bool = False) -> Optional[Path]:
        """
        Executes end-to-end Silver trajectory generation for a given date.
        Saves output to data/historian/silver/date=YYYY-MM-DD/trajectories.parquet.
        """
        out_partition_dir = self.silver_dir / f"date={date_str}"
        out_file = out_partition_dir / "trajectories.parquet"

        if not force and out_file.exists() and out_file.stat().st_size > 0:
            logger.info(
                f"Reusing existing Silver trajectories ({out_file.stat().st_size / (1024 * 1024):.1f} MB) for {date_str} at {out_file}"
            )
            return out_file

        logger.info(f"Processing Silver trajectories for {date_str}...")
        raw_fixes = self.extract_raw_fixes(date_str)
        if not raw_fixes:
            logger.warning(f"No Bronze fixes found for date {date_str}.")
            return None

        clean_df = self.deduplicate_and_compute_kinematics(raw_fixes)
        matched_df = self.match_trips_using_gtfs(clean_df, date_str)

        out_partition_dir.mkdir(parents=True, exist_ok=True)

        table = pa.Table.from_pandas(matched_df, schema=SILVER_SCHEMA, preserve_index=False)
        tmp_file = out_file.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp_file, compression="zstd")
        tmp_file.replace(out_file)

        logger.info(
            f"Silver trajectories complete for {date_str}: {len(matched_df):,} fixes written to {out_file}"
        )
        return out_file

    def rematch_trips_for_date(self, date_str: str) -> Optional[Path]:
        """
        Re-matches GTFS trips on existing Silver trajectories using the active calendar
        and overwrites trajectories.parquet without re-extracting Bronze.
        """
        out_partition_dir = self.silver_dir / f"date={date_str}"
        out_file = out_partition_dir / "trajectories.parquet"
        if not out_file.exists() or out_file.stat().st_size == 0:
            logger.warning(f"No existing Silver trajectory file found at {out_file} to re-match.")
            return None

        logger.info(f"Re-matching GTFS trips for {date_str} on {out_file}...")
        df = pd.read_parquet(out_file)
        matched_df = self.match_trips_using_gtfs(df, date_str)

        table = pa.Table.from_pandas(matched_df, schema=SILVER_SCHEMA, preserve_index=False)
        tmp_file = out_file.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp_file, compression="zstd")
        tmp_file.replace(out_file)

        logger.info(
            f"Silver trajectories re-matched for {date_str}: {len(matched_df):,} fixes written to {out_file}"
        )
        return out_file


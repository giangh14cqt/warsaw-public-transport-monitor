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
from typing import Any, Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.historian.config import paths
from src.historian.matching.geometry import (
    METERS_PER_DEG_LAT,
    METERS_PER_DEG_LON,
    metric_distance,
)
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
    """Parses Warsaw local datetime string 'YYYY-MM-DD HH:MM:SS' into UTC epoch seconds, resolving DST fold."""
    try:
        naive_dt = datetime.datetime.strptime(time_str.strip(), "%Y-%m-%d %H:%M:%S")
        ts0 = int(naive_dt.replace(tzinfo=WARSAW_TZ, fold=0).astimezone(UTC_TZ).timestamp())
        ts1 = int(naive_dt.replace(tzinfo=WARSAW_TZ, fold=1).astimezone(UTC_TZ).timestamp())
        if ts0 != ts1 and tick_ts > 100_000_000:
            return ts0 if abs(ts0 - tick_ts) <= abs(ts1 - tick_ts) else ts1
        return ts0
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
            if tick_ts > 100_000_000 and (tick_ts - gps_ts > 300 or gps_ts - tick_ts > 120):
                continue
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

    def __init__(
        self,
        bronze_dir: Optional[Path] = None,
        silver_dir: Optional[Path] = None,
        bronze_parquet_dir: Optional[Path] = None,
    ):
        self.bronze_dir = bronze_dir or paths.bronze_dir
        if bronze_parquet_dir is not None:
            self.bronze_parquet_dir = bronze_parquet_dir
        elif bronze_dir is not None and bronze_dir != paths.bronze_dir:
            self.bronze_parquet_dir = bronze_dir.parent / "bronze_parquet"
        else:
            self.bronze_parquet_dir = paths.bronze_parquet_dir
        self.silver_dir = silver_dir or paths.silver_dir
        self.silver_dir.mkdir(parents=True, exist_ok=True)
        self.gtfs_versioner = HistorianGTFSVersioner()

    def parse_warsaw_time_to_utc(self, time_str: str, tick_ts: int) -> int:
        """Parses Warsaw local datetime string 'YYYY-MM-DD HH:MM:SS' into UTC epoch seconds."""
        return _parse_warsaw_time_to_utc(time_str, tick_ts)

    def extract_raw_fixes_df(
        self,
        date_str: str,
        max_hours: Optional[List[str]] = None,
        max_workers: Optional[int] = None,
        use_raw_bronze: bool = False,
    ) -> pd.DataFrame:
        """
        Loads raw GPS fixes for a given date as a DataFrame.
        Prefers the pre-unpacked daily Parquet partition (bronze_parquet/date=YYYY-MM-DD/vehicle_fixes.parquet)
        unless use_raw_bronze=True or the Parquet partition is absent, in which case it falls back to
        decompressing raw Bronze .tar.zst/.gz archives.
        """
        pq_file = self.bronze_parquet_dir / f"date={date_str}" / "vehicle_fixes.parquet"
        if not use_raw_bronze and pq_file.exists() and pq_file.stat().st_size > 0:
            try:
                df = pq.read_table(pq_file).to_pandas()
                if max_hours and "hour_utc" in df.columns:
                    df = df[df["hour_utc"].isin(max_hours)].reset_index(drop=True)
                if "hour_utc" in df.columns:
                    df = df.drop(columns=["hour_utc"])
                logger.info(
                    f"Fast-loaded {len(df):,} raw fixes for {date_str} directly from {pq_file} (zero decompression)."
                )
                return df
            except Exception as e:
                logger.warning(
                    f"Failed reading daily Bronze Parquet {pq_file}, falling back to raw .tar.zst archives: {e}"
                )

        fixes = self.extract_raw_fixes(
            date_str=date_str,
            max_hours=max_hours,
            max_workers=max_workers,
            use_raw_bronze=True,
        )
        return pd.DataFrame(fixes) if fixes else pd.DataFrame()

    def extract_raw_fixes(
        self,
        date_str: str,
        max_hours: Optional[List[str]] = None,
        max_workers: Optional[int] = None,
        use_raw_bronze: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Extracts and normalizes raw GPS fixes from Bronze for a given date.
        Uses pre-unpacked bronze_parquet if available (unless use_raw_bronze=True),
        otherwise decompresses hourly .tar.zst/.gz archives in parallel across CPU cores.
        """
        pq_file = self.bronze_parquet_dir / f"date={date_str}" / "vehicle_fixes.parquet"
        if not use_raw_bronze and pq_file.exists() and pq_file.stat().st_size > 0:
            df = self.extract_raw_fixes_df(
                date_str=date_str,
                max_hours=max_hours,
                max_workers=max_workers,
                use_raw_bronze=False,
            )
            return df.to_dict("records") if not df.empty else []

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

    def deduplicate_and_compute_kinematics(
        self, raw_fixes: Union[List[Dict[str, Any]], pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Sorts fixes chronologically by vehicle, collapses repeated identical pings,
        and computes inter-fix delta distance, delta time, and speed using vectorized NumPy arrays.
        """
        if isinstance(raw_fixes, pd.DataFrame):
            if raw_fixes.empty:
                return pd.DataFrame(columns=[f.name for f in SILVER_SCHEMA])
            df = raw_fixes.copy()
        else:
            if not raw_fixes:
                return pd.DataFrame(columns=[f.name for f in SILVER_SCHEMA])
            df = pd.DataFrame(raw_fixes)

        # Sort chronologically per physical vehicle
        df.sort_values(
            by=["vehicle_type", "vehicle_number", "gps_timestamp_utc", "tick_ts"],
            inplace=True,
        )
        df.reset_index(drop=True, inplace=True)

        n = len(df)
        v_type_arr = df["vehicle_type"].to_numpy()
        v_num_arr = df["vehicle_number"].to_numpy()
        lat_arr = df["lat"].to_numpy(dtype=np.float64)
        lon_arr = df["lon"].to_numpy(dtype=np.float64)
        time_str_arr = df["gps_time_str"].to_numpy()

        same_veh = np.zeros(n, dtype=bool)
        if n > 1:
            same_veh[1:] = (v_type_arr[1:] == v_type_arr[:-1]) & (v_num_arr[1:] == v_num_arr[:-1])

        is_dup = np.zeros(n, dtype=bool)
        if n > 1:
            is_dup[1:] = (
                same_veh[1:]
                & (time_str_arr[1:] == time_str_arr[:-1])
                & (lat_arr[1:] == lat_arr[:-1])
                & (lon_arr[1:] == lon_arr[:-1])
            )

        keep_mask = ~is_dup
        keep_idx = np.flatnonzero(keep_mask)
        n_ticks = np.diff(np.append(keep_idx, n)).astype(np.int64)

        res = df.iloc[keep_idx].copy().reset_index(drop=True)
        m = len(res)

        v_type_r = res["vehicle_type"].to_numpy()
        v_num_r = res["vehicle_number"].to_numpy()
        lat_r = res["lat"].to_numpy(dtype=np.float64)
        lon_r = res["lon"].to_numpy(dtype=np.float64)
        ts_r = res["gps_timestamp_utc"].to_numpy(dtype=np.int64)

        same_veh_r = np.zeros(m, dtype=bool)
        if m > 1:
            same_veh_r[1:] = (v_type_r[1:] == v_type_r[:-1]) & (v_num_r[1:] == v_num_r[:-1])

        dt = np.zeros(m, dtype=np.float64)
        dd = np.zeros(m, dtype=np.float64)
        speed = np.zeros(m, dtype=np.float64)

        if m > 1:
            valid_dt = same_veh_r[1:] & (ts_r[1:] > ts_r[:-1])
            dt_step = (ts_r[1:] - ts_r[:-1]).astype(np.float64)
            dt[1:] = np.where(valid_dt, dt_step, 0.0)

            dx = (lon_r[1:] - lon_r[:-1]) * METERS_PER_DEG_LON
            dy = (lat_r[1:] - lat_r[:-1]) * METERS_PER_DEG_LAT
            dd_step = np.hypot(dx, dy)
            dd[1:] = np.where(valid_dt, dd_step, 0.0)

            np.divide(dd[1:], dt[1:], out=speed[1:], where=dt[1:] > 0)

        res["delta_t_s"] = np.round(dt, 2)
        res["delta_d_m"] = np.round(dd, 2)
        res["speed_mps"] = np.round(speed, 2)
        res["is_static"] = (speed < 0.5) & (dd < 5.0)
        res["n_ticks_seen"] = n_ticks
        res["matched_trip_id"] = None

        ordered_cols = [f.name for f in SILVER_SCHEMA]
        return res[ordered_cols]

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
                    ORDER BY trip_start, t.trip_id
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
                    ORDER BY trip_start, t.trip_id
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
                    ORDER BY trip_start, t.trip_id
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

            trip_spans["start_sec"] = [time_to_seconds(x) for x in trip_spans["trip_start"]]
            trip_spans["end_sec"] = [time_to_seconds(x) for x in trip_spans["trip_end"]]

            # Map time of day for vehicle fixes via unique string dictionary lookup
            def extract_sec_of_day(time_str: str) -> int:
                try:
                    parts = str(time_str).rsplit(" ", maxsplit=1)[-1].split(":")
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                except Exception:
                    return 0

            unique_tstrs = df["gps_time_str"].unique()
            tstr_map = {s: extract_sec_of_day(s) for s in unique_tstrs}
            sec_of_day_arr = df["gps_time_str"].map(tstr_map).to_numpy(dtype=np.int64)

            # Build in-memory lookup map: (route_id, brigade) -> (t_ids, t_starts, t_ends)
            raw_trip_map: Dict[tuple, list] = {}
            for r in trip_spans[["route_id", "brigade", "trip_id", "start_sec", "end_sec"]].itertuples(index=False):
                key = (str(r.route_id), str(r.brigade))
                if key not in raw_trip_map:
                    raw_trip_map[key] = []
                raw_trip_map[key].append((r.trip_id, int(r.start_sec), int(r.end_sec)))

            trip_map_np = {}
            for k, v in raw_trip_map.items():
                t_ids = np.array([x[0] for x in v], dtype=object)
                t_starts = np.array([x[1] for x in v], dtype=np.int64)
                t_ends = np.array([x[2] for x in v], dtype=np.int64)
                trip_map_np[k] = (t_ids, t_starts, t_ends)

            # Match each fix to active trip with a 45-minute buffer (preventing severe delay censoring),
            # supporting GTFS 24:00+ midnight schedule conventions via vectorized NumPy broadcasting per brigade.
            matched_arr = np.full(len(df), None, dtype=object)
            INF = 10**9

            for (r_id, brig), idx_arr in df.groupby(["route_id", "brigade"], sort=False).indices.items():
                key = (str(r_id), str(brig))
                if key not in trip_map_np:
                    continue
                t_ids, t_starts, t_ends = trip_map_np[key]
                secs = sec_of_day_arr[idx_arr]

                # Standard sec evaluation: shape (N_fixes, K_trips)
                s = secs[:, None]
                st = t_starts[None, :]
                en = t_ends[None, :]
                in_win1 = (s >= (st - 900)) & (s <= (en + 2700))
                dist1 = np.where(
                    ~in_win1,
                    INF,
                    np.where(
                        s < st,
                        (st - s) + 900,
                        np.where(s <= en, 0, s - en),
                    ),
                )

                # Midnight continuation (sec + 86400) for late-night trips (sec < 18000)
                is_late = secs < 18000
                if np.any(is_late):
                    s2 = (secs + 86400)[:, None]
                    in_win2 = is_late[:, None] & (s2 >= (st - 900)) & (s2 <= (en + 2700))
                    dist2 = np.where(
                        ~in_win2,
                        INF,
                        np.where(
                            s2 < st,
                            (st - s2) + 900,
                            np.where(s2 <= en, 0, s2 - en),
                        ),
                    )
                    best_per_trip = np.minimum(dist1, dist2)
                else:
                    best_per_trip = dist1

                min_dist = best_per_trip.min(axis=1)
                best_k = best_per_trip.argmin(axis=1)
                valid = min_dist < INF
                if np.any(valid):
                    matched_arr[idx_arr[valid]] = t_ids[best_k[valid]]

            df["matched_trip_id"] = matched_arr

        except Exception as e:
            logger.warning(f"Error during GTFS trip matching: {e}")

        return df

    def process_date(
        self,
        date_str: str,
        force: bool = False,
        use_raw_bronze: bool = False,
    ) -> Optional[Path]:
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
        raw_fixes_df = self.extract_raw_fixes_df(date_str, use_raw_bronze=use_raw_bronze)
        if raw_fixes_df.empty:
            logger.warning(f"No Bronze fixes found for date {date_str}.")
            return None

        clean_df = self.deduplicate_and_compute_kinematics(raw_fixes_df)
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
        if "tick_ts" in df.columns and "gps_timestamp_utc" in df.columns:
            dst_fold_mask = (df["tick_ts"] > 100_000_000) & (
                (df["tick_ts"] - df["gps_timestamp_utc"]).between(3400, 3800)
            )
            if dst_fold_mask.any():
                df.loc[dst_fold_mask, "gps_timestamp_utc"] += 3600
            real_ts_mask = df["tick_ts"] > 100_000_000
            valid_age_mask = (~real_ts_mask) | (
                ((df["tick_ts"] - df["gps_timestamp_utc"]) <= 300)
                & ((df["gps_timestamp_utc"] - df["tick_ts"]) <= 120)
            )
            df = df[valid_age_mask].reset_index(drop=True)
        matched_df = self.match_trips_using_gtfs(df, date_str)

        table = pa.Table.from_pandas(matched_df, schema=SILVER_SCHEMA, preserve_index=False)
        tmp_file = out_file.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp_file, compression="zstd")
        tmp_file.replace(out_file)

        logger.info(
            f"Silver trajectories re-matched for {date_str}: {len(matched_df):,} fixes written to {out_file}"
        )
        return out_file


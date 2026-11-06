"""
Warsaw Observed-Delay Historian: Gold Stop-Event Reconstruction Tier.
Reconstructs observed stop arrival and departure events with sub-10s linear interpolation,
calculates exact arrival/departure delays and dwell times, and cross-validates against
zbiorkom GTFS-RT forecasts to quantify forecast error.
"""

from collections import Counter
import datetime
import glob
import gzip
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from google.transit import gtfs_realtime_pb2

from src.historian.config import paths
from src.historian.matching.geometry import (
    LAT_0_DEG,
    LON_0_DEG,
    METERS_PER_DEG_LAT,
    METERS_PER_DEG_LON,
    interpolate_stop_boundary_timestamp,
    metric_distance,
)

logger = logging.getLogger(__name__)

WARSAW_TZ = ZoneInfo("Europe/Warsaw")
UTC_TZ = datetime.timezone.utc

GOLD_SCHEMA = pa.schema(
    [
        ("service_date", pa.string()),
        ("trip_id", pa.string()),
        ("route_id", pa.string()),
        ("brigade", pa.string()),
        ("vehicle_number", pa.string()),
        ("vehicle_type", pa.string()),
        ("stop_sequence", pa.int32()),
        ("stop_id", pa.string()),
        ("stop_name", pa.string()),
        ("scheduled_arrival_utc", pa.int64()),
        ("scheduled_departure_utc", pa.int64()),
        ("actual_arrival_utc", pa.int64()),
        ("actual_departure_utc", pa.int64()),
        ("arrival_delay_seconds", pa.int32()),
        ("departure_delay_seconds", pa.int32()),
        ("dwell_seconds", pa.int32()),
        ("zbk_forecast_delay_s", pa.int32()),
        ("forecast_error_s", pa.int32()),
        ("event_type", pa.string()),
        ("confidence", pa.string()),
        ("min_distance_to_stop_m", pa.float32()),
    ]
)


class GoldStopEventReconstructor:
    """Detects stop geofence crossings, performs sub-10s interpolation, and computes actual delays."""

    def __init__(
        self,
        silver_dir: Optional[Path] = None,
        gold_dir: Optional[Path] = None,
        bronze_dir: Optional[Path] = None,
        stop_radius_meters: float = 25.0,
        require_complete_trips: bool = True,
        bronze_parquet_dir: Optional[Path] = None,
    ):
        self.silver_dir = silver_dir or paths.silver_dir
        self.gold_dir = gold_dir or paths.gold_dir
        self.bronze_dir = bronze_dir or paths.bronze_dir
        if bronze_parquet_dir is not None:
            self.bronze_parquet_dir = bronze_parquet_dir
        elif bronze_dir is not None and bronze_dir != paths.bronze_dir:
            self.bronze_parquet_dir = bronze_dir.parent / "bronze_parquet"
        elif (
            gold_dir is not None
            and gold_dir != paths.gold_dir
            and (gold_dir.parent / "bronze_parquet").exists()
        ):
            self.bronze_parquet_dir = gold_dir.parent / "bronze_parquet"
        else:
            self.bronze_parquet_dir = paths.bronze_parquet_dir
        self.stop_radius = stop_radius_meters
        self.require_complete_trips = require_complete_trips
        (self.gold_dir / "stop_events").mkdir(parents=True, exist_ok=True)

    def _get_active_duckdb(self, date_str: Optional[str] = None) -> str:
        from src.historian.static_gtfs import HistorianGTFSVersioner

        gtfs_parent = (
            self.gold_dir.parent / "gtfs"
            if (self.gold_dir and (self.gold_dir.parent / "gtfs").exists())
            else None
        )
        versioner = HistorianGTFSVersioner(gtfs_dir=gtfs_parent)
        active_db = versioner.get_active_duckdb(date_str)
        if active_db and Path(active_db).exists():
            return str(active_db)

        gtfs_dir = paths.gtfs_dir
        db_files = sorted(gtfs_dir.glob("*/schedule_index.duckdb"))
        if not db_files:
            raise FileNotFoundError("No schedule_index.duckdb found for GTFS timetable lookups.")
        return str(db_files[-1])

    def _parse_hhmmss_to_utc(self, date_str: str, time_str: str) -> int:
        """Converts timetable 'HH:MM:SS' (which can be > 24 for late night trips) to UTC epoch."""
        try:
            parts = str(time_str).strip().split(":")
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            day_offset = h // 24
            h = h % 24

            dt_local = datetime.datetime.strptime(date_str, "%Y-%m-%d").replace(
                hour=h, minute=m, second=s, tzinfo=WARSAW_TZ
            )
            if day_offset > 0:
                dt_local += datetime.timedelta(days=day_offset)
            return int(dt_local.astimezone(UTC_TZ).timestamp())
        except Exception:
            return 0

    def load_zbk_forecast_delays(
        self, date_str: str, use_raw_bronze: bool = False
    ) -> Dict[tuple, Any]:
        """
        Loads GTFS-RT forecast delays for the given date.
        Prefers the pre-unpacked daily Parquet partition (bronze_parquet/date=YYYY-MM-DD/zbk_forecasts.parquet)
        unless use_raw_bronze=True or the Parquet partition is absent, in which case it falls back to
        parsing Bronze zbk_rt protobuf files.
        Returns mapping: (trip_id, seq) -> ("delay"|"time", value)
        """
        forecasts: Dict[tuple, Any] = {}
        pq_file = self.bronze_parquet_dir / f"date={date_str}" / "zbk_forecasts.parquet"
        if not use_raw_bronze and pq_file.exists() and pq_file.stat().st_size > 0:
            try:
                zbk_df = pq.read_table(
                    pq_file,
                    columns=["trip_id", "stop_sequence", "forecast_kind", "forecast_val"],
                ).to_pandas()
                forecasts = dict(
                    zip(
                        zip(zbk_df["trip_id"].tolist(), zbk_df["stop_sequence"].tolist()),
                        zip(zbk_df["forecast_kind"].tolist(), zbk_df["forecast_val"].tolist()),
                    )
                )
                logger.info(
                    f"Fast-loaded {len(forecasts):,} GTFS-RT stop forecasts for {date_str} directly from {pq_file}."
                )
                return forecasts
            except Exception as e:
                logger.warning(
                    f"Failed reading daily zbk_forecasts Parquet {pq_file}, falling back to raw protobufs: {e}"
                )

        from src.historian.bronze import iter_bronze_ticks

        for tick_ts, content, filename in iter_bronze_ticks(
            base_dir=self.bronze_dir,
            source="zbk_rt",
            date_str=date_str,
        ):
            try:
                feed = gtfs_realtime_pb2.FeedMessage()
                feed.ParseFromString(content)

                for entity in feed.entity:
                    if entity.HasField("trip_update"):
                        tu = entity.trip_update
                        trip_id = tu.trip.trip_id if tu.HasField("trip") else ""
                        if not trip_id:
                            continue
                        for stu in tu.stop_time_update:
                            seq = stu.stop_sequence
                            if stu.HasField("arrival"):
                                if stu.arrival.HasField("delay"):
                                    forecasts[(trip_id, seq)] = ("delay", stu.arrival.delay)
                                elif stu.arrival.HasField("time"):
                                    forecasts[(trip_id, seq)] = ("time", stu.arrival.time)
                            elif stu.HasField("departure"):
                                if stu.departure.HasField("delay"):
                                    forecasts[(trip_id, seq)] = ("delay", stu.departure.delay)
                                elif stu.departure.HasField("time"):
                                    forecasts[(trip_id, seq)] = ("time", stu.departure.time)
            except Exception:
                continue

        return forecasts

    def process_date(
        self, date_str: str, force: bool = False, use_raw_bronze: bool = False
    ) -> Optional[Path]:
        """
        Reconstructs all stop arrival/departure events and actual delays for a given date.
        If force=False and gold partition exists, reuses existing file for fast multi-date processing.
        """
        out_partition_dir = self.gold_dir / "stop_events" / f"date={date_str}"
        out_file = out_partition_dir / "stop_events.parquet"
        if not force and out_file.exists() and out_file.stat().st_size > 0:
            logger.info(
                f"Reusing existing Gold stop events ({out_file.stat().st_size / (1024 * 1024):.1f} MB) for {date_str} at {out_file} (use force=True to reprocess)."
            )
            return out_file

        silver_file = self.silver_dir / f"date={date_str}" / "trajectories.parquet"
        if not silver_file.exists():
            logger.warning(f"No Silver trajectory file found at {silver_file}.")
            return None

        logger.info(f"Reconstructing Gold stop events for {date_str} from {silver_file}...")
        silver_df = pd.read_parquet(silver_file)
        if "tick_ts" in silver_df.columns and "gps_timestamp_utc" in silver_df.columns:
            # Repair 1-hour autumn DST fold offset if present on raw cached Silver timestamps
            dst_fold_mask = (silver_df["tick_ts"] > 100_000_000) & (
                (silver_df["tick_ts"] - silver_df["gps_timestamp_utc"]).between(3400, 3800)
            )
            if dst_fold_mask.any():
                silver_df = silver_df.copy()
                silver_df.loc[dst_fold_mask, "gps_timestamp_utc"] += 3600

            # Filter out stale cached onboard timestamps (>5m behind poll tick or >2m in future)
            real_ts_mask = silver_df["tick_ts"] > 100_000_000
            valid_age_mask = (~real_ts_mask) | (
                ((silver_df["tick_ts"] - silver_df["gps_timestamp_utc"]) <= 300)
                & ((silver_df["gps_timestamp_utc"] - silver_df["tick_ts"]) <= 120)
            )
            silver_df = silver_df[valid_age_mask]

        # Keep only fixes that matched a GTFS trip
        matched_df = silver_df[silver_df["matched_trip_id"].notna()].reset_index(drop=True)
        if matched_df.empty:
            logger.warning("No matched trips in Silver trajectories.")
            return None

        # Load static timetable stops for all matched trips via DuckDB
        db_path = self._get_active_duckdb(date_str)
        con = duckdb.connect(db_path, read_only=True)
        unique_trips = matched_df["matched_trip_id"].unique().tolist()

        placeholders = ",".join(["?"] * len(unique_trips))
        stops_query = f"""
            SELECT
                st.trip_id,
                st.stop_sequence,
                st.stop_id,
                st.arrival_time,
                st.departure_time,
                s.stop_name,
                s.stop_lat,
                s.stop_lon
            FROM stop_times st
            JOIN stops s ON st.stop_id = s.stop_id
            WHERE st.trip_id IN ({placeholders})
            ORDER BY st.trip_id, st.stop_sequence
        """
        trip_stops_df = con.execute(stops_query, unique_trips).df()
        con.close()

        if trip_stops_df.empty:
            logger.warning(f"No matching GTFS stop_times found for {date_str}.")
            return None

        # Load zbiorkom benchmark forecasts
        zbk_forecasts = self.load_zbk_forecast_delays(date_str, use_raw_bronze=use_raw_bronze)

        # Pre-cache unique HH:MM:SS timetable strings -> UTC epoch timestamps
        unique_times = pd.unique(
            np.concatenate(
                [
                    trip_stops_df["arrival_time"].to_numpy(),
                    trip_stops_df["departure_time"].to_numpy(),
                ]
            )
        )
        t_cache = {t: self._parse_hhmmss_to_utc(date_str, t) for t in unique_times}

        st_arr_utc_arr = np.array(
            [t_cache[t] for t in trip_stops_df["arrival_time"].to_numpy()], dtype=np.int64
        )
        st_arr_utc_list = st_arr_utc_arr.tolist()
        st_dep_utc_list = [t_cache[t] for t in trip_stops_df["departure_time"].tolist()]
        st_seq_list = [int(x) for x in trip_stops_df["stop_sequence"].tolist()]
        st_stop_id_list = [str(x) for x in trip_stops_df["stop_id"].tolist()]
        st_stop_name_list = [str(x) for x in trip_stops_df["stop_name"].tolist()]
        st_lat_all = trip_stops_df["stop_lat"].to_numpy(dtype=np.float64)
        st_lon_all = trip_stops_df["stop_lon"].to_numpy(dtype=np.float64)
        st_sx_all = (st_lon_all - LON_0_DEG) * METERS_PER_DEG_LON
        st_sy_all = (st_lat_all - LAT_0_DEG) * METERS_PER_DEG_LAT

        # Pre-extract matched trajectory arrays and Warsaw metric projections
        m_veh_num = matched_df["vehicle_number"].astype(str).to_numpy()
        m_veh_type_list = [str(x) for x in matched_df["vehicle_type"].tolist()]
        m_route_id_list = [str(x) for x in matched_df["route_id"].tolist()]
        m_brigade_list = [str(x) for x in matched_df["brigade"].tolist()]
        m_lat = matched_df["lat"].to_numpy(dtype=np.float64)
        m_lon = matched_df["lon"].to_numpy(dtype=np.float64)
        m_ts = matched_df["gps_timestamp_utc"].to_numpy(dtype=np.int64)
        m_fx_all = (m_lon - LON_0_DEG) * METERS_PER_DEG_LON
        m_fy_all = (m_lat - LAT_0_DEG) * METERS_PER_DEG_LAT

        fix_groups = matched_df.groupby("matched_trip_id", sort=False).indices
        stop_groups = trip_stops_df.groupby("trip_id", sort=True).indices

        gold_rows: List[Dict[str, Any]] = []
        retained_trips_cnt = 0
        dropped_incomplete_cnt = 0
        radius = self.stop_radius

        # Process each trip's scheduled stops using 2D vectorized Euclidean distance matrices
        for trip_id, s_idx in stop_groups.items():
            if trip_id not in fix_groups:
                continue

            f_idx = fix_groups[trip_id]
            if len(f_idx) > 1:
                f_idx = f_idx[np.argsort(m_ts[f_idx], kind="quicksort")]

            # Enforce single physical vehicle per trip instance (select dominant vehicle by fix count)
            v_nums = m_veh_num[f_idx]
            first_v = str(v_nums[0])
            if not np.all(v_nums == first_v):
                dominant_veh = str(Counter(v_nums).most_common(1)[0][0])
                f_idx = f_idx[v_nums == dominant_veh]
                vehicle_num = dominant_veh
            else:
                vehicle_num = first_v

            idx0 = int(f_idx[0])
            vehicle_type = m_veh_type_list[idx0]
            route_id = m_route_id_list[idx0]
            brigade = m_brigade_list[idx0]

            fix_times = m_ts[f_idx]
            fix_times_list = fix_times.tolist()
            fix_xs = m_fx_all[f_idx]
            fix_ys = m_fy_all[f_idx]
            n_fixes = len(fix_times_list)
            if n_fixes == 0:
                continue
            first_fix_valid = fix_times_list[0] > 100_000_000

            s_idx_list = s_idx.tolist()
            sched_min_seq = st_seq_list[s_idx_list[0]]
            sched_max_seq = st_seq_list[s_idx_list[-1]]
            for si in s_idx_list:
                sq = st_seq_list[si]
                if sq < sched_min_seq:
                    sched_min_seq = sq
                if sq > sched_max_seq:
                    sched_max_seq = sq

            # Compute full 2D (n_stops x n_fixes) distance matrix and +/- 90 min temporal window
            s_xs = st_sx_all[s_idx]
            s_ys = st_sy_all[s_idx]
            s_arrs = st_arr_utc_arr[s_idx]
            dists_mat = np.hypot(fix_xs[None, :] - s_xs[:, None], fix_ys[None, :] - s_ys[:, None])
            if first_fix_valid:
                time_ok = (s_arrs[:, None] <= 100_000_000) | (
                    np.abs(fix_times[None, :] - s_arrs[:, None]) <= 5400
                )
                masked_dists_mat = np.where(time_ok, dists_mat, 99999.0)
            else:
                masked_dists_mat = dists_mat

            row_min_all = masked_dists_mat.min(axis=1).tolist()
            row_argmin_all = masked_dists_mat.argmin(axis=1).tolist()

            in_rows, in_cols = np.nonzero(masked_dists_mat <= radius)
            in_by_k: Dict[int, List[int]] = {}
            if len(in_rows) > 0:
                for r_k, c_idx in zip(in_rows.tolist(), in_cols.tolist()):
                    if r_k not in in_by_k:
                        in_by_k[r_k] = [c_idx]
                    else:
                        in_by_k[r_k].append(c_idx)

            trip_rows: List[Dict[str, Any]] = []
            min_allowed_idx = 0

            for k, si in enumerate(s_idx_list):
                g_min = row_min_all[k]
                if g_min > 60.0:
                    continue
                g_idx = row_argmin_all[k]
                if g_idx >= min_allowed_idx:
                    min_dist = g_min
                    closest_idx_cached = g_idx
                else:
                    sl = masked_dists_mat[k, min_allowed_idx:]
                    if len(sl) == 0:
                        continue
                    min_dist = float(sl.min())
                    if min_dist > 60.0:
                        continue
                    closest_idx_cached = min_allowed_idx + int(sl.argmin())

                seq = st_seq_list[si]
                stop_id = st_stop_id_list[si]
                stop_name = st_stop_name_list[si]
                sched_arr_utc = st_arr_utc_list[si]
                sched_dep_utc = st_dep_utc_list[si]

                if min_dist <= radius:
                    event_type = "STOPPED"
                    raw_in = in_by_k[k]
                    inside_indices = (
                        [ix for ix in raw_in if ix >= min_allowed_idx]
                        if raw_in[0] < min_allowed_idx
                        else raw_in
                    )
                    first_in = inside_indices[0]
                    last_in = first_in
                    for idx in inside_indices[1:]:
                        dt_in = float(fix_times_list[idx] - fix_times_list[last_in])
                        if dt_in <= 180.0 or (idx == last_in + 1 and dt_in <= 1200.0):
                            last_in = idx
                        else:
                            break
                    min_allowed_idx = last_in

                    dists_k = dists_mat[k]
                    # Sub-10s arrival interpolation
                    if first_in > 0:
                        t_arr = interpolate_stop_boundary_timestamp(
                            t_outside=float(fix_times_list[first_in - 1]),
                            d_outside=float(dists_k[first_in - 1]),
                            t_inside=float(fix_times_list[first_in]),
                            d_inside=float(dists_k[first_in]),
                            radius_meters=radius,
                        )
                        bracket = fix_times_list[first_in] - fix_times_list[first_in - 1]
                    else:
                        t_arr = float(fix_times_list[first_in])
                        bracket = 10.0

                    # Sub-10s departure interpolation
                    if last_in < n_fixes - 1:
                        t_dep = interpolate_stop_boundary_timestamp(
                            t_outside=float(fix_times_list[last_in + 1]),
                            d_outside=float(dists_k[last_in + 1]),
                            t_inside=float(fix_times_list[last_in]),
                            d_inside=float(dists_k[last_in]),
                            radius_meters=radius,
                        )
                    else:
                        t_dep = float(fix_times_list[last_in])

                    actual_arr_utc = int(round(t_arr))
                    actual_dep_utc = int(round(t_dep))
                    dwell_s = max(0, actual_dep_utc - actual_arr_utc)

                    confidence = "HIGH" if bracket <= 12.0 else ("MEDIUM" if bracket <= 25.0 else "LOW")

                    arr_delay_s = (actual_arr_utc - sched_arr_utc) if sched_arr_utc > 0 else None
                    dep_delay_s = (actual_dep_utc - sched_dep_utc) if sched_dep_utc > 0 else None

                else:
                    event_type = "PASSED"
                    # Vehicle passed near stop without stopping (radius < min_dist <= 60.0)
                    min_allowed_idx = closest_idx_cached
                    actual_arr_utc = fix_times_list[closest_idx_cached]
                    actual_dep_utc = actual_arr_utc
                    dwell_s = 0
                    confidence = "MEDIUM"
                    arr_delay_s = (actual_arr_utc - sched_arr_utc) if sched_arr_utc > 0 else None
                    dep_delay_s = (actual_dep_utc - sched_dep_utc) if sched_dep_utc > 0 else None

                # Forecast comparison
                zbk_val = zbk_forecasts.get((trip_id, seq))
                zbk_forecast = None
                if zbk_val is not None:
                    val_type, val = zbk_val
                    if val_type == "delay":
                        zbk_forecast = int(val)
                    elif val_type == "time" and sched_arr_utc > 0:
                        zbk_forecast = int(val - sched_arr_utc)

                forecast_error = (
                    (arr_delay_s - zbk_forecast)
                    if (arr_delay_s is not None and zbk_forecast is not None)
                    else None
                )

                trip_rows.append(
                    {
                        "service_date": date_str,
                        "trip_id": trip_id,
                        "route_id": route_id,
                        "brigade": brigade,
                        "vehicle_number": vehicle_num,
                        "vehicle_type": vehicle_type,
                        "stop_sequence": seq,
                        "stop_id": stop_id,
                        "stop_name": stop_name,
                        "scheduled_arrival_utc": sched_arr_utc,
                        "scheduled_departure_utc": sched_dep_utc,
                        "actual_arrival_utc": actual_arr_utc,
                        "actual_departure_utc": actual_dep_utc,
                        "arrival_delay_seconds": arr_delay_s,
                        "departure_delay_seconds": dep_delay_s,
                        "dwell_seconds": dwell_s,
                        "zbk_forecast_delay_s": zbk_forecast,
                        "forecast_error_s": forecast_error,
                        "event_type": event_type,
                        "confidence": confidence,
                        "min_distance_to_stop_m": round(min_dist, 2),
                    }
                )

            if not trip_rows:
                continue

            # Ensure complete or majority-observed trips are kept while dropping tiny fragments
            if self.require_complete_trips:
                obs_seqs = {r["stop_sequence"] for r in trip_rows}
                has_start = sched_min_seq in obs_seqs
                has_terminal = sched_max_seq in obs_seqs
                n_sched = len(s_idx_list)
                has_majority_coverage = (n_sched >= 6) and (len(obs_seqs) >= max(5, int(0.6 * n_sched)))
                if not ((has_start and has_terminal) or has_majority_coverage):
                    dropped_incomplete_cnt += 1
                    continue

            retained_trips_cnt += 1
            gold_rows.extend(trip_rows)

        if not gold_rows:
            logger.warning(f"No stop events generated for {date_str}.")
            return None

        gold_df = pd.DataFrame(gold_rows)
        out_partition_dir = self.gold_dir / "stop_events" / f"date={date_str}"
        out_partition_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_partition_dir / "stop_events.parquet"

        table = pa.Table.from_pandas(gold_df, schema=GOLD_SCHEMA, preserve_index=False)
        tmp_file = out_file.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp_file, compression="zstd")
        tmp_file.replace(out_file)

        stopped_cnt = len(gold_df[gold_df["event_type"] == "STOPPED"])
        logger.info(
            f"Gold stop events complete for {date_str}: {len(gold_df):,} events ({stopped_cnt:,} STOPPED) across {retained_trips_cnt:,} complete trips ({dropped_incomplete_cnt:,} incomplete trips dropped) written to {out_file}"
        )
        return out_file

    reconstruct_date = process_date

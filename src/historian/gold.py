"""
Warsaw Observed-Delay Historian: Gold Stop-Event Reconstruction Tier.
Reconstructs observed stop arrival and departure events with sub-10s linear interpolation,
calculates exact arrival/departure delays and dwell times, and cross-validates against
zbiorkom GTFS-RT forecasts to quantify forecast error.
"""

import datetime
import glob
import gzip
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from google.transit import gtfs_realtime_pb2

from src.historian.config import paths
from src.historian.matching.geometry import interpolate_stop_boundary_timestamp, metric_distance

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
    ):
        self.silver_dir = silver_dir or paths.silver_dir
        self.gold_dir = gold_dir or paths.gold_dir
        self.bronze_dir = bronze_dir or paths.bronze_dir
        self.stop_radius = stop_radius_meters
        (self.gold_dir / "stop_events").mkdir(parents=True, exist_ok=True)

    def _get_active_duckdb(self) -> str:
        gtfs_dir = paths.gtfs_dir
        db_files = list(gtfs_dir.glob("*/schedule_index.duckdb"))
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

    def load_zbk_forecast_delays(self, date_str: str) -> Dict[tuple, Any]:
        """
        Parses Bronze zbk_rt protobuf files for the given date to extract the latest
        forecasted arrival delay or timestamp for each (trip_id, stop_sequence).
        Returns mapping: (trip_id, seq) -> ("delay"|"time", value)
        """
        forecasts: Dict[tuple, Any] = {}
        zbk_pattern = str(self.bronze_dir / "source=zbk_rt" / f"date={date_str}" / "hour=*" / "*.pb.gz")
        pb_files = sorted(glob.glob(zbk_pattern))

        for pb_file in pb_files:
            try:
                with gzip.open(pb_file, "rb") as f:
                    content = f.read()
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

    def process_date(self, date_str: str) -> Optional[Path]:
        """
        Reconstructs all stop arrival/departure events and actual delays for a given date.
        """
        silver_file = self.silver_dir / f"date={date_str}" / "trajectories.parquet"
        if not silver_file.exists():
            logger.warning(f"No Silver trajectory file found at {silver_file}.")
            return None

        logger.info(f"Reconstructing Gold stop events for {date_str} from {silver_file}...")
        silver_df = pd.read_parquet(silver_file)
        # Keep only fixes that matched a GTFS trip
        matched_df = silver_df[silver_df["matched_trip_id"].notna()].copy()
        if matched_df.empty:
            logger.warning("No matched trips in Silver trajectories.")
            return None

        # Load static timetable stops for all matched trips via DuckDB
        db_path = self._get_active_duckdb()
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

        # Load zbiorkom benchmark forecasts
        zbk_forecasts = self.load_zbk_forecast_delays(date_str)

        # Group fixes by trip_id
        fixes_by_trip = {tid: grp.sort_values("gps_timestamp_utc") for tid, grp in matched_df.groupby("matched_trip_id")}

        gold_rows: List[Dict[str, Any]] = []

        # Process each trip's scheduled stops
        for trip_id, stop_group in trip_stops_df.groupby("trip_id"):
            if trip_id not in fixes_by_trip:
                continue

            trip_fixes = fixes_by_trip[trip_id]
            vehicle_num = str(trip_fixes.iloc[0]["vehicle_number"])
            vehicle_type = str(trip_fixes.iloc[0]["vehicle_type"])
            route_id = str(trip_fixes.iloc[0]["route_id"])
            brigade = str(trip_fixes.iloc[0]["brigade"])

            fix_lats = trip_fixes["lat"].values
            fix_lons = trip_fixes["lon"].values
            fix_times = trip_fixes["gps_timestamp_utc"].values

            for _, stop_row in stop_group.iterrows():
                seq = int(stop_row["stop_sequence"])
                stop_id = str(stop_row["stop_id"])
                stop_name = str(stop_row["stop_name"])
                s_lat = float(stop_row["stop_lat"])
                s_lon = float(stop_row["stop_lon"])

                sched_arr_utc = self._parse_hhmmss_to_utc(date_str, stop_row["arrival_time"])
                sched_dep_utc = self._parse_hhmmss_to_utc(date_str, stop_row["departure_time"])

                # Compute distances from all fixes of this trip to the stop
                dists = [metric_distance(lat, lon, s_lat, s_lon) for lat, lon in zip(fix_lats, fix_lons, strict=False)]
                min_dist = min(dists) if dists else 99999.0

                # Detect entry and exit around stop radius (25m)
                inside_indices = [i for i, d in enumerate(dists) if d <= self.stop_radius]

                actual_arr_utc = None
                actual_dep_utc = None
                event_type = "SKIPPED"
                confidence = "LOW"
                arr_delay_s = None
                dep_delay_s = None
                dwell_s = None

                if inside_indices:
                    event_type = "STOPPED"
                    first_in = inside_indices[0]
                    last_in = inside_indices[-1]

                    # Sub-10s arrival interpolation
                    if first_in > 0:
                        t_arr = interpolate_stop_boundary_timestamp(
                            t_outside=float(fix_times[first_in - 1]),
                            d_outside=dists[first_in - 1],
                            t_inside=float(fix_times[first_in]),
                            d_inside=dists[first_in],
                            radius_meters=self.stop_radius,
                        )
                        bracket = fix_times[first_in] - fix_times[first_in - 1]
                    else:
                        t_arr = float(fix_times[first_in])
                        bracket = 10.0

                    # Sub-10s departure interpolation
                    if last_in < len(fix_times) - 1:
                        t_dep = interpolate_stop_boundary_timestamp(
                            t_outside=float(fix_times[last_in + 1]),
                            d_outside=dists[last_in + 1],
                            t_inside=float(fix_times[last_in]),
                            d_inside=dists[last_in],
                            radius_meters=self.stop_radius,
                        )
                    else:
                        t_dep = float(fix_times[last_in])

                    actual_arr_utc = int(round(t_arr))
                    actual_dep_utc = int(round(t_dep))
                    dwell_s = max(0, actual_dep_utc - actual_arr_utc)

                    confidence = "HIGH" if bracket <= 12.0 else ("MEDIUM" if bracket <= 25.0 else "LOW")

                    if sched_arr_utc > 0:
                        arr_delay_s = actual_arr_utc - sched_arr_utc
                    if sched_dep_utc > 0:
                        dep_delay_s = actual_dep_utc - sched_dep_utc

                elif min_dist <= 60.0:
                    event_type = "PASSED"
                    # Vehicle passed near stop without stopping
                    closest_idx = dists.index(min_dist)
                    actual_arr_utc = int(fix_times[closest_idx])
                    actual_dep_utc = actual_arr_utc
                    dwell_s = 0
                    confidence = "MEDIUM"
                    if sched_arr_utc > 0:
                        arr_delay_s = actual_arr_utc - sched_arr_utc
                    if sched_dep_utc > 0:
                        dep_delay_s = actual_dep_utc - sched_dep_utc
                else:
                    # Vehicle never approached stop (min_dist > 60m); drop unobserved non-event
                    continue

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

                gold_rows.append(
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
            f"Gold stop events complete for {date_str}: {len(gold_df):,} events ({stopped_cnt:,} STOPPED) written to {out_file}"
        )
        return out_file

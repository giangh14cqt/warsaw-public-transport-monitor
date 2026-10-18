"""
Spatial-Temporal Multi-Source Fusion & Feature Mart Assembler.
Performs:
1. Reconstructs vehicle trip trajectories and autoregressive delay lag features.
2. Temporal Join with hourly IMGW synoptic weather telemetry.
3. Spatial Join with OSMnx corridor road topology and traffic signal infrastructure.
4. Anomaly filtering (drops sensor freezes / depot standbys with |arrival_delay| > 120 minutes).
5. Persists the unified modeling matrix to data/processed/feature_mart.parquet.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import duckdb

logger = logging.getLogger(__name__)


class FeatureMartAssembler:
    """Fuses operational transit records, weather metrics, and road network attributes."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.historian_dir = os.path.join(data_dir, "historian")
        self.historian_gold_pattern = os.path.join(
            self.historian_dir, "gold", "stop_events", "date=*", "stop_events.parquet"
        )
        self.historian_processed_dir = os.path.join(self.historian_dir, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.historian_processed_dir, exist_ok=True)
        self.output_parquet = os.path.join(self.processed_dir, "feature_mart.parquet")
        self.historian_output_parquet = os.path.join(
            self.historian_processed_dir, "feature_mart_observed.parquet"
        )

    def _build_exogenous_subqueries(
        self,
        weather_parquet_path: str,
        osm_parquet_path: str,
        calendar_parquet_path: Optional[str] = None,
        interchange_parquet_path: Optional[str] = None,
    ) -> Tuple[str, str, str, str]:
        """Constructs reusable exogenous subqueries for weather, OSM corridors, calendar, and hubs."""
        has_weather = os.path.exists(weather_parquet_path)
        has_osm = os.path.exists(osm_parquet_path)

        cal_path = calendar_parquet_path or os.path.join(
            self.processed_dir, "calendar_lookup.parquet"
        )
        if not os.path.exists(cal_path):
            try:
                from src.core.polish_calendar import generate_calendar_parquet

                generate_calendar_parquet(cal_path)
            except Exception as e:
                logger.warning(f"Could not auto-generate calendar lookup: {e}")
        has_cal = os.path.exists(cal_path)

        hub_path = interchange_parquet_path or os.path.join(
            self.processed_dir, "transit_stop_interchanges.parquet"
        )
        if not os.path.exists(hub_path):
            try:
                from src.exogenous.interchange import extract_default_interchanges

                extract_default_interchanges(output_path=hub_path)
            except Exception as e:
                logger.warning(f"Could not auto-generate stop interchanges: {e}")
        has_hub = os.path.exists(hub_path)

        weather_subquery = (
            f"SELECT * FROM read_parquet('{weather_parquet_path}')"
            if has_weather
            else """
                SELECT
                    NULL::TIMESTAMP AS timestamp,
                    11.3::DOUBLE AS temperature_c,
                    0.0::DOUBLE AS precipitation_mm,
                    3.0::DOUBLE AS wind_speed_ms,
                    75.0::DOUBLE AS relative_humidity,
                    false::BOOLEAN AS freezing_rain_flag
                WHERE 1=0
            """
        )

        if has_osm:
            con = duckdb.connect()
            try:
                osm_cols = [
                    c[0]
                    for c in con.execute(
                        f"DESCRIBE SELECT * FROM read_parquet('{osm_parquet_path}') LIMIT 1"
                    ).fetchall()
                ]
                has_osm_act_path = "actual_path_length_meters" in osm_cols
                has_osm_tort = "tortuosity_ratio" in osm_cols
            except Exception:
                has_osm_act_path = False
                has_osm_tort = False
            finally:
                con.close()

            extra_cols = []
            if not has_osm_act_path:
                extra_cols.append("segment_length_meters AS actual_path_length_meters")
            if not has_osm_tort:
                extra_cols.append("1.0::DOUBLE AS tortuosity_ratio")
            extra_sql = (", " + ", ".join(extra_cols)) if extra_cols else ""
            osm_subquery = f"SELECT * {extra_sql} FROM read_parquet('{osm_parquet_path}')"
        else:
            osm_subquery = """
                SELECT
                    ''::VARCHAR AS stop_id_prev,
                    ''::VARCHAR AS stop_id_curr,
                    ''::VARCHAR AS corridor_name,
                    500.0::DOUBLE AS segment_length_meters,
                    500.0::DOUBLE AS actual_path_length_meters,
                    1.0::DOUBLE AS tortuosity_ratio,
                    2::BIGINT AS signalized_intersection_count,
                    true::BOOLEAN AS is_dedicated_right_of_way,
                    3::BIGINT AS lane_capacity
                WHERE 1=0
            """

        calendar_subquery = (
            f"SELECT calendar_date, is_school_day, is_public_holiday FROM read_parquet('{cal_path}')"
            if has_cal
            else """
                SELECT
                    '2026-10-01'::VARCHAR AS calendar_date,
                    1::BIGINT AS is_school_day,
                    0::BIGINT AS is_public_holiday
                WHERE 1=0
            """
        )

        interchange_subquery = (
            f"SELECT stop_id, interchange_line_count, is_metro_proximity FROM read_parquet('{hub_path}')"
            if has_hub
            else """
                SELECT
                    ''::VARCHAR AS stop_id,
                    1::BIGINT AS interchange_line_count,
                    0::BIGINT AS is_metro_proximity
                WHERE 1=0
            """
        )

        return weather_subquery, osm_subquery, calendar_subquery, interchange_subquery

    def build_historian_fusion_sql(
        self,
        gold_pattern: str,
        weather_parquet_path: str,
        osm_parquet_path: str,
        calendar_parquet_path: Optional[str] = None,
        interchange_parquet_path: Optional[str] = None,
        max_delay_seconds: int = 7200,
        max_records: Optional[int] = None,
        exclude_terminals: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> str:
        """Construct the optimized DuckDB SQL query for Track B observed stop event fusion."""
        limit_clause = f"LIMIT {max_records}" if max_records else ""
        (
            weather_subquery,
            osm_subquery,
            calendar_subquery,
            interchange_subquery,
        ) = self._build_exogenous_subqueries(
            weather_parquet_path, osm_parquet_path, calendar_parquet_path, interchange_parquet_path
        )

        terminal_filter_clause = (
            "AND (t.is_terminal_stop = false OR t.max_seq <= t.min_seq + 1)"
            if exclude_terminals
            else ""
        )

        date_filter_clause = ""
        if start_date or end_date:
            conditions = []
            if start_date:
                conditions.append(f"service_date >= '{start_date[:10]}'")
            if end_date:
                conditions.append(f"service_date <= '{end_date[:10]}'")
            date_filter_clause = "AND " + " AND ".join(conditions)

        return f"""
            WITH raw_stops AS (
                SELECT
                    service_date,
                    trip_id,
                    trip_id || '_' || service_date AS trip_instance_key,
                    route_id,
                    vehicle_number AS vehicle_id,
                    vehicle_type,
                    stop_sequence,
                    stop_id,
                    stop_name,
                    scheduled_arrival_utc,
                    scheduled_departure_utc,
                    actual_arrival_utc,
                    actual_departure_utc,
                    arrival_delay_seconds,
                    departure_delay_seconds,
                    dwell_seconds,
                    zbk_forecast_delay_s,
                    forecast_error_s,
                    confidence
                FROM read_parquet('{gold_pattern}')
                WHERE event_type = 'STOPPED'
                  AND actual_arrival_utc IS NOT NULL
                  AND ABS(arrival_delay_seconds) <= {max_delay_seconds}
                  {date_filter_clause}
                {limit_clause}
            ),
            ordered AS (
                SELECT
                    *,
                    LAG(stop_id) OVER (PARTITION BY trip_instance_key ORDER BY stop_sequence) AS prev_stop_id,
                    LAG(arrival_delay_seconds) OVER (PARTITION BY trip_instance_key ORDER BY stop_sequence) AS prev_stop_delay,
                    LAG(arrival_delay_seconds, 2) OVER (PARTITION BY trip_instance_key ORDER BY stop_sequence) AS prev2_stop_delay,
                    LAG(actual_departure_utc) OVER (PARTITION BY trip_instance_key ORDER BY stop_sequence) AS prev_actual_dep,
                    LAG(scheduled_departure_utc) OVER (PARTITION BY trip_instance_key ORDER BY stop_sequence) AS prev_sched_dep,
                    MAX(stop_sequence) OVER (PARTITION BY trip_instance_key) AS max_seq,
                    MIN(stop_sequence) OVER (PARTITION BY trip_instance_key) AS min_seq
                FROM raw_stops
            ),
            trajectories AS (
                SELECT
                    trip_id,
                    trip_instance_key,
                    service_date AS start_date,
                    route_id,
                    vehicle_id,
                    stop_sequence,
                    stop_id,
                    stop_name,
                    prev_stop_id,
                    arrival_delay_seconds,
                    departure_delay_seconds,
                    dwell_seconds,
                    zbk_forecast_delay_s,
                    forecast_error_s,
                    confidence,
                    prev_stop_delay,
                    prev2_stop_delay,
                    actual_arrival_utc AS rt_arrival_time,
                    actual_departure_utc AS rt_departure_time,
                    to_timestamp(actual_arrival_utc) AS arrival_ts,
                    date_trunc('hour', to_timestamp(actual_arrival_utc)) AS timestamp_bucket,
                    EXTRACT(hour FROM to_timestamp(actual_arrival_utc)) AS hour_of_day,
                    EXTRACT(dayofweek FROM to_timestamp(actual_arrival_utc)) AS day_of_week,
                    CASE WHEN EXTRACT(dayofweek FROM to_timestamp(actual_arrival_utc)) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
                    CASE WHEN EXTRACT(hour FROM to_timestamp(actual_arrival_utc)) IN (7, 8, 9, 16, 17, 18) THEN 1 ELSE 0 END AS is_peak_hour,
                    CASE WHEN EXTRACT(hour FROM to_timestamp(actual_arrival_utc)) IN (7, 8, 9) THEN 1 ELSE 0 END AS is_morning_peak,
                    CASE WHEN EXTRACT(hour FROM to_timestamp(actual_arrival_utc)) IN (16, 17, 18) THEN 1 ELSE 0 END AS is_evening_peak,
                    CASE WHEN prev_stop_delay IS NOT NULL THEN (arrival_delay_seconds - prev_stop_delay) ELSE 0.0 END AS delta_t_run,
                    (departure_delay_seconds - arrival_delay_seconds) AS delta_t_dwell,
                    CASE
                        WHEN prev_actual_dep IS NOT NULL AND actual_arrival_utc > prev_actual_dep
                        THEN (actual_arrival_utc - prev_actual_dep)
                        ELSE NULL
                    END AS observed_run_time_s,
                    CASE WHEN max_seq > min_seq THEN ROUND(CAST(stop_sequence - min_seq AS DOUBLE) / (max_seq - min_seq), 3) ELSE 0.0 END AS trip_progress,
                    CASE WHEN stop_sequence = min_seq THEN true ELSE false END AS is_origin_stop,
                    CASE WHEN stop_sequence = max_seq AND max_seq > min_seq THEN true ELSE false END AS is_terminal_stop,
                    max_seq,
                    min_seq,
                    CASE WHEN vehicle_type = 'tram' OR (TRY_CAST(route_id AS INTEGER) IS NOT NULL AND TRY_CAST(route_id AS INTEGER) < 100) THEN 1 ELSE 0 END AS is_tram,
                    CASE
                        WHEN vehicle_type = 'tram' OR (TRY_CAST(route_id AS INTEGER) BETWEEN 1 AND 79) THEN 'urban_tram'
                        WHEN TRY_CAST(route_id AS INTEGER) BETWEEN 100 AND 599 THEN 'urban_bus'
                        WHEN TRY_CAST(route_id AS INTEGER) BETWEEN 700 AND 899 OR UPPER(route_id) LIKE 'L%' THEN 'suburban_bus'
                        WHEN UPPER(route_id) LIKE 'N%' THEN 'night_bus'
                        ELSE 'other'
                    END AS transit_cohort
                FROM ordered
            ),
            with_headway AS (
                SELECT
                    *,
                    (rt_arrival_time - LAG(rt_arrival_time) OVER (
                        PARTITION BY route_id, stop_id
                        ORDER BY rt_arrival_time
                    )) AS headway_actual_s
                FROM trajectories
            ),
            v_weather AS ({weather_subquery}),
            v_osm AS ({osm_subquery}),
            v_calendar AS ({calendar_subquery}),
            v_interchange AS ({interchange_subquery})
            SELECT
                t.trip_id,
                t.route_id,
                t.vehicle_id,
                t.stop_sequence,
                t.stop_id,
                t.stop_name,
                t.prev_stop_id,
                t.arrival_delay_seconds,
                t.departure_delay_seconds,
                t.dwell_seconds,
                t.zbk_forecast_delay_s,
                t.forecast_error_s,
                t.confidence,
                t.prev_stop_delay,
                t.prev2_stop_delay,
                t.delta_t_run,
                t.delta_t_dwell,
                t.observed_run_time_s,
                t.trip_progress,
                t.is_origin_stop,
                t.is_terminal_stop,
                t.is_tram,
                t.transit_cohort,
                t.arrival_ts,
                t.timestamp_bucket,
                t.hour_of_day,
                t.day_of_week,
                t.is_weekend,
                t.is_peak_hour,
                t.is_morning_peak,
                t.is_evening_peak,
                COALESCE(cal.is_school_day, CASE WHEN t.is_weekend = 1 THEN 0 ELSE 1 END) AS is_school_day,
                COALESCE(cal.is_public_holiday, 0) AS is_public_holiday,
                t.headway_actual_s,
                CASE
                    WHEN t.headway_actual_s >= 10 AND t.headway_actual_s <= 7200
                    THEN t.headway_actual_s - MEDIAN(t.headway_actual_s) OVER (PARTITION BY t.route_id, t.stop_id)
                    ELSE NULL
                END AS headway_deviation,
                COALESCE(hub.interchange_line_count, 1) AS interchange_line_count,
                COALESCE(hub.is_metro_proximity, 0) AS is_metro_proximity,
                COALESCE(w.temperature_c, 11.3) AS temperature_c,
                COALESCE(w.precipitation_mm, 0.0) AS precipitation_mm,
                COALESCE(w.wind_speed_ms, 3.0) AS wind_speed_ms,
                COALESCE(w.relative_humidity, 75.0) AS relative_humidity,
                COALESCE(w.freezing_rain_flag, false) AS freezing_rain_flag,
                osm.corridor_name,
                COALESCE(osm.segment_length_meters, 500.0) AS segment_length_meters,
                COALESCE(osm.actual_path_length_meters, osm.segment_length_meters, 500.0) AS actual_path_length_meters,
                COALESCE(osm.tortuosity_ratio, 1.0) AS tortuosity_ratio,
                COALESCE(osm.signalized_intersection_count, 2) AS signalized_intersection_count,
                COALESCE(osm.is_dedicated_right_of_way, CASE WHEN t.is_tram = 1 THEN true ELSE false END) AS is_dedicated_right_of_way,
                COALESCE(osm.lane_capacity, 3) AS lane_capacity,
                CASE WHEN osm.corridor_name IS NOT NULL THEN 1 ELSE 0 END AS is_study_corridor
            FROM with_headway t
            LEFT JOIN v_weather w ON t.timestamp_bucket = w.timestamp::TIMESTAMP
            LEFT JOIN v_osm osm ON (t.prev_stop_id = osm.stop_id_prev AND t.stop_id = osm.stop_id_curr)
            LEFT JOIN v_calendar cal ON CAST(t.arrival_ts AS DATE)::VARCHAR = cal.calendar_date
            LEFT JOIN v_interchange hub ON t.stop_id = hub.stop_id
            WHERE t.prev_stop_id IS NOT NULL
              {terminal_filter_clause}
              AND ABS(t.delta_t_run) <= 1800
            ORDER BY t.arrival_ts
        """

    def build_fusion_sql(
        self,
        raw_pattern: str,
        weather_parquet_path: str,
        osm_parquet_path: str,
        calendar_parquet_path: Optional[str] = None,
        interchange_parquet_path: Optional[str] = None,
        max_delay_seconds: int = 7200,
        max_records: Optional[int] = None,
        exclude_terminals: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        datasource: str = "legacy",
    ) -> str:
        """Construct the optimized DuckDB SQL query for multi-source spatiotemporal fusion."""
        if datasource == "historian" or "historian" in raw_pattern or "stop_events" in raw_pattern:
            return self.build_historian_fusion_sql(
                gold_pattern=raw_pattern,
                weather_parquet_path=weather_parquet_path,
                osm_parquet_path=osm_parquet_path,
                calendar_parquet_path=calendar_parquet_path,
                interchange_parquet_path=interchange_parquet_path,
                max_delay_seconds=max_delay_seconds,
                max_records=max_records,
                exclude_terminals=exclude_terminals,
                start_date=start_date,
                end_date=end_date,
            )

        limit_clause = f"LIMIT {max_records}" if max_records else ""
        (
            weather_subquery,
            osm_subquery,
            calendar_subquery,
            interchange_subquery,
        ) = self._build_exogenous_subqueries(
            weather_parquet_path, osm_parquet_path, calendar_parquet_path, interchange_parquet_path
        )

        terminal_filter_clause = (
            "AND (t.is_terminal_stop = false OR t.max_seq <= t.min_seq + 1)"
            if exclude_terminals
            else ""
        )

        date_filter_clause = ""
        if start_date or end_date:
            from src.pipeline.ranges import build_date_filter_clause

            filter_expr = build_date_filter_clause(
                "feed_timestamp",
                start_date=start_date,
                end_date=end_date,
                is_epoch_seconds=True,
            )
            if filter_expr and filter_expr != "1=1":
                date_filter_clause = f"AND {filter_expr}"

        # Inspect raw schema to check if start_date exists
        con = duckdb.connect()
        try:
            cols = [
                c[0]
                for c in con.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{raw_pattern}', hive_partitioning=true) LIMIT 1"
                ).fetchall()
            ]
            has_start_date = "start_date" in cols
        except Exception:
            has_start_date = False
        finally:
            con.close()

        if has_start_date:
            trip_key_sql = "trip_id || '_' || COALESCE(start_date, CAST(to_timestamp(feed_timestamp) AS DATE)::VARCHAR)"
            start_date_select = "start_date,"
        else:
            trip_key_sql = "trip_id || '_' || CAST(to_timestamp(COALESCE(rt_arrival_time, feed_timestamp)) AS DATE)::VARCHAR"
            start_date_select = "CAST(to_timestamp(COALESCE(rt_arrival_time, feed_timestamp)) AS DATE)::VARCHAR AS start_date,"

        return f"""
            WITH deduped AS (
                SELECT
                    feed_timestamp,
                    trip_id,
                    {trip_key_sql} AS trip_instance_key,
                    {start_date_select}
                    route_id,
                    vehicle_id,
                    stop_sequence,
                    stop_id,
                    arrival_delay_seconds,
                    departure_delay_seconds,
                    rt_arrival_time,
                    rt_departure_time
                FROM read_parquet('{raw_pattern}', hive_partitioning=true)
                WHERE ABS(arrival_delay_seconds) <= {max_delay_seconds}
                {date_filter_clause}
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY trip_instance_key, stop_sequence
                    ORDER BY feed_timestamp DESC
                ) = 1
                {limit_clause}
            ),
            ordered AS (
                SELECT
                    feed_timestamp,
                    trip_id,
                    trip_instance_key,
                    start_date,
                    route_id,
                    vehicle_id,
                    stop_sequence,
                    stop_id,
                    arrival_delay_seconds,
                    COALESCE(departure_delay_seconds, arrival_delay_seconds) AS departure_delay_seconds,
                    rt_arrival_time,
                    COALESCE(rt_departure_time, rt_arrival_time) AS rt_departure_time,
                    -- Preceding Stop Lags
                    LAG(stop_id) OVER (PARTITION BY trip_instance_key ORDER BY stop_sequence) AS prev_stop_id,
                    LAG(arrival_delay_seconds) OVER (PARTITION BY trip_instance_key ORDER BY stop_sequence) AS prev_stop_delay,
                    LAG(arrival_delay_seconds, 2) OVER (PARTITION BY trip_instance_key ORDER BY stop_sequence) AS prev2_stop_delay,
                    LAG(COALESCE(rt_departure_time, rt_arrival_time)) OVER (PARTITION BY trip_instance_key ORDER BY stop_sequence) AS prev_rt_departure,
                    MAX(stop_sequence) OVER (PARTITION BY trip_instance_key) AS max_seq,
                    MIN(stop_sequence) OVER (PARTITION BY trip_instance_key) AS min_seq
                FROM deduped
            ),
            trajectories AS (
                SELECT
                    trip_id,
                    trip_instance_key,
                    start_date,
                    route_id,
                    vehicle_id,
                    stop_sequence,
                    stop_id,
                    prev_stop_id,
                    arrival_delay_seconds,
                    departure_delay_seconds,
                    prev_stop_delay,
                    prev2_stop_delay,
                    rt_arrival_time,
                    rt_departure_time,
                    to_timestamp(rt_arrival_time) AS arrival_ts,
                    date_trunc('hour', to_timestamp(rt_arrival_time)) AS timestamp_bucket,
                    EXTRACT(hour FROM to_timestamp(rt_arrival_time)) AS hour_of_day,
                    EXTRACT(dayofweek FROM to_timestamp(rt_arrival_time)) AS day_of_week,
                    CASE WHEN EXTRACT(dayofweek FROM to_timestamp(rt_arrival_time)) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
                    CASE WHEN EXTRACT(hour FROM to_timestamp(rt_arrival_time)) IN (7, 8, 9, 16, 17, 18) THEN 1 ELSE 0 END AS is_peak_hour,
                    CASE WHEN EXTRACT(hour FROM to_timestamp(rt_arrival_time)) IN (7, 8, 9) THEN 1 ELSE 0 END AS is_morning_peak,
                    CASE WHEN EXTRACT(hour FROM to_timestamp(rt_arrival_time)) IN (16, 17, 18) THEN 1 ELSE 0 END AS is_evening_peak,
                    -- Running Delay (Delta t_run) & Dwell Delay (Delta t_dwell)
                    CASE WHEN prev_stop_delay IS NOT NULL THEN (arrival_delay_seconds - prev_stop_delay) ELSE 0.0 END AS delta_t_run,
                    (departure_delay_seconds - arrival_delay_seconds) AS delta_t_dwell,
                    -- Observed physical running time in seconds
                    CASE
                        WHEN prev_rt_departure IS NOT NULL AND rt_arrival_time > prev_rt_departure
                        THEN (rt_arrival_time - prev_rt_departure)
                        ELSE NULL
                    END AS observed_run_time_s,
                    -- Trip Progress
                    CASE WHEN max_seq > min_seq THEN ROUND(CAST(stop_sequence - min_seq AS DOUBLE) / (max_seq - min_seq), 3) ELSE 0.0 END AS trip_progress,
                    CASE WHEN stop_sequence = min_seq THEN true ELSE false END AS is_origin_stop,
                    CASE WHEN stop_sequence = max_seq AND max_seq > min_seq THEN true ELSE false END AS is_terminal_stop,
                    max_seq,
                    min_seq,
                    CASE WHEN TRY_CAST(route_id AS INTEGER) IS NOT NULL AND TRY_CAST(route_id AS INTEGER) < 100 THEN 1 ELSE 0 END AS is_tram,
                    CASE
                        WHEN TRY_CAST(route_id AS INTEGER) BETWEEN 1 AND 79 THEN 'urban_tram'
                        WHEN TRY_CAST(route_id AS INTEGER) BETWEEN 100 AND 599 THEN 'urban_bus'
                        WHEN TRY_CAST(route_id AS INTEGER) BETWEEN 700 AND 899 OR UPPER(route_id) LIKE 'L%' THEN 'suburban_bus'
                        WHEN UPPER(route_id) LIKE 'N%' THEN 'night_bus'
                        ELSE 'other'
                    END AS transit_cohort
                FROM ordered
            ),
            with_headway AS (
                SELECT
                    *,
                    (rt_arrival_time - LAG(rt_arrival_time) OVER (
                        PARTITION BY route_id, stop_id
                        ORDER BY rt_arrival_time
                    )) AS headway_actual_s
                FROM trajectories
            ),
            v_weather AS ({weather_subquery}),
            v_osm AS ({osm_subquery}),
            v_calendar AS ({calendar_subquery}),
            v_interchange AS ({interchange_subquery})
            SELECT
                t.trip_id,
                t.route_id,
                t.vehicle_id,
                t.stop_sequence,
                t.stop_id,
                t.prev_stop_id,
                t.arrival_delay_seconds,
                t.departure_delay_seconds,
                t.prev_stop_delay,
                t.prev2_stop_delay,
                t.delta_t_run,
                t.delta_t_dwell,
                t.observed_run_time_s,
                t.trip_progress,
                t.is_origin_stop,
                t.is_terminal_stop,
                t.is_tram,
                t.transit_cohort,
                t.arrival_ts,
                t.timestamp_bucket,
                t.hour_of_day,
                t.day_of_week,
                t.is_weekend,
                t.is_peak_hour,
                t.is_morning_peak,
                t.is_evening_peak,
                COALESCE(cal.is_school_day, CASE WHEN t.is_weekend = 1 THEN 0 ELSE 1 END) AS is_school_day,
                COALESCE(cal.is_public_holiday, 0) AS is_public_holiday,
                t.headway_actual_s,
                CASE
                    WHEN t.headway_actual_s >= 10 AND t.headway_actual_s <= 7200
                    THEN t.headway_actual_s - MEDIAN(t.headway_actual_s) OVER (PARTITION BY t.route_id, t.stop_id)
                    ELSE NULL
                END AS headway_deviation,
                -- Transfer Hub Interchange Exogenous Features
                COALESCE(hub.interchange_line_count, 1) AS interchange_line_count,
                COALESCE(hub.is_metro_proximity, 0) AS is_metro_proximity,
                -- IMGW Weather Exogenous Features
                COALESCE(w.temperature_c, 11.3) AS temperature_c,
                COALESCE(w.precipitation_mm, 0.0) AS precipitation_mm,
                COALESCE(w.wind_speed_ms, 3.0) AS wind_speed_ms,
                COALESCE(w.relative_humidity, 75.0) AS relative_humidity,
                COALESCE(w.freezing_rain_flag, false) AS freezing_rain_flag,
                -- OSMnx Road Topology Exogenous Features
                osm.corridor_name,
                COALESCE(osm.segment_length_meters, 500.0) AS segment_length_meters,
                COALESCE(osm.actual_path_length_meters, osm.segment_length_meters, 500.0) AS actual_path_length_meters,
                COALESCE(osm.tortuosity_ratio, 1.0) AS tortuosity_ratio,
                COALESCE(osm.signalized_intersection_count, 2) AS signalized_intersection_count,
                COALESCE(osm.is_dedicated_right_of_way, CASE WHEN t.is_tram = 1 THEN true ELSE false END) AS is_dedicated_right_of_way,
                COALESCE(osm.lane_capacity, 3) AS lane_capacity,
                CASE WHEN osm.corridor_name IS NOT NULL THEN 1 ELSE 0 END AS is_study_corridor
            FROM with_headway t
            LEFT JOIN v_weather w ON t.timestamp_bucket = w.timestamp::TIMESTAMP
            LEFT JOIN v_osm osm ON (t.prev_stop_id = osm.stop_id_prev AND t.stop_id = osm.stop_id_curr)
            LEFT JOIN v_calendar cal ON CAST(t.arrival_ts AS DATE)::VARCHAR = cal.calendar_date
            LEFT JOIN v_interchange hub ON t.stop_id = hub.stop_id
            WHERE t.prev_stop_id IS NOT NULL  -- Retain stop-to-stop running segments
              {terminal_filter_clause}
              AND ABS(t.delta_t_run) <= 1800   -- Filter extreme segment anomalies (>30 min)
            ORDER BY t.arrival_ts
        """

    def assemble_mart(
        self,
        raw_pattern: Optional[str] = None,
        weather_parquet_path: Optional[str] = None,
        osm_parquet_path: Optional[str] = None,
        calendar_parquet_path: Optional[str] = None,
        interchange_parquet_path: Optional[str] = None,
        output_parquet: Optional[str] = None,
        max_records: Optional[int] = None,
        exclude_terminals: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        datasource: str = "legacy",
    ) -> str:
        """Execute full spatial-temporal join and anomaly filtering in DuckDB."""
        if datasource == "historian" or (
            raw_pattern and ("historian" in raw_pattern or "stop_events" in raw_pattern)
        ):
            in_raw = raw_pattern or self.historian_gold_pattern
            out_path = output_parquet or self.historian_output_parquet
            ds = "historian"
        else:
            in_raw = raw_pattern or os.path.join(self.raw_dir, "**/*.parquet")
            out_path = output_parquet or self.output_parquet
            ds = "legacy"

        in_weather = weather_parquet_path or os.path.join(
            self.processed_dir, "imgw_weather_hourly.parquet"
        )
        in_osm = osm_parquet_path or os.path.join(
            self.processed_dir, "osm_corridor_segments.parquet"
        )
        in_calendar = calendar_parquet_path or os.path.join(
            self.processed_dir, "calendar_lookup.parquet"
        )
        in_interchange = interchange_parquet_path or os.path.join(
            self.processed_dir, "transit_stop_interchanges.parquet"
        )

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        con = duckdb.connect(":memory:")
        sql = self.build_fusion_sql(
            raw_pattern=in_raw,
            weather_parquet_path=in_weather,
            osm_parquet_path=in_osm,
            calendar_parquet_path=in_calendar,
            interchange_parquet_path=in_interchange,
            max_records=max_records,
            exclude_terminals=exclude_terminals,
            start_date=start_date,
            end_date=end_date,
            datasource=ds,
        )

        copy_query = f"COPY ({sql}) TO '{out_path}' (FORMAT PARQUET, COMPRESSION SNAPPY)"
        logger.info(f"Fusing multi-source feature mart [{ds}] from {in_raw}...")
        con.execute(copy_query)
        con.close()

        logger.info(f"Assembled feature mart written to: {out_path}")
        return out_path

    # Backward-compatible alias
    assemble = assemble_mart

    def get_summary(self, parquet_path: Optional[str] = None) -> Dict[str, Any]:
        """Fetch descriptive summary of the assembled feature mart table via DuckDB."""
        path = parquet_path or self.output_parquet
        if not os.path.exists(path):
            return {"status": "empty", "records": 0}

        con = duckdb.connect(":memory:")
        try:
            cols = [
                c[0]
                for c in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()
            ]
        except Exception:
            cols = []

        has_dwell = "dwell_seconds" in cols
        dwell_select = (
            "ROUND(AVG(dwell_seconds), 2) AS avg_dwell_s,"
            if has_dwell
            else "0.0::DOUBLE AS avg_dwell_s,"
        )

        res = con.execute(f"""
            SELECT
                COUNT(*) AS total_observations,
                COUNT(DISTINCT trip_id) AS distinct_trips,
                COUNT(DISTINCT route_id) AS distinct_routes,
                SUM(is_study_corridor) AS study_corridor_observations,
                ROUND(AVG(delta_t_run), 2) AS avg_delta_t_run,
                ROUND(MEDIAN(delta_t_run), 2) AS median_delta_t_run,
                ROUND(STDDEV(delta_t_run), 2) AS std_delta_t_run,
                ROUND(AVG(segment_length_meters), 1) AS avg_segment_meters,
                ROUND(AVG(signalized_intersection_count), 2) AS avg_signals,
                ROUND(AVG(temperature_c), 1) AS avg_temperature_c,
                ROUND(SUM(CASE WHEN is_peak_hour = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS peak_hour_pct,
                ROUND(SUM(CASE WHEN is_tram = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS tram_share_pct,
                {dwell_select}
            FROM read_parquet('{path}')
        """).fetchone()

        by_corr = (
            con.execute(f"""
            SELECT
                COALESCE(corridor_name, 'non_study_network') AS corridor,
                COUNT(*) AS obs,
                ROUND(AVG(delta_t_run), 2) AS avg_delta_t_run,
                ROUND(AVG(segment_length_meters), 1) AS avg_length_m,
                ROUND(AVG(signalized_intersection_count), 2) AS avg_signals
            FROM read_parquet('{path}')
            GROUP BY corridor_name
            ORDER BY obs DESC
        """)
            .df()
            .to_dict(orient="records")
        )

        if not res:
            return {
                "status": "empty",
                "total_observations": 0,
                "distinct_trips": 0,
                "distinct_routes": 0,
                "study_corridor_observations": 0,
                "avg_delta_t_run": 0.0,
                "median_delta_t_run": 0.0,
                "std_delta_t_run": 0.0,
                "avg_segment_meters": 0.0,
                "avg_signals": 0.0,
                "avg_temperature_c": 0.0,
                "peak_hour_pct": 0.0,
                "tram_share_pct": 0.0,
                "avg_dwell_seconds": 0.0,
                "corridors": by_corr,
            }

        return {
            "total_observations": int(res[0]) if res[0] is not None else 0,
            "distinct_trips": int(res[1]) if res[1] is not None else 0,
            "distinct_routes": int(res[2]) if res[2] is not None else 0,
            "study_corridor_observations": int(res[3]) if res[3] is not None else 0,
            "avg_delta_t_run": float(res[4]) if res[4] is not None else 0.0,
            "median_delta_t_run": float(res[5]) if res[5] is not None else 0.0,
            "std_delta_t_run": float(res[6]) if res[6] is not None else 0.0,
            "avg_segment_meters": float(res[7]) if res[7] is not None else 0.0,
            "avg_signals": float(res[8]) if res[8] is not None else 0.0,
            "avg_temperature_c": float(res[9]) if res[9] is not None else 0.0,
            "peak_hour_pct": float(res[10]) if res[10] is not None else 0.0,
            "tram_share_pct": float(res[11]) if res[11] is not None else 0.0,
            "avg_dwell_seconds": float(res[12]) if (len(res) > 12 and res[12] is not None) else 0.0,
            "corridors": by_corr,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    assembler = FeatureMartAssembler()
    print("FeatureMartAssembler ready.")

"""
Spatial-Temporal Multi-Source Fusion & Feature Mart Assembler.
Performs:
1. Temporal Join with hourly IMGW weather telemetry.
2. Spatial Join with OSMnx corridor road topology.
3. Anomaly filtering (drops sensor freezes / depot pullouts with |Delta t| > 120 minutes).
4. Persists the final modeling matrix to data/processed/feature_mart.parquet.
"""

import os
import logging
from typing import Optional
import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureMartAssembler:
    """Fuses operational transit records, weather metrics, and road network attributes."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
        self.output_parquet = os.path.join(self.processed_dir, "feature_mart.parquet")

    def assemble_mart(
        self,
        weather_parquet_path: Optional[str] = None,
        osm_parquet_path: Optional[str] = None,
        max_records: Optional[int] = None,
    ) -> str:
        """Execute full spatial-temporal join and anomaly filtering in DuckDB."""
        con = duckdb.connect(":memory:")

        # 1. Register telemetry
        telemetry_query = f"""
            SELECT 
                feed_timestamp,
                trip_id,
                route_id,
                vehicle_id,
                stop_sequence,
                stop_id,
                arrival_delay_seconds,
                departure_delay_seconds,
                rt_arrival_time,
                rt_departure_time,
                to_timestamp(rt_arrival_time) AS arrival_ts,
                date_trunc('hour', to_timestamp(rt_arrival_time)) AS timestamp_bucket,
                EXTRACT(hour FROM to_timestamp(rt_arrival_time)) AS hour_of_day,
                EXTRACT(dayofweek FROM to_timestamp(rt_arrival_time)) AS day_of_week,
                CASE 
                    WHEN EXTRACT(hour FROM to_timestamp(rt_arrival_time)) IN (7, 8, 9, 16, 17, 18) THEN 1 
                    ELSE 0 
                END AS is_peak_hour
            FROM read_parquet('{self.raw_dir}/**/*.parquet', hive_partitioning=true)
            WHERE ABS(arrival_delay_seconds) <= 7200  -- Anomaly filter: filter out |Delta t| > 120 min
        """
        if max_records:
            telemetry_query += f" LIMIT {max_records}"

        con.execute(f"CREATE TABLE t_telemetry AS {telemetry_query}")

        # 2. Check weather join availability
        weather_path = weather_parquet_path or os.path.join(self.processed_dir, "imgw_weather_hourly.parquet")
        has_weather = os.path.exists(weather_path)
        if has_weather:
            con.execute(f"CREATE VIEW v_weather AS SELECT * FROM read_parquet('{weather_path}')")
        else:
            con.execute("""
                CREATE VIEW v_weather AS 
                SELECT 
                    NULL::TIMESTAMP AS timestamp_bucket,
                    0.0::DOUBLE AS precipitation_mm,
                    15.0::DOUBLE AS temperature_c,
                    false::BOOLEAN AS freezing_rain_flag,
                    10000.0::DOUBLE AS visibility_m
                WHERE 1=0
            """)

        # 3. Check OSM road topology availability
        osm_path = osm_parquet_path or os.path.join(self.processed_dir, "osm_corridor_segments.parquet")
        has_osm = os.path.exists(osm_path)
        if has_osm:
            con.execute(f"CREATE VIEW v_osm AS SELECT * FROM read_parquet('{osm_path}')")
        else:
            con.execute("""
                CREATE VIEW v_osm AS 
                SELECT 
                    ''::VARCHAR AS stop_id_curr,
                    false::BOOLEAN AS is_dedicated_right_of_way,
                    0::BIGINT AS signalized_intersection_count,
                    300.0::DOUBLE AS segment_length_meters
                WHERE 1=0
            """)

        # 4. Perform fused join
        fused_query = f"""
            COPY (
                SELECT 
                    t.*,
                    COALESCE(w.precipitation_mm, 0.0) AS precipitation_mm,
                    COALESCE(w.temperature_c, 15.0) AS temperature_c,
                    COALESCE(w.freezing_rain_flag, false) AS freezing_rain_flag,
                    COALESCE(osm.is_dedicated_right_of_way, false) AS is_dedicated_right_of_way,
                    COALESCE(osm.signalized_intersection_count, 0) AS signalized_intersection_count,
                    COALESCE(osm.segment_length_meters, 350.0) AS segment_length_meters
                FROM t_telemetry t
                LEFT JOIN v_weather w ON t.timestamp_bucket = w.timestamp_bucket
                LEFT JOIN v_osm osm ON t.stop_id = osm.stop_id_curr
            ) TO '{self.output_parquet}' (FORMAT PARQUET, COMPRESSION SNAPPY)
        """
        con.execute(fused_query)
        con.close()
        logger.info(f"Assembled feature mart written to: {self.output_parquet}")
        return self.output_parquet

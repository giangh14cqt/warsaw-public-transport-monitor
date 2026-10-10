"""Unit tests for spatial-temporal multi-source fusion and feature mart assembler."""

import os
import unittest
import tempfile
import pandas as pd
import duckdb

from src.fusion.feature_mart import FeatureMartAssembler


class TestFeatureMartAssembler(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.assembler = FeatureMartAssembler(data_dir=self.test_dir.name)

        # 1. Create mock raw partition
        self.mock_raw_dir = os.path.join(self.test_dir.name, "raw")
        os.makedirs(self.mock_raw_dir, exist_ok=True)
        self.mock_raw_parquet = os.path.join(self.mock_raw_dir, "batch_test.parquet")

        raw_df = pd.DataFrame([
            {
                "feed_timestamp": 1700000000,
                "trip_id": "T1",
                "route_id": "17",
                "vehicle_id": "V1",
                "stop_sequence": 1,
                "stop_id": "1001",
                "arrival_delay_seconds": 30,
                "departure_delay_seconds": 30,
                "rt_arrival_time": 1700000100,  # ~2023-11-14 22:15
                "rt_departure_time": 1700000100,
            },
            {
                "feed_timestamp": 1700000030,
                "trip_id": "T1",
                "route_id": "17",
                "vehicle_id": "V1",
                "stop_sequence": 2,
                "stop_id": "1002",
                "arrival_delay_seconds": 70,
                "departure_delay_seconds": 70,
                "rt_arrival_time": 1700000250,
                "rt_departure_time": 1700000250,
            },
        ])
        raw_df.to_parquet(self.mock_raw_parquet)

        # 2. Create mock OSM corridor segments
        self.mock_osm_parquet = os.path.join(self.test_dir.name, "osm_segments.parquet")
        osm_df = pd.DataFrame([
            {
                "edge_id": "1001_1002",
                "corridor_name": "pulawska",
                "stop_id_prev": "1001",
                "stop_id_curr": "1002",
                "stop_name_prev": "Stop A",
                "stop_name_curr": "Stop B",
                "segment_length_meters": 650.0,
                "is_dedicated_right_of_way": True,
                "signalized_intersection_count": 4,
                "lane_capacity": 3,
            }
        ])
        osm_df.to_parquet(self.mock_osm_parquet)

        # 3. Create mock IMGW weather
        self.mock_weather_parquet = os.path.join(self.test_dir.name, "weather.parquet")
        expected_ts = duckdb.connect(":memory:").execute(
            "SELECT date_trunc('hour', to_timestamp(1700000250)::TIMESTAMP)"
        ).fetchone()[0]
        weather_df = pd.DataFrame([
            {
                "station_id": "12375",
                "station_name": "WARSZAWA",
                "timestamp": pd.to_datetime(expected_ts),
                "timestamp_bucket": pd.to_datetime(expected_ts),
                "temperature_c": 8.5,
                "relative_humidity": 82.0,
                "precipitation_mm": 1.2,
                "wind_speed_ms": 4.5,
                "pressure_hpa": 1012.0,
                "freezing_rain_flag": False,
            }
        ])
        weather_df.to_parquet(self.mock_weather_parquet)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_assemble_mart_end_to_end(self):
        out_parquet = os.path.join(self.test_dir.name, "fused_mart.parquet")
        res_path = self.assembler.assemble_mart(
            raw_pattern=self.mock_raw_parquet,
            weather_parquet_path=self.mock_weather_parquet,
            osm_parquet_path=self.mock_osm_parquet,
            output_parquet=out_parquet,
        )

        self.assertTrue(os.path.exists(res_path))

        # Inspect fused result via DuckDB
        con = duckdb.connect(":memory:")
        fused_df = con.execute(f"SELECT * FROM read_parquet('{res_path}')").df()
        self.assertEqual(len(fused_df), 1)  # 1 stop-to-stop segment (from stop 1 to stop 2)

        row = fused_df.iloc[0]
        self.assertEqual(row["trip_id"], "T1")
        self.assertEqual(row["prev_stop_id"], "1001")
        self.assertEqual(row["stop_id"], "1002")
        # Delta t_run = 70 - 30 = 40
        self.assertEqual(row["delta_t_run"], 40.0)
        # OSM spatial join
        self.assertEqual(row["corridor_name"], "pulawska")
        self.assertEqual(row["segment_length_meters"], 650.0)
        self.assertEqual(row["signalized_intersection_count"], 4)
        self.assertTrue(row["is_dedicated_right_of_way"])
        self.assertEqual(row["is_study_corridor"], 1)
        # Weather temporal join
        self.assertEqual(row["temperature_c"], 8.5)
        self.assertEqual(row["precipitation_mm"], 1.2)
        # Flags
        self.assertEqual(row["is_tram"], 1)

        # Summary check
        summary = self.assembler.get_summary(parquet_path=res_path)
        self.assertEqual(summary["total_observations"], 1)
        self.assertEqual(summary["study_corridor_observations"], 1)
        self.assertEqual(summary["avg_delta_t_run"], 40.0)


if __name__ == "__main__":
    unittest.main()

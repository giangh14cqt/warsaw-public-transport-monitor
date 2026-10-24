"""Unit tests for spatial-temporal multi-source fusion and feature mart assembler."""

import os
import tempfile
import unittest

import duckdb
import pandas as pd

from src.fusion.feature_mart import FeatureMartAssembler


class TestFeatureMartAssembler(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.assembler = FeatureMartAssembler(data_dir=self.test_dir.name)

        # 1. Create mock raw partition
        self.mock_raw_dir = os.path.join(self.test_dir.name, "raw")
        os.makedirs(self.mock_raw_dir, exist_ok=True)
        self.mock_raw_parquet = os.path.join(self.mock_raw_dir, "batch_test.parquet")

        raw_df = pd.DataFrame(
            [
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
            ]
        )
        raw_df.to_parquet(self.mock_raw_parquet)

        # 2. Create mock OSM corridor segments
        self.mock_osm_parquet = os.path.join(self.test_dir.name, "osm_segments.parquet")
        osm_df = pd.DataFrame(
            [
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
            ]
        )
        osm_df.to_parquet(self.mock_osm_parquet)

        # 3. Create mock IMGW weather
        self.mock_weather_parquet = os.path.join(self.test_dir.name, "weather.parquet")
        expected_ts = (
            duckdb.connect(":memory:")
            .execute("SELECT date_trunc('hour', to_timestamp(1700000250)::TIMESTAMP)")
            .fetchone()[0]
        )
        weather_df = pd.DataFrame(
            [
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
            ]
        )
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
        self.assertEqual(row["transit_cohort"], "urban_tram")

        # Summary check
        summary = self.assembler.get_summary(parquet_path=res_path)
        self.assertEqual(summary["total_observations"], 1)
        self.assertEqual(summary["study_corridor_observations"], 1)
        self.assertEqual(summary["avg_delta_t_run"], 40.0)

    def test_cohort_and_terminal_stop_filtering(self):
        # Multi-stop trip testing terminal layover exclusion
        multi_raw_parquet = os.path.join(self.test_dir.name, "multi_raw.parquet")
        raw_df = pd.DataFrame(
            [
                {
                    "feed_timestamp": 1700000000,
                    "trip_id": "T2",
                    "route_id": "190",
                    "vehicle_id": "V2",
                    "stop_sequence": 1,
                    "stop_id": "2001",
                    "arrival_delay_seconds": 10,
                    "departure_delay_seconds": 10,
                    "rt_arrival_time": 1700000100,
                    "rt_departure_time": 1700000100,
                },
                {
                    "feed_timestamp": 1700000030,
                    "trip_id": "T2",
                    "route_id": "190",
                    "vehicle_id": "V2",
                    "stop_sequence": 2,
                    "stop_id": "2002",
                    "arrival_delay_seconds": 25,
                    "departure_delay_seconds": 25,
                    "rt_arrival_time": 1700000200,
                    "rt_departure_time": 1700000200,
                },
                {
                    "feed_timestamp": 1700000060,
                    "trip_id": "T2",
                    "route_id": "190",
                    "vehicle_id": "V2",
                    "stop_sequence": 3,
                    "stop_id": "2003",
                    "arrival_delay_seconds": 90,
                    "departure_delay_seconds": 90,
                    "rt_arrival_time": 1700000300,
                    "rt_departure_time": 1700000300,
                },
                # Suburban route
                {
                    "feed_timestamp": 1700000000,
                    "trip_id": "T3",
                    "route_id": "709",
                    "vehicle_id": "V3",
                    "stop_sequence": 1,
                    "stop_id": "3001",
                    "arrival_delay_seconds": 5,
                    "departure_delay_seconds": 5,
                    "rt_arrival_time": 1700000100,
                    "rt_departure_time": 1700000100,
                },
                {
                    "feed_timestamp": 1700000030,
                    "trip_id": "T3",
                    "route_id": "709",
                    "vehicle_id": "V3",
                    "stop_sequence": 2,
                    "stop_id": "3002",
                    "arrival_delay_seconds": 15,
                    "departure_delay_seconds": 15,
                    "rt_arrival_time": 1700000200,
                    "rt_departure_time": 1700000200,
                },
                # L-line
                {
                    "feed_timestamp": 1700000000,
                    "trip_id": "T4",
                    "route_id": "L41",
                    "vehicle_id": "V4",
                    "stop_sequence": 1,
                    "stop_id": "4001",
                    "arrival_delay_seconds": 0,
                    "departure_delay_seconds": 0,
                    "rt_arrival_time": 1700000100,
                    "rt_departure_time": 1700000100,
                },
                {
                    "feed_timestamp": 1700000030,
                    "trip_id": "T4",
                    "route_id": "L41",
                    "vehicle_id": "V4",
                    "stop_sequence": 2,
                    "stop_id": "4002",
                    "arrival_delay_seconds": 5,
                    "departure_delay_seconds": 5,
                    "rt_arrival_time": 1700000200,
                    "rt_departure_time": 1700000200,
                },
            ]
        )
        raw_df.to_parquet(multi_raw_parquet)

        out_parquet = os.path.join(self.test_dir.name, "multi_fused.parquet")
        res_path = self.assembler.assemble_mart(
            raw_pattern=multi_raw_parquet,
            weather_parquet_path=self.mock_weather_parquet,
            osm_parquet_path=self.mock_osm_parquet,
            output_parquet=out_parquet,
            exclude_terminals=True,
        )

        con = duckdb.connect(":memory:")
        res_df = con.execute(
            f"SELECT trip_id, stop_sequence, route_id, transit_cohort, is_terminal_stop FROM read_parquet('{res_path}') ORDER BY trip_id, stop_sequence"
        ).df()

        # For T2 (3 stops: 1, 2, 3), stop 1 is dropped (origin, no prev_stop), stop 3 is dropped (terminal layover), stop 2 is kept!
        t2_df = res_df[res_df["trip_id"] == "T2"]
        self.assertEqual(len(t2_df), 1)
        self.assertEqual(t2_df.iloc[0]["stop_sequence"], 2)
        self.assertEqual(t2_df.iloc[0]["transit_cohort"], "urban_bus")

        # For T3 (709), cohort is suburban_bus
        t3_df = res_df[res_df["trip_id"] == "T3"]
        self.assertEqual(t3_df.iloc[0]["transit_cohort"], "suburban_bus")

        # For T4 (L41), cohort is suburban_bus
        t4_df = res_df[res_df["trip_id"] == "T4"]
        self.assertEqual(t4_df.iloc[0]["transit_cohort"], "suburban_bus")


    def test_assemble_historian_observed_mart(self):
        # Create mock Gold stop events
        gold_dir = os.path.join(self.test_dir.name, "historian", "gold", "stop_events", "date=2026-10-18")
        os.makedirs(gold_dir, exist_ok=True)
        gold_parquet = os.path.join(gold_dir, "stop_events.parquet")

        gold_df = pd.DataFrame(
            [
                {
                    "service_date": "2026-10-18",
                    "trip_id": "T_OBS_1",
                    "route_id": "190",
                    "brigade": "1",
                    "vehicle_number": "1001",
                    "vehicle_type": "bus",
                    "stop_sequence": 1,
                    "stop_id": "1001",
                    "stop_name": "Stop 1",
                    "scheduled_arrival_utc": 1792330000,
                    "scheduled_departure_utc": 1792330020,
                    "actual_arrival_utc": 1792330010,
                    "actual_departure_utc": 1792330035,
                    "arrival_delay_seconds": 10,
                    "departure_delay_seconds": 15,
                    "dwell_seconds": 25,
                    "zbk_forecast_delay_s": 12,
                    "forecast_error_s": -2,
                    "event_type": "STOPPED",
                    "confidence": "HIGH",
                    "min_distance_to_stop_m": 5.0,
                },
                {
                    "service_date": "2026-10-18",
                    "trip_id": "T_OBS_1",
                    "route_id": "190",
                    "brigade": "1",
                    "vehicle_number": "1001",
                    "vehicle_type": "bus",
                    "stop_sequence": 2,
                    "stop_id": "1002",
                    "stop_name": "Stop 2",
                    "scheduled_arrival_utc": 1792330150,
                    "scheduled_departure_utc": 1792330170,
                    "actual_arrival_utc": 1792330180,
                    "actual_departure_utc": 1792330200,
                    "arrival_delay_seconds": 30,
                    "departure_delay_seconds": 30,
                    "dwell_seconds": 20,
                    "zbk_forecast_delay_s": 40,
                    "forecast_error_s": -10,
                    "event_type": "STOPPED",
                    "confidence": "HIGH",
                    "min_distance_to_stop_m": 8.0,
                },
                {
                    "service_date": "2026-10-18",
                    "trip_id": "T_OBS_1",
                    "route_id": "190",
                    "brigade": "1",
                    "vehicle_number": "1001",
                    "vehicle_type": "bus",
                    "stop_sequence": 3,
                    "stop_id": "1003",
                    "stop_name": "Stop 3",
                    "scheduled_arrival_utc": 1792330300,
                    "scheduled_departure_utc": 1792330320,
                    "actual_arrival_utc": 1792330340,
                    "actual_departure_utc": 1792330360,
                    "arrival_delay_seconds": 40,
                    "departure_delay_seconds": 40,
                    "dwell_seconds": 20,
                    "zbk_forecast_delay_s": 45,
                    "forecast_error_s": -5,
                    "event_type": "STOPPED",
                    "confidence": "HIGH",
                    "min_distance_to_stop_m": 4.0,
                },
            ]
        )
        gold_df.to_parquet(gold_parquet)

        out_observed_parquet = os.path.join(self.test_dir.name, "observed_fused.parquet")
        res_path = self.assembler.assemble_mart(
            raw_pattern=gold_parquet,
            weather_parquet_path=self.mock_weather_parquet,
            osm_parquet_path=self.mock_osm_parquet,
            output_parquet=out_observed_parquet,
            exclude_terminals=False,
            datasource="historian",
        )

        self.assertTrue(os.path.exists(res_path))
        con = duckdb.connect(":memory:")
        fused = con.execute(f"SELECT * FROM read_parquet('{res_path}')").df()
        con.close()

        # Stop 1 is origin (prev_stop is NULL) so stops 2 and 3 remain
        self.assertEqual(len(fused), 2)
        row = fused.iloc[0]
        self.assertEqual(row["trip_id"], "T_OBS_1")
        self.assertEqual(row["stop_sequence"], 2)
        self.assertEqual(row["prev_stop_id"], "1001")
        self.assertEqual(row["prev_stop_delay"], 10)
        self.assertEqual(row["arrival_delay_seconds"], 30)
        self.assertEqual(row["delta_t_run"], 15.0)
        self.assertEqual(row["dwell_seconds"], 20)
        self.assertEqual(row["zbk_forecast_delay_s"], 40)
        self.assertEqual(row["forecast_error_s"], -10)
        self.assertEqual(row["transit_cohort"], "urban_bus")

        # Verify summary handles dwell_seconds
        summary = self.assembler.get_summary(res_path)
        self.assertEqual(summary["total_observations"], 2)
        self.assertEqual(summary["avg_dwell_seconds"], 20.0)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for trajectory reconstruction and delay decomposition component."""

import os
import unittest
import tempfile
import pandas as pd
import numpy as np

from src.fusion.trajectory import TrajectoryReconstructor


class TestTrajectoryReconstructor(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.reconstructor = TrajectoryReconstructor(data_dir=self.test_dir.name)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_compute_delays_and_lags_basic(self):
        df = pd.DataFrame([
            {
                "trip_id": "T1",
                "route_id": "17",
                "stop_sequence": 1,
                "stop_id": "1001",
                "arrival_delay_seconds": 30,
                "departure_delay_seconds": 35,
                "rt_arrival_time": 1000,
                "rt_departure_time": 1005,
            },
            {
                "trip_id": "T1",
                "route_id": "17",
                "stop_sequence": 2,
                "stop_id": "1002",
                "arrival_delay_seconds": 50,
                "departure_delay_seconds": 50,
                "rt_arrival_time": 1120,
                "rt_departure_time": 1120,
            },
            {
                "trip_id": "T1",
                "route_id": "17",
                "stop_sequence": 3,
                "stop_id": "1003",
                "arrival_delay_seconds": 40,
                "departure_delay_seconds": 45,
                "rt_arrival_time": 1250,
                "rt_departure_time": 1255,
            },
        ])

        processed = self.reconstructor.compute_delays_and_lags(df)

        self.assertEqual(len(processed), 3)

        # Stop 1 (origin)
        self.assertTrue(processed.loc[0, "is_origin_stop"])
        self.assertEqual(processed.loc[0, "trip_progress"], 0.0)
        self.assertTrue(pd.isna(processed.loc[0, "prev_stop_id"]))
        self.assertTrue(pd.isna(processed.loc[0, "delta_t_run"]))
        self.assertEqual(processed.loc[0, "delta_t_dwell"], 5.0)

        # Stop 2 (intermediate)
        self.assertEqual(processed.loc[1, "prev_stop_id"], "1001")
        self.assertEqual(processed.loc[1, "prev_stop_delay"], 30.0)
        # Delta t_run = 50 - 30 = 20 seconds
        self.assertEqual(processed.loc[1, "delta_t_run"], 20.0)
        self.assertEqual(processed.loc[1, "delta_t_dwell"], 0.0)
        # Observed runtime: 1120 - 1005 = 115 seconds
        self.assertEqual(processed.loc[1, "observed_run_time_s"], 115.0)
        self.assertEqual(processed.loc[1, "trip_progress"], 0.5)

        # Stop 3 (terminal)
        self.assertEqual(processed.loc[2, "prev_stop_id"], "1002")
        self.assertEqual(processed.loc[2, "prev_stop_delay"], 50.0)
        self.assertEqual(processed.loc[2, "prev2_stop_delay"], 30.0)
        # Delta t_run = 40 - 50 = -10 seconds (recovered time!)
        self.assertEqual(processed.loc[2, "delta_t_run"], -10.0)
        self.assertEqual(processed.loc[2, "delta_t_dwell"], 5.0)
        self.assertEqual(processed.loc[2, "trip_progress"], 1.0)

    def test_headway_deviation(self):
        # Two vehicles on the same route serving the same stop at different times
        df = pd.DataFrame([
            # Trip 1 (runs first)
            {
                "trip_id": "T1",
                "route_id": "17",
                "stop_sequence": 1,
                "stop_id": "1001",
                "arrival_delay_seconds": 10,
                "rt_arrival_time": 1000,
            },
            # Trip 2 (runs 600 seconds later)
            {
                "trip_id": "T2",
                "route_id": "17",
                "stop_sequence": 1,
                "stop_id": "1001",
                "arrival_delay_seconds": 20,
                "rt_arrival_time": 1600,
            },
            # Trip 3 (runs 900 seconds later - wide headway)
            {
                "trip_id": "T3",
                "route_id": "17",
                "stop_sequence": 1,
                "stop_id": "1001",
                "arrival_delay_seconds": 40,
                "rt_arrival_time": 2500,
            },
        ])

        processed = self.reconstructor.compute_delays_and_lags(df)
        t2_row = processed[processed["trip_id"] == "T2"].iloc[0]
        t3_row = processed[processed["trip_id"] == "T3"].iloc[0]

        self.assertEqual(t2_row["headway_actual_s"], 600.0)
        self.assertEqual(t3_row["headway_actual_s"], 900.0)
        # Median headway between 600 and 900 is 750
        self.assertEqual(t2_row["headway_deviation"], -150.0)
        self.assertEqual(t3_row["headway_deviation"], 150.0)

    def test_deduplication_on_repeated_pings(self):
        # Telemetry updates for the same stop sequence
        df = pd.DataFrame([
            {
                "trip_id": "T1",
                "route_id": "17",
                "stop_sequence": 1,
                "stop_id": "1001",
                "arrival_delay_seconds": 10,
                "feed_timestamp": 100,
            },
            {
                "trip_id": "T1",
                "route_id": "17",
                "stop_sequence": 1,
                "stop_id": "1001",
                "arrival_delay_seconds": 15,
                "feed_timestamp": 130,  # Later ping
            },
            {
                "trip_id": "T1",
                "route_id": "17",
                "stop_sequence": 2,
                "stop_id": "1002",
                "arrival_delay_seconds": 25,
                "feed_timestamp": 200,
            },
        ])
        processed = self.reconstructor.compute_delays_and_lags(df)
        self.assertEqual(len(processed), 2)
        # Should keep latest arrival delay of 15s for stop 1
        self.assertEqual(processed.loc[0, "arrival_delay_seconds"], 15)
        # Stop 2 delta t_run = 25 - 15 = 10
        self.assertEqual(processed.loc[1, "delta_t_run"], 10)

    def test_duckdb_parquet_pipeline(self):
        # Create small test parquet partition
        sample_df = pd.DataFrame([
            {
                "feed_timestamp": 1700000000,
                "trip_id": "T1",
                "route_id": "17",
                "vehicle_id": "V1",
                "stop_sequence": 1,
                "stop_id": "1001",
                "arrival_delay_seconds": 30,
                "departure_delay_seconds": 30,
                "rt_arrival_time": 1700000100,
                "rt_departure_time": 1700000100,
            },
            {
                "feed_timestamp": 1700000030,
                "trip_id": "T1",
                "route_id": "17",
                "vehicle_id": "V1",
                "stop_sequence": 2,
                "stop_id": "1002",
                "arrival_delay_seconds": 55,
                "departure_delay_seconds": 55,
                "rt_arrival_time": 1700000220,
                "rt_departure_time": 1700000220,
            },
        ])
        in_path = os.path.join(self.test_dir.name, "input.parquet")
        out_path = os.path.join(self.test_dir.name, "output.parquet")
        sample_df.to_parquet(in_path)

        res_path = self.reconstructor.reconstruct_trajectories(
            input_parquet_pattern=in_path,
            output_parquet=out_path
        )
        self.assertTrue(os.path.exists(res_path))

        summary = self.reconstructor.get_summary(parquet_path=out_path)
        self.assertEqual(summary["total_records"], 2)
        self.assertEqual(summary["distinct_trips"], 1)
        # Stop 2 has delta_t_run = 55 - 30 = 25, stop 1 has 0
        self.assertEqual(summary["segments_with_prev_stop"], 1)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for transit transfer hub & metro proximity extraction component."""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.exogenous.interchange import TransitInterchangeExtractor


class TestTransitInterchangeExtractor(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.extractor = TransitInterchangeExtractor(data_dir=self.test_dir.name)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_haversine_vectorized(self):
        lats = np.array([52.2308, 52.0000])
        lons = np.array([21.0108, 21.0000])
        targets = np.array([[52.2308, 21.0108]])  # Exactly at first point
        min_d = self.extractor.haversine_vectorized(lats, lons, targets)
        self.assertAlmostEqual(min_d[0], 0.0, places=1)
        self.assertGreater(min_d[1], 20000.0)  # >20 km away

    def test_extract_interchanges_mock(self):
        mock_gtfs_dir = os.path.join(self.test_dir.name, "gtfs")
        os.makedirs(mock_gtfs_dir, exist_ok=True)

        stops_df = pd.DataFrame(
            [
                {
                    "stop_id": "1001",
                    "stop_name": "Centrum 01",
                    "stop_lat": 52.2308,
                    "stop_lon": 21.0108,
                },
                {
                    "stop_id": "2001",
                    "stop_name": "Suburban Stop",
                    "stop_lat": 52.0500,
                    "stop_lon": 20.8500,
                },
            ]
        )
        stops_df.to_csv(os.path.join(mock_gtfs_dir, "stops.txt"), index=False)

        trips_df = pd.DataFrame(
            [
                {"trip_id": "T1", "route_id": "17"},
                {"trip_id": "T2", "route_id": "33"},
                {"trip_id": "T3", "route_id": "17"},
                {"trip_id": "T4", "route_id": "709"},
            ]
        )
        trips_df.to_csv(os.path.join(mock_gtfs_dir, "trips.txt"), index=False)

        stop_times_df = pd.DataFrame(
            [
                {"trip_id": "T1", "stop_id": "1001", "stop_sequence": 1},
                {"trip_id": "T2", "stop_id": "1001", "stop_sequence": 1},
                {"trip_id": "T3", "stop_id": "1001", "stop_sequence": 1},
                {"trip_id": "T4", "stop_id": "2001", "stop_sequence": 1},
            ]
        )
        stop_times_df.to_csv(os.path.join(mock_gtfs_dir, "stop_times.txt"), index=False)

        out_parquet = os.path.join(self.test_dir.name, "interchanges.parquet")
        df_res = self.extractor.extract_interchanges(
            gtfs_dir=mock_gtfs_dir, output_parquet=out_parquet
        )

        self.assertTrue(os.path.exists(out_parquet))
        self.assertEqual(len(df_res), 2)

        row_c = df_res[df_res["stop_id"] == "1001"].iloc[0]
        # Route 17 and Route 33 serve stop 1001 -> 2 distinct routes
        self.assertEqual(row_c["interchange_line_count"], 2)
        # Directly at Centrum Metro -> proximity = 1
        self.assertEqual(row_c["is_metro_proximity"], 1)

        row_s = df_res[df_res["stop_id"] == "2001"].iloc[0]
        self.assertEqual(row_s["interchange_line_count"], 1)
        self.assertEqual(row_s["is_metro_proximity"], 0)


if __name__ == "__main__":
    unittest.main()

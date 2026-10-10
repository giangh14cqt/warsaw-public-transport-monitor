"""Unit tests for OSMnx corridor & road topology extraction component."""

import os
import unittest
import tempfile
import pandas as pd

from src.exogenous.osm import OSMCorridorExtractor


class TestOSMCorridorExtractor(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.extractor = OSMCorridorExtractor(
            data_dir=self.test_dir.name, cache_dir=self.test_dir.name
        )

    def tearDown(self):
        self.test_dir.cleanup()

    def test_haversine_distance(self):
        # Warsaw Centralna (52.2288, 21.0032) to Centrum Metro (52.2312, 21.0102)
        dist = self.extractor.haversine_distance(52.2288, 21.0032, 52.2312, 21.0102)
        # Expected great-circle distance is ~550m
        self.assertGreater(dist, 500.0)
        self.assertLess(dist, 600.0)

    def test_compute_segment_metrics(self):
        metrics = self.extractor.compute_segment_metrics(
            edge_id="300101_300102",
            stop_id_prev="300101",
            stop_id_curr="300102",
            stop_name_prev="Plac Zawiszy",
            stop_name_curr="Rondo Daszyńskiego",
            corridor_name="towarowa_okopowa",
            length_meters=850.45,
            is_dedicated_right_of_way=True,
            signal_count=4,
            lane_capacity=3,
        )
        self.assertEqual(metrics["edge_id"], "300101_300102")
        self.assertEqual(metrics["corridor_name"], "towarowa_okopowa")
        self.assertEqual(metrics["segment_length_meters"], 850.5)
        self.assertTrue(metrics["is_dedicated_right_of_way"])
        self.assertEqual(metrics["signalized_intersection_count"], 4)
        self.assertEqual(metrics["lane_capacity"], 3)

    def test_count_signals_along_segment(self):
        # Line from (52.230, 21.000) to (52.240, 21.000)
        signals = pd.DataFrame([
            {"osmid": "sig_1", "lat": 52.235, "lon": 21.000},  # Directly on segment
            {"osmid": "sig_2", "lat": 52.238, "lon": 21.0001}, # Within ~10m buffer
            {"osmid": "sig_3", "lat": 52.235, "lon": 21.050},  # 5km away
        ])
        count = self.extractor.count_signals_along_segment(
            lat1=52.230,
            lon1=21.000,
            lat2=52.240,
            lon2=21.000,
            signals_df=signals,
            buffer_degrees=0.00035,
        )
        self.assertEqual(count, 2)

    def test_save_segments_and_summary(self):
        sample_segments = [
            self.extractor.compute_segment_metrics(
                edge_id="101_102",
                stop_id_prev="101",
                stop_id_curr="102",
                stop_name_prev="Stop A",
                stop_name_curr="Stop B",
                corridor_name="al_jerozolimskie",
                length_meters=500.0,
                is_dedicated_right_of_way=True,
                signal_count=2,
                lane_capacity=3,
            ),
            self.extractor.compute_segment_metrics(
                edge_id="102_103",
                stop_id_prev="102",
                stop_id_curr="103",
                stop_name_prev="Stop B",
                stop_name_curr="Stop C",
                corridor_name="al_jerozolimskie",
                length_meters=450.0,
                is_dedicated_right_of_way=True,
                signal_count=1,
                lane_capacity=3,
            ),
        ]
        out_path = self.extractor.save_segments(sample_segments)
        self.assertTrue(os.path.exists(out_path))

        summary = self.extractor.get_summary()
        self.assertEqual(summary["total_segments"], 2)
        self.assertEqual(summary["total_corridors"], 1)
        self.assertEqual(summary["avg_segment_meters"], 475.0)
        self.assertEqual(summary["total_signals_detected"], 3)
        self.assertEqual(summary["dedicated_row_segments"], 2)

    def test_mock_gtfs_pipeline(self):
        # Create minimal mock GTFS directory
        mock_gtfs_dir = os.path.join(self.test_dir.name, "gtfs")
        os.makedirs(mock_gtfs_dir, exist_ok=True)

        stops_df = pd.DataFrame([
            {
                "stop_id": "1001",
                "stop_name": "Kijowska 01",
                "stop_lat": 52.248,
                "stop_lon": 21.044,
                "zone_id": 1,
                "street": "Targowa",
            },
            {
                "stop_id": "1002",
                "stop_name": "Kijowska 02",
                "stop_lat": 52.250,
                "stop_lon": 21.045,
                "zone_id": 1,
                "street": "Targowa",
            },
        ])
        stops_df.to_csv(os.path.join(mock_gtfs_dir, "stops.txt"), index=False)

        stop_times_df = pd.DataFrame([
            {"trip_id": "T1", "stop_id": "1001", "stop_sequence": 1},
            {"trip_id": "T1", "stop_id": "1002", "stop_sequence": 2},
        ])
        stop_times_df.to_csv(os.path.join(mock_gtfs_dir, "stop_times.txt"), index=False)

        custom_corridors = {
            "targowa": {
                "name": "Targowa",
                "pattern": "Targow",
                "bbox": (21.040, 52.240, 21.050, 52.260),
                "default_lanes": 3,
                "is_dedicated_row": True,
            }
        }

        corr_stops = self.extractor.extract_corridor_stops(
            mock_gtfs_dir, zone_id=1, corridors=custom_corridors
        )
        self.assertEqual(len(corr_stops), 2)

        pairs = self.extractor.extract_adjacent_stop_pairs(
            mock_gtfs_dir, corridor_stops=corr_stops
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs.iloc[0]["stop_id_prev"], "1001")
        self.assertEqual(pairs.iloc[0]["stop_id_curr"], "1002")


if __name__ == "__main__":
    unittest.main()

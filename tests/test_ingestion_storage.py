"""Unit tests for Warsaw transit telemetry ingestion and storage components."""

import unittest
import os
import tempfile
import pandas as pd

from src.ingestion.storage import ParquetPartitionStorage, DuckDBCatalog
from src.ingestion.fetcher import GTFSRTFetcher
from src.ingestion.gtfs_manager import GTFSManager


class TestIngestionStorage(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.storage = ParquetPartitionStorage(base_dir=self.test_dir.name)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_parquet_partition_structure(self):
        sample_records = [
            {
                "feed_timestamp": 1728561600,
                "trip_id": "test_trip_1",
                "route_id": "175",
                "start_date": "20261010",
                "start_time": "12:00:00",
                "vehicle_id": "1/1001",
                "stop_sequence": 1,
                "stop_id": "01",
                "arrival_delay_seconds": 45,
                "departure_delay_seconds": 45,
                "rt_arrival_time": 1728561645,
                "rt_departure_time": 1728561650,
                "record_ingested_at": "2026-10-10T12:00:00Z",
            }
        ]
        count = self.storage.filter_and_buffer(sample_records)
        self.assertEqual(count, 1)
        written_file = self.storage.flush()
        self.assertIsNotNone(written_file)
        self.assertTrue(os.path.exists(written_file))
        self.assertTrue("year=" in written_file)


class TestProxyCompatibility(unittest.TestCase):
    """Verify that root module proxies maintain full backward compatibility."""

    def test_legacy_imports(self):
        from src.fetcher import GTFSRTFetcher as LegacyFetcher
        from src.gtfs_manager import GTFSManager as LegacyGTFSManager
        from src.storage import ParquetPartitionStorage as LegacyStorage

        self.assertIsNotNone(LegacyFetcher)
        self.assertIsNotNone(LegacyGTFSManager)
        self.assertIsNotNone(LegacyStorage)


if __name__ == "__main__":
    unittest.main()

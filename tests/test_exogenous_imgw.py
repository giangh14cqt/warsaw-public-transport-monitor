"""Unit tests for IMGW weather ingestion component."""

import unittest
import os
import tempfile
import pandas as pd
from datetime import datetime

from src.exogenous.imgw import IMGWWeatherHarvester


class TestIMGWWeatherHarvester(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.harvester = IMGWWeatherHarvester(data_dir=self.test_dir.name)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_clean_live_record(self):
        raw_payload = {
            "id_stacji": "12375",
            "stacja": "Warszawa",
            "data_pomiaru": "2026-10-10",
            "godzina_pomiaru": "12",
            "temperatura": "12.5",
            "predkosc_wiatru": "2",
            "wilgotnosc_wzgledna": "94.2",
            "suma_opadu": "1.5",
            "cisnienie": "1010.4",
        }
        cleaned = self.harvester._clean_live_record(raw_payload)
        self.assertIsNotNone(cleaned)
        self.assertEqual(cleaned["station_id"], "12375")
        self.assertEqual(cleaned["temperature_c"], 12.5)
        self.assertEqual(cleaned["precipitation_mm"], 1.5)
        self.assertEqual(cleaned["freezing_rain_flag"], False)
        self.assertEqual(cleaned["relative_humidity"], 94.2)

    def test_freezing_rain_flag(self):
        freezing_payload = {
            "id_stacji": "12375",
            "stacja": "Warszawa",
            "data_pomiaru": "2026-01-15",
            "godzina_pomiaru": "08",
            "temperatura": "-2.0",
            "predkosc_wiatru": "5",
            "wilgotnosc_wzgledna": "98.0",
            "suma_opadu": "3.0",
            "cisnienie": "1005.0",
        }
        cleaned = self.harvester._clean_live_record(freezing_payload)
        self.assertIsNotNone(cleaned)
        self.assertTrue(cleaned["freezing_rain_flag"])

    def test_parquet_upsert_and_summary(self):
        records = [
            {
                "station_id": "12375",
                "station_name": "WARSZAWA",
                "timestamp": pd.to_datetime("2026-09-01 12:00:00"),
                "timestamp_bucket": pd.to_datetime("2026-09-01 12:00:00"),
                "temperature_c": 18.5,
                "relative_humidity": 65.0,
                "precipitation_mm": 0.0,
                "wind_speed_ms": 3.0,
                "pressure_hpa": 1015.0,
                "freezing_rain_flag": False,
            }
        ]
        count = self.harvester.upsert_to_parquet(records)
        self.assertEqual(count, 1)
        self.assertTrue(os.path.exists(self.harvester.output_parquet))

        summary = self.harvester.get_summary()
        self.assertEqual(summary["total_records"], 1)
        self.assertEqual(summary["avg_temperature"], 18.5)


if __name__ == "__main__":
    unittest.main()

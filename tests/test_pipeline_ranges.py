"""
Unit tests for data range inspection and pipeline temporal filtering.
"""

import argparse
import unittest

from src.fusion.feature_mart import FeatureMartAssembler
from src.pipeline.ranges import (
    build_date_filter_clause,
    format_range_report,
    inspect_available_ranges,
    validate_date_range,
)
from src.pipeline.stages import run_stage_ranges


class TestPipelineRanges(unittest.TestCase):
    """Test suite for data range audit and temporal alignment filtering."""

    def test_validate_date_range_valid(self):
        s, e = validate_date_range("2026-10-04", "2026-10-10")
        self.assertEqual(s, "2026-10-04")
        self.assertEqual(e, "2026-10-10")

        s, e = validate_date_range("2026-10-05 06:00:00", "2026-10-08 22:00:00")
        self.assertEqual(s, "2026-10-05 06:00:00")
        self.assertEqual(e, "2026-10-08 22:00:00")

    def test_validate_date_range_inverted_raises(self):
        with self.assertRaises(ValueError):
            validate_date_range("2026-10-10", "2026-10-04")

    def test_validate_date_range_invalid_format_raises(self):
        with self.assertRaises(ValueError):
            validate_date_range("invalid-date", "2026-10-10")

    def test_build_date_filter_clause_epoch(self):
        clause = build_date_filter_clause(
            "feed_timestamp",
            start_date="2026-10-05",
            end_date="2026-10-08",
            is_epoch_seconds=True,
        )
        self.assertIn("to_timestamp(feed_timestamp)", clause)
        self.assertIn("CAST(", clause)
        self.assertIn(">= '2026-10-05'::DATE", clause)
        self.assertIn("<= '2026-10-08'::DATE", clause)

    def test_build_date_filter_clause_timestamp(self):
        clause = build_date_filter_clause(
            "arrival_ts",
            start_date="2026-10-05",
            end_date="2026-10-08",
            is_epoch_seconds=False,
        )
        self.assertIn("arrival_ts AT TIME ZONE", clause)
        self.assertIn(">= '2026-10-05'::DATE", clause)
        self.assertIn("<= '2026-10-08'::DATE", clause)

    def test_build_date_filter_clause_empty(self):
        clause = build_date_filter_clause("arrival_ts")
        self.assertEqual(clause, "1=1")

    def test_inspect_available_ranges_structure(self):
        report = inspect_available_ranges()
        self.assertIn("sources", report)
        self.assertIn("common_overlap", report)
        self.assertIn("recommendations", report)

        sources = report["sources"]
        self.assertIn("raw_telemetry", sources)
        self.assertIn("weather", sources)
        self.assertIn("static_gtfs", sources)
        self.assertIn("osm_corridors", sources)
        self.assertIn("feature_mart", sources)

        # Raw telemetry should be available in active workspace
        raw = sources["raw_telemetry"]
        self.assertEqual(raw["status"], "AVAILABLE")
        self.assertGreater(raw["total_records"], 0)
        self.assertIn("2026-10-04", raw.get("start_date", ""))

        # Check common overlap
        overlap = report["common_overlap"]
        self.assertIsNotNone(overlap.get("start_date"))
        self.assertIsNotNone(overlap.get("end_date"))
        self.assertTrue(overlap.get("is_ready_for_analysis"))

    def test_format_range_report(self):
        report_str = format_range_report()
        self.assertIn("WARSAW TRANSIT TELEMETRY", report_str)
        self.assertIn("Raw Transit Telemetry", report_str)
        self.assertIn("IMGW Synoptic Weather", report_str)
        self.assertIn("Static GTFS Schedule", report_str)
        self.assertIn("OSMnx Road Corridors", report_str)
        self.assertIn("INTERSECTING ANALYTICAL COHORT WINDOW", report_str)
        self.assertIn("ACTIONABLE PIPELINE RUN COMMANDS", report_str)

    def test_run_stage_ranges(self):
        args = argparse.Namespace(data_dir="data")
        res = run_stage_ranges(args)
        self.assertEqual(res["status"], "success")
        self.assertIn("report", res)

    def test_feature_mart_assembler_with_date_range(self):
        assembler = FeatureMartAssembler()
        sql = assembler.build_fusion_sql(
            raw_pattern="data/raw/**/*.parquet",
            weather_parquet_path="data/processed/imgw_weather_hourly.parquet",
            osm_parquet_path="data/processed/osm_corridor_segments.parquet",
            start_date="2026-10-05",
            end_date="2026-10-07",
            max_records=100,
        )
        self.assertIn("CAST(to_timestamp(feed_timestamp)", sql)
        self.assertIn(">= '2026-10-05'::DATE", sql)
        self.assertIn("<= '2026-10-07'::DATE", sql)


if __name__ == "__main__":
    unittest.main()

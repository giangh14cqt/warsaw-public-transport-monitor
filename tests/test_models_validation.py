"""
Unit tests for Purged Temporal Block Validation Strategy and Metrics Harness.
Validates:
- Chronological temporal block partitioning.
- Boundary trajectory purging (eliminates in-flight trips across cutoffs).
- Embargo gap enforcement between temporal partitions.
- Zero-leakage audit verification and DataLeakageError detection.
- Walk-forward temporal cross-validation folds.
- Standardized regression metrics calculation and formatting.
"""

import os
import unittest
import numpy as np
import pandas as pd

from src.models.validation import PurgedTemporalBlockSplitter, DataLeakageError
from src.models.metrics import (
    compute_regression_metrics,
    format_metrics_table,
    compare_models_table,
)


class TestModelsValidation(unittest.TestCase):
    """Test suite for PurgedTemporalBlockSplitter and regression metrics."""

    def setUp(self):
        """Create synthetic transit telemetry dataset with strictly organized temporal sequence."""
        base_time = pd.Timestamp("2026-10-05 06:00:00", tz="UTC")
        records = []

        # Trip 1: Train (06:00 - 07:00)
        for i in range(5):
            records.append({
                "trip_id": "trip_train_1",
                "vehicle_id": "veh_101",
                "arrival_ts": base_time + pd.Timedelta(minutes=i * 15),
                "arrival_delay_seconds": 10 + i * 2,
            })

        # Trip 2: Train (08:00 - 09:00)
        for i in range(5):
            records.append({
                "trip_id": "trip_train_2",
                "vehicle_id": "veh_102",
                "arrival_ts": base_time + pd.Timedelta(hours=2, minutes=i * 15),
                "arrival_delay_seconds": 15 + i * 3,
            })

        # Trip 3: Boundary-crossing trip spanning across train-val cutoff (11:30 - 13:10)
        for i in range(6):
            records.append({
                "trip_id": "trip_crossing_train_val",
                "vehicle_id": "veh_103",
                "arrival_ts": base_time + pd.Timedelta(hours=5, minutes=30 + i * 20),
                "arrival_delay_seconds": 20,
            })

        # Trip 4: Val (14:30 - 15:30)
        for i in range(5):
            records.append({
                "trip_id": "trip_val_1",
                "vehicle_id": "veh_104",
                "arrival_ts": base_time + pd.Timedelta(hours=8, minutes=30 + i * 15),
                "arrival_delay_seconds": 25,
            })

        # Trip 5: Val (16:30 - 17:30)
        for i in range(5):
            records.append({
                "trip_id": "trip_val_2",
                "vehicle_id": "veh_105",
                "arrival_ts": base_time + pd.Timedelta(hours=10, minutes=30 + i * 15),
                "arrival_delay_seconds": 30,
            })

        # Trip 6: Boundary-crossing trip spanning across val-test cutoff (18:00 - 19:40)
        for i in range(6):
            records.append({
                "trip_id": "trip_crossing_val_test",
                "vehicle_id": "veh_106",
                "arrival_ts": base_time + pd.Timedelta(hours=12, minutes=i * 20),
                "arrival_delay_seconds": 35,
            })

        # Trip 7: Test (20:30 - 21:30)
        for i in range(5):
            records.append({
                "trip_id": "trip_test_1",
                "vehicle_id": "veh_107",
                "arrival_ts": base_time + pd.Timedelta(hours=14, minutes=30 + i * 15),
                "arrival_delay_seconds": 40,
            })

        # Trip 8: Test (22:30 - 23:30)
        for i in range(5):
            records.append({
                "trip_id": "trip_test_2",
                "vehicle_id": "veh_108",
                "arrival_ts": base_time + pd.Timedelta(hours=16, minutes=30 + i * 15),
                "arrival_delay_seconds": 45,
            })

        self.df = pd.DataFrame(records)

    def test_chronological_ordering_and_embargo(self):
        """Verify splits are strictly chronological and enforce embargo gap."""
        splitter = PurgedTemporalBlockSplitter(
            time_col="arrival_ts",
            embargo_minutes=30.0,
            purge_trips=True,
        )
        train_df, val_df, test_df = splitter.split(self.df, train_ratio=0.35, val_ratio=0.35)

        self.assertFalse(train_df.empty)
        self.assertFalse(val_df.empty)
        self.assertFalse(test_df.empty)

        # Monotonicity checks
        train_max = train_df["arrival_ts"].max()
        val_min = val_df["arrival_ts"].min()
        val_max = val_df["arrival_ts"].max()
        test_min = test_df["arrival_ts"].min()

        self.assertLess(train_max, val_min)
        self.assertLess(val_max, test_min)

        # Embargo gap check
        train_val_gap = (val_min - train_max).total_seconds()
        self.assertGreaterEqual(train_val_gap, 30.0 * 60.0 - 1.0)

        val_test_gap = (test_min - val_max).total_seconds()
        self.assertGreaterEqual(val_test_gap, 30.0 * 60.0 - 1.0)

    def test_boundary_trajectory_purging(self):
        """Verify trips that cross the split boundary are purged from evaluation."""
        splitter = PurgedTemporalBlockSplitter(
            time_col="arrival_ts",
            embargo_minutes=15.0,
            purge_trips=True,
            purge_mode="drop_both",
        )
        train_df, val_df, test_df = splitter.split(self.df, train_ratio=0.35, val_ratio=0.35)

        # The boundary-crossing trips must not be present in train, val, or test
        self.assertNotIn("trip_crossing_train_val", val_df["trip_id"].values)
        self.assertNotIn("trip_crossing_train_val", train_df["trip_id"].values)
        self.assertNotIn("trip_crossing_val_test", val_df["trip_id"].values)
        self.assertNotIn("trip_crossing_val_test", test_df["trip_id"].values)

    def test_verify_no_leakage_audit_success(self):
        """Verify leakage audit passes on clean partitioned splits."""
        splitter = PurgedTemporalBlockSplitter(
            time_col="arrival_ts",
            embargo_minutes=30.0,
            purge_trips=True,
        )
        train_df, val_df, test_df = splitter.split(self.df, train_ratio=0.35, val_ratio=0.35)
        audit = splitter.verify_no_leakage(train_df, val_df, test_df, raise_on_error=True)

        self.assertTrue(audit["is_leakage_free"])
        self.assertEqual(len(audit["violations"]), 0)
        self.assertGreater(audit["embargo_gap_train_val_seconds"], 0)
        self.assertGreater(audit["embargo_gap_val_test_seconds"], 0)

    def test_verify_no_leakage_detects_inversion(self):
        """Verify audit raises DataLeakageError on inverted timestamps."""
        splitter = PurgedTemporalBlockSplitter(time_col="arrival_ts", embargo_minutes=10.0)
        train_df, val_df, test_df = splitter.split(self.df, train_ratio=0.35, val_ratio=0.35)

        # Artificially inject temporal inversion (put later row into train)
        leaked_train = pd.concat([train_df, val_df.iloc[-1:]], ignore_index=True)

        with self.assertRaises(DataLeakageError):
            splitter.verify_no_leakage(leaked_train, val_df, test_df, raise_on_error=True)

    def test_verify_no_leakage_detects_boundary_overlap(self):
        """Verify audit raises DataLeakageError if a boundary trip leaks into validation."""
        splitter = PurgedTemporalBlockSplitter(time_col="arrival_ts", embargo_minutes=0.0)
        train_df, val_df, test_df = splitter.split(self.df, train_ratio=0.35, val_ratio=0.35, embargo_minutes=0.0)

        # Inject shared trip in both near boundary
        boundary_row = train_df.iloc[-1:].copy()
        boundary_row["arrival_ts"] = val_df["arrival_ts"].min() + pd.Timedelta(seconds=5)
        leaked_val = pd.concat([val_df, boundary_row], ignore_index=True)

        with self.assertRaises(DataLeakageError):
            splitter.verify_no_leakage(train_df, leaked_val, test_df, raise_on_error=True)

    def test_strict_zero_trip_id_overlap(self):
        """Verify strict mode completely eliminates any trip_id overlap."""
        splitter = PurgedTemporalBlockSplitter(
            time_col="arrival_ts",
            strict_zero_trip_id_overlap=True,
        )
        train_df, val_df, test_df = splitter.split(self.df, train_ratio=0.35, val_ratio=0.35)

        train_trips = set(train_df["trip_id"])
        val_trips = set(val_df["trip_id"])
        test_trips = set(test_df["trip_id"])

        self.assertTrue(train_trips.isdisjoint(val_trips))
        self.assertTrue(val_trips.isdisjoint(test_trips))
        self.assertTrue(train_trips.isdisjoint(test_trips))

    def test_walk_forward_cv(self):
        """Verify expanding walk-forward temporal cross-validation folds."""
        splitter = PurgedTemporalBlockSplitter(time_col="arrival_ts", embargo_minutes=15.0)
        folds = list(splitter.walk_forward_cv(self.df, n_splits=2, train_ratio=0.3, val_ratio=0.3))

        self.assertGreaterEqual(len(folds), 1)
        for fold_idx, (train_fold, val_fold) in enumerate(folds):
            self.assertFalse(train_fold.empty)
            self.assertFalse(val_fold.empty)
            self.assertLess(train_fold["arrival_ts"].max(), val_fold["arrival_ts"].min())

    def test_regression_metrics_calculations(self):
        """Verify standardized regression metrics harness."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 19.0, 31.0, 38.0])

        metrics = compute_regression_metrics(y_true, y_pred)

        # Expected MAE = (|2| + |-1| + |1| + |-2|) / 4 = 6 / 4 = 1.5
        self.assertAlmostEqual(metrics["mae"], 1.5, places=3)
        # Expected RMSE = sqrt((4 + 1 + 1 + 4)/4) = sqrt(2.5) ~ 1.5811
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(2.5), places=3)
        # Expected WAPE = (6 / 100) * 100 = 6.0%
        self.assertAlmostEqual(metrics["wape"], 6.0, places=3)
        self.assertGreater(metrics["r2"], 0.95)
        self.assertGreater(metrics["pearson_r"], 0.98)

        # Formatting table
        formatted = format_metrics_table(metrics, "TestModel")
        self.assertIn("TestModel", formatted)
        self.assertIn("mae", formatted)

        # Compare models table
        comp_df = compare_models_table({"M1": metrics, "M2": metrics})
        self.assertEqual(len(comp_df), 2)
        self.assertIn("mae", comp_df.columns)

    def test_regression_metrics_edge_cases(self):
        """Verify metrics handle edge cases (empty, zeros, NaNs)."""
        # Empty inputs
        m_empty = compute_regression_metrics([], [])
        self.assertEqual(m_empty["mae"], 0.0)

        # Zero denominator / constant target
        m_const = compute_regression_metrics([5.0, 5.0], [5.0, 5.0])
        self.assertEqual(m_const["mae"], 0.0)
        self.assertEqual(m_const["r2"], 1.0)

        # NaNs in predictions
        m_nan = compute_regression_metrics([10.0, 20.0, np.nan], [12.0, 18.0, 15.0])
        self.assertAlmostEqual(m_nan["mae"], 2.0, places=2)

    def test_production_feature_mart_zero_leakage(self):
        """Verify zero leakage protocol on production feature mart parquet if present."""
        mart_path = "data/processed/feature_mart.parquet"
        if not os.path.exists(mart_path):
            self.skipTest("Feature mart parquet not found locally.")

        import duckdb
        con = duckdb.connect()
        sample_df = con.execute(
            f"SELECT trip_id, vehicle_id, arrival_ts, arrival_delay_seconds "
            f"FROM '{mart_path}' USING SAMPLE 50000"
        ).df()

        splitter = PurgedTemporalBlockSplitter(
            time_col="arrival_ts",
            embargo_minutes=30.0,
            purge_trips=True,
        )
        train_df, val_df, test_df = splitter.split(sample_df, train_ratio=0.6, val_ratio=0.2)

        audit = splitter.verify_no_leakage(train_df, val_df, test_df, raise_on_error=True)
        self.assertTrue(audit["is_leakage_free"])
        self.assertGreaterEqual(audit["embargo_gap_train_val_seconds"], 30.0 * 60.0 - 5.0)
        self.assertGreaterEqual(audit["embargo_gap_val_test_seconds"], 30.0 * 60.0 - 5.0)


if __name__ == "__main__":
    unittest.main()

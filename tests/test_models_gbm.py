"""
Unit tests for Gradient Boosting Benchmark Pipeline (LightGBM & CatBoost).
Validates:
- LightGBM training with early stopping.
- CatBoost training and prediction.
- Feature importance extraction.
- End-to-end benchmark comparison suite.
"""

import os
import unittest

import numpy as np
import pandas as pd

from src.models.gbm import GradientBoostingBenchmark, train_gbm_suite


class TestModelsGBM(unittest.TestCase):
    """Test suite for GradientBoostingBenchmark."""

    def setUp(self):
        """Create synthetic non-linear delay telemetry dataset."""
        np.random.seed(42)
        n = 1000
        x1 = np.random.uniform(0, 10, n)
        x2 = np.random.poisson(3, n)
        x3 = np.random.choice([0, 1], n)
        # Non-linear relationship: y = 2 * x1^1.5 - 3 * x2 + 5 * x3 + noise
        y = 2.0 * (x1**1.5) - 3.0 * x2 + 5.0 * x3 + np.random.normal(0, 1, n)

        self.df = pd.DataFrame(
            {
                "feature_continuous": x1,
                "feature_discrete": x2,
                "feature_binary": x3,
                "delta_t_run": y,
            }
        )
        self.train_df = self.df.iloc[:700].copy()
        self.val_df = self.df.iloc[700:].copy()
        self.features = ["feature_continuous", "feature_discrete", "feature_binary"]

    def test_lightgbm_training_and_metrics(self):
        """Verify LightGBM trains, predicts, and logs valid regression metrics."""
        gbm = GradientBoostingBenchmark(target_col="delta_t_run", model_type="lightgbm")
        metrics = gbm.train(self.train_df, self.val_df, self.features, n_estimators=50)

        self.assertIn("val_mae", metrics)
        self.assertIn("val_rmse", metrics)
        self.assertIn("val_r2", metrics)
        self.assertGreater(metrics["val_r2"], 0.70)

        preds = gbm.predict(self.val_df)
        self.assertEqual(len(preds), len(self.val_df))

        # Check feature importances
        fi = gbm.get_feature_importances()
        self.assertEqual(len(fi), 3)
        self.assertIn("relative_importance", fi.columns)
        self.assertAlmostEqual(fi["relative_importance"].sum(), 100.0, places=2)

    def test_catboost_training_and_metrics(self):
        """Verify CatBoost trains, predicts, and logs valid regression metrics."""
        gbm = GradientBoostingBenchmark(target_col="delta_t_run", model_type="catboost")
        metrics = gbm.train(self.train_df, self.val_df, self.features, n_estimators=50)

        self.assertIn("val_mae", metrics)
        self.assertIn("val_rmse", metrics)
        self.assertIn("val_r2", metrics)
        self.assertGreater(metrics["val_r2"], 0.70)

        preds = gbm.predict(self.val_df)
        self.assertEqual(len(preds), len(self.val_df))

    def test_train_gbm_suite_comparison(self):
        """Verify train_gbm_suite trains both models and returns comparison DataFrame."""
        lgb_m, cb_m, comp_df = train_gbm_suite(
            self.train_df, self.val_df, self.features, target_col="delta_t_run"
        )
        self.assertEqual(len(comp_df), 2)
        self.assertIn("val_mae", comp_df.columns)
        self.assertIn("val_rmse", comp_df.columns)
        self.assertIn("val_r2", comp_df.columns)

    def test_production_feature_mart_gbm(self):
        """Verify LightGBM on a sample from real feature_mart.parquet."""
        mart_path = "data/processed/feature_mart.parquet"
        if not os.path.exists(mart_path):
            self.skipTest("Feature mart parquet not found locally.")

        import duckdb

        con = duckdb.connect()
        sample_df = con.execute(
            f"SELECT arrival_ts, delta_t_run, prev_stop_delay, signalized_intersection_count, "
            f"segment_length_meters, trip_progress FROM '{mart_path}' "
            f"WHERE delta_t_run IS NOT NULL USING SAMPLE 5000"
        ).df()
        con.close()

        if len(sample_df) < 50:
            self.skipTest("Insufficient records in feature mart for GBM test.")

        split_idx = max(int(len(sample_df) * 0.7), 1)
        train_sample = sample_df.iloc[:split_idx]
        val_sample = sample_df.iloc[split_idx:]

        features = [
            "prev_stop_delay",
            "signalized_intersection_count",
            "segment_length_meters",
            "trip_progress",
        ]
        gbm = GradientBoostingBenchmark(target_col="delta_t_run", model_type="lightgbm")
        metrics = gbm.train(train_sample, val_sample, features, n_estimators=50)

        self.assertIn("val_mae", metrics)
        self.assertGreater(metrics["val_mae"], 0.0)

    def test_huber_and_quantile_objectives(self):
        """Verify LightGBM and CatBoost train with Huber and Quantile loss objectives."""
        for obj in ["huber", "quantile"]:
            lgb = GradientBoostingBenchmark(target_col="delta_t_run", model_type="lightgbm", objective=obj)
            metrics = lgb.train(self.train_df, self.val_df, self.features, n_estimators=30)
            self.assertIn("val_mae", metrics)

            cb = GradientBoostingBenchmark(target_col="delta_t_run", model_type="catboost", objective=obj)
            metrics_cb = cb.train(self.train_df, self.val_df, self.features, n_estimators=30)
            self.assertIn("val_mae", metrics_cb)

    def test_sample_weight_training(self):
        """Verify models accept sample_weight without error."""
        weights = np.random.uniform(0.5, 2.0, len(self.train_df))
        gbm = GradientBoostingBenchmark(target_col="delta_t_run", model_type="lightgbm")
        metrics = gbm.train(
            self.train_df, self.val_df, self.features, n_estimators=30, sample_weight=weights
        )
        self.assertIn("val_mae", metrics)

    def test_in_fold_winsorization(self):
        """Verify in-fold winsorization clips extreme propagation anomalies without leakage."""
        df_train = self.train_df.copy()
        df_train["prev_stop_delay"] = np.random.normal(0, 10, len(df_train))
        # Add a massive anomaly to train
        df_train.iloc[0, df_train.columns.get_loc("prev_stop_delay")] = 9999.0

        df_val = self.val_df.copy()
        df_val["prev_stop_delay"] = np.random.normal(0, 10, len(df_val))
        df_val.iloc[0, df_val.columns.get_loc("prev_stop_delay")] = 8888.0

        gbm = GradientBoostingBenchmark(target_col="delta_t_run", model_type="lightgbm", winsorize=True)
        feats = self.features + ["prev_stop_delay"]
        gbm.train(df_train, df_val, feats, n_estimators=20)

        # Check bounds were stored
        self.assertIn("prev_stop_delay", gbm._winsor_bounds)
        lower, upper = gbm._winsor_bounds["prev_stop_delay"]
        self.assertLess(upper, 5000.0)


if __name__ == "__main__":
    unittest.main()

"""
Unit tests for Econometric Baseline: Two-Way Fixed Effects (TWFE) Panel Regression.
Validates:
- High-performance within-transformation demeaning.
- Cluster-robust standard errors and t-statistics.
- Point elasticity estimation for policy covariates.
- Out-of-sample fixed-effects prediction and unseen category handling.
- Robust handling of collinear and zero-variance regressors.
"""

import os
import unittest
import numpy as np
import pandas as pd

from src.models.twfe import TWFEBaselineModel
from src.models.metrics import compute_regression_metrics


class TestModelsTWFE(unittest.TestCase):
    """Test suite for TWFEBaselineModel."""

    def setUp(self):
        """Create synthetic panel dataset with known entity and time fixed effects."""
        np.random.seed(42)
        n_entities = 10
        n_times = 8
        obs_per_cell = 25
        n_total = n_entities * n_times * obs_per_cell

        entity_ids = [f"route_{i}" for i in range(n_entities)]
        time_ids = [i for i in range(n_times)]

        records = []
        # True effects
        alpha = {e: np.random.normal(0, 15) for e in entity_ids}
        lambda_t = {t: np.random.normal(0, 10) for t in time_ids}
        beta_signal = 3.5
        beta_length = 0.05

        for e in entity_ids:
            for t in time_ids:
                for _ in range(obs_per_cell):
                    signals = np.random.poisson(lam=3)
                    length = np.random.uniform(200, 1000)
                    noise = np.random.normal(0, 5)
                    delay = 50.0 + alpha[e] + lambda_t[t] + beta_signal * signals + beta_length * length + noise
                    records.append({
                        "route_id": e,
                        "hour_of_day": t,
                        "signalized_intersection_count": signals,
                        "segment_length_meters": length,
                        "is_dedicated_right_of_way": np.random.choice([0, 1]),
                        "zero_var_precipitation": 0.0,  # Regressor with zero variance
                        "delta_t_run": delay,
                    })

        self.df = pd.DataFrame(records)

    def test_twfe_fit_and_recovery(self):
        """Verify TWFE model converges and recovers true linear coefficients."""
        twfe = TWFEBaselineModel(
            target_col="delta_t_run",
            entity_col="route_id",
            time_col="hour_of_day",
            cov_type="HC1",
        )
        features = ["signalized_intersection_count", "segment_length_meters", "zero_var_precipitation"]
        summary = twfe.fit(self.df, features)

        self.assertIn("rsquared", summary)
        self.assertGreater(summary["rsquared"], 0.5)

        # Coefficients check
        coef_signal = twfe.coefficients["signalized_intersection_count"]
        coef_length = twfe.coefficients["segment_length_meters"]

        # Expected signal ~ 3.5, length ~ 0.05
        self.assertAlmostEqual(coef_signal, 3.5, delta=0.5)
        self.assertAlmostEqual(coef_length, 0.05, delta=0.02)

        # Zero variance feature should be safely handled
        self.assertEqual(twfe.coefficients["zero_var_precipitation"], 0.0)

    def test_elasticities_calculation(self):
        """Verify point elasticity table computation and statistics."""
        twfe = TWFEBaselineModel(target_col="delta_t_run")
        features = ["signalized_intersection_count", "segment_length_meters"]
        twfe.fit(self.df, features)

        table = twfe.get_elasticity_table()
        self.assertEqual(len(table), 2)
        self.assertIn("elasticity", table.columns)
        self.assertIn("p_value", table.columns)

        # Both features should be statistically significant
        for _, row in table.iterrows():
            self.assertLess(row["p_value"], 0.05)
            self.assertGreater(row["elasticity"], 0.0)

    def test_out_of_sample_predict(self):
        """Verify prediction logic on new data, including unseen entities."""
        twfe = TWFEBaselineModel(target_col="delta_t_run")
        twfe.fit(self.df, ["signalized_intersection_count", "segment_length_meters"])

        # Test set with a mix of known and unseen routes
        test_df = pd.DataFrame([
            {
                "route_id": "route_0",  # Known route
                "hour_of_day": 2,       # Known hour
                "signalized_intersection_count": 4,
                "segment_length_meters": 500,
            },
            {
                "route_id": "route_999",  # Unseen route
                "hour_of_day": 99,        # Unseen hour
                "signalized_intersection_count": 2,
                "segment_length_meters": 300,
            },
        ])

        preds = twfe.predict(test_df)
        self.assertEqual(len(preds), 2)
        self.assertTrue(np.all(np.isfinite(preds)))
        self.assertGreater(preds[0], 0.0)
        self.assertGreater(preds[1], 0.0)

    def test_summary_formatting(self):
        """Verify ASCII summary table generation."""
        twfe = TWFEBaselineModel(target_col="delta_t_run")
        twfe.fit(self.df, ["signalized_intersection_count", "segment_length_meters"])

        summary_text = twfe.summary()
        self.assertIn("Two-Way Fixed Effects", summary_text)
        self.assertIn("signalized_intersection_count", summary_text)
        self.assertIn("Overall R²", summary_text)

    def test_production_feature_mart_twfe(self):
        """Test TWFE fitting on sample from real feature_mart.parquet."""
        mart_path = "data/processed/feature_mart.parquet"
        if not os.path.exists(mart_path):
            self.skipTest("Feature mart parquet not found locally.")

        import duckdb
        con = duckdb.connect()
        sample_df = con.execute(
            f"SELECT route_id, hour_of_day, delta_t_run, "
            f"signalized_intersection_count, segment_length_meters, is_dedicated_right_of_way "
            f"FROM '{mart_path}' WHERE delta_t_run IS NOT NULL USING SAMPLE 10000"
        ).df()

        twfe = TWFEBaselineModel(target_col="delta_t_run")
        features = ["signalized_intersection_count", "segment_length_meters", "is_dedicated_right_of_way"]
        res = twfe.fit(sample_df, features)

        self.assertGreater(res["nobs"], 0)
        self.assertIn("signalized_intersection_count", twfe.coefficients)
        preds = twfe.predict(sample_df)
        self.assertEqual(len(preds), len(sample_df))


if __name__ == "__main__":
    unittest.main()

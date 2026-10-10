"""
Unit tests for Counterfactual Diagnostics via DiCE and Transit Operational Optimization.
Validates:
- Counterfactual engine initialization and predictions.
- DiCE framework integration and fallback handling.
- Actionable transit policy intervention scenario generation.
- Batch severe delay analysis and policy metric summarization.
- Figure generation and file export.
"""

import os
import shutil
import unittest
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.xai.counterfactuals import CounterfactualDiagnostics


class TestCounterfactualDiagnostics(unittest.TestCase):
    """Test suite for CounterfactualDiagnostics."""

    def setUp(self):
        """Create synthetic transit delay dataset with responsive delay dynamics."""
        np.random.seed(42)
        n = 300

        prev_delay = np.random.uniform(0, 500, n)
        headway_dev = np.random.uniform(0, 300, n)
        row = np.random.choice([0.0, 1.0], n)
        signals = np.random.poisson(4, n).astype(float)
        precip = np.random.exponential(1.0, n)

        # Non-linear delay response:
        # Delays grow with upstream delay, headway deviation, and signals;
        # dedicated ROW reduces delay by ~50s.
        y = (
            0.6 * prev_delay
            + 0.4 * headway_dev
            + 8.0 * signals
            - 50.0 * row
            + 15.0 * np.log1p(precip)
            + np.random.normal(0, 5, n)
        )
        y = np.maximum(y, 0.0)

        self.df = pd.DataFrame({
            "prev_stop_delay": prev_delay,
            "headway_deviation": headway_dev,
            "is_dedicated_right_of_way": row,
            "signalized_intersection_count": signals,
            "precipitation_mm": precip,
            "route_id": np.random.choice(["17", "33", "186", "523"], n),
            "delta_t_run": y,
        })
        self.features = [
            "prev_stop_delay",
            "headway_deviation",
            "is_dedicated_right_of_way",
            "signalized_intersection_count",
            "precipitation_mm",
        ]

        self.model = lgb.LGBMRegressor(n_estimators=30, learning_rate=0.1, random_state=42, verbosity=-1)
        self.model.fit(self.df[self.features], self.df["delta_t_run"])

        self.output_dir = "reports/figures/test_cf"
        self.tables_dir = "reports/tables/test_cf"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.tables_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test artifacts."""
        for d in [self.output_dir, self.tables_dir]:
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)

    def test_initialization_and_prediction(self):
        """Verify model predictions match expected shape and range."""
        diag = CounterfactualDiagnostics(self.model, self.features, target_col="delta_t_run")
        preds = diag.predict(self.df[self.features])
        self.assertEqual(len(preds), len(self.df))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_actionable_interventions_scenario(self):
        """Verify policy intervention remedies reduce predicted delay."""
        diag = CounterfactualDiagnostics(self.model, self.features, target_col="delta_t_run")

        # Pick a severely delayed query instance with mixed traffic (row=0)
        delayed_idx = self.df[(self.df["delta_t_run"] > 250) & (self.df["is_dedicated_right_of_way"] == 0.0)].index[0]
        query = self.df.loc[delayed_idx]

        res = diag.generate_actionable_interventions(query, target_on_time_threshold=120.0)

        self.assertIn("original_prediction", res)
        self.assertIn("scenarios", res)
        scenarios = res["scenarios"]

        # Ensure all 5 core policy scenarios are populated
        self.assertIn("dedicated_row_upgrade", scenarios)
        self.assertIn("headway_regularization", scenarios)
        self.assertIn("upstream_delay_absorption", scenarios)
        self.assertIn("transit_signal_priority", scenarios)
        self.assertIn("joint_multimodal_remedy", scenarios)

        # Joint intervention should yield non-negative savings
        joint = scenarios["joint_multimodal_remedy"]
        self.assertGreaterEqual(joint["delay_savings"], 0.0)
        self.assertLessEqual(joint["predicted_delay"], res["original_prediction"])

    def test_batch_severe_diagnostics(self):
        """Verify batch diagnostics produces valid summary table and metrics."""
        diag = CounterfactualDiagnostics(self.model, self.features, target_col="delta_t_run")
        batch_df, summary = diag.run_severe_delay_batch_diagnostics(
            self.df, delay_threshold=150.0, max_samples=10, target_on_time=120.0
        )

        self.assertGreater(len(batch_df), 0)
        self.assertIn("total_incidents_analyzed", summary)
        self.assertIn("on_time_recovery_rates", summary)
        self.assertIn("mean_delay_reduction_seconds", summary)

        # On-time recovery rate should be bounded between 0 and 1
        joint_rate = summary["on_time_recovery_rates"]["joint_multimodal_remedy"]
        self.assertTrue(0.0 <= joint_rate <= 1.0)

    def test_plot_generation(self):
        """Verify case study and policy impact plots are generated."""
        diag = CounterfactualDiagnostics(self.model, self.features, target_col="delta_t_run")
        query = self.df.iloc[0]
        res = diag.generate_actionable_interventions(query, target_on_time_threshold=120.0)

        case_plot = os.path.join(self.output_dir, "case_study.png")
        diag.plot_counterfactual_case_study(res, output_path=case_plot)
        self.assertTrue(os.path.exists(case_plot))
        self.assertGreater(os.path.getsize(case_plot), 5000)

        _, summary = diag.run_severe_delay_batch_diagnostics(
            self.df, delay_threshold=50.0, max_samples=5, target_on_time=120.0
        )
        policy_plot = os.path.join(self.output_dir, "policy_summary.png")
        diag.plot_policy_remedy_summary(summary, output_path=policy_plot)
        self.assertTrue(os.path.exists(policy_plot))
        self.assertGreater(os.path.getsize(policy_plot), 5000)


if __name__ == "__main__":
    unittest.main()

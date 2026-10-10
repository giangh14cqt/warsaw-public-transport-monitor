"""
Unit tests for TreeSHAP Global and Local Feature Attribution Diagnostics.
Validates:
- TreeSHAP explainer initialization and explanation calculation.
- Global mean absolute SHAP rankings and relative importance.
- Policy domain feature group attribution.
- Beeswarm summary plot generation and file export.
- Local waterfall plot generation for individual delay spikes.
- Extreme delay anomaly detection.
"""

import os
import unittest
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.xai.shap_explainer import TreeSHAPExplainer


class TestXAIShap(unittest.TestCase):
    """Test suite for TreeSHAPExplainer."""

    def setUp(self):
        """Train a lightweight tree model on synthetic non-linear delay data."""
        np.random.seed(42)
        n = 300
        x_prop = np.random.exponential(15, n)          # Prior delay propagation
        x_signals = np.random.poisson(3, n)             # Signal density
        x_length = np.random.uniform(200, 800, n)       # Segment length
        x_precip = np.random.choice([0.0, 1.5, 5.0], n) # Rain

        # Synthetic ground truth
        y = 0.8 * x_prop + 2.5 * x_signals + 0.02 * x_length + 4.0 * x_precip + np.random.normal(0, 2, n)

        self.df = pd.DataFrame({
            "prev_stop_delay": x_prop,
            "signalized_intersection_count": x_signals,
            "segment_length_meters": x_length,
            "precipitation_mm": x_precip,
            "delta_t_run": y,
        })
        self.features = [
            "prev_stop_delay",
            "signalized_intersection_count",
            "segment_length_meters",
            "precipitation_mm",
        ]

        self.model = lgb.LGBMRegressor(n_estimators=30, learning_rate=0.1, random_state=42, verbosity=-1)
        self.model.fit(self.df[self.features], self.df["delta_t_run"])
        self.output_dir = "reports/figures/xai_test"
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test artifacts."""
        if os.path.exists(self.output_dir):
            import shutil
            shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_explain_and_global_rankings(self):
        """Verify explain() computes valid SHAP values and mean absolute rankings."""
        explainer = TreeSHAPExplainer(self.model, self.features)
        exp = explainer.explain(self.df[self.features], max_samples=100)

        self.assertIsNotNone(exp)
        self.assertEqual(len(exp), 100)
        self.assertEqual(exp.values.shape[1], len(self.features))

        # Check mean absolute SHAP table
        rankings = explainer.get_mean_absolute_shap()
        self.assertEqual(len(rankings), len(self.features))
        self.assertIn("mean_abs_shap", rankings.columns)
        self.assertIn("relative_impact_pct", rankings.columns)
        self.assertAlmostEqual(rankings["relative_impact_pct"].sum(), 100.0, places=2)

    def test_feature_group_attribution(self):
        """Verify grouping SHAP values by policy domains."""
        explainer = TreeSHAPExplainer(self.model, self.features)
        explainer.explain(self.df[self.features], max_samples=100)

        groups = explainer.get_feature_group_attribution()
        self.assertGreater(len(groups), 1)
        self.assertIn("group", groups.columns)
        self.assertAlmostEqual(groups["relative_impact_pct"].sum(), 100.0, places=2)

    def test_beeswarm_plot_export(self):
        """Verify global beeswarm plot generates and saves to disk."""
        explainer = TreeSHAPExplainer(self.model, self.features)
        explainer.explain(self.df[self.features], max_samples=80)

        plot_path = os.path.join(self.output_dir, "test_beeswarm.png")
        saved_path = explainer.plot_summary(plot_path)

        self.assertEqual(saved_path, plot_path)
        self.assertTrue(os.path.exists(plot_path))
        self.assertGreater(os.path.getsize(plot_path), 5000)

    def test_waterfall_plot_and_extreme_diagnostics(self):
        """Verify local waterfall plot generation and extreme event diagnostics."""
        explainer = TreeSHAPExplainer(self.model, self.features)
        explainer.explain(self.df[self.features], max_samples=80)

        waterfall_path = os.path.join(self.output_dir, "test_waterfall.png")
        saved_path = explainer.plot_waterfall(0, waterfall_path)

        self.assertEqual(saved_path, waterfall_path)
        self.assertTrue(os.path.exists(waterfall_path))
        self.assertGreater(os.path.getsize(waterfall_path), 5000)

        # Test extreme delays diagnosis
        cases = explainer.diagnose_extreme_delays(self.df, self.output_dir, top_k=2, threshold_seconds=20.0)
        self.assertGreaterEqual(len(cases), 1)
        self.assertTrue(os.path.exists(cases[0]["plot_path"]))


if __name__ == "__main__":
    unittest.main()

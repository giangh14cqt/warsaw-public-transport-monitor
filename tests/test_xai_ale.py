"""
Unit tests for Non-Linear Threshold Detection via Accumulated Local Effects (ALE).
Validates:
- 1D ALE curve computation and centering.
- Non-linear tipping point and inflection detection.
- Chart generation with rug density plot.
- Multi-feature ALE suite execution.
"""

import os
import unittest
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.xai.ale_curves import ALENonLinearDetector


class TestXAIAle(unittest.TestCase):
    """Test suite for ALENonLinearDetector."""

    def setUp(self):
        """Create synthetic non-linear dataset with a sharp tipping point at x=3.0."""
        np.random.seed(42)
        n = 500
        x1 = np.random.uniform(0, 8, n)
        x2 = np.random.poisson(3, n)

        # Clear bifurcation at x1 = 3.0: below 3 slope is 0.5, above 3 slope is 6.0
        y = np.where(x1 <= 3.0, 0.5 * x1, 1.5 + 6.0 * (x1 - 3.0)) + 2.0 * x2 + np.random.normal(0, 0.5, n)

        self.df = pd.DataFrame({
            "precipitation_mm": x1,
            "signalized_intersection_count": x2,
            "delta_t_run": y,
        })
        self.features = ["precipitation_mm", "signalized_intersection_count"]

        self.model = lgb.LGBMRegressor(n_estimators=40, learning_rate=0.1, random_state=42, verbosity=-1)
        self.model.fit(self.df[self.features], self.df["delta_t_run"])

        self.output_dir = "reports/figures/xai_test_ale"
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test artifacts."""
        if os.path.exists(self.output_dir):
            import shutil
            shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_compute_ale_values(self):
        """Verify ALE computation produces monotonic or expected piecewise response."""
        detector = ALENonLinearDetector(self.model, self.features)
        ale_df = detector.compute_ale("precipitation_mm", self.df, n_bins=20)

        self.assertGreater(len(ale_df), 10)
        self.assertIn("grid_val", ale_df.columns)
        self.assertIn("ale_centered", ale_df.columns)

        # Centered ALE should have mean close to zero
        self.assertAlmostEqual(ale_df["ale_centered"].mean(), 0.0, delta=2.0)

    def test_tipping_point_detection(self):
        """Verify tipping point near 3.0 is correctly detected by curvature analysis."""
        detector = ALENonLinearDetector(self.model, self.features)
        ale_df = detector.compute_ale("precipitation_mm", self.df, n_bins=25)
        tipping_points = detector.detect_tipping_points(ale_df, feature_name="precipitation_mm")

        self.assertGreater(len(tipping_points), 0)
        # Check that one detected point is near the synthetic threshold 3.0 (+/- 1.0)
        thresholds = [tp["threshold_value"] for tp in tipping_points]
        near_three = any(abs(t - 3.0) <= 1.2 for t in thresholds)
        self.assertTrue(near_three, f"Expected threshold near 3.0, detected {thresholds}")

    def test_plot_ale_export(self):
        """Verify plot_ale exports valid PNG image."""
        detector = ALENonLinearDetector(self.model, self.features)
        ale_df = detector.compute_ale("precipitation_mm", self.df, n_bins=15)
        tipping = detector.detect_tipping_points(ale_df, feature_name="precipitation_mm")

        chart_path = os.path.join(self.output_dir, "ale_test_precip.png")
        saved = detector.plot_ale("precipitation_mm", ale_df, tipping, self.df, chart_path)

        self.assertEqual(saved, chart_path)
        self.assertTrue(os.path.exists(chart_path))
        self.assertGreater(os.path.getsize(chart_path), 5000)

    def test_run_suite(self):
        """Verify run_suite iterates over multiple features."""
        detector = ALENonLinearDetector(self.model, self.features)
        res = detector.run_suite(self.df, self.features, output_dir=self.output_dir)

        self.assertIn("precipitation_mm", res)
        self.assertIn("signalized_intersection_count", res)
        self.assertTrue(os.path.exists(res["precipitation_mm"]["chart_path"]))


if __name__ == "__main__":
    unittest.main()

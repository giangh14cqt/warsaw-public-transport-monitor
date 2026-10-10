"""
Unit tests for Infrastructure Buffering and Feature Interaction Quantification.
Validates:
- Pairwise SHAP interaction matrix computation.
- Counterfactual infrastructure buffering effect calculation.
- 2D interaction plot generation and file saving.
- Multi-shock interaction suite execution.
"""

import os
import unittest
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.xai.interactions import InteractionQuantifier


class TestXAIInteractions(unittest.TestCase):
    """Test suite for InteractionQuantifier."""

    def setUp(self):
        """Create synthetic dataset where dedicated ROW buffers delay shocks."""
        np.random.seed(42)
        n = 400
        shock = np.random.uniform(0, 10, n)
        row = np.random.choice([0.0, 1.0], n)
        signals = np.random.poisson(3, n)

        # In mixed traffic (row=0), delay escalates by 5s per shock unit;
        # on dedicated ROW (row=1), delay escalates by only 1.5s per shock unit
        y = 5.0 * shock * (1.0 - 0.7 * row) - 3.0 * row + 2.0 * signals + np.random.normal(0, 0.5, n)

        self.df = pd.DataFrame({
            "prev_stop_delay": shock,
            "is_dedicated_right_of_way": row,
            "signalized_intersection_count": signals,
            "delta_t_run": y,
        })
        self.features = ["prev_stop_delay", "is_dedicated_right_of_way", "signalized_intersection_count"]

        self.model = lgb.LGBMRegressor(n_estimators=30, learning_rate=0.1, random_state=42, verbosity=-1)
        self.model.fit(self.df[self.features], self.df["delta_t_run"])

        self.output_dir = "reports/figures/xai_test_interactions"
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test artifacts."""
        if os.path.exists(self.output_dir):
            import shutil
            shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_interaction_matrix(self):
        """Verify SHAP interaction matrix produces valid M x M tensor."""
        quantifier = InteractionQuantifier(self.model, self.features)
        mat, feats = quantifier.compute_interaction_matrix(self.df, self.features, max_samples=50)

        self.assertEqual(mat.shape, (len(self.features), len(self.features)))
        self.assertEqual(len(feats), len(self.features))

    def test_buffering_effect_calculation(self):
        """Verify dedicated ROW buffering effect is positive and mitigates delay escalation."""
        quantifier = InteractionQuantifier(self.model, self.features)
        res = quantifier.compute_buffering_effect(
            self.df, shock_feature="prev_stop_delay", infra_feature="is_dedicated_right_of_way"
        )

        self.assertIn("buffering_effect_seconds_saved", res)
        self.assertIn("buffering_mitigation_pct", res)

        # Buffering should be positive (dedicated ROW saved seconds during delay shock)
        self.assertGreater(res["buffering_effect_seconds_saved"], 0.0)
        self.assertGreater(res["buffering_mitigation_pct"], 20.0)

    def test_plot_interaction(self):
        """Verify plot_interaction generates PNG file."""
        quantifier = InteractionQuantifier(self.model, self.features)
        chart_path = os.path.join(self.output_dir, "test_interaction.png")
        saved = quantifier.plot_interaction("prev_stop_delay", "is_dedicated_right_of_way", self.df, chart_path)

        self.assertEqual(saved, chart_path)
        self.assertTrue(os.path.exists(chart_path))
        self.assertGreater(os.path.getsize(chart_path), 5000)

    def test_run_suite(self):
        """Verify run_suite produces results across multiple shocks."""
        quantifier = InteractionQuantifier(self.model, self.features)
        res = quantifier.run_suite(self.df, shock_features=["prev_stop_delay", "signalized_intersection_count"], output_dir=self.output_dir)

        self.assertIn("prev_stop_delay", res)
        self.assertTrue(os.path.exists(res["prev_stop_delay"]["chart_path"]))


if __name__ == "__main__":
    unittest.main()

"""
End-to-End Pipeline Integration Test Suite.
Verifies complete pipeline automation, CLI arguments, multi-stage orchestration,
and artifact generation from raw data to benchmark tables and xAI diagnostics.
"""

import os
import sys
import shutil
import subprocess
import unittest
import argparse
import pandas as pd

from run_pipeline import (
    run_stage_exogenous,
    run_stage_fusion,
    run_stage_benchmark,
)


class TestPipelineE2E(unittest.TestCase):
    """Integration test suite for master pipeline runner."""

    def setUp(self):
        """Set up test directories."""
        self.test_output_dir = "reports/test_e2e"
        os.makedirs(self.test_output_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test artifacts."""
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir, ignore_errors=True)

    def test_cli_help(self):
        """Verify CLI runner responds to --help with all stage options."""
        res = subprocess.run(
            [sys.executable, "run_pipeline.py", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("preflight", res.stdout)
        self.assertIn("exogenous", res.stdout)
        self.assertIn("fusion", res.stdout)
        self.assertIn("benchmark", res.stdout)
        self.assertIn("xai", res.stdout)

    def test_exogenous_stage_execution(self):
        """Verify exogenous data sources are present or harvestable."""
        args = argparse.Namespace(dry_run=True)
        res = run_stage_exogenous(args)
        self.assertEqual(res["status"], "success")
        self.assertTrue(os.path.exists(res["weather_path"]))
        self.assertTrue(os.path.exists(res["osm_path"]))

    def test_fusion_and_benchmark_stages(self):
        """Verify fusion verification and fast benchmark execution."""
        fusion_args = argparse.Namespace(
            sample_size=1000,
            force_rebuild=False,
            raw_pattern="data/raw/**/*.parquet",
        )
        fusion_res = run_stage_fusion(fusion_args)
        self.assertEqual(fusion_res["status"], "success")
        self.assertTrue(os.path.exists(fusion_res["mart_path"]))

        # Benchmark on small sample
        bench_args = argparse.Namespace(
            sample_size=2000,
            output_dir=self.test_output_dir,
        )
        bench_res = run_stage_benchmark(bench_args)
        self.assertEqual(bench_res["status"], "success")
        self.assertTrue(os.path.exists(bench_res["summary_csv"]))

        df_bench = pd.read_csv(bench_res["summary_csv"])
        self.assertGreaterEqual(len(df_bench), 3)
        self.assertIn("model", df_bench.columns)
        self.assertIn("val_mae", df_bench.columns)

    def test_cli_execution_benchmark_stage(self):
        """Verify full subprocess execution of CLI for benchmark stage."""
        res = subprocess.run(
            [
                sys.executable,
                "run_pipeline.py",
                "--stage",
                "benchmark",
                "--sample-size",
                "3000",
                "--output-dir",
                self.test_output_dir,
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("STAGE 4: ECONOMETRIC & MACHINE LEARNING BENCHMARK MODELS", res.stdout)
        self.assertIn("PIPELINE EXECUTION COMPLETE", res.stdout)


if __name__ == "__main__":
    unittest.main()

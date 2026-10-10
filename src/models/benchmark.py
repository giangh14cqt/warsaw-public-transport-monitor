"""
Comparative Benchmark Runner: TWFE vs. Gradient Boosted Trees.
Executes purged temporal block validation and outputs comparative performance metrics.
"""

import logging
from typing import Dict, Any
import pandas as pd
from src.models.validation import PurgedTemporalBlockSplitter
from src.models.twfe import TWFEBaselineModel
from src.models.gbm import GradientBoostingBenchmark

logger = logging.getLogger(__name__)


def run_benchmark(feature_mart_df: pd.DataFrame) -> pd.DataFrame:
    """Run end-to-end benchmark comparing TWFE and LightGBM."""
    splitter = PurgedTemporalBlockSplitter()
    train_df, val_df, test_df = splitter.split(feature_mart_df)

    feature_cols = [
        "hour_of_day",
        "day_of_week",
        "is_peak_hour",
        "is_dedicated_right_of_way",
        "signalized_intersection_count",
        "segment_length_meters",
        "precipitation_mm",
        "temperature_c",
    ]
    # Filter features present in dataframe
    valid_features = [c for c in feature_cols if c in train_df.columns]

    # 1. TWFE Baseline
    twfe = TWFEBaselineModel()
    twfe_res = twfe.fit(train_df, valid_features)

    # 2. Gradient Boosting
    gbm = GradientBoostingBenchmark()
    gbm_res = gbm.train(train_df, val_df, valid_features)

    benchmark_summary = pd.DataFrame([
        {"model": "TWFE Panel Regression", "val_mae": None, "val_rmse": None, "r2": twfe_res.get("rsquared")},
        {"model": "LightGBM Regressor", "val_mae": gbm_res.get("val_mae"), "val_rmse": gbm_res.get("val_rmse"), "r2": gbm_res.get("val_r2")},
    ])
    return benchmark_summary

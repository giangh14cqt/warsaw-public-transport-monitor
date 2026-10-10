"""
Comparative Benchmark Runner: TWFE vs. Gradient Boosted Trees.
Executes purged temporal block validation and outputs comparative performance metrics.
"""

import logging
from typing import Dict, Any
import pandas as pd
from src.models.validation import PurgedTemporalBlockSplitter
from src.models.metrics import compute_regression_metrics
from src.models.twfe import TWFEBaselineModel
from src.models.gbm import GradientBoostingBenchmark

logger = logging.getLogger(__name__)


def run_benchmark(feature_mart_df: pd.DataFrame) -> pd.DataFrame:
    """Run end-to-end benchmark comparing TWFE and LightGBM with leakage audit."""
    splitter = PurgedTemporalBlockSplitter(embargo_minutes=30.0, purge_trips=True)
    train_df, val_df, test_df = splitter.split(feature_mart_df)
    audit = splitter.verify_no_leakage(train_df, val_df, test_df, raise_on_error=True)
    logger.info(
        f"Validation audit verified: Zero leakage across {audit['train_rows']} train, "
        f"{audit['val_rows']} val, {audit['test_rows']} test records."
    )

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
    twfe_preds = twfe.predict(val_df)
    target_col = "arrival_delay_seconds" if "arrival_delay_seconds" in val_df.columns else "delta_t_run"
    twfe_metrics = compute_regression_metrics(val_df[target_col], twfe_preds, prefix="val_")

    # 2. Gradient Boosting
    gbm = GradientBoostingBenchmark(target_col=target_col)
    gbm_res = gbm.train(train_df, val_df, valid_features)

    benchmark_summary = pd.DataFrame([
        {
            "model": "TWFE Panel Regression",
            "val_mae": twfe_metrics.get("val_mae"),
            "val_rmse": twfe_metrics.get("val_rmse"),
            "val_wape": twfe_metrics.get("val_wape"),
            "r2": twfe_res.get("rsquared") if twfe_res.get("rsquared") is not None else twfe_metrics.get("val_r2"),
        },
        {
            "model": "LightGBM Regressor",
            "val_mae": gbm_res.get("val_mae"),
            "val_rmse": gbm_res.get("val_rmse"),
            "val_wape": gbm_res.get("val_wape"),
            "r2": gbm_res.get("val_r2"),
        },
    ])
    return benchmark_summary

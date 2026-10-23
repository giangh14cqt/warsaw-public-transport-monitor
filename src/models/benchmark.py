"""
Comparative Benchmark Runner: TWFE vs. LightGBM vs. CatBoost.
Executes purged temporal block validation and outputs comparative performance metrics.
"""

import logging
import os
from typing import Dict, List, Optional

import duckdb
import numpy as np
import pandas as pd

from src.core.config import features, paths
from src.core.constants import classify_cohort_series
from src.models.gbm import GradientBoostingBenchmark
from src.models.metrics import compute_regression_metrics
from src.models.twfe import TWFEBaselineModel
from src.models.validation import PurgedTemporalBlockSplitter

logger = logging.getLogger(__name__)


def classify_cohort(df: pd.DataFrame) -> pd.Series:
    """Classify Warsaw transit line into operational cohort."""
    return classify_cohort_series(df)


def compute_regime_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, val_df: pd.DataFrame
) -> Dict[str, Optional[float]]:
    """Evaluates validation MAE across orthogonal regimes: dry vs rain and peak vs offpeak."""
    res: Dict[str, Optional[float]] = {
        "val_mae_dry": None,
        "val_mae_rain": None,
        "val_mae_peak": None,
        "val_mae_offpeak": None,
    }
    if len(y_true) == 0:
        return res

    if "precipitation_mm" in val_df.columns:
        rain_mask = (val_df["precipitation_mm"] > 0).to_numpy()
        if rain_mask.sum() > 0:
            res["val_mae_rain"] = round(float(np.mean(np.abs(y_true[rain_mask] - y_pred[rain_mask]))), 2)
        if (~rain_mask).sum() > 0:
            res["val_mae_dry"] = round(float(np.mean(np.abs(y_true[~rain_mask] - y_pred[~rain_mask]))), 2)

    if "is_peak_hour" in val_df.columns:
        peak_mask = (val_df["is_peak_hour"] == 1).to_numpy()
        if peak_mask.sum() > 0:
            res["val_mae_peak"] = round(float(np.mean(np.abs(y_true[peak_mask] - y_pred[peak_mask]))), 2)
        if (~peak_mask).sum() > 0:
            res["val_mae_offpeak"] = round(float(np.mean(np.abs(y_true[~peak_mask] - y_pred[~peak_mask]))), 2)

    return res


def run_benchmark(
    feature_mart_df: pd.DataFrame,
    sample_size: int = 100000,
    cohort: Optional[str] = None,
    exclude_terminals: bool = True,
    objective: str = "huber",
) -> pd.DataFrame:
    """
    Run end-to-end comparative benchmark across:
    1. Econometric Two-Way Fixed Effects Panel Regression
    2. LightGBM Non-Linear Gradient Boosting Regressor
    3. CatBoost Non-Linear Gradient Boosting Regressor

    Parameters
    ----------
    feature_mart_df : pd.DataFrame
        Input feature mart records.
    sample_size : int
        Maximum observations to evaluate.
    cohort : Optional[str]
        Optional transit cohort to filter on ('urban_tram', 'urban_bus', 'suburban_bus', 'all').
    exclude_terminals : bool
        Whether to filter out terminal turnaround layover stops (default: True).
    objective : str
        Loss objective for gradient boosting: 'huber' (default), 'quantile', or 'regression'.
    """
    df = feature_mart_df.copy()

    # 1. Terminal Layover Filtering (Removes statutory driver layovers at pętle)
    if exclude_terminals:
        if "is_terminal_stop" in df.columns:
            n_before = len(df)
            df = df[df["is_terminal_stop"] == False].copy()
            logger.info(
                f"Filtered {n_before - len(df):,} terminal layover stops via is_terminal_stop flag."
            )
        elif "trip_progress" in df.columns:
            n_before = len(df)
            df = df[df["trip_progress"] < 0.99].copy()
            logger.info(
                f"Filtered {n_before - len(df):,} terminal layover stops via trip_progress < 0.99."
            )

    # 2. Cohort Filtering
    if cohort and cohort != "all":
        cohort_series = classify_cohort(df)
        df["_cohort"] = cohort_series
        df = df[df["_cohort"] == cohort].drop(columns=["_cohort"]).copy()
        logger.info(f"Filtered dataset to cohort '{cohort}': {len(df):,} records remaining.")

    if len(df) > sample_size and sample_size > 0 and "inverse_sampling_weight" not in df.columns:
        logger.info(f"Subsampling {sample_size} rows from {len(df)} rows for benchmark efficiency.")
        df = df.sample(n=sample_size, random_state=42).copy()

    # Determine primary target variable
    target_col = "arrival_delay_seconds" if "arrival_delay_seconds" in df.columns else "delta_t_run"

    splitter = PurgedTemporalBlockSplitter(embargo_minutes=30.0, purge_trips=True)
    train_df, val_df, test_df = splitter.split(df, train_ratio=0.6, val_ratio=0.2)
    audit = splitter.verify_no_leakage(train_df, val_df, test_df, raise_on_error=True)
    logger.info(
        f"Validation audit verified: Zero leakage across {audit['train_rows']} train, "
        f"{audit['val_rows']} val, {audit['test_rows']} test records."
    )

    valid_features = [c for c in features.benchmark_features if c in train_df.columns]

    # Primary hold-out evaluation partition (unseen test partition)
    eval_df = test_df if (test_df is not None and len(test_df) > 0) else val_df
    logger.info(f"Evaluating benchmark models on hold-out partition ({len(eval_df):,} records, target={target_col})...")

    # 1. TWFE Panel Regression Baseline
    logger.info("Training TWFE baseline model...")
    twfe = TWFEBaselineModel(target_col=target_col, entity_col="route_id", time_col="hour_of_day")
    twfe.fit(train_df, valid_features)
    twfe_preds = twfe.predict(eval_df)
    twfe_metrics = compute_regression_metrics(eval_df[target_col], twfe_preds, prefix="val_")
    twfe_regimes = compute_regime_metrics(eval_df[target_col].to_numpy(), twfe_preds, eval_df)

    # 2. LightGBM Non-Linear Regressor
    logger.info(f"Training LightGBM regressor (objective={objective})...")
    lgb_gbm = GradientBoostingBenchmark(target_col=target_col, model_type="lightgbm", objective=objective)
    lgb_metrics = lgb_gbm.train(
        train_df, val_df, valid_features, n_estimators=200, learning_rate=0.05
    )
    lgb_preds = lgb_gbm.predict(eval_df)
    lgb_test_metrics = compute_regression_metrics(eval_df[target_col], lgb_preds, prefix="val_")
    lgb_regimes = compute_regime_metrics(eval_df[target_col].to_numpy(), lgb_preds, eval_df)

    # 3. CatBoost Non-Linear Regressor
    logger.info(f"Training CatBoost regressor (objective={objective})...")
    cb_gbm = GradientBoostingBenchmark(target_col=target_col, model_type="catboost", objective=objective)
    cb_metrics = cb_gbm.train(
        train_df, val_df, valid_features, n_estimators=200, learning_rate=0.05
    )
    cb_preds = cb_gbm.predict(eval_df)
    cb_test_metrics = compute_regression_metrics(eval_df[target_col], cb_preds, prefix="val_")
    cb_regimes = compute_regime_metrics(eval_df[target_col].to_numpy(), cb_preds, eval_df)

    benchmark_summary = pd.DataFrame(
        [
            {
                "model": "TWFE Panel Regression",
                "val_mae": twfe_metrics.get("val_mae"),
                "val_rmse": twfe_metrics.get("val_rmse"),
                "val_medae": twfe_metrics.get("val_medae"),
                "val_wape": twfe_metrics.get("val_wape"),
                "val_r2": twfe_metrics.get("val_r2"),
                "val_mae_dry": twfe_regimes.get("val_mae_dry"),
                "val_mae_rain": twfe_regimes.get("val_mae_rain"),
                "val_mae_peak": twfe_regimes.get("val_mae_peak"),
                "val_mae_offpeak": twfe_regimes.get("val_mae_offpeak"),
            },
            {
                "model": "LightGBM Regressor",
                "val_mae": lgb_test_metrics.get("val_mae"),
                "val_rmse": lgb_test_metrics.get("val_rmse"),
                "val_medae": lgb_test_metrics.get("val_medae"),
                "val_wape": lgb_test_metrics.get("val_wape"),
                "val_r2": lgb_test_metrics.get("val_r2"),
                "val_mae_dry": lgb_regimes.get("val_mae_dry"),
                "val_mae_rain": lgb_regimes.get("val_mae_rain"),
                "val_mae_peak": lgb_regimes.get("val_mae_peak"),
                "val_mae_offpeak": lgb_regimes.get("val_mae_offpeak"),
            },
            {
                "model": "CatBoost Regressor",
                "val_mae": cb_test_metrics.get("val_mae"),
                "val_rmse": cb_test_metrics.get("val_rmse"),
                "val_medae": cb_test_metrics.get("val_medae"),
                "val_wape": cb_test_metrics.get("val_wape"),
                "val_r2": cb_test_metrics.get("val_r2"),
                "val_mae_dry": cb_regimes.get("val_mae_dry"),
                "val_mae_rain": cb_regimes.get("val_mae_rain"),
                "val_mae_peak": cb_regimes.get("val_mae_peak"),
                "val_mae_offpeak": cb_regimes.get("val_mae_offpeak"),
            },
        ]
    )
    return benchmark_summary


def run_stratified_benchmark(
    feature_mart_df: pd.DataFrame,
    sample_size: int = 100000,
    cohorts: Optional[List[str]] = None,
    exclude_terminals: bool = True,
    objective: str = "huber",
) -> pd.DataFrame:
    """
    Run comparative benchmarks across pooled data and stratified transit cohorts:
    - urban_tram (Trams on tracks/ROW)
    - urban_bus (City core routes 100-599)
    - suburban_bus (Regional feeder routes 700-899 & L-lines)
    """
    target_cohorts = cohorts or ["pooled", "urban_tram", "urban_bus", "suburban_bus"]
    all_results = []

    for c in target_cohorts:
        cohort_name = c if c != "pooled" else "all"
        logger.info(f"\n{'=' * 70}\nRunning benchmark for cohort: {c.upper()}\n{'=' * 70}")
        try:
            summary = run_benchmark(
                feature_mart_df,
                sample_size=sample_size,
                cohort=cohort_name,
                exclude_terminals=exclude_terminals,
                objective=objective,
            )
            summary.insert(0, "cohort", c)
            all_results.append(summary)
        except Exception as e:
            logger.error(f"Failed benchmark for cohort {c}: {e}")

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    mart_path = paths.get_feature_mart_str()
    if os.path.exists(mart_path):
        con = duckdb.connect()
        logger.info(f"Loading feature mart from {mart_path}...")
        df_sample = con.execute(
            f"SELECT * FROM '{mart_path}' USING SAMPLE 100000 (reservoir, 42)"
        ).df()
        summary = run_benchmark(df_sample)
        print("\n" + "=" * 78)
        print("MODEL BENCHMARK COMPARISON SUMMARY (HOLD-OUT VALIDATION)")
        print("=" * 78)
        print(summary.to_string(index=False))
        print("=" * 78 + "\n")
    else:
        print(f"Error: Feature mart artifact not found at {mart_path}")

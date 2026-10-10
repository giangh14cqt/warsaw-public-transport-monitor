"""
Comparative Benchmark Runner: TWFE vs. LightGBM vs. CatBoost.
Executes purged temporal block validation and outputs comparative performance metrics.
"""

import os
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import duckdb

from src.models.validation import PurgedTemporalBlockSplitter
from src.models.metrics import compute_regression_metrics, compare_models_table
from src.models.twfe import TWFEBaselineModel
from src.models.gbm import GradientBoostingBenchmark

logger = logging.getLogger(__name__)


def classify_cohort(df: pd.DataFrame) -> pd.Series:
    """Classify Warsaw transit line into operational cohort."""
    if "transit_cohort" in df.columns:
        return df["transit_cohort"]

    def _get_cohort(row) -> str:
        r_str = str(row.get("route_id", "")).strip()
        is_tram = row.get("is_tram", 0)
        if is_tram == 1 or (r_str.isdigit() and 1 <= int(r_str) <= 79):
            return "urban_tram"
        if r_str.isdigit() and 100 <= int(r_str) <= 599:
            return "urban_bus"
        if (r_str.isdigit() and 700 <= int(r_str) <= 899) or r_str.upper().startswith("L"):
            return "suburban_bus"
        if r_str.upper().startswith("N"):
            return "night_bus"
        return "other"

    return df.apply(_get_cohort, axis=1)


def run_benchmark(
    feature_mart_df: pd.DataFrame,
    sample_size: int = 100000,
    cohort: Optional[str] = None,
    exclude_terminals: bool = True,
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
    """
    df = feature_mart_df.copy()

    # 1. Terminal Layover Filtering (Removes statutory driver layovers at pętle)
    if exclude_terminals:
        if "is_terminal_stop" in df.columns:
            n_before = len(df)
            df = df[df["is_terminal_stop"] == False].copy()
            logger.info(f"Filtered {n_before - len(df):,} terminal layover stops via is_terminal_stop flag.")
        elif "trip_progress" in df.columns:
            n_before = len(df)
            df = df[df["trip_progress"] < 0.99].copy()
            logger.info(f"Filtered {n_before - len(df):,} terminal layover stops via trip_progress < 0.99.")

    # 2. Cohort Filtering
    if cohort and cohort != "all":
        cohort_series = classify_cohort(df)
        df["_cohort"] = cohort_series
        df = df[df["_cohort"] == cohort].drop(columns=["_cohort"]).copy()
        logger.info(f"Filtered dataset to cohort '{cohort}': {len(df):,} records remaining.")

    if len(df) > sample_size:
        logger.info(f"Subsampling {sample_size} rows from {len(df)} rows for benchmark efficiency.")
        df = df.sample(n=sample_size, random_state=42).copy()

    # Determine primary target variable
    target_col = "delta_t_run" if "delta_t_run" in df.columns else "arrival_delay_seconds"

    splitter = PurgedTemporalBlockSplitter(embargo_minutes=30.0, purge_trips=True)
    train_df, val_df, test_df = splitter.split(df, train_ratio=0.6, val_ratio=0.2)
    audit = splitter.verify_no_leakage(train_df, val_df, test_df, raise_on_error=True)
    logger.info(
        f"Validation audit verified: Zero leakage across {audit['train_rows']} train, "
        f"{audit['val_rows']} val, {audit['test_rows']} test records."
    )

    feature_cols = [
        "prev_stop_delay",
        "signalized_intersection_count",
        "segment_length_meters",
        "is_dedicated_right_of_way",
        "trip_progress",
        "headway_deviation",
        "hour_of_day",
        "day_of_week",
        "is_peak_hour",
        "is_tram",
        "lane_capacity",
        "precipitation_mm",
        "temperature_c",
    ]
    valid_features = [c for c in feature_cols if c in train_df.columns]

    # 1. TWFE Panel Regression Baseline
    logger.info("Training TWFE baseline model...")
    twfe = TWFEBaselineModel(target_col=target_col, entity_col="route_id", time_col="hour_of_day")
    twfe_res = twfe.fit(train_df, valid_features)
    twfe_preds = twfe.predict(val_df)
    twfe_metrics = compute_regression_metrics(val_df[target_col], twfe_preds, prefix="val_")

    # 2. LightGBM Non-Linear Regressor
    logger.info("Training LightGBM regressor...")
    lgb_gbm = GradientBoostingBenchmark(target_col=target_col, model_type="lightgbm")
    lgb_metrics = lgb_gbm.train(train_df, val_df, valid_features, n_estimators=200, learning_rate=0.05)

    # 3. CatBoost Non-Linear Regressor
    logger.info("Training CatBoost regressor...")
    cb_gbm = GradientBoostingBenchmark(target_col=target_col, model_type="catboost")
    cb_metrics = cb_gbm.train(train_df, val_df, valid_features, n_estimators=200, learning_rate=0.05)

    benchmark_summary = pd.DataFrame([
        {
            "model": "TWFE Panel Regression",
            "val_mae": twfe_metrics.get("val_mae"),
            "val_rmse": twfe_metrics.get("val_rmse"),
            "val_medae": twfe_metrics.get("val_medae"),
            "val_wape": twfe_metrics.get("val_wape"),
            "val_r2": twfe_metrics.get("val_r2"),
        },
        {
            "model": "LightGBM Regressor",
            "val_mae": lgb_metrics.get("val_mae"),
            "val_rmse": lgb_metrics.get("val_rmse"),
            "val_medae": lgb_metrics.get("val_medae"),
            "val_wape": lgb_metrics.get("val_wape"),
            "val_r2": lgb_metrics.get("val_r2"),
        },
        {
            "model": "CatBoost Regressor",
            "val_mae": cb_metrics.get("val_mae"),
            "val_rmse": cb_metrics.get("val_rmse"),
            "val_medae": cb_metrics.get("val_medae"),
            "val_wape": cb_metrics.get("val_wape"),
            "val_r2": cb_metrics.get("val_r2"),
        },
    ])
    return benchmark_summary


def run_stratified_benchmark(
    feature_mart_df: pd.DataFrame,
    sample_size: int = 100000,
    cohorts: Optional[List[str]] = None,
    exclude_terminals: bool = True,
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
        logger.info(f"\n{'='*70}\nRunning benchmark for cohort: {c.upper()}\n{'='*70}")
        try:
            summary = run_benchmark(
                feature_mart_df,
                sample_size=sample_size,
                cohort=cohort_name,
                exclude_terminals=exclude_terminals,
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
    mart_path = "data/processed/feature_mart.parquet"
    if os.path.exists(mart_path):
        con = duckdb.connect()
        logger.info(f"Loading feature mart from {mart_path}...")
        df_sample = con.execute(f"SELECT * FROM '{mart_path}' USING SAMPLE 100000 (reservoir, 42)").df()
        summary = run_benchmark(df_sample)
        print("\n" + "=" * 78)
        print("MODEL BENCHMARK COMPARISON SUMMARY (HOLD-OUT VALIDATION)")
        print("=" * 78)
        print(summary.to_string(index=False))
        print("=" * 78 + "\n")
    else:
        print(f"Error: Feature mart artifact not found at {mart_path}")

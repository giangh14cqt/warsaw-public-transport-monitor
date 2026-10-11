"""
Global & Local Feature Attribution Diagnostics using TreeSHAP.
Decomposes non-linear gradient boosting predictions into exact additive Shapley values:
    f(x) = E[f(X)] + sum_j phi_j(x)

Capabilities:
1. Global Feature Attribution:
   - Mean absolute SHAP values (E[|phi_j|]) ranking feature impact.
   - Grouped attribution across Operational, Infrastructure, Temporal, and Meteorological drivers.
   - Beeswarm summary plots displaying feature magnitude vs. direction of delay push.
2. Local Root-Cause Diagnostics:
   - Identifies extreme delay anomalies (Delta t > 10 minutes).
   - Generates local Waterfall plots isolating individual causal bottlenecks.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import shap

from src.core.config import features, paths
from src.visualization.plots import plot_shap_beeswarm, plot_shap_waterfall

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_GROUPS = features.feature_groups


class TreeSHAPExplainer:
    """Computes exact TreeSHAP values for global and local transit delay attribution."""

    def __init__(self, model_obj: Any, feature_names: Optional[List[str]] = None):
        """
        Initialize TreeSHAPExplainer.

        Parameters
        ----------
        model_obj : Any
            Trained model instance (e.g. LGBMRegressor, CatBoostRegressor, or GradientBoostingBenchmark).
        feature_names : List[str], optional
            Names of input features.
        """
        if hasattr(model_obj, "model") and model_obj.model is not None:
            self.model = model_obj.model
            self.feature_names = feature_names or getattr(model_obj, "feature_cols", None)
        else:
            self.model = model_obj
            self.feature_names = feature_names

        self.explainer: Optional[shap.TreeExplainer] = None
        self.explanation: Optional[shap.Explanation] = None
        self.shap_values: Optional[np.ndarray] = None
        self.base_value: float = 0.0
        self.X_explained: Optional[pd.DataFrame] = None

    def fit_explainer(self) -> shap.TreeExplainer:
        """Initialize shap.TreeExplainer on the underlying tree ensemble."""
        try:
            self.explainer = shap.TreeExplainer(self.model)
            if hasattr(self.explainer, "expected_value"):
                ev = self.explainer.expected_value
                self.base_value = float(ev[0]) if isinstance(ev, (list, np.ndarray)) else float(ev)
            return self.explainer
        except Exception as e:
            logger.error(f"Failed to initialize shap.TreeExplainer: {e}")
            raise

    def explain(
        self,
        X: pd.DataFrame,
        max_samples: int = 2000,
        random_state: int = 42,
    ) -> shap.Explanation:
        """
        Compute TreeSHAP values for input dataset X.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix to explain (typically out-of-sample hold-out partition).
        max_samples : int
            Maximum number of samples to explain (default 2000 for high efficiency).
        random_state : int
            Seed for deterministic sampling.

        Returns
        -------
        shap.Explanation
            Full SHAP explanation object with values, base_values, and data.
        """
        if self.explainer is None:
            self.fit_explainer()

        if len(X) > max_samples:
            logger.info(
                f"Subsampling {max_samples} from {len(X)} records for TreeSHAP explanation."
            )
            X_eval = X.sample(n=max_samples, random_state=random_state).copy()
        else:
            X_eval = X.copy()

        # Clean non-numeric or missing values
        for c in X_eval.columns:
            if X_eval[c].dtype == bool:
                X_eval[c] = X_eval[c].astype(float)
            elif str(X_eval[c].dtype) == "category":
                X_eval[c] = X_eval[c].cat.codes.astype(float)
            elif X_eval[c].dtype == object:
                X_eval[c] = pd.to_numeric(X_eval[c], errors="coerce").fillna(0.0)
            elif X_eval[c].isna().any():
                X_eval[c] = X_eval[c].fillna(0.0)

        self.X_explained = X_eval
        if self.feature_names is None:
            self.feature_names = list(X_eval.columns)

        explanation = self.explainer(X_eval)
        self.explanation = explanation
        self.shap_values = explanation.values
        if hasattr(explanation, "base_values"):
            bv = explanation.base_values
            self.base_value = float(bv[0]) if isinstance(bv, (list, np.ndarray)) else float(bv)

        logger.info(
            f"Computed TreeSHAP values across {len(X_eval)} observations and {X_eval.shape[1]} features. "
            f"Base value E[f(X)]={self.base_value:.2f}s"
        )
        return explanation

    def get_mean_absolute_shap(self) -> pd.DataFrame:
        """Calculate mean absolute SHAP value for each feature."""
        if self.shap_values is None or self.X_explained is None:
            raise RuntimeError("Must call explain() before calculating feature importance.")

        mean_abs = np.mean(np.abs(self.shap_values), axis=0)
        df = (
            pd.DataFrame(
                {
                    "feature": self.X_explained.columns,
                    "mean_abs_shap": mean_abs,
                }
            )
            .sort_values(by="mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

        total = df["mean_abs_shap"].sum()
        df["relative_impact_pct"] = (df["mean_abs_shap"] / total * 100.0) if total > 0 else 0.0
        return df

    def get_feature_group_attribution(
        self,
        group_mapping: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Aggregate SHAP importance by policy domain:
        Operational, Infrastructure, Temporal, Meteorology.
        """
        mapping = group_mapping or DEFAULT_FEATURE_GROUPS
        df = self.get_mean_absolute_shap()
        df["group"] = df["feature"].map(mapping).fillna("Other")

        grouped = df.groupby("group")["mean_abs_shap"].sum().reset_index()
        total = grouped["mean_abs_shap"].sum()
        grouped["relative_impact_pct"] = (
            (grouped["mean_abs_shap"] / total * 100.0) if total > 0 else 0.0
        )
        return grouped.sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    def plot_summary(
        self,
        output_path: Optional[str] = "reports/figures/xai/shap_beeswarm_summary.png",
        max_display: int = 15,
    ) -> str:
        """
        Generate and export global TreeSHAP Beeswarm summary plot.
        """
        if self.explanation is None:
            raise RuntimeError("Must call explain() before plotting summary.")

        if not output_path:
            return ""

        return plot_shap_beeswarm(self.explanation, output_path, max_display=max_display)

    def plot_waterfall(
        self,
        sample_idx: int,
        output_path: Optional[str] = None,
        max_display: int = 10,
    ) -> str:
        """
        Generate and export local TreeSHAP Waterfall plot for observation sample_idx.
        """
        if self.explanation is None:
            raise RuntimeError("Must call explain() before plotting waterfall.")

        if sample_idx < 0 or sample_idx >= len(self.explanation):
            raise IndexError(
                f"Sample index {sample_idx} out of range [0, {len(self.explanation) - 1}]."
            )

        if not output_path:
            return ""

        return plot_shap_waterfall(
            self.explanation[sample_idx],
            output_path,
            sample_idx=sample_idx,
            max_display=max_display,
        )

    def find_extreme_delay_events(
        self,
        df: pd.DataFrame,
        threshold_seconds: float = 600.0,
        target_col: str = "delta_t_run",
    ) -> pd.DataFrame:
        """
        Find extreme delay occurrences (Delta t > threshold_seconds).
        """
        t_col = target_col if target_col in df.columns else "arrival_delay_seconds"
        extreme_df = (
            df[df[t_col] >= threshold_seconds].sort_values(by=t_col, ascending=False).copy()
        )
        logger.info(
            f"Found {len(extreme_df)} extreme delay events with {t_col} >= {threshold_seconds}s"
        )
        return extreme_df

    def diagnose_extreme_delays(
        self,
        df: pd.DataFrame,
        output_dir: str = "reports/figures/xai",
        top_k: int = 3,
        threshold_seconds: float = 300.0,
    ) -> List[Dict[str, Any]]:
        """
        Diagnose top extreme delay events with local waterfall plots.
        """
        os.makedirs(output_dir, exist_ok=True)
        extreme_events = self.find_extreme_delay_events(df, threshold_seconds=threshold_seconds)
        if extreme_events.empty:
            logger.info(
                f"No events with delay >= {threshold_seconds}s; selecting top {top_k} max delays."
            )
            extreme_events = df.sort_values(
                by="delta_t_run" if "delta_t_run" in df.columns else "arrival_delay_seconds",
                ascending=False,
            )

        top_events = extreme_events.head(top_k)
        case_studies = []

        for idx, (original_idx, row) in enumerate(top_events.iterrows()):
            # Find matching index in self.X_explained
            if self.X_explained is not None and original_idx in self.X_explained.index:
                loc_idx = self.X_explained.index.get_loc(original_idx)
            else:
                loc_idx = idx if self.X_explained is not None and idx < len(self.X_explained) else 0

            target_val = float(row.get("delta_t_run", row.get("arrival_delay_seconds", 0.0)))
            route_id = str(row.get("route_id", "unknown"))
            stop_id = str(row.get("stop_id", "unknown"))

            plot_file = os.path.join(output_dir, f"shap_waterfall_extreme_case_{idx + 1}.png")
            self.plot_waterfall(loc_idx, output_path=plot_file)

            case_studies.append(
                {
                    "case_id": idx + 1,
                    "original_row_id": int(original_idx)
                    if isinstance(original_idx, (int, np.integer))
                    else original_idx,
                    "route_id": route_id,
                    "stop_id": stop_id,
                    "observed_delay_seconds": target_val,
                    "plot_path": plot_file,
                }
            )

        return case_studies


def run_shap_pipeline(
    feature_mart_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    sample_size: int = 30000,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute end-to-end TreeSHAP attribution diagnostics workflow."""
    import duckdb

    from src.models.gbm import GradientBoostingBenchmark
    from src.models.validation import PurgedTemporalBlockSplitter
    from src.pipeline.ranges import build_date_filter_clause

    mart_path_str = feature_mart_path or paths.get_feature_mart_str()
    out_dir_str = output_dir or str(paths.figures_xai)
    os.makedirs(out_dir_str, exist_ok=True)
    con = duckdb.connect()
    logger.info("Loading feature mart sample for TreeSHAP analysis...")
    try:
        if start_date or end_date:
            date_where = build_date_filter_clause(
                "arrival_ts", start_date=start_date, end_date=end_date
            )
            sample_query = f"SELECT * FROM '{mart_path_str}' WHERE {date_where} USING SAMPLE {sample_size} (reservoir, 42)"
        else:
            sample_query = (
                f"SELECT * FROM '{mart_path_str}' USING SAMPLE {sample_size} (reservoir, 42)"
            )
        df = con.execute(sample_query).df()
    finally:
        con.close()
    if "is_terminal_stop" in df.columns:
        df = df[df["is_terminal_stop"] == False].copy()
    elif "trip_progress" in df.columns:
        df = df[df["trip_progress"] < 0.99].copy()

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
        "precipitation_mm",
        "temperature_c",
    ]
    valid_features = [c for c in feature_cols if c in df.columns]

    splitter = PurgedTemporalBlockSplitter(embargo_minutes=30.0, purge_trips=True)
    train_df, val_df, test_df = splitter.split(df, train_ratio=0.6, val_ratio=0.2)

    logger.info("Training LightGBM model for SHAP attribution...")
    gbm = GradientBoostingBenchmark(target_col="delta_t_run", model_type="lightgbm")
    gbm.train(train_df, val_df, valid_features, n_estimators=100)

    logger.info("Computing TreeSHAP values...")
    explainer = TreeSHAPExplainer(gbm)
    explainer.explain(test_df[valid_features], max_samples=1000)

    beeswarm_path = explainer.plot_summary(os.path.join(out_dir_str, "shap_beeswarm_summary.png"))
    mean_shap_df = explainer.get_mean_absolute_shap()
    group_df = explainer.get_feature_group_attribution()
    cases = explainer.diagnose_extreme_delays(test_df, out_dir_str, top_k=2)

    return {
        "beeswarm_path": beeswarm_path,
        "mean_shap": mean_shap_df.to_dict(orient="records"),
        "group_attribution": group_df.to_dict(orient="records"),
        "cases": cases,
    }


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    mart_path = paths.get_feature_mart_str()
    if not os.path.exists(mart_path):
        print(f"Error: {mart_path} not found.")
        sys.exit(1)

    import duckdb

    from src.models.gbm import GradientBoostingBenchmark
    from src.models.validation import PurgedTemporalBlockSplitter

    con = duckdb.connect()
    logger.info("Loading feature mart sample for TreeSHAP analysis...")
    df = con.execute(f"SELECT * FROM '{mart_path}' USING SAMPLE 30000").df()

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
        "precipitation_mm",
        "temperature_c",
    ]
    valid_features = [c for c in feature_cols if c in df.columns]

    splitter = PurgedTemporalBlockSplitter(embargo_minutes=30.0)
    train_df, val_df, test_df = splitter.split(df, train_ratio=0.6, val_ratio=0.2)

    logger.info("Training LightGBM model for SHAP attribution...")
    gbm = GradientBoostingBenchmark(target_col="delta_t_run", model_type="lightgbm")
    gbm.train(train_df, val_df, valid_features, n_estimators=100)

    logger.info("Computing TreeSHAP values...")
    explainer = TreeSHAPExplainer(gbm)
    explainer.explain(test_df[valid_features], max_samples=1000)

    # 1. Global summary plot
    beeswarm_path = explainer.plot_summary("reports/figures/xai/shap_beeswarm_summary.png")
    print(f"Global Beeswarm Plot exported to: {beeswarm_path}")

    # 2. Mean absolute SHAP table
    mean_shap_df = explainer.get_mean_absolute_shap()
    print("\n" + "=" * 60)
    print("TOP GLOBAL TREESHAP ATTRIBUTIONS")
    print("=" * 60)
    print(mean_shap_df.to_string(index=False))

    # 3. Grouped domain attribution
    group_df = explainer.get_feature_group_attribution()
    print("\n" + "=" * 60)
    print("POLICY DOMAIN FEATURE ATTRIBUTION")
    print("=" * 60)
    print(group_df.to_string(index=False))

    # 4. Extreme delay local waterfall plots
    cases = explainer.diagnose_extreme_delays(test_df, "reports/figures/xai", top_k=2)
    print("\nExtreme delay case studies diagnosed:")
    for c in cases:
        print(
            f"  Case #{c['case_id']}: Route {c['route_id']}, Delay = {c['observed_delay_seconds']:.1f}s -> {c['plot_path']}"
        )

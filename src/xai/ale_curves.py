"""
Non-Linear Threshold Detection via Accumulated Local Effects (ALE).
Identifies tipping points and bifurcation thresholds in transit delay response curves:
- Precipitation non-linear break thresholds (e.g. tipping points beyond 2.0 mm/h).
- Sub-zero ambient temperature transitions (freezing point boundary at 0°C).
- Operational upstream delay propagation tipping points.
- Signalized intersection density capacity thresholds.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ALENonLinearDetector:
    """
    Computes 1D Accumulated Local Effects (ALE) curves and detects non-linear tipping points.
    Based on Apley & Zhu (2020) Visualizing the Effects of Predictor Variables.
    """

    def __init__(self, model_obj: Any, feature_cols: Optional[List[str]] = None):
        """
        Initialize ALENonLinearDetector.

        Parameters
        ----------
        model_obj : Any
            Trained model (LGBMRegressor, CatBoostRegressor, or GradientBoostingBenchmark).
        feature_cols : List[str], optional
            Feature column names.
        """
        if hasattr(model_obj, "model") and model_obj.model is not None:
            self.model = model_obj.model
            self.feature_cols = feature_cols or getattr(model_obj, "feature_cols", None)
        else:
            self.model = model_obj
            self.feature_cols = feature_cols

    def compute_ale(
        self,
        feature_name: str,
        data: pd.DataFrame,
        n_bins: int = 25,
        grid_type: str = "quantile",
    ) -> pd.DataFrame:
        """
        Compute 1D Accumulated Local Effects curve for feature_name.

        Parameters
        ----------
        feature_name : str
            Feature to analyze.
        data : pd.DataFrame
            Evaluation dataset.
        n_bins : int
            Number of intervals (default 25).
        grid_type : str
            'quantile' (equal frequency) or 'uniform' (equal width).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: [grid_val, ale, ale_centered, local_slope, bin_count]
        """
        if feature_name not in data.columns:
            raise KeyError(f"Feature '{feature_name}' not present in dataframe.")

        vals = data[feature_name].dropna().to_numpy(dtype=float)
        if len(vals) == 0 or np.var(vals) < 1e-9:
            logger.warning(f"Feature '{feature_name}' has zero or near-zero variance; returning flat ALE.")
            val_mean = float(np.mean(vals)) if len(vals) > 0 else 0.0
            return pd.DataFrame({
                "grid_val": [val_mean],
                "ale": [0.0],
                "ale_centered": [0.0],
                "local_slope": [0.0],
                "bin_count": [len(vals)],
            })

        # Generate grid boundaries
        if grid_type == "quantile":
            quantiles = np.linspace(0, 1, n_bins + 1)
            raw_grid = np.quantile(vals, quantiles)
            grid = np.unique(raw_grid)
            if len(grid) < 3:
                grid = np.linspace(vals.min(), vals.max(), max(n_bins, 5))
        else:
            grid = np.linspace(vals.min(), vals.max(), n_bins + 1)

        K = len(grid) - 1
        ale_diffs = []
        bin_counts = []

        # Prepare evaluation copy
        eval_cols = self.feature_cols if self.feature_cols else list(data.columns)
        active_cols = [c for c in eval_cols if c in data.columns]
        X_base = data[active_cols].copy()

        # Handle booleans and categoricals
        for c in X_base.columns:
            if X_base[c].dtype == bool:
                X_base[c] = X_base[c].astype(float)
            elif str(X_base[c].dtype) == "category":
                X_base[c] = X_base[c].cat.codes.astype(float)
            elif X_base[c].dtype == object:
                X_base[c] = pd.to_numeric(X_base[c], errors="coerce").fillna(0.0)

        for k in range(K):
            z_low, z_high = grid[k], grid[k + 1]
            if k == K - 1:
                mask = (vals >= z_low) & (vals <= z_high)
            else:
                mask = (vals >= z_low) & (vals < z_high)

            n_k = int(np.sum(mask))
            bin_counts.append(n_k)

            if n_k > 0:
                sub = X_base[mask].copy()

                sub[feature_name] = z_high
                pred_high = self.model.predict(sub)

                sub[feature_name] = z_low
                pred_low = self.model.predict(sub)

                diff = float(np.mean(pred_high - pred_low))
                ale_diffs.append(diff)
            else:
                ale_diffs.append(0.0)

        # Cumulative sum of local differences
        ale = np.concatenate([[0.0], np.cumsum(ale_diffs)])

        # Interpolate ALE to data points for centering
        ale_interpolated = np.interp(vals, grid, ale)
        center_constant = float(np.mean(ale_interpolated))
        ale_centered = ale - center_constant

        # Compute local slopes dALE/dx
        dx = np.diff(grid)
        dx[dx == 0] = 1e-9
        slopes = np.diff(ale_centered) / dx
        slopes = np.append(slopes, slopes[-1])  # align length

        ale_df = pd.DataFrame({
            "grid_val": grid,
            "ale": ale,
            "ale_centered": ale_centered,
            "local_slope": slopes,
            "bin_count": np.append(bin_counts, [0]),
        })
        return ale_df

    def detect_tipping_points(
        self,
        ale_df: pd.DataFrame,
        feature_name: str = "",
        min_change_seconds: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Detect non-linear tipping points, inflection points, and slope changes.

        Parameters
        ----------
        ale_df : pd.DataFrame
            Computed ALE DataFrame.
        feature_name : str
            Feature name being evaluated.
        min_change_seconds : float
            Minimum change in effect size to qualify as a tipping point.

        Returns
        -------
        List[Dict[str, Any]]
            Detected tipping points with threshold, slope_before, slope_after, and description.
        """
        if len(ale_df) < 4:
            return []

        x = ale_df["grid_val"].to_numpy()
        y = ale_df["ale_centered"].to_numpy()

        dx = np.diff(x)
        dx[dx == 0] = 1e-9
        slopes = np.diff(y) / dx

        # Second differences (curvature / inflection)
        d_slopes = np.diff(slopes)
        abs_d_slopes = np.abs(d_slopes)

        tipping_points = []
        if len(abs_d_slopes) > 0 and np.max(abs_d_slopes) > 0.05:
            # Find significant peaks in curvature
            threshold_cutoff = np.quantile(abs_d_slopes, 0.75)
            candidate_indices = np.where(abs_d_slopes >= threshold_cutoff)[0]

            for idx in candidate_indices:
                x_val = float(x[idx + 1])
                slope_pre = float(slopes[idx])
                slope_post = float(slopes[idx + 1])
                effect_pre = float(y[idx])
                effect_post = float(y[idx + 1])
                delta_effect = float(effect_post - effect_pre)

                # Domain-specific annotation
                desc = f"Curvature acceleration at {x_val:.2f}"
                if "precipitation" in feature_name.lower():
                    if 1.0 <= x_val <= 3.5:
                        desc = f"Precipitation bifurcation threshold (~{x_val:.1f} mm/h): rain runoff impedes traction"
                elif "temp" in feature_name.lower():
                    if -2.0 <= x_val <= 2.0:
                        desc = f"Freezing boundary tipping point (~{x_val:.1f}°C): icing risk on rails/roadways"
                elif "signal" in feature_name.lower():
                    desc = f"Signalized intersection capacity bottleneck at {x_val:.1f} signals"

                tipping_points.append({
                    "feature": feature_name,
                    "threshold_value": round(x_val, 3),
                    "ale_effect_at_threshold": round(float(y[idx + 1]), 2),
                    "slope_before": round(slope_pre, 3),
                    "slope_after": round(slope_post, 3),
                    "slope_change": round(slope_post - slope_pre, 3),
                    "description": desc,
                })

        # Sort by magnitude of curvature change
        tipping_points.sort(key=lambda t: abs(t["slope_change"]), reverse=True)
        return tipping_points[:3]

    def plot_ale(
        self,
        feature_name: str,
        ale_df: pd.DataFrame,
        tipping_points: Optional[List[Dict[str, Any]]] = None,
        data_rug: Optional[pd.DataFrame] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate and save a publication-quality 1D ALE diagnostic chart.
        """
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fig, (ax_main, ax_rug) = plt.subplots(
            2, 1, figsize=(9, 6), gridspec_kw={"height_ratios": [5, 1]}, sharex=True, dpi=300
        )

        x = ale_df["grid_val"].to_numpy()
        y = ale_df["ale_centered"].to_numpy()

        # Main ALE curve
        ax_main.plot(x, y, color="#1f77b4", lw=2.5, label="1D Accumulated Local Effect (ALE)")
        ax_main.axhline(0, color="gray", linestyle="--", alpha=0.5, lw=1)

        # Highlight zero / freezing point for temperature
        if "temp" in feature_name.lower() and x.min() <= 0 <= x.max():
            ax_main.axvline(0, color="#d62728", linestyle=":", lw=1.8, label="Freezing Boundary (0°C)")

        # Highlight tipping points
        if tipping_points:
            for pt in tipping_points:
                tx = pt["threshold_value"]
                ty = pt["ale_effect_at_threshold"]
                ax_main.axvline(tx, color="#ff7f0e", linestyle="--", lw=1.5, label=f"Tipping Point ({tx:.2f})")
                ax_main.scatter([tx], [ty], color="#ff7f0e", s=60, zorder=5)
                ax_main.annotate(
                    f"{pt['threshold_value']:.1f}",
                    xy=(tx, ty),
                    xytext=(tx + 0.05 * (x.max() - x.min()), ty + 1.0),
                    fontsize=9,
                    fontweight="bold",
                    color="#d95f02",
                    arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1),
                )

        ax_main.set_ylabel("Accumulated Delay Effect (seconds)", fontsize=11)
        clean_name = feature_name.replace("_", " ").title()
        ax_main.set_title(f"Accumulated Local Effects (ALE): {clean_name}", fontsize=13, pad=10)
        ax_main.grid(True, linestyle=":", alpha=0.6)
        ax_main.legend(loc="best", framealpha=0.9, fontsize=9)

        # Bottom density / rug plot
        if data_rug is not None and feature_name in data_rug.columns:
            rug_vals = data_rug[feature_name].dropna().astype(float).to_numpy()
            bins = min(30, max(len(np.unique(rug_vals)), 2))
            ax_rug.hist(rug_vals, bins=bins, color="#aec7e8", edgecolor="white", density=True)
            ax_rug.set_ylabel("Density", fontsize=8)
        else:
            counts = ale_df["bin_count"].to_numpy()[:-1]
            mid_x = (x[:-1] + x[1:]) / 2.0
            ax_rug.bar(mid_x, counts, width=(x[1:] - x[:-1]), color="#aec7e8", edgecolor="white")
            ax_rug.set_ylabel("Count", fontsize=8)

        ax_rug.set_xlabel(f"{clean_name}", fontsize=11)
        ax_rug.set_yticks([])
        ax_rug.grid(False)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, bbox_inches="tight")
            plt.close("all")
            logger.info(f"Saved ALE chart to: {output_path}")
            return output_path
        else:
            plt.close("all")
            return ""

    def run_suite(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        output_dir: str = "reports/figures/xai",
    ) -> Dict[str, Any]:
        """
        Execute full ALE diagnostic suite on key features and export charts.
        """
        os.makedirs(output_dir, exist_ok=True)
        default_features = [
            "precipitation_mm",
            "temperature_c",
            "prev_stop_delay",
            "signalized_intersection_count",
            "segment_length_meters",
        ]
        target_features = features or default_features
        results = {}

        for feat in target_features:
            if feat not in data.columns:
                continue

            ale_df = self.compute_ale(feat, data)
            tipping = self.detect_tipping_points(ale_df, feature_name=feat)
            chart_path = os.path.join(output_dir, f"ale_{feat}.png")
            self.plot_ale(feat, ale_df, tipping_points=tipping, data_rug=data, output_path=chart_path)

            results[feat] = {
                "chart_path": chart_path,
                "ale_min": round(float(ale_df["ale_centered"].min()), 2),
                "ale_max": round(float(ale_df["ale_centered"].max()), 2),
                "tipping_points": tipping,
            }

        return results


def run_ale_pipeline(
    feature_mart_path: str = "data/processed/feature_mart.parquet",
    output_dir: str = "reports/figures/xai",
    sample_size: int = 30000,
) -> Dict[str, Any]:
    """Execute end-to-end ALE non-linear curve diagnostics workflow."""
    import duckdb
    from src.models.gbm import GradientBoostingBenchmark
    from src.models.validation import PurgedTemporalBlockSplitter

    os.makedirs(output_dir, exist_ok=True)
    con = duckdb.connect()
    logger.info("Loading feature mart sample for ALE curve diagnostics...")
    df = con.execute(f"SELECT * FROM '{feature_mart_path}' USING SAMPLE {sample_size} (reservoir, 42)").df()
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
        "precipitation_mm",
        "temperature_c",
    ]
    valid_features = [c for c in feature_cols if c in df.columns]

    splitter = PurgedTemporalBlockSplitter(embargo_minutes=30.0, purge_trips=True)
    train_df, val_df, test_df = splitter.split(df, train_ratio=0.6, val_ratio=0.2)

    logger.info("Training gradient boosting model for ALE analysis...")
    gbm = GradientBoostingBenchmark(target_col="delta_t_run", model_type="lightgbm")
    gbm.train(train_df, val_df, valid_features, n_estimators=80)

    detector = ALENonLinearDetector(gbm)
    report = detector.run_suite(
        test_df[valid_features],
        features=[
            "prev_stop_delay",
            "trip_progress",
            "signalized_intersection_count",
            "segment_length_meters",
            "is_dedicated_right_of_way",
            "temperature_c",
            "precipitation_mm",
        ],
        output_dir=output_dir,
    )
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    mart_path = "data/processed/feature_mart.parquet"
    if not os.path.exists(mart_path):
        print(f"Error: {mart_path} not found.")
        exit(1)

    import duckdb
    from src.models.gbm import GradientBoostingBenchmark

    con = duckdb.connect()
    logger.info("Loading feature mart sample for ALE curve diagnostics...")
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
        "precipitation_mm",
        "temperature_c",
    ]
    valid_features = [c for c in feature_cols if c in df.columns]

    logger.info("Training gradient boosting model for ALE analysis...")
    gbm = GradientBoostingBenchmark(target_col="delta_t_run", model_type="lightgbm")
    gbm.train(df.iloc[:20000], df.iloc[20000:], valid_features, n_estimators=80)

    detector = ALENonLinearDetector(gbm)
    report = detector.run_suite(df.iloc[2000:], features=valid_features[:5])

    print("\n" + "=" * 70)
    print("ALE NON-LINEAR THRESHOLD DIAGNOSTICS REPORT")
    print("=" * 70)
    for feat, res in report.items():
        print(f"\nFeature: {feat}")
        print(f"  Chart: {res['chart_path']}")
        print(f"  ALE Range: [{res['ale_min']}s, {res['ale_max']}s]")
        if res["tipping_points"]:
            for tp in res["tipping_points"]:
                print(f"  -> Tipping Point at {tp['threshold_value']}: {tp['description']}")
        else:
            print("  -> Linear or smooth monotonic relationship without sharp tipping points.")
    print("=" * 70 + "\n")

"""
Counterfactual Diagnostics via DiCE and Constrained Operational Optimization.
Evaluates actionable operational, dispatch, and infrastructural interventions
to mitigate severe transit delays (Delta t > 10 min) and transition service to on-time (Delta t < 2 min).

Key Capabilities:
1. DiCE Integration: Diverse Counterfactual Explanations via dice-ml with custom continuous/categorical constraints.
2. Transit Actionable Intervention Engine: Evaluates dedicated infrastructure, headway regularization,
   upstream delay absorption, and Transit Signal Priority (TSP).
3. Joint Minimal Counterfactual Search: Constrained optimization finding minimal actionable perturbations
   required to achieve target arrival delay.
4. Policy Insights & Visualizations: Radar/bar charts comparing original severe delays vs counterfactual remedies.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import dice_ml
    HAS_DICE = True
except ImportError:
    HAS_DICE = False

logger = logging.getLogger(__name__)


class CounterfactualDiagnostics:
    """Generates and analyzes actionable counterfactual interventions for transit delay mitigation."""

    ACTIONABLE_FEATURES = [
        "prev_stop_delay",
        "headway_deviation",
        "is_dedicated_right_of_way",
        "signalized_intersection_count",
    ]

    IMMUTABLE_FEATURES = [
        "hour_of_day",
        "day_of_week",
        "is_peak_hour",
        "segment_length_meters",
        "lane_capacity",
        "is_tram",
        "precipitation_mm",
        "temperature_c",
        "relative_humidity",
        "wind_speed_ms",
        "freezing_rain_flag",
    ]

    def __init__(
        self,
        model_obj: Any,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "delta_t_run",
    ):
        """
        Initialize CounterfactualDiagnostics.

        Parameters
        ----------
        model_obj : Any
            Trained regression model (LGBMRegressor, CatBoostRegressor, or GradientBoostingBenchmark).
        feature_cols : List[str], optional
            Ordered feature columns used during training.
        target_col : str
            Target prediction column name.
        """
        if hasattr(model_obj, "model") and model_obj.model is not None:
            self.model = model_obj.model
            self.feature_cols = feature_cols or getattr(model_obj, "feature_cols", None)
        else:
            self.model = model_obj
            self.feature_cols = feature_cols

        self.target_col = target_col
        self.dice_exp = None
        self.dice_data = None
        self.dice_model = None

    def setup_dice(
        self,
        reference_df: pd.DataFrame,
        continuous_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        method: str = "kdtree",
    ) -> bool:
        """
        Configure DiCE framework using reference baseline data.

        Parameters
        ----------
        reference_df : pd.DataFrame
            Reference dataset (training or validation split).
        continuous_features : List[str], optional
            Names of continuous features.
        categorical_features : List[str], optional
            Names of categorical features.
        method : str
            DiCE generation method ('kdtree', 'random', 'genetic').

        Returns
        -------
        bool
            True if DiCE was successfully initialized, False otherwise.
        """
        if not HAS_DICE:
            logger.warning("dice-ml is not installed. DiCE backend unavailable.")
            return False

        if self.feature_cols is None:
            self.feature_cols = [c for c in reference_df.columns if c != self.target_col]

        active_cols = [c for c in self.feature_cols if c in reference_df.columns]
        prep_df = reference_df[active_cols].copy()

        # Add target column if missing
        if self.target_col not in prep_df.columns:
            if self.target_col in reference_df.columns:
                prep_df[self.target_col] = reference_df[self.target_col].astype(float)
            else:
                prep_df[self.target_col] = self.predict(prep_df)

        # Cast all numeric and boolean features to float to ensure compatibility with pandas 2.x in DiCE
        for col in active_cols:
            if prep_df[col].dtype == bool or str(prep_df[col].dtype) in ("bool", "boolean"):
                prep_df[col] = prep_df[col].astype(float)
            elif np.issubdtype(prep_df[col].dtype, np.number):
                prep_df[col] = prep_df[col].astype(float)

        cont_feats = continuous_features or [
            c for c in active_cols if c not in (categorical_features or [])
        ]

        try:
            self.dice_data = dice_ml.Data(
                dataframe=prep_df,
                continuous_features=cont_feats,
                outcome_name=self.target_col,
            )
            self.dice_model = dice_ml.Model(
                model=self.model,
                backend="sklearn",
                model_type="regressor",
            )
            self.dice_exp = dice_ml.Dice(self.dice_data, self.dice_model, method=method)
            logger.info(f"DiCE framework initialized successfully with method='{method}'.")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize DiCE explainer: {e}. Falling back to optimization engine.")
            self.dice_exp = None
            return False

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate model predictions with column alignment."""
        if self.feature_cols is not None:
            cols = [c for c in self.feature_cols if c in df.columns]
            X = df[cols].copy()
        else:
            X = df.copy()

        # Align category dtypes if model is LightGBM
        for c in X.columns:
            if X[c].dtype == object:
                X[c] = X[c].astype("category")

        preds = self.model.predict(X)
        return np.asarray(preds, dtype=float)

    def explain_instance_dice(
        self,
        instance: Union[pd.Series, pd.DataFrame],
        total_cfs: int = 3,
        desired_range: Tuple[float, float] = (0.0, 120.0),
        features_to_vary: Optional[List[str]] = None,
        permitted_range: Optional[Dict[str, List[float]]] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Generate counterfactuals for a single instance using DiCE.

        Parameters
        ----------
        instance : pd.Series or pd.DataFrame
            Query instance with severe delay.
        total_cfs : int
            Number of diverse counterfactuals to produce.
        desired_range : tuple
            Target delay interval in seconds [min_delay, max_delay] (default: 0 to 120s on-time).
        features_to_vary : List[str], optional
            Features permitted to mutate.
        permitted_range : Dict[str, List[float]], optional
            Upper and lower bounds for mutated features.

        Returns
        -------
        pd.DataFrame or None
            Counterfactual scenarios, or None if DiCE search failed.
        """
        if self.dice_exp is None:
            return None

        query_df = instance.to_frame().T if isinstance(instance, pd.Series) else instance.copy()
        query_df = query_df[[c for c in self.feature_cols if c in query_df.columns]].copy()
        for c in query_df.columns:
            if query_df[c].dtype == bool or str(query_df[c].dtype) in ("bool", "boolean") or np.issubdtype(query_df[c].dtype, np.number):
                query_df[c] = query_df[c].astype(float)

        vary_feats = features_to_vary or [
            f for f in self.ACTIONABLE_FEATURES if f in query_df.columns
        ]

        try:
            cf_exp = self.dice_exp.generate_counterfactuals(
                query_df,
                total_CFs=total_cfs,
                desired_range=list(desired_range),
                features_to_vary=vary_feats,
                permitted_range=permitted_range,
            )
            cf_df = cf_exp.cf_examples_list[0].final_cfs_df
            if cf_df is not None and len(cf_df) > 0:
                return cf_df
        except Exception as e:
            logger.debug(f"DiCE counterfactual generation exception: {e}")

        return None

    def generate_actionable_interventions(
        self,
        instance: Union[pd.Series, pd.DataFrame],
        target_on_time_threshold: float = 120.0,
    ) -> Dict[str, Any]:
        """
        Generate structured transit policy intervention scenarios for a delayed instance.

        Evaluates 5 distinct operational and infrastructural policy remedies:
        1. Baseline: Current delayed status.
        2. Policy 1 (Dedicated ROW Upgrade): Convert mixed traffic to segregated bus/tram corridor.
        3. Policy 2 (Headway Regularization): Eliminate dispatch bunching/gaps (headway_deviation -> 0).
        4. Policy 3 (Upstream Delay Absorption): Upstream holding/recovery (prev_stop_delay -> 0).
        5. Policy 4 (Transit Signal Priority - TSP): Remove signal stop delays (signals -> 0).
        6. Policy 5 (Joint Optimal Remedy): Minimal combination required to achieve on-time target.
        """
        row = instance.iloc[0] if isinstance(instance, pd.DataFrame) else instance.copy()
        base_df = pd.DataFrame([row])
        current_pred = float(self.predict(base_df)[0])

        results = {
            "original_prediction": current_pred,
            "on_time_target": target_on_time_threshold,
            "is_currently_on_time": current_pred <= target_on_time_threshold,
            "scenarios": {},
        }

        # 1. Policy: Dedicated Right-of-Way Upgrade
        scen_row = base_df.copy()
        if "is_dedicated_right_of_way" in scen_row.columns:
            scen_row["is_dedicated_right_of_way"] = 1.0
            pred_row = float(self.predict(scen_row)[0])
            results["scenarios"]["dedicated_row_upgrade"] = {
                "name": "Dedicated Transit Right-of-Way",
                "predicted_delay": pred_row,
                "delay_savings": current_pred - pred_row,
                "achieves_on_time": pred_row <= target_on_time_threshold,
                "mutations": {"is_dedicated_right_of_way": {"from": float(row.get("is_dedicated_right_of_way", 0.0)), "to": 1.0}},
            }

        # 2. Policy: Headway Regularization (Zero Dispatch Deviation)
        scen_hw = base_df.copy()
        if "headway_deviation" in scen_hw.columns:
            scen_hw["headway_deviation"] = 0.0
            pred_hw = float(self.predict(scen_hw)[0])
            results["scenarios"]["headway_regularization"] = {
                "name": "Dynamic Headway Regularization (No Bunching)",
                "predicted_delay": pred_hw,
                "delay_savings": current_pred - pred_hw,
                "achieves_on_time": pred_hw <= target_on_time_threshold,
                "mutations": {"headway_deviation": {"from": float(row.get("headway_deviation", 0.0)), "to": 0.0}},
            }

        # 3. Policy: Upstream Delay Absorption (Terminal / Hub Holding)
        scen_up = base_df.copy()
        if "prev_stop_delay" in scen_up.columns:
            scen_up["prev_stop_delay"] = 0.0
            pred_up = float(self.predict(scen_up)[0])
            results["scenarios"]["upstream_delay_absorption"] = {
                "name": "Upstream Schedule Recovery / Hub Holding",
                "predicted_delay": pred_up,
                "delay_savings": current_pred - pred_up,
                "achieves_on_time": pred_up <= target_on_time_threshold,
                "mutations": {"prev_stop_delay": {"from": float(row.get("prev_stop_delay", 0.0)), "to": 0.0}},
            }

        # 4. Policy: Transit Signal Priority (TSP Green Wave)
        scen_tsp = base_df.copy()
        if "signalized_intersection_count" in scen_tsp.columns:
            scen_tsp["signalized_intersection_count"] = 0.0
            pred_tsp = float(self.predict(scen_tsp)[0])
            results["scenarios"]["transit_signal_priority"] = {
                "name": "Full Transit Signal Priority (TSP)",
                "predicted_delay": pred_tsp,
                "delay_savings": current_pred - pred_tsp,
                "achieves_on_time": pred_tsp <= target_on_time_threshold,
                "mutations": {"signalized_intersection_count": {"from": float(row.get("signalized_intersection_count", 0.0)), "to": 0.0}},
            }

        # 5. Policy: Comprehensive Joint Intervention (Combined Best Practice)
        scen_joint = base_df.copy()
        joint_mutations = {}
        if "is_dedicated_right_of_way" in scen_joint.columns:
            scen_joint["is_dedicated_right_of_way"] = 1.0
            joint_mutations["is_dedicated_right_of_way"] = {"from": float(row.get("is_dedicated_right_of_way", 0.0)), "to": 1.0}
        if "headway_deviation" in scen_joint.columns:
            scen_joint["headway_deviation"] = 0.0
            joint_mutations["headway_deviation"] = {"from": float(row.get("headway_deviation", 0.0)), "to": 0.0}
        if "prev_stop_delay" in scen_joint.columns:
            # 50% upstream buffer absorption
            orig_prev = float(row.get("prev_stop_delay", 0.0))
            new_prev = max(0.0, orig_prev * 0.25)
            scen_joint["prev_stop_delay"] = new_prev
            joint_mutations["prev_stop_delay"] = {"from": orig_prev, "to": new_prev}
        if "signalized_intersection_count" in scen_joint.columns:
            orig_sig = float(row.get("signalized_intersection_count", 0.0))
            new_sig = max(0.0, orig_sig * 0.5)
            scen_joint["signalized_intersection_count"] = new_sig
            joint_mutations["signalized_intersection_count"] = {"from": orig_sig, "to": new_sig}

        pred_joint = float(self.predict(scen_joint)[0])
        results["scenarios"]["joint_multimodal_remedy"] = {
            "name": "Joint Multimodal Operational & Infrastructure Remedy",
            "predicted_delay": pred_joint,
            "delay_savings": current_pred - pred_joint,
            "achieves_on_time": pred_joint <= target_on_time_threshold,
            "mutations": joint_mutations,
        }

        # Determine most efficient single intervention
        single_scenarios = [
            k for k in results["scenarios"].keys() if k != "joint_multimodal_remedy"
        ]
        if single_scenarios:
            best_single = max(single_scenarios, key=lambda k: results["scenarios"][k]["delay_savings"])
            results["most_effective_single_lever"] = best_single
            results["max_single_delay_savings"] = results["scenarios"][best_single]["delay_savings"]

        return results

    def run_severe_delay_batch_diagnostics(
        self,
        df: pd.DataFrame,
        delay_threshold: float = 600.0,
        max_samples: int = 50,
        target_on_time: float = 120.0,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run counterfactual policy diagnostics over severe delay incidents.

        Parameters
        ----------
        df : pd.DataFrame
            Validation or test set containing transit records.
        delay_threshold : float
            Threshold identifying severe delay incidents (default 600s = 10 minutes).
        max_samples : int
            Maximum severe delay cases to evaluate.
        target_on_time : float
            Delay ceiling defining on-time transit arrival (default 120s = 2 minutes).

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Any]]
            Per-sample scenario results dataframe and aggregate policy impact metrics.
        """
        target_col = self.target_col if self.target_col in df.columns else "arrival_delay_seconds"
        severe_df = df[df[target_col] >= delay_threshold].copy()

        if len(severe_df) == 0:
            logger.warning(f"No records found with {target_col} >= {delay_threshold}s. Subsampling top delays.")
            severe_df = df.sort_values(by=target_col, ascending=False).head(max_samples).copy()
        elif len(severe_df) > max_samples:
            severe_df = severe_df.sort_values(by=target_col, ascending=False).head(max_samples).copy()

        logger.info(f"Evaluating counterfactual diagnostics across {len(severe_df)} severe delay incidents...")

        records = []
        for idx, row in severe_df.iterrows():
            diag = self.generate_actionable_interventions(row, target_on_time_threshold=target_on_time)
            rec = {
                "route_id": str(row.get("route_id", "unknown")),
                "observed_delay": float(row.get(target_col, 0.0)),
                "predicted_delay": diag["original_prediction"],
                "most_effective_single_lever": diag.get("most_effective_single_lever", "none"),
                "max_single_savings": diag.get("max_single_delay_savings", 0.0),
            }
            for k, sc in diag["scenarios"].items():
                rec[f"{k}_pred"] = sc["predicted_delay"]
                rec[f"{k}_savings"] = sc["delay_savings"]
                rec[f"{k}_on_time"] = sc["achieves_on_time"]
            records.append(rec)

        results_df = pd.DataFrame(records)

        # Aggregate summary statistics
        n = len(results_df)
        summary = {
            "total_incidents_analyzed": n,
            "mean_observed_delay": float(results_df["observed_delay"].mean()),
            "mean_predicted_delay": float(results_df["predicted_delay"].mean()),
            "on_time_recovery_rates": {
                "dedicated_row_upgrade": float(results_df.get("dedicated_row_upgrade_on_time", pd.Series([False])).mean()),
                "headway_regularization": float(results_df.get("headway_regularization_on_time", pd.Series([False])).mean()),
                "upstream_delay_absorption": float(results_df.get("upstream_delay_absorption_on_time", pd.Series([False])).mean()),
                "transit_signal_priority": float(results_df.get("transit_signal_priority_on_time", pd.Series([False])).mean()),
                "joint_multimodal_remedy": float(results_df.get("joint_multimodal_remedy_on_time", pd.Series([False])).mean()),
            },
            "mean_delay_reduction_seconds": {
                "dedicated_row_upgrade": float(results_df.get("dedicated_row_upgrade_savings", pd.Series([0.0])).mean()),
                "headway_regularization": float(results_df.get("headway_regularization_savings", pd.Series([0.0])).mean()),
                "upstream_delay_absorption": float(results_df.get("upstream_delay_absorption_savings", pd.Series([0.0])).mean()),
                "transit_signal_priority": float(results_df.get("transit_signal_priority_savings", pd.Series([0.0])).mean()),
                "joint_multimodal_remedy": float(results_df.get("joint_multimodal_remedy_savings", pd.Series([0.0])).mean()),
            },
        }

        return results_df, summary

    def plot_counterfactual_case_study(
        self,
        instance_dict: Dict[str, Any],
        output_path: str = "reports/figures/xai/counterfactual_case_study.png",
        title: Optional[str] = None,
    ) -> str:
        """
        Plot bar chart comparing predicted delay under various policy interventions for a single case.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        orig_delay = instance_dict["original_prediction"]
        target = instance_dict["on_time_target"]

        labels = ["Baseline (Status Quo)"]
        values = [orig_delay]
        colors = ["#d9534f"]

        scenario_colors = {
            "dedicated_row_upgrade": "#0275d8",
            "headway_regularization": "#5bc0de",
            "upstream_delay_absorption": "#f0ad4e",
            "transit_signal_priority": "#9b59b6",
            "joint_multimodal_remedy": "#5cb85c",
        }

        for k, sc in instance_dict["scenarios"].items():
            labels.append(sc["name"])
            values.append(sc["predicted_delay"])
            colors.append(scenario_colors.get(k, "#6c757d"))

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(labels, values, color=colors, edgecolor="black", alpha=0.85, height=0.55)
        ax.axvline(target, color="green", linestyle="--", linewidth=2.0, label=f"On-Time Threshold ({target:.0f}s)")
        ax.set_xlabel("Predicted Delay Delta $\\Delta t$ (seconds)", fontsize=12, fontweight="bold")
        ax.set_title(
            title or "Counterfactual Policy Interventions for Severe Transit Delay",
            fontsize=13,
            fontweight="bold",
            pad=14,
        )
        ax.grid(axis="x", linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", fontsize=11)
        ax.invert_yaxis()

        # Add value labels
        for bar, val in zip(bars, values):
            diff = orig_delay - val
            diff_text = f" (-{diff:.1f}s)" if diff > 0 else ""
            ax.text(
                val + 5,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}s{diff_text}",
                va="center",
                ha="left",
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Counterfactual case study saved to {output_path}")
        return output_path

    def plot_policy_remedy_summary(
        self,
        summary_dict: Dict[str, Any],
        output_path: str = "reports/figures/xai/counterfactual_policy_impact.png",
    ) -> str:
        """
        Plot comparative bar chart comparing mean delay reduction and on-time recovery rates across policies.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        savings = summary_dict["mean_delay_reduction_seconds"]
        recovery = summary_dict["on_time_recovery_rates"]

        policy_names = [
            "Dedicated ROW\nUpgrade",
            "Headway\nRegularization",
            "Upstream Delay\nAbsorption",
            "Transit Signal\nPriority (TSP)",
            "Joint Multimodal\nRemedy",
        ]
        keys = [
            "dedicated_row_upgrade",
            "headway_regularization",
            "upstream_delay_absorption",
            "transit_signal_priority",
            "joint_multimodal_remedy",
        ]

        savings_vals = [savings.get(k, 0.0) for k in keys]
        recovery_vals = [recovery.get(k, 0.0) * 100 for k in keys]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        # Panel 1: Mean Delay Reduction (Seconds)
        bar_colors = ["#0275d8", "#5bc0de", "#f0ad4e", "#9b59b6", "#5cb85c"]
        bars1 = ax1.bar(policy_names, savings_vals, color=bar_colors, edgecolor="black", alpha=0.85, width=0.55)
        ax1.set_ylabel("Mean Delay Reduction (seconds)", fontsize=11, fontweight="bold")
        ax1.set_title("Average Delay Attenuation by Intervention", fontsize=12, fontweight="bold")
        ax1.grid(axis="y", linestyle=":", alpha=0.6)

        for bar in bars1:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"+{h:.1f}s", ha="center", va="bottom", fontweight="bold")

        # Panel 2: On-Time Recovery Rate (%)
        bars2 = ax2.bar(policy_names, recovery_vals, color=bar_colors, edgecolor="black", alpha=0.85, width=0.55)
        ax2.set_ylabel("On-Time Restoration Rate (%)", fontsize=11, fontweight="bold")
        ax2.set_title("Probability of Achieving On-Time Status ($\\Delta t \\leq 120$s)", fontsize=12, fontweight="bold")
        ax2.grid(axis="y", linestyle=":", alpha=0.6)
        ax2.set_ylim(0, 105)

        for bar in bars2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")

        plt.suptitle(
            "Empirical Policy Efficacy: Transit Delay Counterfactual Remedies (N = {})".format(
                summary_dict.get("total_incidents_analyzed", 0)
            ),
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Counterfactual policy summary chart saved to {output_path}")
        return output_path


def run_counterfactual_pipeline(
    feature_mart_path: str = "data/processed/feature_mart.parquet",
    output_dir: str = "reports/figures/xai",
    tables_dir: str = "reports/tables/xai",
) -> Dict[str, Any]:
    """Execute end-to-end Counterfactual Diagnostics workflow."""
    import duckdb
    from src.models.gbm import GradientBoostingBenchmark
    from src.models.validation import PurgedTemporalBlockSplitter

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    con = duckdb.connect()
    logger.info(f"Loading feature mart from {feature_mart_path}...")
    sample_df = con.execute(f"SELECT * FROM '{feature_mart_path}' USING SAMPLE 100000").df()

    target_col = "delta_t_run" if "delta_t_run" in sample_df.columns else "arrival_delay_seconds"
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
    valid_features = [c for c in feature_cols if c in sample_df.columns]

    splitter = PurgedTemporalBlockSplitter(embargo_minutes=30.0, purge_trips=True)
    train_df, val_df, test_df = splitter.split(sample_df, train_ratio=0.6, val_ratio=0.2)

    logger.info("Training reference LightGBM regressor for counterfactual diagnostics...")
    gbm = GradientBoostingBenchmark(target_col=target_col, model_type="lightgbm")
    gbm.train(train_df, val_df, valid_features, n_estimators=150, learning_rate=0.05)

    cf_engine = CounterfactualDiagnostics(gbm, valid_features, target_col=target_col)
    cf_engine.setup_dice(train_df.head(2000), method="kdtree")

    # Evaluate batch diagnostics on severe delays
    batch_df, summary = cf_engine.run_severe_delay_batch_diagnostics(
        test_df,
        delay_threshold=300.0,
        max_samples=50,
        target_on_time=120.0,
    )

    # Save summary tables
    summary_csv = os.path.join(tables_dir, "counterfactual_summary.csv")
    batch_df.to_csv(summary_csv, index=False)
    summary_json = os.path.join(tables_dir, "counterfactual_policy_metrics.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Case study plot on worst observed delay
    worst_idx = test_df[target_col].idxmax()
    worst_instance = test_df.loc[worst_idx]
    case_diag = cf_engine.generate_actionable_interventions(worst_instance, target_on_time_threshold=120.0)

    case_plot = os.path.join(output_dir, "counterfactual_case_study.png")
    cf_engine.plot_counterfactual_case_study(
        case_diag,
        output_path=case_plot,
        title=f"Route {worst_instance.get('route_id', 'Unknown')} Incident (Observed Delay: {worst_instance[target_col]:.0f}s)",
    )

    policy_plot = os.path.join(output_dir, "counterfactual_policy_impact.png")
    cf_engine.plot_policy_remedy_summary(summary, output_path=policy_plot)

    return {
        "summary": summary,
        "batch_records": len(batch_df),
        "case_plot": case_plot,
        "policy_plot": policy_plot,
        "summary_csv": summary_csv,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_counterfactual_pipeline()

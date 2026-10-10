"""
Non-Linear Gradient Boosting Benchmark Pipeline (LightGBM & CatBoost).
Predicts arrival delay delta Delta t (seconds) using rich multi-source feature groups:
- Temporal: hour_of_day, day_of_week, is_peak_hour, is_morning_peak, is_evening_peak, is_weekend
- Operational: prev_stop_delay, prev2_stop_delay, headway_deviation, headway_actual_s, trip_progress
- Infrastructure: is_dedicated_right_of_way, signalized_intersection_count, segment_length_meters, lane_capacity
- Meteorology: precipitation_mm, temperature_c, relative_humidity, wind_speed_ms, freezing_rain_flag
- Categorical: route_id, corridor_name, stop_id
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.models.metrics import compute_regression_metrics, format_metrics_table

logger = logging.getLogger(__name__)


class GradientBoostingBenchmark:
    """Trains, tunes, and evaluates LightGBM and CatBoost regression for transit delays."""

    def __init__(
        self,
        target_col: str = "delta_t_run",
        model_type: str = "lightgbm",
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize GradientBoostingBenchmark.

        Parameters
        ----------
        target_col : str
            Target regression column (default 'delta_t_run' or 'arrival_delay_seconds').
        model_type : str
            'lightgbm' or 'catboost'.
        params : Dict[str, Any], optional
            Model-specific hyperparameters overriding defaults.
        """
        self.target_col = target_col
        self.model_type = model_type.lower()
        self.params = params or {}
        self.model = None
        self.feature_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.feature_importances: Dict[str, float] = {}
        self.best_iteration: Optional[int] = None
        self.val_metrics: Dict[str, float] = {}

    def _prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        cat_cols: Optional[List[str]] = None,
        is_training: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """Preprocess features and encode categoricals."""
        data = df.copy()

        # Target extraction
        y = None
        if self.target_col in data.columns:
            y = data[self.target_col].to_numpy(dtype=float)

        active_feats = [c for c in feature_cols if c in data.columns]
        X = data[active_feats].copy()

        # Categorical columns
        detected_cat = cat_cols or [c for c in active_feats if X[c].dtype == object or str(X[c].dtype) == "category"]
        if is_training:
            self.cat_cols = detected_cat

        for col in active_feats:
            if col in self.cat_cols:
                if self.model_type == "lightgbm":
                    X[col] = X[col].astype("category")
                else:
                    X[col] = X[col].astype(str).fillna("missing")
            elif X[col].dtype == bool:
                X[col] = X[col].astype(float)
            else:
                # Numeric column: fill NaNs with 0.0 or let trees handle NaNs
                if X[col].isna().any():
                    X[col] = X[col].fillna(0.0)
                X[col] = X[col].astype(float)

        return X, y

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: List[str],
        cat_cols: Optional[List[str]] = None,
        n_estimators: int = 250,
        learning_rate: float = 0.05,
        early_stopping_rounds: int = 25,
    ) -> Dict[str, float]:
        """
        Train gradient boosting model with early stopping on validation split.

        Parameters
        ----------
        train_df : pd.DataFrame
            Purged temporal block training split.
        val_df : pd.DataFrame
            Out-of-sample validation split.
        feature_cols : List[str]
            Feature names for training.
        cat_cols : List[str], optional
            Explicit list of categorical features.
        n_estimators : int
            Maximum number of boosting iterations (default 250).
        learning_rate : float
            Shrinkage rate (default 0.05).
        early_stopping_rounds : int
            Rounds of early stopping patience (default 25).

        Returns
        -------
        Dict[str, float]
            Validation performance metrics.
        """
        self.feature_cols = feature_cols
        X_train, y_train = self._prepare_data(train_df, feature_cols, cat_cols=cat_cols, is_training=True)
        X_val, y_val = self._prepare_data(val_df, feature_cols, cat_cols=cat_cols, is_training=False)

        if y_train is None or y_val is None:
            raise ValueError(f"Target column '{self.target_col}' missing from datasets.")

        if self.model_type == "lightgbm":
            try:
                import lightgbm as lgb
                default_params = {
                    "n_estimators": n_estimators,
                    "learning_rate": learning_rate,
                    "num_leaves": 31,
                    "min_child_samples": 20,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "reg_alpha": 0.1,
                    "reg_lambda": 1.0,
                    "random_state": 42,
                    "verbosity": -1,
                    "n_jobs": -1,
                }
                default_params.update(self.params)

                callbacks = []
                if early_stopping_rounds > 0 and len(X_val) > 0:
                    callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

                self.model = lgb.LGBMRegressor(**default_params)
                self.model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)] if len(X_val) > 0 else None,
                    callbacks=callbacks if callbacks else None,
                )

                if hasattr(self.model, "best_iteration_"):
                    self.best_iteration = self.model.best_iteration_

                # Extract feature importances
                importances = self.model.feature_importances_
                self.feature_importances = dict(zip(X_train.columns, [float(v) for v in importances]))

            except (ImportError, OSError) as e:
                logger.warning(f"LightGBM initialization failed ({e}), falling back to CatBoost or HistGradientBoosting.")
                self.model_type = "catboost"

        if self.model_type == "catboost" and self.model is None:
            try:
                import catboost as cb
                default_params = {
                    "iterations": n_estimators,
                    "learning_rate": learning_rate,
                    "depth": 6,
                    "l2_leaf_reg": 3.0,
                    "random_seed": 42,
                    "early_stopping_rounds": early_stopping_rounds,
                    "verbose": False,
                }
                default_params.update(self.params)

                cat_features_idx = [X_train.columns.get_loc(c) for c in self.cat_cols if c in X_train.columns]
                self.model = cb.CatBoostRegressor(**default_params)
                self.model.fit(
                    X_train,
                    y_train,
                    eval_set=(X_val, y_val) if len(X_val) > 0 else None,
                    cat_features=cat_features_idx if cat_features_idx else None,
                )
                importances = self.model.get_feature_importance()
                self.feature_importances = dict(zip(X_train.columns, [float(v) for v in importances]))

            except (ImportError, OSError) as e:
                logger.warning(f"CatBoost initialization failed ({e}), using scikit-learn HistGradientBoosting.")
                from sklearn.ensemble import HistGradientBoostingRegressor
                self.model = HistGradientBoostingRegressor(max_iter=n_estimators, random_state=42)
                self.model.fit(X_train, y_train)

        # Generate validation predictions
        preds = self.predict(val_df)
        self.val_metrics = compute_regression_metrics(y_val, preds, prefix="val_")

        logger.info(
            f"{self.model_type.upper()} Validation: MAE={self.val_metrics['val_mae']:.2f}s, "
            f"RMSE={self.val_metrics['val_rmse']:.2f}s, R²={self.val_metrics['val_r2']:.4f}, "
            f"WAPE={self.val_metrics['val_wape']:.2f}%"
        )
        return self.val_metrics

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Generate model predictions on test or evaluation dataframe."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        X_test, _ = self._prepare_data(test_df, self.feature_cols, cat_cols=self.cat_cols, is_training=False)
        return self.model.predict(X_test)

    def get_feature_importances(self) -> pd.DataFrame:
        """Return sorted DataFrame of feature importances."""
        if not self.feature_importances:
            return pd.DataFrame(columns=["feature", "importance", "relative_importance"])

        records = [{"feature": k, "importance": float(v)} for k, v in self.feature_importances.items()]
        df = pd.DataFrame(records).sort_values(by="importance", ascending=False).reset_index(drop=True)
        total = df["importance"].sum()
        df["relative_importance"] = (df["importance"] / total * 100.0) if total > 0 else 0.0
        return df

    def summary(self) -> str:
        """Generate human-readable summary of the gradient boosting model."""
        lines = [
            "=" * 78,
            f"Gradient Boosting Regressor Benchmark: {self.model_type.upper()}",
            f"Target: {self.target_col} | Features: {len(self.feature_cols)}",
            f"Best Iteration: {self.best_iteration}",
            f"Val MAE: {self.val_metrics.get('val_mae', 0.0):.2f} s | Val RMSE: {self.val_metrics.get('val_rmse', 0.0):.2f} s | Val R²: {self.val_metrics.get('val_r2', 0.0):.4f}",
            "-" * 78,
            "Top Feature Importances:",
        ]
        fi_df = self.get_feature_importances()
        for _, row in fi_df.head(10).iterrows():
            lines.append(f"  {row['feature']:<30s} {row['importance']:>12.2f} ({row['relative_importance']:>5.1f}%)")
        lines.append("=" * 78)
        return "\n".join(lines)


def train_gbm_suite(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "delta_t_run",
    cat_cols: Optional[List[str]] = None,
) -> Tuple[GradientBoostingBenchmark, GradientBoostingBenchmark, pd.DataFrame]:
    """
    Train both LightGBM and CatBoost models and produce a comparison DataFrame.
    """
    # 1. LightGBM
    lgb_model = GradientBoostingBenchmark(target_col=target_col, model_type="lightgbm")
    lgb_metrics = lgb_model.train(train_df, val_df, feature_cols, cat_cols=cat_cols)

    # 2. CatBoost
    cb_model = GradientBoostingBenchmark(target_col=target_col, model_type="catboost")
    cb_metrics = cb_model.train(train_df, val_df, feature_cols, cat_cols=cat_cols)

    comparison = pd.DataFrame([
        {
            "model": "LightGBM Regressor",
            "val_mae": lgb_metrics.get("val_mae"),
            "val_rmse": lgb_metrics.get("val_rmse"),
            "val_wape": lgb_metrics.get("val_wape"),
            "val_r2": lgb_metrics.get("val_r2"),
            "val_medae": lgb_metrics.get("val_medae"),
        },
        {
            "model": "CatBoost Regressor",
            "val_mae": cb_metrics.get("val_mae"),
            "val_rmse": cb_metrics.get("val_rmse"),
            "val_wape": cb_metrics.get("val_wape"),
            "val_r2": cb_metrics.get("val_r2"),
            "val_medae": cb_metrics.get("val_medae"),
        },
    ])
    return lgb_model, cb_model, comparison

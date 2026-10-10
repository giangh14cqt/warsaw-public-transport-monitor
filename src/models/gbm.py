"""
Non-Linear Gradient Boosting Pipeline (LightGBM / CatBoost).
Predicts arrival delay delta Delta t (seconds) using rich feature groups:
- Temporal: hour_of_day, day_of_week, is_peak_hour
- Operational: prev_stop_delay, headway_deviation
- Infrastructure: is_dedicated_right_of_way, signalized_intersection_count, segment_length_meters
- Meteorology: precipitation_mm, temperature_c, freezing_rain_flag
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class GradientBoostingBenchmark:
    """Trains and validates LightGBM / CatBoost regression for transit delays."""

    def __init__(self, target_col: str = "arrival_delay_seconds"):
        self.target_col = target_col
        self.model = None
        self.feature_cols: List[str] = []

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: List[str],
        n_estimators: int = 200,
        learning_rate: float = 0.05,
    ) -> Dict[str, float]:
        """Train gradient boosting regressor and compute validation metrics."""
        self.feature_cols = feature_cols
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[self.target_col].fillna(0)

        X_val = val_df[feature_cols].fillna(0)
        y_val = val_df[self.target_col].fillna(0)

        try:
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=42,
                verbosity=-1,
            )
            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_val)

            mae = float(np.mean(np.abs(y_val - preds)))
            rmse = float(np.sqrt(np.mean((y_val - preds) ** 2)))
            ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
            r2 = float(1 - (np.sum((y_val - preds) ** 2) / ss_tot)) if ss_tot > 0 else 0.0

            metrics = {"val_mae": mae, "val_rmse": rmse, "val_r2": r2}
            logger.info(f"LightGBM validation: MAE={mae:.2f}s, RMSE={rmse:.2f}s, R²={r2:.4f}")
            return metrics
        except ImportError:
            logger.warning("LightGBM not installed. Using basic scikit-learn regressor fallback.")
            from sklearn.ensemble import HistGradientBoostingRegressor
            self.model = HistGradientBoostingRegressor(max_iter=n_estimators, random_state=42)
            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_val)
            mae = float(np.mean(np.abs(y_val - preds)))
            rmse = float(np.sqrt(np.mean((y_val - preds) ** 2)))
            return {"val_mae": mae, "val_rmse": rmse, "val_r2": 0.0}

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Generate predictions on test set."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        X_test = test_df[self.feature_cols].fillna(0)
        return self.model.predict(X_test)

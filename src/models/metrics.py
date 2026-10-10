"""
Standardized Evaluation Metrics Harness for Transit Delay Prediction Models.
Computes robust, industry-standard regression metrics:
- MAE: Mean Absolute Error (seconds)
- RMSE: Root Mean Squared Error (seconds)
- R²: Coefficient of Determination
- WAPE: Weighted Absolute Percentage Error (%)
- MedAE: Median Absolute Error (seconds)
- MaxAE: Maximum Absolute Error (seconds)
- Pearson R: Pearson correlation coefficient between actual and predicted delays
"""

import logging
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_regression_metrics(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute standardized regression evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth transit delay targets.
    y_pred : array-like
        Model predicted delay targets.
    prefix : str, optional
        Optional prefix for metric dict keys (e.g., 'val_' or 'test_').

    Returns
    -------
    Dict[str, float]
        Dictionary of standardized regression metrics:
        - {prefix}mae: Mean Absolute Error (seconds)
        - {prefix}rmse: Root Mean Squared Error (seconds)
        - {prefix}r2: R-squared coefficient of determination
        - {prefix}wape: Weighted Absolute Percentage Error (%)
        - {prefix}medae: Median Absolute Error (seconds)
        - {prefix}max_error: Maximum Absolute Error (seconds)
        - {prefix}pearson_r: Pearson correlation coefficient
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()

    if len(y_t) == 0 or len(y_p) == 0:
        return {
            f"{prefix}mae": 0.0,
            f"{prefix}rmse": 0.0,
            f"{prefix}r2": 0.0,
            f"{prefix}wape": 0.0,
            f"{prefix}medae": 0.0,
            f"{prefix}max_error": 0.0,
            f"{prefix}pearson_r": 0.0,
        }

    if len(y_t) != len(y_p):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_t)} vs len(y_pred)={len(y_p)}"
        )

    # Filter out NaNs / Infs if present
    valid_mask = np.isfinite(y_t) & np.isfinite(y_p)
    if not np.all(valid_mask):
        n_dropped = int(np.sum(~valid_mask))
        logger.warning(f"Dropping {n_dropped} non-finite entries from metric calculation")
        y_t = y_t[valid_mask]
        y_p = y_p[valid_mask]

    if len(y_t) == 0:
        return {
            f"{prefix}mae": 0.0,
            f"{prefix}rmse": 0.0,
            f"{prefix}r2": 0.0,
            f"{prefix}wape": 0.0,
            f"{prefix}medae": 0.0,
            f"{prefix}max_error": 0.0,
            f"{prefix}pearson_r": 0.0,
        }

    errors = y_p - y_t
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    medae = float(np.median(abs_errors))
    max_error = float(np.max(abs_errors))

    # R-squared calculation
    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    ss_res = float(np.sum(errors ** 2))
    if ss_tot > 1e-9:
        r2 = float(1.0 - (ss_res / ss_tot))
    else:
        # If ground truth has zero variance
        r2 = 1.0 if ss_res < 1e-9 else 0.0

    # WAPE (Weighted Absolute Percentage Error) = sum(|y_true - y_pred|) / sum(|y_true|) * 100%
    denom_wape = float(np.sum(np.abs(y_t)))
    if denom_wape > 1e-9:
        wape = float((np.sum(abs_errors) / denom_wape) * 100.0)
    else:
        wape = 0.0 if np.sum(abs_errors) < 1e-9 else 100.0

    # Pearson correlation coefficient
    std_t = float(np.std(y_t))
    std_p = float(np.std(y_p))
    if std_t > 1e-9 and std_p > 1e-9:
        corr_matrix = np.corrcoef(y_t, y_p)
        pearson_r = float(corr_matrix[0, 1])
    else:
        pearson_r = 0.0

    return {
        f"{prefix}mae": round(mae, 4),
        f"{prefix}rmse": round(rmse, 4),
        f"{prefix}r2": round(r2, 4),
        f"{prefix}wape": round(wape, 4),
        f"{prefix}medae": round(medae, 4),
        f"{prefix}max_error": round(max_error, 4),
        f"{prefix}pearson_r": round(pearson_r, 4),
    }


def format_metrics_table(metrics: Dict[str, float], model_name: str = "") -> str:
    """Format evaluation metrics as an ASCII string table."""
    header = f"=== Benchmark Performance: {model_name} ===" if model_name else "=== Benchmark Performance ==="
    lines = [header]
    for k, v in metrics.items():
        if "wape" in k:
            lines.append(f"  {k:15s}: {v:8.2f}%")
        elif "r2" in k or "pearson_r" in k:
            lines.append(f"  {k:15s}: {v:8.4f}")
        else:
            lines.append(f"  {k:15s}: {v:8.2f} s")
    return "\n".join(lines)


def compare_models_table(models_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Convert a dictionary of model metrics into a comparison DataFrame.
    
    Parameters
    ----------
    models_metrics : Dict[str, Dict[str, float]]
        e.g., {"TWFE": {"mae": 32.1, ...}, "LightGBM": {"mae": 18.4, ...}}
    """
    records = []
    for model_name, metrics in models_metrics.items():
        row = {"model": model_name}
        row.update(metrics)
        records.append(row)
    df = pd.DataFrame(records)
    if "model" in df.columns:
        df = df.set_index("model")
    return df

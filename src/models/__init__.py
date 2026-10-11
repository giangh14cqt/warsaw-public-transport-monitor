"""
Modeling & Validation Subsystem.
Includes Purged Temporal Block splitting, Econometric TWFE baseline,
Gradient Boosting pipelines, and Standardized Evaluation Metrics.
"""

from src.models.gbm import GradientBoostingBenchmark
from src.models.metrics import (
    compare_models_table,
    compute_regression_metrics,
    format_metrics_table,
)
from src.models.twfe import TWFEBaselineModel
from src.models.validation import DataLeakageError, PurgedTemporalBlockSplitter

__all__ = [
    "PurgedTemporalBlockSplitter",
    "DataLeakageError",
    "compute_regression_metrics",
    "format_metrics_table",
    "compare_models_table",
    "TWFEBaselineModel",
    "GradientBoostingBenchmark",
]

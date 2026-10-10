"""
Modeling & Validation Subsystem.
Includes Purged Temporal Block splitting, Econometric TWFE baseline,
Gradient Boosting pipelines, and Standardized Evaluation Metrics.
"""

from src.models.validation import PurgedTemporalBlockSplitter, DataLeakageError
from src.models.metrics import (
    compute_regression_metrics,
    format_metrics_table,
    compare_models_table,
)
from src.models.twfe import TWFEBaselineModel
from src.models.gbm import GradientBoostingBenchmark

__all__ = [
    "PurgedTemporalBlockSplitter",
    "DataLeakageError",
    "compute_regression_metrics",
    "format_metrics_table",
    "compare_models_table",
    "TWFEBaselineModel",
    "GradientBoostingBenchmark",
]

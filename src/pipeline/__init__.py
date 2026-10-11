"""
Warsaw Transit Telemetry & Attribution System: Pipeline Orchestration Layer.
Encapsulates individual pipeline stage execution logic.
"""

from src.pipeline.ranges import (
    build_date_filter_clause,
    format_range_report,
    inspect_available_ranges,
    print_range_report,
    validate_date_range,
)
from src.pipeline.stages import (
    banner,
    get_parquet_row_count,
    run_stage_benchmark,
    run_stage_exogenous,
    run_stage_fusion,
    run_stage_preflight,
    run_stage_ranges,
    run_stage_xai,
)

__all__ = [
    "banner",
    "get_parquet_row_count",
    "run_stage_preflight",
    "run_stage_exogenous",
    "run_stage_fusion",
    "run_stage_benchmark",
    "run_stage_xai",
    "run_stage_ranges",
    "inspect_available_ranges",
    "print_range_report",
    "format_range_report",
    "build_date_filter_clause",
    "validate_date_range",
]

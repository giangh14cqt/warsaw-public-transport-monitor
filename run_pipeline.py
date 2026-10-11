#!/usr/bin/env python3
"""
Warsaw Transit Delay Telemetry & Attribution System: Master Pipeline Runner.
Orchestrates end-to-end execution across:
0. Temporal Range Audit: Inspect available data ranges across raw, weather, GTFS, and OSM sources.
1. Pre-Flight Integrity Check: GTFS schedule archive & live GTFS-RT feed validation.
2. Exogenous Data Fusion: IMGW weather harvesting & OSMnx corridor road topology.
3. Feature Mart Assembly: DuckDB spatiotemporal fusion and delay lag decomposition.
4. Econometric & Machine Learning Benchmarks: TWFE panel regression, LightGBM, and CatBoost.
5. Explainable AI Diagnostics: TreeSHAP attribution, ALE curves, interaction buffering, and DiCE counterfactuals.

Usage:
    # 1. Audit available date ranges across all data sources
    python3 run_pipeline.py --check-ranges

    # 2. Run full pipeline on specific date range
    python3 run_pipeline.py --stage all --start-date 2026-10-04 --end-date 2026-10-10

    # 3. Train models on specific workweek
    python3 run_pipeline.py --stage benchmark --start-date 2026-10-05 --end-date 2026-10-09 --sample-size 50000

    # 4. Generate xAI diagnostics on selected date range
    python3 run_pipeline.py --stage xai --start-date 2026-10-04 --end-date 2026-10-10
"""

import argparse
import logging
import sys
import time

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline_runner")

__all__ = [
    "banner",
    "get_parquet_row_count",
    "run_stage_preflight",
    "run_stage_exogenous",
    "run_stage_fusion",
    "run_stage_benchmark",
    "run_stage_xai",
    "run_stage_ranges",
    "main",
]


def main():
    parser = argparse.ArgumentParser(
        description="Warsaw Transit Telemetry & Attribution System: Master Pipeline Runner."
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=[
            "preflight",
            "exogenous",
            "fusion",
            "benchmark",
            "xai",
            "ranges",
            "check-ranges",
            "all",
        ],
        help="Pipeline stage to execute (default: all).",
    )
    parser.add_argument(
        "--check-ranges",
        action="store_true",
        help="Inspect and display available date ranges across all data sources and exit.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for data range selection (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS, Europe/Warsaw local time).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for data range selection (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS, Europe/Warsaw local time).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50000,
        help="Subsample size for benchmark and xAI stages (default: 50,000; 0 for all).",
    )
    parser.add_argument(
        "--fusion-sample-size",
        type=int,
        default=None,
        help="Optional row limit on raw telemetry during fusion stage (default: None, full dataset).",
    )
    parser.add_argument(
        "--raw-pattern",
        type=str,
        default="data/raw/**/*.parquet",
        help="Hive-partitioned raw Parquet glob pattern.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory for reports and figures (default: reports).",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild of feature_mart.parquet even if already present.",
    )
    parser.add_argument(
        "--cohort",
        type=str,
        default="stratified",
        choices=["all", "stratified", "urban_tram", "urban_bus", "suburban_bus"],
        help="Transit cohort to evaluate: all (pooled), stratified (all cohorts), or specific cohort.",
    )
    parser.add_argument(
        "--include-terminals",
        action="store_true",
        help="Include terminal layover stops (default is to exclude for noise reduction).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode for preflight and testing without touching production paths.",
    )

    args = parser.parse_args()

    # Range inspection shortcut
    if args.check_ranges or args.stage in ("ranges", "check-ranges"):
        run_stage_ranges(args)
        return 0

    # Validate custom date range if supplied
    if args.start_date or args.end_date:
        from src.pipeline.ranges import validate_date_range

        args.start_date, args.end_date = validate_date_range(args.start_date, args.end_date)

    total_t0 = time.time()
    banner("WARSAW PUBLIC TRANSPORT DELAY ATTRIBUTION PIPELINE", fill="#")
    range_info = f" | Date Range: {args.start_date or 'earliest'} -> {args.end_date or 'latest'}"
    logger.info(
        f"Target Stage: {args.stage.upper()} | Sample Size: {args.sample_size:,}{range_info}"
    )

    stages_executed = {}

    try:
        if args.stage in ("preflight", "all"):
            stages_executed["preflight"] = run_stage_preflight(args)

        if args.stage in ("exogenous", "all"):
            stages_executed["exogenous"] = run_stage_exogenous(args)

        if args.stage in ("fusion", "all"):
            stages_executed["fusion"] = run_stage_fusion(args)

        if args.stage in ("benchmark", "all"):
            stages_executed["benchmark"] = run_stage_benchmark(args)

        if args.stage in ("xai", "all"):
            stages_executed["xai"] = run_stage_xai(args)

        total_duration = time.time() - total_t0
        banner("PIPELINE EXECUTION COMPLETE: ALL STAGES PASSED", fill="#")
        logger.info(f"Total end-to-end execution time: {total_duration:.2f}s.")
        print("\nSUMMARY OF GENERATED ARTIFACTS:")
        print("  - Models Benchmark: reports/tables/benchmark_comparison.csv")
        print("  - TreeSHAP Beeswarm: reports/figures/xai/shap_beeswarm_summary.png")
        print(
            "  - ALE Curves: reports/figures/xai/ale_precipitation_curve.png, ale_temperature_curve.png"
        )
        print("  - Infrastructure Buffering: reports/figures/xai/interaction_headway_row.png")
        print(
            "  - Counterfactual Policy Impact: reports/figures/xai/counterfactual_policy_impact.png"
        )
        print("  - Counterfactual Summary: reports/tables/xai/counterfactual_summary.csv")
        print("\nPipeline execution concluded successfully.\n")
        return 0

    except Exception as e:
        logger.exception(f"Pipeline failed during execution: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Warsaw Transit Delay Telemetry & Attribution System: Master Pipeline Runner.
Orchestrates end-to-end execution across:
1. Pre-Flight Integrity Check: GTFS schedule archive & live GTFS-RT feed validation.
2. Exogenous Data Fusion: IMGW weather harvesting & OSMnx corridor road topology.
3. Feature Mart Assembly: DuckDB spatiotemporal fusion and delay lag decomposition.
4. Econometric & Machine Learning Benchmarks: TWFE panel regression, LightGBM, and CatBoost.
5. Explainable AI Diagnostics: TreeSHAP attribution, ALE curves, interaction buffering, and DiCE counterfactuals.

Usage:
    python3 run_pipeline.py --stage all
    python3 run_pipeline.py --stage benchmark --sample-size 50000
    python3 run_pipeline.py --stage xai
"""

import os
import sys
import time
import argparse
import logging
from typing import Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline_runner")


def banner(text: str, fill: str = "="):
    """Print visually distinguished stage banner."""
    width = 80
    print("\n" + fill * width)
    print(f"  {text}".center(width))
    print(fill * width + "\n")


def run_stage_preflight(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute pre-flight system integrity checks."""
    banner("STAGE 1: PRE-FLIGHT INTEGRITY CHECK")
    from src.gtfs_manager import GTFSManager
    from src.fetcher import GTFSRTFetcher
    from src.storage import ParquetPartitionStorage, DuckDBCatalog

    t0 = time.time()
    logger.info("Verifying static GTFS schedule manager...")
    gtfs_mgr = GTFSManager(gtfs_base_dir="data/gtfs")
    unpack_dir = gtfs_mgr.sync_weekly_gtfs()
    logger.info(f"Static GTFS unpack directory: {unpack_dir}")

    logger.info("Connecting to live Warsaw GTFS-RT endpoint...")
    fetcher = GTFSRTFetcher()
    feed_bytes = fetcher.fetch_feed()
    records, stats = fetcher.parse_feed(feed_bytes, gtfs_mgr)
    logger.info(f"Parsed {len(records)} vehicle observation records from live feed.")

    logger.info("Verifying Parquet buffer flush...")
    test_storage = ParquetPartitionStorage(base_dir="data/raw_test" if args.dry_run else "data/raw")
    buffered = test_storage.filter_and_buffer(records)
    part_path = test_storage.flush()

    duration = time.time() - t0
    logger.info(f"Pre-flight gate passed in {duration:.2f}s.")
    return {
        "status": "success",
        "duration_seconds": duration,
        "feed_bytes": len(feed_bytes),
        "parsed_records": len(records),
        "test_partition": part_path,
    }


def run_stage_exogenous(args: argparse.Namespace) -> Dict[str, Any]:
    """Verify or harvest exogenous IMGW weather and OSM corridor topology."""
    banner("STAGE 2: EXOGENOUS FEATURE MART INTEGRATION")
    t0 = time.time()

    weather_path = "data/processed/imgw_weather_hourly.parquet"
    osm_path = "data/processed/osm_corridor_segments.parquet"

    # Weather check
    if os.path.exists(weather_path):
        logger.info(f"Found active IMGW weather telemetry at {weather_path} ({os.path.getsize(weather_path):,} bytes).")
    else:
        logger.info("Harvesting IMGW-PIB hourly synoptic weather telemetry...")
        from src.exogenous.weather import fetch_imgw_synoptic_archive
        fetch_imgw_synoptic_archive(output_path=weather_path)

    # OSM corridors check
    if os.path.exists(osm_path):
        logger.info(f"Found active OSMnx road corridor topology at {osm_path} ({os.path.getsize(osm_path):,} bytes).")
    else:
        logger.info("Extracting OSMnx corridor infrastructure attributes...")
        from src.exogenous.osm_corridors import extract_default_corridors
        extract_default_corridors(output_path=osm_path)

    duration = time.time() - t0
    return {
        "status": "success",
        "duration_seconds": duration,
        "weather_path": weather_path,
        "osm_path": osm_path,
    }


def run_stage_fusion(args: argparse.Namespace) -> Dict[str, Any]:
    """Fuse raw telemetry, weather, and road topology into unified feature mart."""
    banner("STAGE 3: SPATIOTEMPORAL FUSION & FEATURE MART ASSEMBLY")
    from src.fusion.feature_mart import FeatureMartAssembler

    t0 = time.time()
    assembler = FeatureMartAssembler(data_dir="data")
    mart_path = assembler.output_parquet

    max_records = args.sample_size if (args.sample_size and args.sample_size > 0) else None

    if os.path.exists(mart_path) and not args.force_rebuild:
        logger.info(f"Existing feature mart verified at {mart_path} ({os.path.getsize(mart_path):,} bytes).")
        import duckdb
        con = duckdb.connect()
        total_rows = con.execute(f"SELECT count(*) FROM '{mart_path}'").fetchone()[0]
        logger.info(f"Feature mart contains {total_rows:,} rows.")
    else:
        logger.info(f"Assembling unified feature mart into {mart_path}...")
        df_mart = assembler.assemble(
            raw_pattern=args.raw_pattern,
            max_records=max_records,
            save_parquet=True,
        )
        total_rows = len(df_mart)
        logger.info(f"Successfully assembled {total_rows:,} feature records.")

    duration = time.time() - t0
    return {
        "status": "success",
        "duration_seconds": duration,
        "mart_path": mart_path,
        "total_rows": total_rows,
    }


def run_stage_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    """Train econometric TWFE baseline and gradient boosting benchmark models."""
    banner("STAGE 4: ECONOMETRIC & MACHINE LEARNING BENCHMARK MODELS")
    import duckdb
    from src.models.benchmark import run_benchmark

    t0 = time.time()
    mart_path = "data/processed/feature_mart.parquet"
    if not os.path.exists(mart_path):
        raise FileNotFoundError(f"Feature mart not found at {mart_path}. Run --stage fusion first.")

    sample_size = args.sample_size if args.sample_size > 0 else 100000
    con = duckdb.connect()
    logger.info(f"Loading {sample_size:,} sample rows from {mart_path}...")
    df_sample = con.execute(f"SELECT * FROM '{mart_path}' USING SAMPLE {sample_size}").df()

    logger.info("Executing purged temporal block benchmark (TWFE vs. LightGBM vs. CatBoost)...")
    summary_df = run_benchmark(df_sample, sample_size=sample_size)

    tables_dir = os.path.join(args.output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    summary_csv = os.path.join(tables_dir, "benchmark_comparison.csv")
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"Benchmark comparison table exported to {summary_csv}")

    duration = time.time() - t0
    print("\n" + summary_df.to_string(index=False) + "\n")
    return {
        "status": "success",
        "duration_seconds": duration,
        "summary_csv": summary_csv,
        "benchmark_results": summary_df.to_dict(orient="records"),
    }


def run_stage_xai(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute complete explainable AI attribution and diagnostics suite."""
    banner("STAGE 5: EXPLAINABLE AI (xAI) ATTRIBUTION & COUNTERFACTUAL DIAGNOSTICS")
    t0 = time.time()

    from src.xai.shap_explainer import run_shap_pipeline
    from src.xai.ale_curves import run_ale_pipeline
    from src.xai.interactions import run_interaction_pipeline
    from src.xai.counterfactuals import run_counterfactual_pipeline

    figures_dir = os.path.join(args.output_dir, "figures", "xai")
    tables_dir = os.path.join(args.output_dir, "tables", "xai")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    mart_path = "data/processed/feature_mart.parquet"

    # 1. TreeSHAP Attribution
    logger.info("1/4: Computing global TreeSHAP beeswarm and local waterfall attributions...")
    shap_res = run_shap_pipeline(feature_mart_path=mart_path, output_dir=figures_dir)

    # 2. ALE Non-Linear Thresholds
    logger.info("2/4: Computing Accumulated Local Effects (ALE) non-linear curves...")
    ale_res = run_ale_pipeline(feature_mart_path=mart_path, output_dir=figures_dir)

    # 3. Infrastructure Buffering & Interactions
    logger.info("3/4: Quantifying dedicated right-of-way infrastructure buffering...")
    inter_res = run_interaction_pipeline(feature_mart_path=mart_path, output_dir=figures_dir)

    # 4. DiCE Counterfactual Diagnostics
    logger.info("4/4: Generating actionable counterfactual intervention diagnostics...")
    cf_res = run_counterfactual_pipeline(
        feature_mart_path=mart_path, output_dir=figures_dir, tables_dir=tables_dir
    )

    duration = time.time() - t0
    logger.info(f"xAI diagnostics completed in {duration:.2f}s.")
    return {
        "status": "success",
        "duration_seconds": duration,
        "shap_summary": shap_res,
        "ale_summary": ale_res,
        "interaction_summary": inter_res,
        "counterfactual_summary": cf_res,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Warsaw Transit Telemetry & Attribution System: Master Pipeline Runner."
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["preflight", "exogenous", "fusion", "benchmark", "xai", "all"],
        help="Pipeline stage to execute (default: all).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50000,
        help="Subsample size for benchmark and xAI stages (default: 50,000; 0 for all).",
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
        "--dry-run",
        action="store_true",
        help="Dry run mode for preflight and testing without touching production paths.",
    )

    args = parser.parse_args()

    total_t0 = time.time()
    banner("WARSAW PUBLIC TRANSPORT DELAY ATTRIBUTION PIPELINE", fill="#")
    logger.info(f"Target Stage: {args.stage.upper()} | Sample Size: {args.sample_size:,}")

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
        print("  - ALE Curves: reports/figures/xai/ale_precipitation_curve.png, ale_temperature_curve.png")
        print("  - Infrastructure Buffering: reports/figures/xai/interaction_headway_row.png")
        print("  - Counterfactual Policy Impact: reports/figures/xai/counterfactual_policy_impact.png")
        print("  - Counterfactual Summary: reports/tables/xai/counterfactual_summary.csv")
        print("\nPipeline execution concluded successfully.\n")
        return 0

    except Exception as e:
        logger.exception(f"Pipeline failed during execution: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

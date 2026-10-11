"""
Pipeline Stage Executors for Warsaw Transit Telemetry & Attribution System.
Implements the 5 core stages:
1. Pre-Flight Integrity Check
2. Exogenous Data Fusion (IMGW Weather & OSM Corridors)
3. Spatiotemporal Fusion & Feature Mart Assembly
4. Econometric & Machine Learning Benchmarks
5. Explainable AI Diagnostics & Policy Counterfactuals
"""

import argparse
import logging
import os
import time
from typing import Any, Dict

from src.core.config import paths

logger = logging.getLogger(__name__)


def banner(text: str, fill: str = "=") -> None:
    """Print visually distinguished stage banner."""
    width = 80
    print("\n" + fill * width)
    print(f"  {text}".center(width))
    print(fill * width + "\n")


def get_parquet_row_count(parquet_path: str) -> int:
    """Safely count rows in a Parquet file via DuckDB, returning 0 if empty or None."""
    if not os.path.exists(parquet_path) or os.path.getsize(parquet_path) == 0:
        return 0
    import duckdb

    con = duckdb.connect()
    try:
        res = con.execute(f"SELECT count(*) FROM '{parquet_path}'").fetchone()
        if res and res[0] is not None:
            return int(res[0])
        return 0
    except Exception as e:
        logger.warning(f"Could not count rows in {parquet_path}: {e}")
        return 0
    finally:
        con.close()


def run_stage_preflight(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute pre-flight system integrity checks."""
    banner("STAGE 1: PRE-FLIGHT INTEGRITY CHECK")
    from src.fetcher import GTFSRTFetcher
    from src.gtfs_manager import GTFSManager
    from src.storage import ParquetPartitionStorage

    t0 = time.time()
    logger.info("Verifying static GTFS schedule manager...")
    gtfs_mgr = GTFSManager(gtfs_base_dir=str(paths.gtfs_dir))
    unpack_dir = gtfs_mgr.sync_weekly_gtfs()
    logger.info(f"Static GTFS unpack directory: {unpack_dir}")

    logger.info("Connecting to live Warsaw GTFS-RT endpoint...")
    fetcher = GTFSRTFetcher()
    feed_bytes = fetcher.fetch_feed()
    records, stats = fetcher.parse_feed(feed_bytes, gtfs_mgr)
    logger.info(f"Parsed {len(records)} vehicle observation records from live feed.")

    logger.info("Verifying Parquet buffer flush...")
    test_storage = ParquetPartitionStorage(
        base_dir="data/raw_test" if getattr(args, "dry_run", False) else str(paths.raw_dir)
    )
    test_storage.filter_and_buffer(records)
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

    weather_path = str(paths.weather_hourly)
    osm_path = str(paths.osm_corridors)
    force_rebuild = getattr(args, "force_rebuild", False)

    # Weather check
    if os.path.exists(weather_path) and not force_rebuild:
        logger.info(
            f"Found active IMGW weather telemetry at {weather_path} ({os.path.getsize(weather_path):,} bytes)."
        )
    else:
        logger.info("Harvesting IMGW-PIB hourly synoptic weather telemetry...")
        from src.exogenous.imgw import fetch_imgw_synoptic_archive

        fetch_imgw_synoptic_archive(output_path=weather_path)

    # OSM corridors check
    if os.path.exists(osm_path) and not force_rebuild:
        logger.info(
            f"Found active OSMnx road corridor topology at {osm_path} ({os.path.getsize(osm_path):,} bytes)."
        )
    else:
        logger.info("Extracting OSMnx corridor infrastructure attributes...")
        from src.exogenous.osm import extract_default_corridors

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
    assembler = FeatureMartAssembler(data_dir=str(paths.data_dir))
    mart_path = assembler.output_parquet

    # Subsample only if explicitly requested for fusion or in dry-run mode
    max_records = getattr(args, "fusion_sample_size", None)
    if max_records is None and getattr(args, "dry_run", False):
        max_records = 5000

    force_rebuild = getattr(args, "force_rebuild", False)
    raw_pat = getattr(args, "raw_pattern", None)

    if os.path.exists(mart_path) and not force_rebuild:
        logger.info(
            f"Existing feature mart verified at {mart_path} ({os.path.getsize(mart_path):,} bytes)."
        )
        total_rows = get_parquet_row_count(mart_path)
        logger.info(f"Feature mart contains {total_rows:,} rows.")
    else:
        logger.info(f"Assembling unified feature mart into {mart_path}...")
        assembler.assemble_mart(
            raw_pattern=raw_pat,
            output_parquet=mart_path,
            max_records=max_records,
        )
        total_rows = get_parquet_row_count(mart_path)
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

    from src.models.benchmark import run_benchmark, run_stratified_benchmark

    t0 = time.time()
    mart_path = paths.get_feature_mart_str()
    if not os.path.exists(mart_path):
        raise FileNotFoundError(f"Feature mart not found at {mart_path}. Run --stage fusion first.")

    sample_size = getattr(args, "sample_size", 100000)
    if sample_size is None or sample_size <= 0:
        sample_size = 100000
    exclude_terminals = not getattr(args, "include_terminals", False)
    cohort_arg = getattr(args, "cohort", "stratified")

    con = duckdb.connect()
    logger.info(f"Loading {sample_size:,} sample rows from {mart_path}...")
    try:
        df_sample = con.execute(
            f"SELECT * FROM '{mart_path}' USING SAMPLE {sample_size} (reservoir, 42)"
        ).df()
    finally:
        con.close()

    output_dir = getattr(args, "output_dir", "reports")
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    summary_csv = os.path.join(tables_dir, "benchmark_comparison.csv")

    if df_sample.empty:
        logger.warning(f"Feature mart at {mart_path} contains 0 records; skipping benchmark.")
        return {
            "status": "empty",
            "duration_seconds": time.time() - t0,
            "summary_csv": summary_csv,
            "benchmark_results": [],
        }

    if cohort_arg == "stratified":
        logger.info(
            "Executing stratified benchmark across pooled, urban tram, urban bus, and suburban bus cohorts..."
        )
        cohort_df = run_stratified_benchmark(
            df_sample, sample_size=sample_size, exclude_terminals=exclude_terminals
        )
        cohort_csv = os.path.join(tables_dir, "benchmark_by_cohort.csv")
        cohort_df.to_csv(cohort_csv, index=False)
        logger.info(f"Stratified cohort benchmark table exported to {cohort_csv}")

        # Also extract pooled benchmark as primary comparison table
        pooled_df = (
            cohort_df[cohort_df["cohort"] == "pooled"]
            .drop(columns=["cohort"])
            .reset_index(drop=True)
        )
        if pooled_df.empty:
            pooled_df = run_benchmark(
                df_sample, sample_size=sample_size, exclude_terminals=exclude_terminals
            )
        pooled_df.to_csv(summary_csv, index=False)
        logger.info(f"Pooled benchmark comparison table exported to {summary_csv}")
        summary_df = cohort_df
    else:
        target_cohort = None if cohort_arg == "all" else cohort_arg
        logger.info(
            f"Executing benchmark for cohort '{cohort_arg}' (exclude_terminals={exclude_terminals})..."
        )
        summary_df = run_benchmark(
            df_sample,
            sample_size=sample_size,
            cohort=target_cohort,
            exclude_terminals=exclude_terminals,
        )
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

    from src.xai.ale_curves import run_ale_pipeline
    from src.xai.counterfactuals import run_counterfactual_pipeline
    from src.xai.interactions import run_interaction_pipeline
    from src.xai.shap_explainer import run_shap_pipeline

    output_dir = getattr(args, "output_dir", "reports")
    figures_dir = os.path.join(output_dir, "figures", "xai")
    tables_dir = os.path.join(output_dir, "tables", "xai")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    mart_path = paths.get_feature_mart_str()

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

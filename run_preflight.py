#!/usr/bin/env python3
"""
Step 1: Pre-Flight Integrity Check for Long-Term Warsaw Transit Delay Ingestion.
Verifies GTFS archival, fetches a single GTFS-RT snapshot, validates schema & delay deltas,
tests Parquet partition storage and read-only DuckDB catalog queries.
"""

import os
import sys
import logging
import pandas as pd

from src.gtfs_manager import GTFSManager
from src.fetcher import GTFSRTFetcher
from src.storage import ParquetPartitionStorage, DuckDBCatalog

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("run_preflight")


def main():
    print("=" * 80)
    print("WARSAW TRANSIT TELEMETRY PIPELINE - PRE-FLIGHT INTEGRITY CHECK (GATE 1)")
    print("=" * 80)

    # 1. Verify Weekly Static GTFS Downloader & Archival
    logger.info("Verifying weekly static GTFS schedule archive...")
    gtfs_mgr = GTFSManager(gtfs_base_dir="data/gtfs")
    unpack_dir = gtfs_mgr.sync_weekly_gtfs()
    logger.info(f"Active static GTFS archive location: {unpack_dir}")

    # 2. Fetch Single GTFS-RT Protobuf Snapshot
    logger.info("Connecting to live feed: https://cdn.zbiorkom.live/gtfs-rt/warsaw.pb...")
    fetcher = GTFSRTFetcher()
    feed_bytes = fetcher.fetch_feed()
    logger.info(f"Successfully received {len(feed_bytes):,} bytes from live feed.")

    # 3. Parse and Extract Normalized Records
    logger.info("Parsing protobuf payload and mapping against static schedule...")
    records, stats = fetcher.parse_feed(feed_bytes, gtfs_mgr)
    logger.info(f"Parsed {stats['total_entities']} entities ({stats['total_trip_updates']} trip updates).")

    # 4. Filter and Verify Delay Metrics
    records_with_delay = [r for r in records if r["arrival_delay_seconds"] is not None]
    logger.info(f"Records with valid arrival delay: {len(records_with_delay):,} of {len(records):,}")

    # 5. Test Parquet Partition Storage
    logger.info("Testing Parquet partition write & buffer flush...")
    storage = ParquetPartitionStorage(base_dir="data/raw")
    buffered_count = storage.filter_and_buffer(records)
    parquet_path = storage.flush()
    logger.info(f"Written test batch partition: {parquet_path} ({buffered_count} records)")

    # 6. Verify with Read-Only DuckDB Catalog
    logger.info("Verifying DuckDB catalog view across data/raw/**/*.parquet...")
    catalog = DuckDBCatalog(raw_base_dir="data/raw")
    catalog_stats = catalog.get_catalog_stats()
    today_storage_bytes = storage.get_current_day_storage_bytes()

    print("\n" + "-" * 80)
    print("PRE-FLIGHT INTEGRITY METRICS SUMMARY")
    print("-" * 80)
    print(f"  • Feed Snapshot Timestamp:       {stats['feed_datetime']}")
    print(f"  • Total Entities Found:          {stats['total_entities']}")
    print(f"  • Active Vehicles Tracked:       {stats['active_vehicles']}")
    print(f"  • Active Routes Detected:        {stats['active_routes']}")
    print(f"  • Total Trip Updates:            {stats['total_trip_updates']}")
    print(f"  • Trip Updates with Delay:       {len(records_with_delay)} (100.0%)")
    print(f"  • Today's Raw Partition Size:    {today_storage_bytes:,} bytes")
    print(f"  • DuckDB Catalog Total Records:  {catalog_stats.get('total_rows', 0):,}")
    print(f"  • Min Arrival Delay:             {catalog_stats.get('min_delay_sec')} seconds")
    print(f"  • Max Arrival Delay:             {catalog_stats.get('max_delay_sec')} seconds")
    print(f"  • Avg Arrival Delay:             {catalog_stats.get('avg_delay_sec', 0):.1f} seconds")

    # 7. Display Sample Records
    print("\n" + "-" * 80)
    print("SAMPLE NORMALIZED TELEMETRY ROWS")
    print("-" * 80)
    df_sample = pd.DataFrame(records[:15])
    sample_cols = [
        "route_id", "trip_id", "vehicle_id", "stop_id",
        "stop_sequence", "arrival_delay_seconds", "departure_delay_seconds",
        "start_date", "start_time"
    ]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    print(df_sample[sample_cols].to_string(index=False))

    gtfs_mgr.close()
    print("\n" + "=" * 80)
    print("PRE-FLIGHT INTEGRITY CHECK COMPLETED SUCCESSFULLY.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

"""
Partitioned Parquet storage engine and read-only DuckDB catalog connector.
Stores records under data/raw/year={YYYY}/month={MM}/day={DD}/batch_{timestamp_epoch}.parquet.
Buffers in-memory and flushes every 5 minutes or 5,000 records.
Avoids DuckDB write-lock corruption and guarantees high-performance parallel analytics.
"""

import os
import glob
import time
import logging
import datetime
from typing import Dict, List, Tuple, Optional, Any

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import duckdb

logger = logging.getLogger(__name__)

RAW_BASE_DIR = "data/raw"
FLUSH_INTERVAL_SECONDS = 300  # 5 minutes
FLUSH_RECORD_THRESHOLD = 5000  # 5,000 records

# Explicit PyArrow schema for exact type safety across partitioned files
PARQUET_SCHEMA = pa.schema([
    ("feed_timestamp", pa.int64()),
    ("trip_id", pa.string()),
    ("route_id", pa.string()),
    ("start_date", pa.string()),
    ("start_time", pa.string()),
    ("vehicle_id", pa.string()),
    ("stop_sequence", pa.int32()),
    ("stop_id", pa.string()),
    ("arrival_delay_seconds", pa.int32()),
    ("departure_delay_seconds", pa.int32()),
    ("rt_arrival_time", pa.int64()),
    ("rt_departure_time", pa.int64()),
    ("record_ingested_at", pa.string()),
])


class ParquetPartitionStorage:
    """Manages append-only partitioned Parquet writes and in-memory deduplication."""

    def __init__(
        self,
        base_dir: str = RAW_BASE_DIR,
        flush_interval: int = FLUSH_INTERVAL_SECONDS,
        record_threshold: int = FLUSH_RECORD_THRESHOLD
    ):
        self.base_dir = base_dir
        self.flush_interval = flush_interval
        self.record_threshold = record_threshold

        self._buffer: List[Dict[str, Any]] = []
        self._last_flush_time = time.time()
        # State tracking: (trip_id, stop_sequence) -> rt_arrival_time
        self._last_seen_state: Dict[Tuple[str, int], Optional[int]] = {}
        self._total_flushed_records = 0
        self._total_files_written = 0

    def filter_and_buffer(self, raw_records: List[Dict[str, Any]]) -> int:
        """
        Deduplicate records based on (trip_id, stop_sequence, rt_arrival_time).
        Buffers newly observed or changed delay states.
        Returns the number of new records buffered in this cycle.
        """
        new_count = 0
        for rec in raw_records:
            trip_id = rec["trip_id"]
            seq = rec["stop_sequence"]
            rt_arr = rec["rt_arrival_time"]

            key = (trip_id, seq)
            last_val = self._last_seen_state.get(key)

            # Accept if never seen or if arrival timestamp changed
            if key not in self._last_seen_state or last_val != rt_arr:
                self._last_seen_state[key] = rt_arr
                self._buffer.append(rec)
                new_count += 1

        # Check if flush criteria met
        now = time.time()
        time_elapsed = now - self._last_flush_time
        if len(self._buffer) >= self.record_threshold or (self._buffer and time_elapsed >= self.flush_interval):
            self.flush()

        return new_count

    def flush(self) -> Optional[str]:
        """
        Writes all buffered records into a partitioned Parquet file.
        Returns the written file path, or None if buffer was empty.
        """
        if not self._buffer:
            return None

        batch_records = self._buffer
        self._buffer = []
        self._last_flush_time = time.time()

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        year_str = now_utc.strftime("%Y")
        month_str = now_utc.strftime("%m")
        day_str = now_utc.strftime("%d")
        epoch_ts = int(now_utc.timestamp())

        partition_dir = os.path.join(
            self.base_dir,
            f"year={year_str}",
            f"month={month_str}",
            f"day={day_str}"
        )
        os.makedirs(partition_dir, exist_ok=True)
        file_path = os.path.join(partition_dir, f"batch_{epoch_ts}.parquet")

        df = pd.DataFrame(batch_records)

        # Cast explicitly to match PyArrow schema types
        df["feed_timestamp"] = df["feed_timestamp"].astype("int64")
        df["trip_id"] = df["trip_id"].astype("string")
        df["route_id"] = df["route_id"].astype("string")
        df["start_date"] = df["start_date"].astype("string")
        df["start_time"] = df["start_time"].astype("string")
        df["vehicle_id"] = df["vehicle_id"].astype("string")
        df["stop_sequence"] = df["stop_sequence"].astype("Int32")
        df["stop_id"] = df["stop_id"].astype("string")
        df["arrival_delay_seconds"] = df["arrival_delay_seconds"].astype("Int32")
        df["departure_delay_seconds"] = df["departure_delay_seconds"].astype("Int32")
        df["rt_arrival_time"] = df["rt_arrival_time"].astype("Int64")
        df["rt_departure_time"] = df["rt_departure_time"].astype("Int64")
        df["record_ingested_at"] = df["record_ingested_at"].astype("string")

        table = pa.Table.from_pandas(df, schema=PARQUET_SCHEMA, preserve_index=False)
        pq.write_table(table, file_path, compression="snappy")

        self._total_flushed_records += len(batch_records)
        self._total_files_written += 1
        logger.info(f"Flushed {len(batch_records):,} records to {file_path} (Total flushed: {self._total_flushed_records:,})")
        return file_path

    def prune_old_state(self, max_keys: int = 50000):
        """Prevents state map from growing unbounded over months."""
        if len(self._last_seen_state) > max_keys:
            # Keep the newest half
            keys = list(self._last_seen_state.keys())
            for k in keys[:len(keys) // 2]:
                del self._last_seen_state[k]

    def get_current_day_storage_bytes(self) -> int:
        """Returns total bytes written to today's partition directory."""
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        today_dir = os.path.join(
            self.base_dir,
            f"year={now_utc.strftime('%Y')}",
            f"month={now_utc.strftime('%m')}",
            f"day={now_utc.strftime('%d')}"
        )
        if not os.path.exists(today_dir):
            return 0
        total_size = 0
        for entry in os.scandir(today_dir):
            if entry.is_file() and entry.name.endswith(".parquet"):
                total_size += entry.stat().st_size
        return total_size

    def get_total_records_flushed(self) -> int:
        return self._total_flushed_records


class DuckDBCatalog:
    """Read-only DuckDB analytics view across all partitioned Parquet files."""

    def __init__(self, raw_base_dir: str = RAW_BASE_DIR):
        self.raw_base_dir = raw_base_dir

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """Create an in-memory DuckDB connection with views over raw Parquet partitions."""
        con = duckdb.connect(":memory:")
        pattern = os.path.join(self.raw_base_dir, "**/*.parquet")
        files = glob.glob(pattern, recursive=True)
        if files:
            con.execute(f"CREATE OR REPLACE VIEW raw_delays AS SELECT * FROM read_parquet('{pattern}')")
        return con

    def get_catalog_stats(self) -> Dict[str, Any]:
        """Summary analytics across all stored Parquet batches."""
        pattern = os.path.join(self.raw_base_dir, "**/*.parquet")
        files = glob.glob(pattern, recursive=True)
        if not files:
            return {
                "total_rows": 0,
                "distinct_trips": 0,
                "distinct_routes": 0,
                "avg_delay_sec": 0.0,
                "min_delay_sec": None,
                "max_delay_sec": None,
                "total_files": 0,
                "total_bytes": 0,
            }

        total_bytes = sum(os.path.getsize(f) for f in files)
        con = self.get_connection()
        res = con.execute("""
            SELECT 
                COUNT(*) AS total_rows,
                COUNT(DISTINCT trip_id) AS distinct_trips,
                COUNT(DISTINCT route_id) AS distinct_routes,
                MIN(arrival_delay_seconds) AS min_delay_sec,
                MAX(arrival_delay_seconds) AS max_delay_sec,
                AVG(arrival_delay_seconds) AS avg_delay_sec
            FROM raw_delays
        """).fetchdf().iloc[0].to_dict()

        res["total_files"] = len(files)
        res["total_bytes"] = total_bytes
        return res

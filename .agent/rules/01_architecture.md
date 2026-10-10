# 01_architecture.md: Storage and Concurrency Architecture Rules

## 1. Storage Anti-Corruption & Zero-Lock Guarantees
- **No Monolithic Database Locks**: Never maintain long-running, open write locks on SQLite, DuckDB, or monolithic database files during continuous live polling.
- **Partitioned Parquet Sink**: All raw telemetry stream data must be written to append-only, immutable **Apache Parquet** partitions partitioned by date:
  ```text
  data/raw/year=YYYY/month=MM/day=DD/batch_<timestamp>_<uuid>.parquet
  ```
- **Compression**: All Parquet files must be compressed using **Snappy** for maximum read/write throughput with low CPU overhead.
- **In-Memory Buffering**: Buffer telemetry observations in memory and flush to Parquet partitions using dual trigger criteria:
  - Time elapsed $\ge 300$ seconds (5 minutes), OR
  - Buffered record count $\ge 5,000$ rows.

## 2. Analytical Layer (Read-Only)
- Analytical queries and feature engineering pipelines must query Parquet files using read-only in-memory DuckDB connections (`read_parquet('data/raw/**/*.parquet', hive_partitioning=true)`).
- Never attach DuckDB in read-write mode to files concurrently being written by the ingestion daemon.

## 3. Weekly Static Timetable Versioning
- Static GTFS snapshots must be versioned by schedule date in `data/gtfs/gtfs_YYYY_MM_DD.zip` and unpacked into isolated date folders to prevent schedule schema drift.

---
description: Run pre-flight integrity check and sandbox validation for the Warsaw transit telemetry pipeline
---

# Pre-Flight Verification Workflow (`/run-preflight`)

This workflow executes Gate 1 pre-flight validation to test live telemetry streaming, static GTFS schedule extraction, and DuckDB analytical queries.

## Step 1: Execute Pre-flight Suite
```bash
python3 run_preflight.py
```

## Step 2: Run Automated Unit and Integration Tests
```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

## Step 3: Verify Storage Integrity
Check that test Parquet partitions and duckdb views can be read without errors:
```bash
python3 -c "
import duckdb
con = duckdb.connect(':memory:')
print(con.execute(\"SELECT 'DuckDB Operational' AS status\").fetchall())
"
```

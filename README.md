# Warsaw Public Transport Delay Telemetry Engine

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Storage-Parquet](https://img.shields.io/badge/storage-Apache%20Parquet-orange.svg)](https://parquet.apache.org/)
[![Engine-DuckDB](https://img.shields.io/badge/analytics-DuckDB-yellow.svg)](https://duckdb.org/)
[![Docker](https://img.shields.io/badge/deployment-Docker%20Compose-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A high-frequency, resilient telemetry ingestion engine designed for multi-month continuous data collection of Warsaw public transit (buses and trams) from [zbiorkom.live](https://zbiorkom.live).

Built as the foundational telemetry pipeline for University of Warsaw (UW) Master's thesis research on **Traffic Congestion, Transit Delay Propagation, and Explainable AI (xAI)**.

---

## Architecture Overview

```
                                 ┌────────────────────────────────────────────────┐
                                 │   https://cdn.zbiorkom.live/gtfs-rt/warsaw.pb  │
                                 └───────────────────────┬────────────────────────┘
                                                         │ (Poll every 30s)
                                                         ▼
┌─────────────────────────────────┐           ┌──────────────────────┐
│  Weekly Static GTFS Archive     │           │   GTFSRTFetcher      │
│  (cdn.zbiorkom.live/gtfs/*.zip) │──────────▶│   - Protobuf Parser  │
│  Indexed in embedded DuckDB     │ (O(1) Map)│   - Tenacity Retries │
└─────────────────────────────────┘           └──────────┬───────────┘
                                                         │
                                                         ▼
                                              ┌──────────────────────┐
                                              │ State Deduplication  │
                                              │ (trip, seq, rt_time) │
                                              └──────────┬───────────┘
                                                         │ (Buffer 5 min / 5k rows)
                                                         ▼
                                    ┌───────────────────────────────────────────┐
                                    │    Hive-Partitioned Apache Parquet        │
                                    │ data/raw/year=YYYY/month=MM/day=DD/*.pq   │
                                    └────────────────────┬──────────────────────┘
                                                         │
                                ┌────────────────────────┴────────────────────────┐
                                ▼                                                 ▼
                ┌───────────────────────────────┐                 ┌───────────────────────────────┐
                │     Read-Only DuckDB Engine   │                 │   Daily Cloud Backup (rclone) │
                │   Zero-Lock Analytics & ML    │                 │   Incremental to Google Drive │
                └───────────────────────────────┘                 └───────────────────────────────┘
```

### Key Architectural Highlights
- **Storage Anti-Corruption**: Never maintains an open, long-running write lock on a monolithic database. Telemetry is flushed to immutable, append-only **Apache Parquet partitions** compressed with **Snappy**.
- **State Deduplication**: Only persists records where `(trip_id, stop_sequence, rt_arrival_time)` has changed since the previous cycle, discarding duplicate observations for stationary vehicles.
- **Circuit Breakers**:
  - Exponential backoff with jitter on transient network hiccups ($3\text{s}, 6\text{s}, 12\text{s}, 24\text{s}$).
  - Automatic 5-minute pause on persistent HTTP 429 (Rate Limit).
  - 15-minute persistent network outage detection.
  - Fail-safe stop on Protobuf `DecodeError` schema mismatches.
- **Dead Man's Switch**: Sends start, periodic 15-minute telemetry health reports, and failure alerts to [Healthchecks.io](https://healthchecks.io).
- **Schedule Drift & DST Resilience**: Automatically refreshes static GTFS timetables weekly and handles `Europe/Warsaw` Daylight Saving Time (CEST $\to$ CET) transitions.

---

## Telemetry Schema

Each record captured in the partitioned Parquet files adheres to the following typed schema:

| Column | Type | Description |
| :--- | :--- | :--- |
| `feed_timestamp` | `INT64` | Epoch timestamp of the live GTFS-RT feed header |
| `trip_id` | `STRING` | Unique GTFS trip identifier (e.g., `6916331-863617`) |
| `route_id` | `STRING` | Route / line identifier (e.g., `1`, `9`, `24`, `175`, `523`) |
| `start_date` | `STRING` | Scheduled service date (`YYYYMMDD`) |
| `start_time` | `STRING` | Scheduled trip departure time (`HH:MM:SS`) |
| `vehicle_id` | `STRING` | Physical vehicle identifier (e.g., `3/5894`) |
| `stop_sequence` | `INT32` | Ordinal stop index along the route |
| `stop_id` | `STRING` | Exact physical stop post code in Warsaw ZTM |
| `arrival_delay_seconds` | `INT32` | **Arrival delay delta** in seconds ($>0$ late, $<0$ early, $=0$ on-time) |
| `departure_delay_seconds`| `INT32` | **Departure delay delta** in seconds |
| `rt_arrival_time` | `INT64` | Live observed/estimated arrival Unix timestamp |
| `rt_departure_time` | `INT64` | Live observed/estimated departure Unix timestamp |
| `record_ingested_at` | `STRING` | UTC ISO timestamp when the collector recorded the state |

---

## Directory Structure

```text
.
├── .agent/                   # Antigravity IDE agent rules & workflows
│   ├── rules/
│   │   ├── 01_architecture.md   # Parquet partitioning, zero-lock concurrency rules
│   │   ├── 02_error_handling.md # Circuit breakers (HTTP 429 backoff, rate limiting)
│   │   └── 03_code_style.md     # Python standards (uv, ruff, backwards compatibility)
│   └── workflows/
│       ├── run-preflight.md     # /run-preflight trigger
│       ├── harvest-weather.md   # /harvest-weather trigger
│       └── eval-models.md       # /eval-models trigger
├── .dockerignore
├── .env.example              # Configuration template (Heartbeat URL, paths)
├── .gitignore                # Protects data/, logs/, and virtual environment
├── Dockerfile                # Production container definition (Python 3.12-slim)
├── docker-compose.yml        # Multi-month daemon service with resource bounds
├── pyproject.toml            # Modern project specification & tool configuration
├── requirements.txt          # Production ingestion dependencies
├── run_preflight.py          # Pre-flight sandbox verification test
├── collector_daemon.py       # Main continuous polling daemon with signal traps
├── backup_to_gdrive.sh       # Automated rclone backup script for Google Drive
├── pull_from_gdrive.sh       # Syncs partitions down from Google Drive for local analytics
├── src/
│   ├── ingestion/            # Phase 1: GTFS-RT pollers, static archiver & Parquet sink
│   ├── exogenous/            # Phase 2: IMGW weather harvester & OSMnx road topology
│   ├── fusion/               # Phase 3: Trajectory reconstruction & multi-source joins
│   ├── models/               # Phase 4: TWFE panel regression & LightGBM/CatBoost
│   ├── xai/                  # Phase 5: TreeSHAP, ALE curves & DiCE counterfactuals
│   ├── fetcher.py            # Backward-compatible proxy to src.ingestion.fetcher
│   ├── gtfs_manager.py       # Backward-compatible proxy to src.ingestion.gtfs_manager
│   └── storage.py            # Backward-compatible proxy to src.ingestion.storage
├── tests/                    # Unit tests, pre-flight and schema sanity checks
├── data/
│   ├── raw/                  # Partitioned Parquet sink (year=YYYY/month=MM/day=DD/)
│   ├── gtfs/                 # Archived weekly static GTFS snapshots
│   └── processed/            # Fused feature marts (feature_mart.parquet)
└── logs/
    ├── collector_health.log  # 15-minute rotating heartbeat log
    └── backup.log            # Audit trail of Google Drive cloud backups
```

---

## Quickstart & Local Setup

### 1. Prerequisites
- Python 3.12+
- `uv` (recommended) or `pip`

```bash
# Clone the repository
git clone https://github.com/giangh14cqt/warsaw-public-transport-monitor.git
cd warsaw-public-transport-monitor

# Create virtual environment and install dependencies
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Pre-Flight Verification
Run the sandbox test to verify live feed connectivity, static schedule matching, and Parquet writing:
```bash
python3 run_preflight.py
```

---

## Production Deployment (Remote / On-Premise Server)

### 1. Configuration
Copy the environment template and set your [Healthchecks.io](https://healthchecks.io) heartbeat ping URL:
```bash
cp .env.example .env
nano .env
```
```env
HEARTBEAT_URL=https://hc-ping.com/YOUR-UUID-HERE
```

### 2. Run with Docker Compose
```bash
docker compose up -d --build
```

### 3. Monitoring
- **Inspect live polling cycles**:
  ```bash
  docker compose logs -f --tail 30
  ```
- **Inspect periodic 15-minute health heartbeats**:
  ```bash
  cat logs/collector_health.log
  ```
- **Check container resource status**:
  ```bash
  docker compose ps
  ```

---

## Querying Data During Ingestion

Because files are written as append-only Parquet partitions, **you can inspect and query the data at any time without write locks or pausing the collector**:

### 1. CLI Summary (Using DuckDB)
```bash
python3 -c "
import duckdb
con = duckdb.connect(':memory:')
print(con.execute('''
    SELECT 
        COUNT(*) AS total_records,
        COUNT(DISTINCT trip_id) AS active_trips,
        COUNT(DISTINCT route_id) AS active_routes,
        ROUND(AVG(arrival_delay_seconds), 1) AS avg_delay_sec,
        MIN(arrival_delay_seconds) AS min_delay_sec,
        MAX(arrival_delay_seconds) AS max_delay_sec
    FROM read_parquet('data/raw/**/*.parquet', hive_partitioning=true)
''').fetchdf())
"
```

### 2. Python / Jupyter Notebook
```python
import duckdb

con = duckdb.connect(":memory:")

# Query specific route on a specific day with partition pruning:
df = con.execute("""
    SELECT 
        route_id, 
        trip_id, 
        stop_id, 
        stop_sequence, 
        arrival_delay_seconds, 
        record_ingested_at
    FROM read_parquet('data/raw/**/*.parquet', hive_partitioning=true)
    WHERE year = 2026 AND month = 10 AND day = 4 AND route_id = '175'
    ORDER BY record_ingested_at DESC
""").fetchdf()

print(df.head())
```

---

## Automated Google Drive Backup Setup

To safeguard against local drive failures, the pipeline includes an automated incremental backup script powered by `rclone`.

1. **Configure rclone with Google Drive**:
   ```bash
   rclone config
   # Name remote: gdrive (or specify RCLONE_REMOTE in your .env)
   ```
2. **Test manual backup**:
   ```bash
   ./backup_to_gdrive.sh
   ```
3. **Automate with cron (Runs daily at 03:00 AM UTC)**:
   ```bash
   crontab -e
   ```
   Add the following entry:
   ```cron
   0 3 * * * /path/to/warsaw-public-transport-monitor/backup_to_gdrive.sh > /dev/null 2>&1
   ```

### Pulling Data Down for Local Analytics

To work locally with fresh telemetry captured on the remote server without touching the running daemon, sync partitions down from Google Drive:
```bash
./pull_from_gdrive.sh
```
This runs an incremental `rclone copy --update`, fetching newly archived Parquet partitions into `data/raw/` while preserving existing local files.

---

## Master Pipeline Runner (`run_pipeline.py`)

The repository includes a unified, reproducible CLI orchestrator that executes the entire analytical lifecycle from ingestion verification to explainable AI diagnostics:

```bash
# Execute complete end-to-end pipeline across all stages
python3 run_pipeline.py --stage all

# Run specific analytical stages
python3 run_pipeline.py --stage preflight       # Live GTFS-RT feed and static archive validation
python3 run_pipeline.py --stage exogenous       # IMGW weather & OSMnx road network topology
python3 run_pipeline.py --stage fusion          # Spatiotemporal fusion into feature_mart.parquet
python3 run_pipeline.py --stage benchmark       # TWFE panel regression vs LightGBM vs CatBoost
python3 run_pipeline.py --stage xai             # TreeSHAP, ALE curves, buffering & DiCE counterfactuals

# Fast iteration with sample size
python3 run_pipeline.py --stage benchmark --sample-size 50000
```

---

## Empirical Benchmark Results (Hold-Out Validation)

Evaluated under strict **Purged Temporal Block Splitting** with 30-minute embargo buffers to prevent autoregressive data leakage ($N = 100,000$ sample, $N_{\text{val}} = 18,970$):

| Model Architecture | Out-of-Sample MAE | Out-of-Sample RMSE | Median AE | WAPE (%) | Validation $R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Econometric TWFE Baseline** (Route & Hour FE) | 29.44s | 46.06s | 21.74s | 102.77% | -0.0365 |
| **LightGBM Regressor** (Non-linear Gradient Boosting) | **28.21s** | **44.91s** | **20.42s** | **98.48%** | **+0.0142** |
| **CatBoost Regressor** (Symmetric Oblivious Trees) | **28.23s** | **44.96s** | **20.51s** | **98.54%** | **+0.0122** |

*Econometric Elasticities*: In TWFE linear specification, signal density is statistically significant ($\beta = +1.19\text{s}/\text{signal}, p < 0.0001, \varepsilon = +0.5032$), and dedicated right-of-way provides $-1.85$s direct baseline relief. Gradient boosting achieves notable predictive uplift by capturing non-linear threshold bifurcations.

---

## Explainable AI (xAI) Diagnostics Summary

1. **TreeSHAP Global Delay Decomposition**:
   - Operational Delay Propagation (`prev_stop_delay`): **29.70%** relative contribution.
   - Temporal Rhythm (`hour_of_day`, `peak_hour`): **22.49%**.
   - Dispatch Regularity (`headway_deviation`): **20.70%**.
   - Fleet Type (`is_tram`): **7.86%**.
   - Infrastructure Geometry & Bottlenecks: **4.93%**.
2. **ALE Non-Linear Tipping Points**:
   - Upstream delay exhibits saturation at $+13.0$s segment delay delta once prior delay exceeds $300$s.
   - Route progression displays acute congestion inflections at **74%** and **94%** of total trip length.
   - Dedicated Right-of-Way provides a discontinuous step-function insulating barrier ($-1.78$s).
3. **Infrastructure Buffering & Interaction Effects**:
   - Dedicated transit corridors mitigate headway deviation shocks by **+3.59s (39.0% reduction in delay escalation)**.
4. **Counterfactual Remedies (DiCE)**:
   - Dynamic Headway Regularization alone recovers **+2.99s per segment**, restoring **100%** of severe delay cases back to on-time status ($\Delta t \le 120$s).

---

## Test Suite Execution

Run the complete 54-test suite validating all ingestion, storage, econometric, gradient boosting, xAI, and pipeline runner components:

```bash
# Run all unit and integration tests
.venv/bin/python3 -m unittest discover tests

# Run specific E2E pipeline integration suite
.venv/bin/python3 -m unittest tests/test_pipeline_e2e.py
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
Data provided via [zbiorkom.live](https://zbiorkom.live) and Zarząd Transportu Miejskiego w Warszawie (ZTM).

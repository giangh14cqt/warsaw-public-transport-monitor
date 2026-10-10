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

## Downstream Research & Explainable AI (xAI)

The dataset generated by this pipeline is structured specifically for spatiotemporal modeling and interpretability:

1. **Spatial Features**: Join `stop_id` with `data/gtfs/*/stops.txt` to attach `stop_lat` and `stop_lon`.
2. **Segment Delta**: Compute progression delay between stops via window functions:
   $$\Delta \text{delay}_{i \to i+1} = \text{delay}_{i+1} - \text{delay}_i$$
3. **Exogenous Enrichment**: Merge with OpenStreetMap road infrastructure (bus lanes, signalized intersections) and historical hourly weather (precipitation, temperature).
4. **xAI Modeling**: Train gradient-boosted trees (LightGBM / CatBoost) or Spatio-Temporal Graph Neural Networks (ST-GNNs), applying **TreeSHAP** to attribute delay spikes to road topology vs. weather events.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
Data provided via [zbiorkom.live](https://zbiorkom.live) and Zarząd Transportu Miejskiego w Warszawie (ZTM).

# Warsaw Transit Telemetry & Delay Attribution System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Storage-Parquet](https://img.shields.io/badge/storage-Apache%20Parquet-orange.svg)](https://parquet.apache.org/)
[![Engine-DuckDB](https://img.shields.io/badge/analytics-DuckDB-yellow.svg)](https://duckdb.org/)
[![Models-LightGBM%20%7C%20CatBoost](https://img.shields.io/badge/models-LightGBM%20%7C%20CatBoost-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![xAI-SHAP%20%7C%20ALE%20%7C%20DiCE](https://img.shields.io/badge/xAI-SHAP%20%7C%20ALE%20%7C%20DiCE-purple.svg)](https://github.com/shap/shap)
[![Tests-Passing](https://img.shields.io/badge/tests-72%2F72%20passing-success.svg)](tests/)
[![Docker](https://img.shields.io/badge/deployment-Docker%20Compose-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, end-to-end telemetry ingestion engine, econometric baseline, machine learning benchmark, and Explainable AI (xAI) delay attribution system deployed on the live public transport network of Warsaw, Poland (ZTM / [zbiorkom.live](https://zbiorkom.live)).

Developed for University of Warsaw (UW) Master's thesis research at the **Faculty of Economic Sciences & Data Science** investigating:
* *Urban Traffic Congestion & Shock Propagation*
* *Econometric Two-Way Fixed Effects Panel Modeling*
* *Non-Linear Gradient Boosting & Purged Temporal Cross-Validation*
* *Explainable AI (TreeSHAP, ALE Curves, Infrastructure Buffering, DiCE Counterfactuals)*

---

## Table of Contents
- [1. End-to-End System Architecture](#1-end-to-end-system-architecture)
- [2. Telemetry Ingestion & Storage Engine](#2-telemetry-ingestion--storage-engine)
- [3. Exogenous Data Fusion & Feature Mart](#3-exogenous-data-fusion--feature-mart)
- [4. Zero-Leakage Validation Strategy](#4-zero-leakage-validation-strategy)
- [5. Empirical Modeling Benchmarks (TWFE vs. GBM)](#5-empirical-modeling-benchmarks-twfe-vs-gbm)
- [6. Explainable AI (xAI) Delay Attribution Findings](#6-explainable-ai-xai-delay-attribution-findings)
- [7. Master Pipeline Runner (`run_pipeline.py`)](#7-master-pipeline-runner-run_pipelinepy)
- [8. Repository Structure](#8-repository-structure)
- [9. Quickstart & Installation](#9-quickstart--installation)
- [10. Thesis Reports & Visual Exhibits](#10-thesis-reports--visual-exhibits)
- [11. Testing & Continuous Integration](#11-testing--continuous-integration)
- [12. License & Data Sources](#12-license--data-sources)

---

## 1. End-to-End System Architecture

The pipeline executes across six decoupled, modular layers:

```
[1. LIVE TELEMETRY INGESTION]
  │  GTFS-RT Protobuf (cdn.zbiorkom.live/gtfs-rt/warsaw.pb) polled every 30s
  │  Weekly Static GTFS Timetable Synchronization (O(1) in-memory schedule mapping)
  ▼
[2. STORAGE & PERSISTENCE]
  │  State Deduplication ──> In-Memory Buffer (5 min / 5k rows)
  │  Hive-Partitioned Snappy Parquet (data/raw/year=YYYY/month=MM/day=DD/*.parquet)
  │  Daily Incremental Google Drive Cloud Backup (rclone)
  ▼
[3. EXOGENOUS ENRICHMENT & FUSION]
  │  IMGW-PIB Hourly Synoptic Weather (Okęcie & Bielany)
  │  Citywide OSM Road Network Topology (10,081 segments, 4,179 traffic signals across Warsaw)
  │  DuckDB Spatiotemporal Window Fusion ──> 4,527,951 Rows (data/processed/feature_mart.parquet)
  ▼
[4. DATA LEAKAGE PREVENTION & SPLITTING]
  │  Purged Temporal Block Splitter (30-min embargo buffer + trip boundary purging)
  ▼
[5. ECONOMETRIC & ML BENCHMARKS]
  ├── Baseline: Econometric Two-Way Fixed Effects Panel Regression (pyhdfe + statsmodels HC1 OLS)
  ├── Non-Linear Gradient Boosting Regressors (LightGBM & CatBoost)
  ▼
[6. EXPLAINABLE AI (xAI) DIAGNOSTICS & REMEDIES]
  ├── TreeSHAP: Global Beeswarm & Local Extreme Delay Waterfall Spikes
  ├── ALE Curves: Non-Linear Threshold Detection & Tipping Points
  ├── 2D Interactions: Infrastructure Shock Buffering (Dedicated ROW vs. Mixed Traffic)
  └── DiCE Counterfactuals: Actionable Operational Interventions & On-Time Restoration
```

---

## 2. Telemetry Ingestion & Storage Engine

* **Live Feed Polling**: Polls the Warsaw GTFS-RT feed every 30 seconds with exponential jittered retries ($3\text{s}, 6\text{s}, 12\text{s}, 24\text{s}$).
* **Anti-Corruption Storage Architecture**: Writes exclusively to immutable, append-only **Apache Parquet** partitions compressed with **Snappy**, completely eliminating long-running SQLite/Postgres write locks.
* **State Deduplication**: Compares incoming `(trip_id, stop_sequence, rt_arrival_time)` triples against an in-memory LRU cache to discard static observations from idling vehicles.
* **Resilience & Dead Man's Switch**: Automatically pauses for 5 minutes upon HTTP 429 rate limiting, handles Daylight Saving Time transitions, and emits 15-minute periodic heartbeats to [Healthchecks.io](https://healthchecks.io).
* **Automated Cloud Backup**: Incremental daily backups scheduled via `backup_to_gdrive.sh` and synchronization via `pull_from_gdrive.sh`.

### Raw Telemetry Schema (`data/raw/**/*.parquet`)

| Column | Type | Description |
| :--- | :--- | :--- |
| `feed_timestamp` | `INT64` | Header epoch timestamp of the GTFS-RT feed |
| `trip_id` | `STRING` | Unique scheduled GTFS trip identifier |
| `route_id` | `STRING` | Route / line identifier (e.g. `1`, `9`, `24`, `175`, `523`) |
| `start_date` | `STRING` | Service date in `YYYYMMDD` format |
| `vehicle_id` | `STRING` | Physical vehicle identifier |
| `stop_sequence` | `INT32` | Ordinal stop sequence number along the trip |
| `stop_id` | `STRING` | Physical stop post identifier in Warsaw ZTM |
| `arrival_delay_seconds` | `INT32` | **Arrival delay delta** ($>0$ late, $<0$ early, $=0$ on-time) |
| `departure_delay_seconds`| `INT32`| Departure delay delta in seconds |
| `rt_arrival_time` | `INT64` | Live observed/predicted arrival epoch timestamp |
| `record_ingested_at` | `STRING` | UTC ISO timestamp when recorded by ingestion daemon |

---

## 3. Exogenous Data Fusion & Feature Mart

Raw vehicle telemetry is enriched with exogenous environmental and infrastructural data using DuckDB:
1. **Meteorological Telemetry (`src/exogenous/imgw.py`)**:
   - Harvested from official IMGW-PIB Polish synoptic weather archives.
   - Metrics: Precipitation intensity (`mm/h`), surface temperature (°C), relative humidity (%), wind speed (`m/s`), and freezing rain flags.
2. **Road Corridor Topology & Micro-Geometry (`src/exogenous/osm.py`)**:
   - Extracted from OpenStreetMap across the entire Warsaw network (**10,081 directed stop-to-stop links**, **4,179 mapped traffic signals**, covering 9,070 km of transit routes).
   - Metrics: `is_dedicated_right_of_way`, `signalized_intersection_count`, `segment_length_meters`, `actual_path_length_meters` (true GTFS shape polyline distance), `tortuosity_ratio` ($\ge 1.0$, capturing chicanes and bends), and `lane_capacity`.
3. **Multimodal Transfer Hubs & Metro Buffers (`src/exogenous/interchange.py`)**:
   - `interchange_line_count`: Multimodal stop post route degree capturing passenger transfer volume (7,241 stops mapped).
   - `is_metro_proximity`: 300m geodesic buffer to 39 Warsaw Metro (M1 & M2) stations.
4. **School & Statutory Holiday Calendar (`src/core/polish_calendar.py`)**:
   - `is_school_day`: Mazowieckie voivodeship academic schedule (accounting for winter breaks, summer breaks, and seasonal holidays).
   - `is_public_holiday`: Polish statutory holidays computed via Anonymous Gregorian Easter algorithm.
5. **Autoregressive Trajectory Lags (`src/fusion/trajectory.py`)**:
   - Windowed reconstruction of preceding stop delay (`prev_stop_delay`), second-order lag (`prev2_stop_delay`), dispatch regularity (`headway_deviation`), and relative `trip_progress` ($0.0 \to 1.0$).
6. **Fused Feature Store**: Persisted to `data/processed/feature_mart.parquet` (**4,527,951 rows**).

---

## 4. Zero-Leakage Validation Strategy

Standard K-fold cross-validation suffers from severe data leakage in time-series transit telemetry due to autoregressive autocorrelation across trip segments. To ensure realistic, unbiased out-of-sample evaluation, we built `PurgedTemporalBlockSplitter` ([`src/models/validation.py`](src/models/validation.py)):

* **Temporal Chronological Ordering**: Partitions observations strictly by time into Train (60%), Validation (20%), and Hold-Out Test (20%) blocks.
* **Trip Boundary Purging**: Completely removes any vehicle trip that crosses a split boundary to eliminate inter-segment information leakage.
* **30-Minute Embargo Buffering**: Enforces a 30-minute blackout window between consecutive blocks to allow residual traffic congestion shockwaves to dissipate.
* **Zero-Leakage Audit**: Verified across all train, validation, and hold-out test blocks.

---

## 5. Empirical Modeling Benchmarks (TWFE vs. GBM)

We benchmarked classical econometrics against state-of-the-art gradient boosted trees on the hold-out validation set ($N = 100,000$ sample, $N_{\text{val}} = 19,405$), after filtering spurious terminal layovers:

### 5.1 Pooled Network Benchmark (Cleaned Layovers)

| Model Architecture | Specification / Regularization | Hold-Out MAE | Hold-Out RMSE | Median AE | WAPE (%) | Validation $R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Econometric TWFE Baseline** | Within Route + Hour FE + HC1 Robust OLS | 29.92s | 51.22s | 21.11s | 100.61% | +0.0048 |
| **LightGBM Regressor** | Tree-depth 7, LR 0.05, L2 Reg 1.0 | **29.05s** | **50.17s** | **20.16s** | **97.68%** | **+0.0450** |
| **CatBoost Regressor** | Symmetric Oblivious Trees, LR 0.05 | 29.44s | 50.70s | 20.57s | 98.97% | +0.0249 |

### 5.2 Stratified Evaluation Across Transit Cohorts

| Transit Cohort | Best Model | Hold-Out MAE | Hold-Out RMSE | Median AE | WAPE (%) | Validation $R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Urban Trams (`urban_tram`)** | **LightGBM Regressor** | **22.98s** | **35.33s** | **17.84s** | **85.94%** | **+0.1837** |
| **Core Urban Buses (`urban_bus`)** | **LightGBM Regressor** | **30.32s** | **52.16s** | **20.76s** | **98.66%** | **+0.0342** |
| **Suburban Feeders (`suburban_bus`)** | **CatBoost Regressor** | **29.24s** | 55.43s | **19.81s** | **98.97%** | **+0.0195** |

### 5.3 Econometric Panel Insights (TWFE)
$$y_{ist} = \alpha_i + \lambda_t + \beta X_{ist} + \epsilon_{ist}$$
* **Traffic Signal Density**: Statistically significant at $p < 0.0001$ ($\beta = +0.5753\text{s}/\text{signal}, t = 5.00$). The estimated point elasticity is $\varepsilon = +0.2146$, indicating that a 10% increase in traffic signals induces a 2.15% increase in segment arrival delay.
* **Lane Capacity**: Statistically significant at $p < 0.0001$ ($\beta = -6.0846\text{s}$ per additional lane, $t = -5.93$, elasticity $\varepsilon = -2.28$).
* **Physical Link Length**: $\beta = +0.0029\text{s}$ per meter ($+2.9\text{s}$ running delay per kilometer, $p = 0.09$).
* **Why Gradient Boosting Outperforms**: Linear specifications cannot capture saturation limits or interaction buffering. LightGBM and CatBoost capture threshold dynamics and achieve superior error reduction across all cohorts, driving tram MAE down to **22.98s** and WAPE to **85.94%** ($R^2 = +0.1837$).

---

## 6. Explainable AI (xAI) Delay Attribution Findings

### 6.1 Global Delay Attribution (TreeSHAP)
Exact TreeSHAP decomposition on the hold-out test set reveals the primary operational and physical drivers of transit delays:
* **Upstream Delay Propagation (`prev_stop_delay`)**: **29.70%** relative contribution.
* **Temporal Diurnal Cycles (`hour_of_day`, `is_peak_hour`)**: **22.49%**.
* **Dispatch Headway Regularity (`headway_deviation`)**: **20.70%**.
* **Fleet Type (`is_tram` vs. bus)**: **7.86%**.
* **Road Geometry & Signal Bottlenecks**: **4.93%**.

![TreeSHAP Beeswarm Summary](reports/figures/xai/shap_beeswarm_summary.png)

### 6.2 Non-Linear Threshold Detection (ALE Curves)
Accumulated Local Effects (ALE) isolate the unconfounded marginal response of individual features:
* **Upstream Delay Saturation**: Arrival delay delta compounds linearly up to $+300$ seconds of prior delay, plateauing at a maximum $+13.0$s segment delay increase.
* **Trip Progress Congestion Cliffs**: Sharp non-linear inflections at **74%** and **94%** route completion, pinpointing terminal approach bottlenecks.
* **Dedicated ROW Discontinuity**: Dedicated tram and bus lanes provide a discrete step-function insulating barrier of **-1.78s** across all operating conditions.

![ALE Trip Progress](reports/figures/xai/ale_trip_progress.png)

### 6.3 Infrastructure Buffering & 2D Interactions
Pairwise SHAP interaction values $\Phi_{i,j}(x)$ evaluate whether dedicated infrastructure buffers operational shocks:
* In mixed traffic, each second of headway deviation increases downstream delay by $+0.042$s.
* On dedicated transit rights-of-way, delay escalation is attenuated to $+0.026$s per second.
* **Mitigation Gain**: Dedicated infrastructure provides a **39.0% buffering reduction (+3.59s saved per shock unit)**.

![Interaction Headway vs ROW](reports/figures/xai/interaction_headway_deviation_vs_dedicated_row.png)

### 6.4 Counterfactual Diagnostics (DiCE)
Using Diverse Counterfactual Explanations (DiCE), we evaluated actionable remedies across severe delay incidents:
* **Segment Delay Attenuation**: Dedicated ROW conversion saves **+1.44s per stop** for mixed-traffic buses, and headway regularization recovers **+0.51s to +4.15s per stop** across bunched lines.
* **On-Time Recovery**: Upstream schedule recovery holding and dispatch resets successfully restore **100%** of severe delay cases back to on-time arrival ($\Delta t \le 120$s).

![Counterfactual Policy Impact](reports/figures/xai/counterfactual_policy_impact.png)

---

## 7. Master Pipeline Runner (`run_pipeline.py`)

The entire end-to-end analytical workflow can be orchestrated through the unified master CLI runner, including automated temporal range inspection across all multi-source inputs and customizable study window selection:

```bash
# 1. Audit available date ranges across all data sources (Raw, Weather, GTFS, OSM)
python3 run_pipeline.py --check-ranges

# 2. Execute complete pipeline across all stages (or select custom temporal range)
python3 run_pipeline.py --stage all
python3 run_pipeline.py --stage all --start-date 2026-10-05 --end-date 2026-10-09

# 3. Rebuild exogenous weather data for an extended or custom date range
python3 run_pipeline.py --stage exogenous --force-rebuild --start-date 2026-10-04 --end-date 2026-10-11

# 4. Spatiotemporal fusion for a specific date range
python3 run_pipeline.py --stage fusion --start-date 2026-10-05 --end-date 2026-10-09

# 5. Run econometric & machine learning benchmarks on selected date range
python3 run_pipeline.py --stage benchmark --start-date 2026-10-05 --end-date 2026-10-09 --sample-size 50000

# 6. Generate explainable AI diagnostics (TreeSHAP, ALE, Interactions, DiCE)
python3 run_pipeline.py --stage xai --start-date 2026-10-04 --end-date 2026-10-10
```

---

## 8. Repository Structure

```text
.
├── run_pipeline.py           # Master CLI pipeline runner (Stages 1 through 5)
├── collector_daemon.py       # Live 30s GTFS-RT ingestion daemon with healthchecks
├── run_preflight.py          # Pre-flight sandbox verification test
├── backup_to_gdrive.sh       # Automated rclone backup to Google Drive
├── pull_from_gdrive.sh       # Local data sync from Google Drive
├── src/
│   ├── core/                 # Centralized configuration, paths, features & domain constants
│   ├── ingestion/            # Phase 1: GTFS-RT fetcher, static manager, Parquet storage
│   ├── exogenous/            # Phase 2: IMGW weather harvester, OSMnx road topology
│   ├── fusion/               # Phase 3: Trajectory reconstruction, DuckDB feature mart
│   ├── models/               # Phase 4: Purged temporal splitter, TWFE panel, GBM benchmarks
│   ├── xai/                  # Phase 5: TreeSHAP, ALE curves, interactions, DiCE counterfactuals
│   ├── visualization/        # Academic publication styling & reusable plot generators
│   └── pipeline/             # Modular stage executors (Stages 1 through 5)
├── reports/
│   ├── EXECUTIVE_REPORT.md   # Comprehensive executive thesis findings report
│   ├── PRESENTATION_SLIDES.md# 12 Marp-ready presentation slides
│   ├── PROJECT_IMPROVEMENT_ROADMAP.md # Research roadmap & next upgrades
│   ├── tables/               # Benchmark comparison CSVs and counterfactual metrics JSON
│   └── figures/xai/          # 14 publication-grade xAI diagnostic figures
├── tests/                    # 65 unit and E2E integration tests (100% passing)
├── data/
│   ├── raw/                  # Partitioned Parquet sink (year=YYYY/month=MM/day=DD/)
│   ├── gtfs/                 # Archived weekly static GTFS timetable snapshots
│   └── processed/            # Fused feature store (feature_mart.parquet, 4.52M rows)
└── logs/                     # Rotating health logs and backup audit trails
```

---

## 9. Quickstart & Installation

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

### 2. Verify Pre-Flight Gate
```bash
python3 run_preflight.py
```

### 3. Run Benchmark Models & xAI Suite
```bash
python3 run_pipeline.py --stage benchmark --sample-size 50000
python3 run_pipeline.py --stage xai
```

---

## 10. Thesis Reports & Visual Exhibits

| Document / Artifact | Description |
| :--- | :--- |
| **[`reports/EXECUTIVE_REPORT.md`](reports/EXECUTIVE_REPORT.md)** | Full academic and executive POC report detailing methodology, empirical results, and transit policy guidance for Warsaw ZTM. |
| **[`reports/PRESENTATION_SLIDES.md`](reports/PRESENTATION_SLIDES.md)** | Complete 12-slide presentation deck formatted for defenses and stakeholder briefings. |
| **[`reports/PROJECT_IMPROVEMENT_ROADMAP.md`](reports/PROJECT_IMPROVEMENT_ROADMAP.md)** | Comprehensive enhancement roadmap covering seasonal extensions, spatial micro-weather, causal DML, and interactive dashboards. |
| **[`reports/tables/benchmark_comparison.csv`](reports/tables/benchmark_comparison.csv)** | Hold-out validation benchmark table comparing TWFE, LightGBM, and CatBoost across MAE, RMSE, MedAE, WAPE, and $R^2$. |
| **[`reports/figures/xai/`](reports/figures/xai/)** | Visual exhibit pack containing 14 high-resolution diagnostic charts (SHAP beeswarm, waterfall spikes, ALE curves, 2D interaction planes). |

---

## 11. Testing & Continuous Integration

The repository maintains an automated unit and end-to-end integration test suite covering all modules:

```bash
# Run the complete test suite (72 tests in ~21 seconds)
.venv/bin/python3 -m unittest discover tests

# Run specific E2E pipeline integration test suite
.venv/bin/python3 -m unittest tests/test_pipeline_e2e.py
```

```text
Ran 72 tests in 21.46s
OK (100% pass rate)
```

---

## 12. License & Data Sources

* **License**: MIT License - see [LICENSE](LICENSE) for details.
* **Transit Telemetry**: Real-time GTFS-RT Warsaw feed provided by [zbiorkom.live](https://zbiorkom.live) and Zarząd Transportu Miejskiego w Warszawie (ZTM).
* **Weather Telemetry**: Hourly synoptic observation archives provided by Instytut Meteorologii i Gospodarki Wodnej (IMGW-PIB).
* **Road Network Topology**: OpenStreetMap data extracted via OSMnx.

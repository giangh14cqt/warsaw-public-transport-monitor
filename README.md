# Warsaw Transit Telemetry & Delay Attribution System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Storage-Parquet](https://img.shields.io/badge/storage-Apache%20Parquet-orange.svg)](https://parquet.apache.org/)
[![Engine-DuckDB](https://img.shields.io/badge/analytics-DuckDB-yellow.svg)](https://duckdb.org/)
[![Models-LightGBM%20%7C%20CatBoost](https://img.shields.io/badge/models-LightGBM%20%7C%20CatBoost-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![xAI-SHAP%20%7C%20ALE%20%7C%20DiCE](https://img.shields.io/badge/xAI-SHAP%20%7C%20ALE%20%7C%20DiCE-purple.svg)](https://github.com/shap/shap)
[![Tests-Passing](https://img.shields.io/badge/tests-72%2F72%20passing-success.svg)](tests/)
[![Docker](https://img.shields.io/badge/deployment-Docker%20Compose-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete, end-to-end data pipeline, econometric baseline, machine learning benchmark, and Explainable AI (xAI) delay attribution system for the public transit network of Warsaw, Poland (ZTM / [zbiorkom.live](https://zbiorkom.live)).

Developed for Master's thesis research at the **Faculty of Economic Sciences & Data Science, University of Warsaw (UW)**, this project studies:
* *Urban Traffic Congestion & Delay Propagation*
* *Econometric Two-Way Fixed Effects (TWFE) Panel Regression*
* *Non-Linear Gradient Boosting (LightGBM & CatBoost) with Purged Temporal Cross-Validation*
* *Explainable AI (TreeSHAP, ALE Curves, Infrastructure Shock Buffering, and DiCE Counterfactuals)*

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

The pipeline processes transit telemetry through six modular, independent stages:

```
[1. LIVE TELEMETRY INGESTION]
  │  Polls Warsaw GTFS-RT Protobuf (cdn.zbiorkom.live/gtfs-rt/warsaw.pb) every 30s
  │  Synchronizes with weekly static GTFS timetable schedules in memory
  ▼
[2. STORAGE & PERSISTENCE]
  │  Deduplicates vehicle states using an in-memory buffer (5 min / 5k records)
  │  Writes Hive-partitioned Snappy Parquet (data/raw/year=YYYY/month=MM/day=DD/*.parquet)
  │  Runs daily incremental backups to Google Drive via rclone
  ▼
[3. EXOGENOUS ENRICHMENT & FUSION]
  │  IMGW-PIB hourly synoptic weather telemetry (Warszawa-Okęcie & Bielany)
  │  Citywide OpenStreetMap road network (10,081 stop-to-stop links, 4,179 traffic signals)
  │  DuckDB spatiotemporal join produces 4,527,951 rows (data/processed/feature_mart.parquet)
  ▼
[4. DATA LEAKAGE PREVENTION & SPLITTING]
  │  Purged Temporal Block Splitter (30-minute embargo buffer + trip boundary purging)
  ▼
[5. ECONOMETRIC & ML BENCHMARKS]
  ├── Baseline: Econometric Two-Way Fixed Effects Panel Regression (pyhdfe + statsmodels HC1 OLS)
  ├── Non-Linear Gradient Boosting Regressors (LightGBM & CatBoost)
  ▼
[6. EXPLAINABLE AI (xAI) DIAGNOSTICS & REMEDIES]
  ├── TreeSHAP: Global beeswarm importance and local waterfall delay breakdown
  ├── ALE Curves: Non-linear threshold detection and congestion tipping points
  ├── 2D Interactions: Infrastructure shock buffering (Dedicated ROW vs. Mixed Traffic)
  └── DiCE Counterfactuals: Actionable operational interventions to restore on-time arrivals
```

---

## 2. Telemetry Ingestion & Storage Engine

* **Live Feed Polling**: Polls the Warsaw GTFS-RT feed every 30 seconds with exponential retry delays ($3\text{s}, 6\text{s}, 12\text{s}, 24\text{s}$).
* **Reliable Parquet Storage**: Writes directly to append-only **Apache Parquet** files compressed with **Snappy**. This avoids database write locks and keeps raw historical data safe and immutable.
* **State Deduplication**: Checks incoming `(trip_id, stop_sequence, rt_arrival_time)` triples against an in-memory LRU cache to ignore unchanged observations when vehicles are stopped.
* **Fault Tolerance & Health Checks**: Pauses automatically for 5 minutes when hitting HTTP 429 rate limits, handles Daylight Saving Time shifts smoothly, and sends a heartbeat ping every 15 minutes to [Healthchecks.io](https://healthchecks.io).
* **Automated Cloud Backup**: Backs up daily partitions to Google Drive via `backup_to_gdrive.sh` and syncs them locally using `pull_from_gdrive.sh`.

### Raw Telemetry Schema (`data/raw/**/*.parquet`)

| Column | Type | Description |
| :--- | :--- | :--- |
| `feed_timestamp` | `INT64` | Header epoch timestamp of the GTFS-RT feed |
| `trip_id` | `STRING` | Unique scheduled GTFS trip identifier |
| `route_id` | `STRING` | Transit line identifier (e.g., `1`, `9`, `24`, `175`, `523`) |
| `start_date` | `STRING` | Service date in `YYYYMMDD` format |
| `vehicle_id` | `STRING` | Physical vehicle identifier |
| `stop_sequence` | `INT32` | Stop sequence index along the scheduled trip |
| `stop_id` | `STRING` | Physical stop identifier in the Warsaw ZTM network |
| `arrival_delay_seconds` | `INT32` | **Arrival delay delta** ($>0$ late, $<0$ early, $=0$ on time) |
| `departure_delay_seconds`| `INT32`| Departure delay delta in seconds |
| `rt_arrival_time` | `INT64` | Observed or predicted arrival epoch timestamp |
| `record_ingested_at` | `STRING` | UTC ISO timestamp when recorded by the ingestion daemon |

---

## 3. Exogenous Data Fusion & Feature Mart

Using DuckDB, the pipeline enriches raw vehicle records with environmental, calendar, and infrastructure data:

1. **Meteorological Telemetry (`src/exogenous/imgw.py`)**:
   - Ingests official IMGW-PIB hourly synoptic weather observations from Warsaw stations.
   - Variables: Precipitation intensity (`mm/h`), surface temperature (°C), relative humidity (%), wind speed (`m/s`), and freezing rain flags.
2. **Road Network Topology & Geometry (`src/exogenous/osm.py`)**:
   - Extracted from OpenStreetMap across the entire Warsaw network (**10,081 directed stop-to-stop links**, **4,179 mapped traffic signals**, covering 9,070 km of transit routes).
   - Variables: `is_dedicated_right_of_way`, `signalized_intersection_count`, `segment_length_meters`, `actual_path_length_meters` (true GTFS shape polyline distance), `tortuosity_ratio` ($\ge 1.0$, capturing bends and curves), and `lane_capacity`.
3. **Multimodal Transfer Hubs & Metro Access (`src/exogenous/interchange.py`)**:
   - `interchange_line_count`: Total lines serving each stop post (7,241 stops mapped) to proxy transfer passenger boarding pressure.
   - `is_metro_proximity`: 300-meter geodesic buffer around all 39 Warsaw Metro (M1 & M2) stations.
4. **School & Public Holiday Calendar (`src/core/polish_calendar.py`)**:
   - `is_school_day`: Mazowieckie voivodeship academic calendar (accounting for winter breaks, summer breaks, and exam periods).
   - `is_public_holiday`: Polish statutory holidays computed with the Anonymous Gregorian Easter algorithm.
5. **Autoregressive Trajectory Lags (`src/fusion/trajectory.py`)**:
   - Lagged features tracking delays at previous stops (`prev_stop_delay`, `prev2_stop_delay`), dispatch spacing (`headway_deviation`), and relative `trip_progress` ($0.0 \to 1.0$).
6. **Unified Feature Store**: Saved to `data/processed/feature_mart.parquet` (**4,527,951 rows**).

---

## 4. Zero-Leakage Validation Strategy

Standard K-fold cross-validation causes severe data leakage when used on transit time-series data. Because consecutive stops on the same trip are closely linked in time, random splits accidentally allow future segment information to leak into training models.

To ensure realistic, unbiased evaluation, we implemented `PurgedTemporalBlockSplitter` ([`src/models/validation.py`](src/models/validation.py)):

* **Strict Chronological Ordering**: Partitions observations forward in time into Train (60%), Validation (20%), and Hold-Out Test (20%) blocks.
* **Trip Boundary Purging**: Completely drops any vehicle trip that crosses a split boundary, preventing trajectory information from leaking across folds.
* **30-Minute Embargo Window**: Enforces a 30-minute blackout buffer between consecutive splits so lingering traffic congestion shockwaves can dissipate.
* **Zero-Leakage Audit**: Verified and tested across all train, validation, and hold-out test splits.

---

## 5. Empirical Modeling Benchmarks (TWFE vs. GBM)

We evaluated classical econometric panel models against non-linear gradient boosted trees on the hold-out validation set ($N = 100,000$ sample, $N_{\text{val}} = 19,405$), after removing terminal layover rest breaks:

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
* **Traffic Signal Density**: Statistically significant at $p < 0.0001$ ($\beta = +0.5753\text{s}/\text{signal}, t = 5.00$). The estimated point elasticity is $\varepsilon = +0.2146$, meaning a 10% increase in traffic signal count increases segment delay by 2.15%.
* **Lane Capacity**: Statistically significant at $p < 0.0001$ ($\beta = -6.0846\text{s}$ per additional lane, $t = -5.93$, elasticity $\varepsilon = -2.28$).
* **Physical Link Length**: $\beta = +0.0029\text{s}$ per meter ($+2.9\text{s}$ running delay per kilometer, $p = 0.09$).
* **Why Gradient Boosting Outperforms Linear TWFE**: Linear panel regression assumes effects are purely additive. In contrast, transit delays exhibit threshold tipping points and non-linear interactions. LightGBM and CatBoost capture these non-linearities, reducing Urban Tram MAE to **22.98s** and WAPE to **85.94%** ($R^2 = +0.1837$).

---

## 6. Explainable AI (xAI) Delay Attribution Findings

### 6.1 Global Delay Attribution (TreeSHAP)
Exact TreeSHAP decomposition on the hold-out test set reveals the primary operational and infrastructure drivers of transit delays:
* **Upstream Delay Propagation (`prev_stop_delay`)**: **29.70%** relative contribution.
* **Time of Day & Rush Hour Cycles (`hour_of_day`, `is_peak_hour`)**: **22.49%**.
* **Dispatch Headway Regularity (`headway_deviation`)**: **20.70%**.
* **Fleet Type (`is_tram` vs. bus)**: **7.86%**.
* **Road Geometry & Signal Bottlenecks**: **4.93%**.

![TreeSHAP Beeswarm Summary](reports/figures/xai/shap_beeswarm_summary.png)

### 6.2 Non-Linear Threshold Detection (ALE Curves)
Accumulated Local Effects (ALE) isolate the pure marginal effect of individual features without bias from correlated variables:
* **Upstream Delay Saturation**: Delays grow linearly until upstream delays exceed $+300$ seconds, plateauing at a maximum addition of $+13.0$ seconds to segment delay.
* **Trip Progress Bottlenecks**: Delays jump sharply at **74%** and **94%** route completion, pinpointing congestion near central transfer hubs and terminal approaches.
* **Dedicated Right-of-Way Effect**: Segregated tram tracks and bus lanes act as a structural barrier, providing a direct step-function reduction of **-1.78s** in segment delay across all operating conditions.

![ALE Trip Progress](reports/figures/xai/ale_trip_progress.png)

### 6.3 Infrastructure Buffering & 2D Interactions
Pairwise SHAP interaction values $\Phi_{i,j}(x)$ show whether dedicated infrastructure protects vehicles from operational shocks:
* In mixed traffic, each extra second of headway deviation increases downstream delay by $+0.042$s.
* On dedicated transit rights-of-way, this delay escalation is reduced to $+0.026$s per second.
* **Buffering Benefit**: Dedicated infrastructure absorbs **39.0% of the shock (+3.59s saved per shock unit)**.

![Interaction Headway vs ROW](reports/figures/xai/interaction_headway_deviation_vs_dedicated_row.png)

### 6.4 Counterfactual Diagnostics (DiCE)
Using Diverse Counterfactual Explanations (DiCE), we tested how specific operational remedies would change severe delay incidents:
* **Segment Delay Reduction**: Converting mixed-traffic lanes to dedicated right-of-way saves an average of **+1.44s per stop** for buses. Regularizing vehicle dispatch headways saves **+0.51s to +4.15s per stop** on bunched lines.
* **On-Time Schedule Recovery**: For vehicles facing heavy delays ($\Delta t > 300\text{s}$), physical lane improvements prevent further delay accumulation, but cannot erase several minutes of past delay over one short stop. Restoring on-time status ($\le 120\text{s}$) requires upstream schedule recovery holding or dispatch interval resets, which achieve a **100% on-time recovery rate**.

![Counterfactual Policy Impact](reports/figures/xai/counterfactual_policy_impact.png)

---

## 7. Master Pipeline Runner (`run_pipeline.py`)

The unified CLI runner automates the entire end-to-end workflow, including data range inspection across all inputs and customizable study window selection:

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
├── collector_daemon.py       # Live 30s GTFS-RT ingestion daemon with health checks
├── run_preflight.py          # Pre-flight environment and sandbox verification test
├── backup_to_gdrive.sh       # Automated rclone backup to Google Drive
├── pull_from_gdrive.sh       # Local data sync from Google Drive
├── src/
│   ├── core/                 # Configuration, paths, feature registry, and calendar helpers
│   ├── ingestion/            # Phase 1: GTFS-RT poller, static schedule manager, Parquet storage
│   ├── exogenous/            # Phase 2: IMGW weather harvester, OSMnx road network topology
│   ├── fusion/               # Phase 3: Trajectory reconstruction, DuckDB feature mart
│   ├── models/               # Phase 4: Purged temporal splitter, TWFE panel, GBM benchmarks
│   ├── xai/                  # Phase 5: TreeSHAP, ALE curves, interactions, DiCE counterfactuals
│   ├── visualization/        # Academic publication styling & reusable plot generators
│   └── pipeline/             # Modular stage runners (Stages 1 through 5)
├── reports/
│   ├── EXECUTIVE_REPORT.md   # Comprehensive executive thesis findings report
│   ├── PRESENTATION_SLIDES.md# 12 Marp-ready presentation slides
│   ├── PROJECT_IMPROVEMENT_ROADMAP.md # Research roadmap & next upgrades
│   ├── tables/               # Benchmark comparison CSVs and counterfactual metrics JSON
│   └── figures/xai/          # Publication-ready xAI diagnostic figures
├── tests/                    # 72 unit and E2E integration tests (100% passing)
├── data/
│   ├── raw/                  # Partitioned Parquet sink (year=YYYY/month=MM/day=DD/)
│   ├── gtfs/                 # Archived weekly static GTFS timetable snapshots
│   └── processed/            # Fused feature store (feature_mart.parquet, 4.52M rows)
└── logs/                     # Daemon health logs and backup audit trails
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
| **[`reports/EXECUTIVE_REPORT.md`](reports/EXECUTIVE_REPORT.md)** | Full academic and executive report detailing methodology, empirical results, and transit policy guidance for Warsaw ZTM. |
| **[`reports/PRESENTATION_SLIDES.md`](reports/PRESENTATION_SLIDES.md)** | Complete 12-slide presentation deck formatted for thesis defenses and stakeholder briefings. |
| **[`reports/PROJECT_IMPROVEMENT_ROADMAP.md`](reports/PROJECT_IMPROVEMENT_ROADMAP.md)** | Enhancement roadmap covering seasonal telemetry, spatial radar weather, causal machine learning, and interactive dashboards. |
| **[`reports/tables/benchmark_comparison.csv`](reports/tables/benchmark_comparison.csv)** | Hold-out validation benchmark table comparing TWFE, LightGBM, and CatBoost across MAE, RMSE, MedAE, WAPE, and $R^2$. |
| **[`reports/figures/xai/`](reports/figures/xai/)** | Diagnostic visual exhibit pack containing 14 high-resolution charts (SHAP beeswarm, waterfall plots, ALE curves, 2D interaction surfaces). |

---

## 11. Testing & Continuous Integration

The repository includes automated unit and end-to-end integration tests covering all pipeline stages:

```bash
# Run the complete test suite (72 tests in ~20 seconds)
.venv/bin/python3 -m unittest discover tests

# Run the end-to-end pipeline integration test
.venv/bin/python3 -m unittest tests/test_pipeline_e2e.py
```

```text
Ran 72 tests in 19.17s
OK (100% pass rate)
```

---

## 12. License & Data Sources

* **License**: MIT License - see [LICENSE](LICENSE) for details.
* **Transit Telemetry**: Real-time GTFS-RT Warsaw feed provided by [zbiorkom.live](https://zbiorkom.live) and Zarząd Transportu Miejskiego w Warszawie (ZTM).
* **Weather Telemetry**: Hourly synoptic observations provided by Instytut Meteorologii i Gospodarki Wodnej (IMGW-PIB).
* **Road Network Topology**: OpenStreetMap data extracted via OSMnx.

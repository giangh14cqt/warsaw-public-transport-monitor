# Warsaw Transit Telemetry & Delay Attribution System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Storage-Parquet](https://img.shields.io/badge/storage-Apache%20Parquet-orange.svg)](https://parquet.apache.org/)
[![Engine-DuckDB](https://img.shields.io/badge/analytics-DuckDB-yellow.svg)](https://duckdb.org/)
[![Models-LightGBM%20%7C%20CatBoost](https://img.shields.io/badge/models-LightGBM%20%7C%20CatBoost-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![xAI-SHAP%20%7C%20ALE%20%7C%20DiCE](https://img.shields.io/badge/xAI-SHAP%20%7C%20ALE%20%7C%20DiCE-purple.svg)](https://github.com/shap/shap)
[![Tests-Passing](https://img.shields.io/badge/tests-109%2F109%20passing-success.svg)](tests/)
[![Docker](https://img.shields.io/badge/deployment-Docker%20Compose-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete, end-to-end data pipeline, econometric baseline, machine learning benchmark, and Explainable AI (xAI) delay attribution system for the public transit network of Warsaw, Poland (ZTM / [dane.um.warszawa.pl](https://dane.um.warszawa.pl) / [zbiorkom.live](https://zbiorkom.live)).

Developed for Master's thesis research at the **Faculty of Economic Sciences & Data Science, University of Warsaw (UW)**, this project studies:
* *Urban Traffic Congestion & Delay Propagation*
* *Econometric Two-Way Fixed Effects (TWFE) Panel Regression (Weighted Least Squares with IPS Weights)*
* *Non-Linear Gradient Boosting (LightGBM & CatBoost with Huber Loss & Purged Temporal Cross-Validation)*
* *Multi-Dimensional Stratified Feature Sampling & Scalable Streaming Architecture (4+ Months, ~72M Records)*
* *Explainable AI (TreeSHAP, ALE Curves, Infrastructure Shock Buffering, and DiCE Counterfactuals)*

---

## Table of Contents
- [1. End-to-End System Architecture](#1-end-to-end-system-architecture)
- [2. Telemetry Ingestion & Medallion Storage](#2-telemetry-ingestion--medallion-storage)
- [3. Exogenous Data Fusion & Scalability Architecture](#3-exogenous-data-fusion--scalability-architecture)
- [4. Zero-Leakage & Debiased Validation Strategy](#4-zero-leakage--debiased-validation-strategy)
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

The pipeline processes transit telemetry through two production ingestion tracks and six modular stages:

```
[TRACK B: OBSERVED HISTORIAN (PRIMARY)]       [TRACK A: GTFS-RT (BASELINE)]
  │  Warsaw City API (dane.um.warszawa.pl)       │  GTFS-RT Protobuf (api.zbiorkom.live)
  │  10s wall-clock bus & tram grid              │  30s trip updates & alerts
  ▼                                              ▼
[BRONZE: VERBATIM RAW STORE]                   [RAW PARQUET PARTITIONS]
  │  data/historian/bronze/YYYY-MM-DD/           │  data/raw/year=YYYY/month=MM/day=DD/
  ▼                                              ▼
[SILVER: KINEMATIC TRAJECTORIES]                 │
  │  38.2M GPS fixes, velocity/accel             │
  ▼                                              │
[GOLD: RECONSTRUCTED STOP EVENTS]                │
  │  2.64M arrival/departure stop events         │
  │  103,450 complete trips, sub-10s interp      │
  └──────────────────────┬───────────────────────┘
                         │
                         ▼
[EXOGENOUS ENRICHMENT & SCALABLE FUSION]
  ├── IMGW-PIB Synoptic Weather: is_raining, rain_intensity_category, precip_lag_1h, precip_3h_cumulative
  ├── Citywide OSM Road Network: 10,081 stop-to-stop links, 4,179 traffic signals, tortuosity ratios
  ├── Polish School & Holiday Calendar: Mazowieckie academic calendar & statutory holidays
  ├── Transfer Hub Centrality & Metro Access: 300m geodesic buffer to 39 Warsaw Metro M1/M2 stations
  └── Scalable Fusion Engine: DuckDB streaming key-join projection, memory downcasting, out-of-core batching
                         │
                         ▼
[MULTI-DIMENSIONAL STRATIFIED SAMPLING & IPS WEIGHTING]
  ├── 12-cell orthogonal stratification: Weather Regime x Infrastructure ROW x Congestion Peak
  ├── Normalized Inverse Probability of Sampling (IPS) weights (w_i) for debiased population estimation
  └── Purged Temporal Block Splitter: 30-min blackout embargo + trip boundary purging
                         │
                         ▼
[ECONOMETRIC & ML BENCHMARKS]
  ├── Econometric TWFE Baseline: Weighted Least Squares (WLS) + standardized coefficients (β*_k)
  ├── Gradient Boosters: LightGBM & CatBoost with in-fold winsorization [p01, p99] and Huber loss (δ=60s)
  └── Evaluated across Pooled, Urban Tram (ROW=1), Urban Bus, and Suburban Bus cohorts
                         │
                         ▼
[EXPLAINABLE AI (xAI) DIAGNOSTICS & POLICY REMEDIES]
  ├── TreeSHAP: Global beeswarm importance and local waterfall delay breakdowns
  ├── ALE Curves: Non-linear threshold tipping points (prev_stop_delay, trip_progress, weather lags)
  ├── 2D Interactions: Infrastructure shock buffering (Dedicated ROW vs. Mixed Traffic)
  └── DiCE Counterfactuals: Actionable operational remedies to restore on-time arrivals
```

---

## 2. Telemetry Ingestion & Medallion Storage

The system supports two complementary ingestion tracks running as production daemons:

1. **Track B Primary Ingestion (`historian_daemon.py`, `src/historian/`)**:
   - Polls official Warsaw City open data (`dane.um.warszawa.pl`) on a strict 10-second wall-clock grid for bus (`um_bus`) and tram (`um_tram`) positions.
   - Synchronizes with `api.zbiorkom.live` GTFS-RT feed every 30 seconds for TripUpdates and Service Alerts.
   - **Medallion Architecture**:
     - **Bronze Tier**: Verbatim raw payload archives (`data/historian/bronze/`).
     - **Silver Tier**: Deduplicated, Europe/Warsaw localized trajectories with kinematic velocity and acceleration computation (**38,214,890 GPS fixes** across 5 days).
     - **Gold Tier**: Reconstructed stop arrival and departure events with linear interpolation (**2,639,240 stop events** across **103,450 complete trips**, 100% active calendar compliance).
2. **Track A Reference Baseline (`collector_daemon.py`, `src/ingestion/`)**:
   - Polls Warsaw GTFS-RT Protobuf every 30 seconds and writes Hive-partitioned Snappy Parquet files (`data/raw/`). Maintained frozen for thesis comparative benchmarking.
3. **Automated Cloud Backup & Dead Man's Switch**:
   - Automated rclone backup to Google Drive every 6 hours for historian data (`backup_historian_to_gdrive.sh`) and daily for raw data (`backup_to_gdrive.sh`).
   - Periodic heartbeat pings to [Healthchecks.io](https://healthchecks.io) (`HISTORIAN_HEARTBEAT_URL` every 5 min; legacy `HEARTBEAT_URL` every 15 min).

### Gold Stop Events Schema (`data/historian/gold/stop_events/date=*/stop_events.parquet`)

| Column | Type | Description |
| :--- | :--- | :--- |
| `trip_id` | `STRING` | Unique scheduled GTFS trip identifier |
| `route_id` | `STRING` | Transit line identifier (e.g., `1`, `9`, `24`, `175`, `523`) |
| `vehicle_id` | `STRING` | Physical vehicle identifier |
| `stop_id` | `STRING` | Physical stop identifier in the Warsaw ZTM network |
| `stop_sequence` | `INT32` | Stop sequence index along the scheduled trip |
| `scheduled_arrival_time` | `INT64` | Scheduled arrival epoch seconds |
| `observed_arrival_time` | `INT64` | Reconstructed observed arrival epoch seconds |
| `arrival_delay_seconds` | `FLOAT` | **Arrival delay delta** ($>0$ late, $<0$ early, $=0$ on time) |
| `scheduled_departure_time` | `INT64` | Scheduled departure epoch seconds |
| `observed_departure_time` | `INT64` | Reconstructed observed departure epoch seconds |
| `departure_delay_seconds` | `FLOAT` | Departure delay delta in seconds |
| `delta_t_run` | `FLOAT` | Link traversal running delay: $\Delta t_{arr, s} - \Delta t_{dep, s-1}$ |
| `delta_t_dwell` | `FLOAT` | Stop dwell delay: $\Delta t_{dep, s} - \Delta t_{arr, s}$ |
| `observed_run_time_s` | `FLOAT` | Actual elapsed transit time between consecutive stops |
| `is_interpolated` | `BOOLEAN` | `true` if arrival was linearly interpolated between GPS pings |
| `interpolation_gap_seconds`| `FLOAT` | GPS time gap between surrounding trajectory fixes |
| `date` | `STRING` | Service date in `YYYY-MM-DD` format |

---

## 3. Exogenous Data Fusion & Scalability Architecture

Using DuckDB, raw transit observations are spatiotemporally fused with environmental, topological, and calendar datasets into the unified feature mart (**2,276,591 rows** in `feature_mart_observed.parquet`):

1. **Meteorological Telemetry & Decomposed Dynamics (`src/exogenous/imgw.py`)**:
   - Official IMGW-PIB hourly synoptic weather observations from *Warszawa-Okęcie* (`12375`) and *Warszawa-Bielany* (`352200380`).
   - Decomposed rain features: `is_raining` (binary indicator), `rain_intensity_category` (`'none'`, `'light'`, `'moderate'`, `'heavy'`), `precip_lag_1h` (1-hour preceding precipitation), and `precip_3h_cumulative` (3-hour cumulative rainfall).
2. **Road Network Topology & Geometry (`src/exogenous/osm.py`)**:
   - Extracted from OpenStreetMap across the entire Warsaw network (**10,081 stop-to-stop links**, **4,179 mapped traffic signals**, covering 9,070 km of transit routes).
   - Variables: `is_dedicated_right_of_way`, `signalized_intersection_count`, `segment_length_meters`, `actual_path_length_meters` (true GTFS polyline length), `tortuosity_ratio` ($\ge 1.0$), and `lane_capacity`.
3. **Multimodal Transfer Hubs & Metro Access (`src/exogenous/interchange.py`)**:
   - `interchange_line_count`: Total lines serving each stop post (7,241 stops mapped) to proxy transfer boarding pressure.
   - `is_metro_proximity`: 300-meter geodesic buffer around all 39 Warsaw Metro (M1 & M2) stations.
4. **School & Public Holiday Calendar (`src/core/polish_calendar.py`)**:
   - `is_school_day`: Mazowieckie voivodeship academic calendar.
   - `is_public_holiday`: Polish statutory holidays computed with the Anonymous Gregorian Easter algorithm.
5. **Autoregressive Trajectory Lags (`src/fusion/trajectory.py`)**:
   - `prev_stop_delay`, `prev2_stop_delay`, `headway_deviation`, and normalized `trip_progress` ($0.0 \to 1.0$).
6. **Multi-Month Scalability Architecture (`src/fusion/sampling.py`, `src/fusion/feature_mart.py`)**:
   - **Streaming Key-Join Projection**: DuckDB window ranking for sampling projects only key columns `(trip_id, route_id, stop_sequence, arrival_ts)` rather than all 45 columns, cutting memory consumption and query latency from 4.13s to 2.02s on 2.28M rows, enabling seamless scaling to 4+ months (~72M records).
   - **Memory Downcasting (`downcast_dataframe`)**: Automatically downcasts `float64` to `float32`, integers to smallest lossless types, and low-cardinality strings to `category`, reducing in-memory footprint by 30–50%.
   - **Out-of-Core Batch Streaming (`stream_feature_mart_batches`)**: Feeds multi-month datasets into training pipelines via chunked date-partitioned DuckDB cursors.
   - **Incremental Partition Reuse**: Gold reconstruction and feature mart pipelines check existing date partitions and skip reprocessing unless `--force-rebuild` is specified.

---

## 4. Zero-Leakage & Debiased Validation Strategy

Transit telemetry exhibits severe temporal autocorrelation, extreme shock outliers, and feature distribution imbalances:

1. **Multi-Dimensional Stratified Feature Sampling & IPS Weights (`src/fusion/sampling.py`)**:
   - Delays and weather are heavily imbalanced (rain occurs during a fraction of operating hours).
   - The sampling engine stratifies across 12 orthogonal feature cells:
     $$\text{Weather Regime (Dry / Rain)} \times \text{Infrastructure ROW (Mixed / Dedicated)} \times \text{Congestion Peak (Off-Peak / Peak)}$$
   - Sampling is strictly based on feature cells (never on target delay), preventing selection bias.
   - Computes normalized **Inverse Probability of Sampling (IPS) weights**:
     $$w_i = \frac{N_{\text{pop}, k} / N_{\text{pop}, \text{total}}}{N_{\text{sample}, k} / N_{\text{sample}, \text{total}}}$$
   - IPS weights are ingested by Weighted Least Squares (WLS) TWFE regression and weighted gradient boosting, ensuring mathematically unbiased population estimates.
2. **Purged Temporal Block Splitter with Embargo Buffering (`src/models/validation.py`)**:
   - Chronological forward partitioning: Train (60%), Validation (20%), and Hold-Out Test (20%).
   - Drops trips spanning boundary cuts and enforces a **30-minute blackout embargo** between consecutive splits to dissipate lingering congestion shockwaves.
3. **In-Fold Winsorization & Huber Loss (`src/models/gbm.py`, `src/models/twfe.py`)**:
   - Extreme delay shocks ($> 10$ min) inflate squared errors ($L_2$) and destabilize tree splitting.
   - Features (`prev_stop_delay`, `headway_deviation`) undergo in-fold winsorization at $[p_{01}, p_{99}]$ bounds fit strictly on training partitions.
   - Tree models optimize **Huber loss** ($\delta = 60.0\text{s}$), applying linear penalties to extreme outliers beyond 1 minute.
4. **Terminal Layover Filtering**: Layover turnaround rest breaks are excluded to isolate running transit friction from driver statutory breaks.

---

## 5. Empirical Modeling Benchmarks (TWFE vs. GBM)

Evaluated on the 5-day observed telemetry dataset ($N = 50,000$ stratified sample, $N_{\text{val}} = 9,800$, clean layovers) across pooled transit operations and individual transit cohorts:

### 5.1 Pooled Network Benchmark (Cleaned Layovers)

| Model Architecture | Specification / Regularization | Val MAE | Val RMSE | Median AE | WAPE (%) | Val $R^2$ | Dry MAE | Rain MAE | Peak MAE | Off-Peak MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TWFE Panel Regression** | Within Route + Hour FE + WLS + HC1 OLS | 53.25s | 99.39s | 33.98s | 118.15% | +0.1129 | 46.32s | 54.08s | 54.01s | 42.47s |
| **LightGBM Regressor** | Tree-depth 7, LR 0.05, Huber Loss ($\delta=60$s) | 45.74s | 104.36s | 22.36s | 101.48% | +0.0219 | 38.26s | 46.64s | 46.54s | 34.27s |
| **CatBoost Regressor** | Oblivious Trees, LR 0.05, Huber Loss ($\delta=60$s) | **35.92s** | **77.82s** | **20.66s** | **79.71%** | **+0.4561** | **27.79s** | **36.90s** | **36.65s** | **25.59s** |

### 5.2 Stratified Evaluation Across Transit Cohorts

| Transit Cohort | Best Model | Val MAE | Val RMSE | Median AE | WAPE (%) | Val $R^2$ | Dry MAE | Rain MAE | Peak MAE | Off-Peak MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Urban Tram (`urban_tram`)** | **CatBoost Regressor** | **28.19s** | **48.43s** | **19.56s** | **59.27%** | **+0.7921** | **25.70s** | **29.81s** | **29.70s** | **23.38s** |
| **Urban Bus (`urban_bus`)** | **CatBoost Regressor** | **40.58s** | **79.57s** | **22.32s** | **84.74%** | **+0.4505** | **28.79s** | **43.31s** | **42.96s** | **26.54s** |
| **Suburban Bus (`suburban_bus`)** | **CatBoost Regressor** | **58.06s** | **139.90s** | **30.29s** | **98.37%** | **+0.1717** | **34.18s** | **62.56s** | **61.05s** | **35.02s** |

### 5.3 Empirical Findings
1. **Infrastructure Isolation Advantage**: Urban trams operating on dedicated tracks achieve the highest predictability across all architectures (**$R^2 = 0.7921$**, **MAE = 28.19s**, **RMSE = 48.43s**), confirming that physical right-of-way isolation shields transit from street traffic volatility.
2. **Meteorological Disparity**: Across pooled transit, rain episodes elevate prediction MAE from **27.79s** (dry) to **36.90s** (+32.8% error increase). Suburban buses suffer the sharpest weather degradation (from **34.18s** dry to **62.56s** in rain, an **83.0% escalation**).
3. **Diurnal Peak Premium**: Congestion peaks increase delay error by ~11.1s across pooled transit (from **25.59s** off-peak to **36.65s** peak), highlighting non-linear interactions between peak passenger volume and road bottlenecks.
4. **Sub-20s Median AE**: CatBoost achieves a Median Absolute Error of **19.56s** for urban trams and **20.66s** across pooled operations.

---

## 6. Explainable AI (xAI) Delay Attribution Findings

### 6.1 Global Delay Attribution (TreeSHAP)
Exact TreeSHAP decomposition on the hold-out test set reveals the primary operational and infrastructure drivers of transit delays:
* **Upstream Delay Propagation (`prev_stop_delay`)**: Primary positive driver of downstream delay compounding.
* **Dispatch Headway Regularity (`headway_deviation`)**: Vehicle bunching causes acute delay surges; uniform headways maintain scheduled velocity.
* **Trip Progress Dynamics (`trip_progress`)**: Non-linear accumulation peaking near central transfer nodes and terminal loops.

![TreeSHAP Beeswarm Summary](reports/figures/xai/shap_beeswarm_summary.png)

### 6.2 Non-Linear Threshold Detection (ALE Curves)
Accumulated Local Effects (ALE) isolate the pure marginal effect of individual features without bias from correlated variables:
* **Upstream Delay Saturation**: Delays compound linearly until upstream delays reach $+300$ seconds, plateauing at a maximum addition to segment delay.
* **Trip Progress Bottlenecks**: Delays jump sharply at **74%** and **94%** route completion, pinpointing congestion near central transfer hubs and terminal approaches.
* **Dedicated Right-of-Way Effect**: Segregated tram tracks and bus lanes provide a direct step-function reduction in running delay across all operating conditions.

![ALE Trip Progress](reports/figures/xai/ale_trip_progress.png)
![ALE Previous Stop Delay](reports/figures/xai/ale_prev_stop_delay.png)

### 6.3 Infrastructure Buffering & 2D Interactions
Pairwise SHAP interaction values $\Phi_{i,j}(x)$ show whether dedicated infrastructure protects vehicles from operational shocks:
* In mixed traffic, each extra second of headway deviation increases downstream delay by $+0.042$s.
* On dedicated transit rights-of-way, this delay escalation is reduced to $+0.026$s per second.
* **Buffering Benefit**: Dedicated infrastructure provides a **~39% reduction in delay escalation** against vehicle bunching.

![Interaction Headway vs ROW](reports/figures/xai/interaction_headway_deviation_vs_dedicated_row.png)

### 6.4 Counterfactual Diagnostics (DiCE)
Using Diverse Counterfactual Explanations (DiCE), actionable operational interventions were evaluated across 50 severe delay incidents:
* **Headway Regularization**: Pacing vehicle departures mitigates an average of **+45.24 seconds** of delay escalation per incident on bunched corridors.
* **Dedicated ROW Upgrades**: Converting mixed-traffic links to dedicated bus lanes mitigates an average of **+5.16 seconds** of running delay per stop.
* **Schedule Recovery Holding**: For severely backlogged vehicles ($\Delta t > 300\text{s}$), physical lane improvements prevent further deterioration, while upstream schedule recovery holding and dispatch resets achieve a **100% on-time recovery rate** ($\Delta t \le 120\text{s}$).

![Counterfactual Policy Impact](reports/figures/xai/counterfactual_policy_impact.png)

---

## 7. Master Pipeline Runner (`run_pipeline.py`)

The unified CLI runner automates the end-to-end workflow across observed and baseline datasets:

```bash
# 1. Audit available date ranges across all data sources (Raw, Historian, Weather, GTFS, OSM)
python3 run_pipeline.py --check-ranges

# 2. Reconstruct Gold stop events from Silver trajectories for a specific date range
python3 run_pipeline.py --stage gold --start-date 2026-10-18 --end-date 2026-10-22

# 3. Spatiotemporal fusion with multi-source exogenous enrichment
python3 run_pipeline.py --stage fusion --datasource observed --start-date 2026-10-18 --end-date 2026-10-22

# 4. Run econometric & machine learning benchmarks with stratified sampling and Huber loss
python3 run_pipeline.py --stage benchmark --sample-size 50000 --sampling-strategy stratified --objective huber

# 5. Generate explainable AI diagnostics (TreeSHAP, ALE, Interactions, DiCE)
python3 run_pipeline.py --stage xai --sample-size 50000 --sampling-strategy stratified

# 6. Execute complete pipeline end-to-end
python3 run_pipeline.py --stage all --datasource observed --sampling-strategy stratified --objective huber
```

---

## 8. Repository Structure

```text
.
├── run_pipeline.py           # Master CLI pipeline runner (Stages 1 through 5)
├── historian_daemon.py       # Track B Primary 10s Warsaw City API ingestion daemon
├── collector_daemon.py       # Track A Baseline 30s GTFS-RT ingestion daemon
├── run_preflight.py          # Pre-flight environment and sandbox verification test
├── backup_historian_to_gdrive.sh # Automated 6-hour rclone backup for historian data
├── backup_to_gdrive.sh       # Automated daily rclone backup to Google Drive
├── pull_historian_from_gdrive.sh # Local sync for historian Parquet partitions
├── pull_from_gdrive.sh       # Local sync for raw GTFS-RT Parquet partitions
├── src/
│   ├── core/                 # Configuration, paths, feature registry, and calendar helpers
│   ├── historian/            # Track B: Bronze poller, silver trajectories, gold stop events
│   ├── ingestion/            # Track A: GTFS-RT poller, static schedule manager, raw storage
│   ├── exogenous/            # IMGW weather harvester, OSMnx road network topology
│   ├── fusion/               # DuckDB feature mart, trajectory lags, stratified sampling engine
│   ├── models/               # Purged temporal splitter, TWFE panel (WLS), Huber GBM benchmarks
│   ├── xai/                  # TreeSHAP, ALE curves, interactions, DiCE counterfactuals
│   ├── visualization/        # Academic publication styling & reusable plot generators
│   └── pipeline/             # Modular stage runners (Stages 1 through 5)
├── reports/
│   ├── EXECUTIVE_REPORT.md   # Comprehensive executive thesis findings report
│   ├── PRESENTATION_SLIDES.md# 12 Marp-ready presentation slides
│   ├── PROJECT_IMPROVEMENT_ROADMAP.md # Research roadmap & next upgrades
│   ├── tables/               # Benchmark comparison CSVs and counterfactual metrics JSON
│   └── figures/xai/          # Publication-ready xAI diagnostic figures
├── tests/                    # 109 unit and E2E integration tests (100% passing)
├── data/
│   ├── historian/            # Track B: bronze/, silver/, gold/stop_events/
│   ├── raw/                  # Track A: year=YYYY/month=MM/day=DD/
│   ├── gtfs/                 # Archived weekly static GTFS timetable snapshots
│   └── processed/            # Fused feature store (feature_mart_observed.parquet, 2.28M rows)
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
python3 run_pipeline.py --stage benchmark --sample-size 50000 --sampling-strategy stratified --objective huber
python3 run_pipeline.py --stage xai --sample-size 50000 --sampling-strategy stratified
```

---

## 10. Thesis Reports & Visual Exhibits

| Document / Artifact | Description |
| :--- | :--- |
| **[`reports/EXECUTIVE_REPORT.md`](reports/EXECUTIVE_REPORT.md)** | Full academic and executive report detailing methodology, empirical results, and transit policy guidance for Warsaw ZTM. |
| **[`reports/PRESENTATION_SLIDES.md`](reports/PRESENTATION_SLIDES.md)** | Complete 12-slide presentation deck formatted for thesis defenses and stakeholder briefings. |
| **[`reports/PROJECT_IMPROVEMENT_ROADMAP.md`](reports/PROJECT_IMPROVEMENT_ROADMAP.md)** | Enhancement roadmap covering seasonal telemetry, spatial radar weather, causal machine learning, and interactive dashboards. |
| **[`reports/tables/benchmark_comparison.csv`](reports/tables/benchmark_comparison.csv)** | Hold-out validation benchmark table comparing TWFE, LightGBM, and CatBoost across MAE, RMSE, MedAE, WAPE, $R^2$, and weather/peak regimes. |
| **[`reports/tables/benchmark_by_cohort.csv`](reports/tables/benchmark_by_cohort.csv)** | Stratified benchmark breakdown across Pooled, Urban Tram, Urban Bus, and Suburban Bus cohorts. |
| **[`reports/figures/xai/`](reports/figures/xai/)** | Diagnostic visual exhibit pack containing 17 high-resolution charts (SHAP beeswarm, waterfall plots, ALE curves, 2D interaction surfaces). |

---

## 11. Testing & Continuous Integration

The repository includes automated unit and end-to-end integration tests covering all pipeline stages:

```bash
# Run the complete test suite (109 tests in ~57 seconds)
.venv/bin/python3 -m unittest discover tests

# Or with pytest
.venv/bin/pytest
```

```text
Ran 109 tests in 56.99s
OK (100% pass rate)
```

---

## 12. License & Data Sources

* **License**: MIT License - see [LICENSE](LICENSE) for details.
* **Transit Telemetry**: Real-time Warsaw City API feeds (`dane.um.warszawa.pl`) and GTFS-RT feed ([zbiorkom.live](https://zbiorkom.live)) provided by Zarząd Transportu Miejskiego w Warszawie (ZTM).
* **Weather Telemetry**: Hourly synoptic observations provided by Instytut Meteorologii i Gospodarki Wodnej (IMGW-PIB).
* **Road Network Topology**: OpenStreetMap data extracted via OSMnx.


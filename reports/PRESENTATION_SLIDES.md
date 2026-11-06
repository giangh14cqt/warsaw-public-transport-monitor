# Warsaw Public Transport Telemetry & Delay Attribution
## Econometric Benchmarking & Explainable AI (xAI) Diagnostics

**Master's Thesis Presentation | University of Warsaw (UW)**  
**Author**: Giang Truong Do  
**Advisor & Department**: Faculty of Economic Sciences & Data Science  
**Date**: November 2026  

---

## Slide 1: Problem Statement & Research Objectives

### The Challenge of Urban Transit Delays
- Public transit delays are non-linear and driven by complex interactions:
  - **Operational disruptions**: Vehicle bunching and autoregressive delays carried over from prior stops.
  - **Infrastructure bottlenecks**: Traffic signal friction and conflicts in mixed street traffic.
  - **Weather dynamics**: Rain onset, precipitation accumulation, and cold-weather road adhesion.
- **Core Research Questions**:
  1. *How do weather shocks and road bottlenecks propagate through urban transit networks?*
  2. *How effectively does dedicated public transit infrastructure buffer against operational shocks?*
  3. *Can explainable machine learning uncover root causes and evaluate actionable operational remedies?*

---

## Slide 2: End-to-End Pipeline Architecture

```text
[Track B: Warsaw City API (10s Grid)]      [Track A: GTFS-RT Protobuf (30s)]
  │ (dane.um.warszawa.pl bus & tram)          │ (api.zbiorkom.live baseline)
  ▼                                           ▼
[Bronze Daily Parquet Store]               [Raw Hive Parquet Partitions]
  │ (201.4M GPS fixes across 20 days)         │
  ▼                                           │
[Silver Kinematic Trajectories]               │
  │ (174.4M GPS fixes, DST-safe, <=300s age)  │
  ▼                                           │
[Gold Reconstructed Stop Events] ─────────────┘
  │ (13.94M arrival/departure events across 542,406 trips)
  ▼
[Embedded Zero-Lock DuckDB Equi-Join Streaming Fusion]
   ├── IMGW-PIB Synoptic Weather (Warszawa-Okęcie WMO 12375, 1,539 hrs, 1h/3h lags)
   ├── Citywide OSM Road Network (10,081 links, 4,179 signals, 99.52% match)
   ├── Polish School & Holiday Calendar (Mazowieckie academic schedule)
   └── Metro M1/M2 Station Buffers & Transfer Hub Centrality
  ▼
[Full-Dataset & Multi-Dimensional Stratified IPS Weighting]
   └── 11.80M adjacent stop-to-stop observations (100% range-invariant features)
  ▼
[Purged Temporal Block Validation (30m Embargo + Winsorization + Huber Loss)]
   ├── Econometric TWFE Baseline (Weighted Least Squares + Route-Clustered CRVE)
   ├── Non-Linear Gradient Boosters (LightGBM & CatBoost with Huber Loss δ=60s)
   └── xAI Suite (TreeSHAP, ALE Curves, 2D Interactions, DiCE Counterfactuals)
```

---

## Slide 3: Telemetry & Feature Engineering Scale

- **Observed Ground-Truth Tier (Track B Historian, Oct 18 – Nov 6, 2026 — 20 Days)**:
  - **Bronze Daily Parquet**: **201,368,817 raw GPS fixes** unpacked on the remote server into fast daily Parquet partitions.
  - **Silver Trajectories**: **174,417,823 GPS fixes** deduplicated, DST-fold resolved, stale fixes filtered ($\le 300\text{s}$), with vectorized kinematic velocity & acceleration (**10.2× speedup**).
  - **Gold Stop Events**: **13,940,036 reconstructed stop arrival/departure events** (`13,026,746` `STOPPED`) across **542,406 trips** with sub-10s linear interpolation, single-vehicle trip enforcement, **90.7% retention of severely delayed trips ($>600\text{s}$)**, and 100% active calendar compliance.
- **Multi-Source Spatiotemporal Feature Mart (`feature_mart_observed.parquet`, 11,796,973 rows)**:
  - Operational: `prev_stop_delay`, `prev2_stop_delay`, date-partitioned deterministic `headway_deviation`, `trip_progress`.
  - Infrastructure (**99.52% exact OSM corridor match**): `lane_capacity`, `signalized_intersection_count`, `actual_path_length_meters`, `tortuosity_ratio`, `is_dedicated_right_of_way`.
  - Transfer Hubs: `interchange_line_count`, `is_metro_proximity` (300m buffer to 39 M1/M2 stations).
  - Decomposed Weather (`Warszawa-Okęcie WMO 12375`): `precipitation_mm`, `is_raining`, `rain_intensity_category`, `precip_lag_1h`, `precip_3h_cumulative`, `temperature_c`.
- **Multi-Month Scalability**: DuckDB hash-equi joins (**11.80M rows fused in 39.7s**), memory downcasting (`downcast_dataframe`), and 100% bit-exact historical feature invariance under date-range expansion.

---

## Slide 4: Validation Strategy: Zero Data Leakage & Debiasing

### 1. Full-Dataset & Multi-Dimensional Stratified IPS Weighting
- **Full-Scale Evaluation**: Evaluated across all **11,796,973** fused records (`Train = 7,040,134`, `Val = 2,272,547`, `Test = 2,293,756`), alongside 12-cell orthogonal stratified sampling ($\text{Weather} \times \text{ROW} \times \text{Peak}$).
- **Normalized Inverse Probability of Sampling (IPS) Weights**:
  $$w_i = \frac{N_{\text{pop}, k} / N_{\text{pop}, \text{total}}}{N_{\text{sample}, k} / N_{\text{sample}, \text{total}}}$$
  Applied in Weighted Least Squares (WLS) TWFE regression and gradient boosting.

### 2. Boundary-Scoped Purged Temporal Block Splitting
- **Trip Boundary Purging**: Active vehicle trips crossing split boundaries within $\pm 4\text{h}$ (`Train–Val = 2,379`, `Val–Test = 2,425`) are completely dropped without discarding recurring weekday `trip_id`s on other service dates.
- **Embargo Buffering**: 30-minute blackout window between train, validation, and test sets.
- **In-Fold Winsorization & Huber Loss**: Features and training targets winsorized strictly on training folds; Huber loss ($\delta=60.0\text{s}$) prevents quadratic penalty blowups on extreme delay shocks.

---

## Slide 5: Model Benchmark Performance

### Hold-Out Test Results ($N_{\text{test}}=2{,}293{,}756$ Full Dataset, Target: $\Delta t_{\text{run}}$, Cleaned Layovers)

| Transit Cohort | Model Architecture | Test MAE | Test RMSE | Median AE | WAPE | Test $R^2$ | Dry MAE | Rain MAE | Peak MAE | Off-Peak MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pooled** *(All, $2.29\text{M}$)* | **TWFE Panel (WLS)** | 27.65s | 41.91s | 19.00s | 99.51% | +0.0266 | 27.77s | 24.36s | 31.05s | 25.70s |
| | **LightGBM (Huber)** | **25.35s** | **39.55s** | **16.67s** | **91.25%** | **+0.1329** | **25.47s** | **22.27s** | **28.61s** | **23.49s** |
| | **CatBoost (Huber)** | 26.06s | 40.42s | 17.23s | 93.81% | +0.0945 | 26.19s | 22.78s | 29.44s | 24.13s |
| **Urban Tram** *($599\text{k}$)* | **TWFE Panel (WLS)** | 24.54s | 34.76s | 17.00s | 98.82% | +0.0232 | 24.56s | 23.86s | 25.11s | 24.20s |
| | **LightGBM (Huber)** | **18.59s** | **27.43s** | **12.65s** | **74.86%** | **+0.3918** | **18.62s** | **17.78s** | **19.29s** | **18.17s** |
| | **CatBoost (Huber)** | 20.51s | 29.58s | 14.27s | 82.59% | +0.2927 | 20.53s | 19.74s | 21.12s | 20.14s |
| **Urban Bus** *($1.53\text{M}$)* | **TWFE Panel (WLS)** | 28.76s | 44.25s | 19.52s | 99.53% | +0.0282 | 28.94s | 24.33s | 33.54s | 26.08s |
| | **LightGBM (Huber)** | **26.39s** | **41.79s** | **17.04s** | **91.33%** | **+0.1332** | **26.56s** | **22.25s** | **30.92s** | **23.85s** |
| | **CatBoost (Huber)** | 27.06s | 42.71s | 17.41s | 93.64% | +0.0949 | 27.24s | 22.69s | 31.72s | 24.44s |
| **Suburban Bus** *($165\text{k}$)* | **TWFE Panel (WLS)** | 28.08s | 42.96s | 20.12s | 98.45% | +0.0346 | 28.18s | 25.12s | 31.41s | 26.18s |
| | **LightGBM (Huber)** | **23.88s** | **38.28s** | **16.73s** | **83.74%** | **+0.2331** | **23.97s** | **21.40s** | **26.92s** | **22.15s** |
| | **CatBoost (Huber)** | 25.35s | 40.08s | 17.90s | 88.87% | +0.1593 | 25.45s | 22.63s | 28.48s | 23.56s |

### Key Empirical Findings:
1. **Non-Linear Boost over Linear TWFE**: On first-differenced link running delay ($\Delta t_{\text{run}}$), LightGBM increases out-of-sample $R^2$ by **4.7x to 16.9x** over TWFE ($+0.1329$ vs $+0.0266$ pooled; $+0.3918$ on Urban Trams; $+0.2331$ on Suburban Buses).
2. **Infrastructure Isolation Advantage**: Urban trams on dedicated tracks achieve the lowest errors (**LightGBM MAE = 18.59s**, **MedAE = 12.65s**, **WAPE = 74.86%**, **$R^2 = +0.3918$**)—confirming physical segregation insulates transit from traffic volatility.
3. **Diurnal Peak Disparity**: Rush hour increases Urban Bus MAE by **+7.07s/stop** (`23.85s` $\to$ `30.92s`) and Suburban Bus MAE by **+4.77s/stop** (`22.15s` $\to$ `26.92s`) vs only **+1.12s/stop** on segregated Urban Trams (`18.17s` $\to$ `19.29s`).

---

## Slide 6: Econometric Panel Specification (TWFE with WLS & CRVE)

### Model Specification
$$\Delta t_{\text{run}, ist} = \alpha_i + \lambda_t + \beta_1 \text{Signals}_{is} + \beta_2 \text{Lanes}_{is} + \beta_3 \text{Length}_{is} + \beta_4 \text{Progress}_{ist} + \epsilon_{ist}$$
- Estimated via **Weighted Least Squares (WLS)** with iterative multi-way fixed-effects demeaning (`route_id` $\times$ `hour_of_day`, $N = 7{,}040{,}134$ training rows) and route-clustered CRVE standard errors.
- **Variance Decomposition**:
  - Pooled TWFE: Overall $R^2 = 0.0280$, Within-$R^2 = 0.0092$ (Hold-out Test $R^2 = 0.0266$).
  - Suburban Bus TWFE: Overall $R^2 = 0.0360$, Within-$R^2 = 0.0157$ (Hold-out Test $R^2 = 0.0346$).
- **Non-Linear Advantage of Tree Boosters**: Linear TWFE assumes additive shocks across heterogeneous road links. Gradient boosting captures non-linear signal, tortuosity, and upstream recovery thresholds, reaching $R^2 = 0.3918$ on trams and $0.2331$ on suburban buses.

---

## Slide 7: Global Delay Attribution (TreeSHAP)

### Decomposing Delay Drivers Across Operational Domains
- **Fleet & Corridor Topology (`signalized_intersection_count` [SHAP `3.12`], `tortuosity_ratio` [`1.88`], `is_tram` [`1.67`], `actual_path_length_meters` [`1.66`], `segment_length_meters` [`1.03`])**: Primary structural determinants of inter-stop running time variance.
- **Upstream Delay & Recovery (`headway_deviation` [`2.91`], `prev_stop_delay` [`2.89`], `trip_progress` [`1.40`], and interaction `prev_stop_delay × headway_deviation` [`1.31`])**: Captures early-departure pacing, schedule slack absorption, upstream delay carryover, and vehicle bunching dynamics.
- **Exact Additivity**: Verified across 24 features ($E[f(X)] = +5.25\text{s}$ positive net link running delay drift, $\max |\Delta| = 2.34 \times 10^{-13}\text{s}$).

*Exhibit*: `reports/figures/xai/shap_beeswarm_summary.png`, `reports/figures/xai/shap_waterfall_extreme_case_1.png`

---

## Slide 8: Non-Linear Threshold Detection (ALE Curves)

### Identifying Operational Tipping Points
1. **Early Pacing, Slack Recovery & Saturation (`prev_stop_delay`)**:
   - Vehicles $>60\text{s}$ ahead of schedule (`-79.0s` threshold) lose their lead on the next link (`+13s` to `+31s`), consistent with early-running avoidance (hypothesis); moderate delays (`0s–250s`) trigger `-3.7s/link` schedule recovery before saturating under severe gridlock (`>300s`).
2. **Route Progress Congestion Transitions (`trip_progress`)**:
   - Initial departure merge friction (`+3s` to `+5s`) transitions at `0.10–0.25` route completion into steady mid-route schedule recovery (`-0.5s` to `-1.4s`).
3. **Dedicated Right-of-Way (ROW) Protection (`is_dedicated_right_of_way`)**:
   - Provides a negative step-function reduction in link delay escalation across operating conditions.

*Exhibit*: `reports/figures/xai/ale_trip_progress.png`, `reports/figures/xai/ale_prev_stop_delay.png`

---

## Slide 9: Infrastructure Buffering & 2D Interactions

### Does Dedicated Infrastructure Protect Against Operational Shocks?
- **Pairwise SHAP Interaction Matrix**: $\Phi_{i, j}(x)$ exported to `reports/tables/xai/shap_interaction_matrix.csv`.
- **Empirical Buffering Metrics**:
  - **Upstream Delay (`prev_stop_delay`)**: Dedicated ROW mitigates **3.26s per stop link (46.7% reduction)** of delay escalation under upstream congestion shocks.
  - **Signalized Intersections (`signalized_intersection_count`)**: Dedicated ROW saves **1.50s/link (12.6% reduction)** across high-signal corridors.
  - **Segment Distance (`segment_length_meters`)**: Dedicated ROW saves **0.73s/link (8.4% reduction)** on long inter-stop segments.

*Exhibit*: `reports/figures/xai/interaction_headway_deviation_vs_dedicated_row.png`

---

## Slide 10: Actionable Counterfactual Remedies (DiCE)

### Operational Remedies for Delayed Vehicles (50 Severe Incident Case Studies, Mean Delay 529.0s)

| Operational Remedy | Mean Delay Reduction | On-Time Restoration Rate ($\le 120$s) |
| :--- | :--- | :--- |
| **Dedicated ROW Upgrade ($ROW \to 1$)** | **+0.72s / stop link** saved | **0.0%** (single link cannot erase 8.8 min of prior delay) |
| **Dynamic Headway Regularization** | **+0.38s / stop link** saved | **0.0%** (prevents bunching escalation per stop) |
| **Transit Signal Priority ($TSP$)** | **+3.45s / stop link** saved | **0.0%** (cumulative corridor savings ~86s per 25-stop trip) |
| **Upstream Delay Absorption (Turnaround Recovery)** | **+331.00s / incident** saved | **30.0%** (absorbs accumulated backlog at turnaround/hubs) |
| **Joint Multimodal Remedy** | **+337.97s / incident** saved | **30.0%** (highest recovery + downstream link resilience) |

- **Key Policy Takeaway**: Link-level physical & signal upgrades (TSP + headway pacing + dedicated ROW) prevent incremental delay accumulation along a 25-stop corridor (~85–110s cumulative trip savings) and buffer against upstream and signal shocks (13%–47% reduction), while turnaround schedule recovery absorption is essential to clear large pre-existing upstream backlogs.

*Exhibit*: `reports/figures/xai/counterfactual_policy_impact.png`, `reports/figures/xai/counterfactual_case_study.png`

---

## Slide 11: Policy Recommendations for Warsaw ZTM

1. **Implement Dynamic Headway Pacing**:
   - Shift dispatch optimization from static schedule adherence to active headway holding at major transfer hubs (e.g., Rondo Daszyńskiego, Dworzec Centralny) to prevent delay cascading.
2. **Target Dedicated ROW Upgrades**:
   - Prioritize bus lane allocation on mixed-traffic corridors with severe peak sensitivity (Urban & Suburban feeder corridors, Puławska, Al. Jerozolimskie).
3. **Deploy Corridor-Wide Transit Signal Priority (TSP)**:
   - Target TSP green-extension at high-density signalized intersections ($\ge 5$ signals/link) to save ~3.45s per link (~86s per 25-stop route).

---

## Slide 12: Engineering & Open Science Deliverables

- **Codebase**: Fully reproducible, modular Python package.
- **Production Daemons**: 10s Warsaw City API poller (`historian_daemon.py`) + GTFS-RT poller (`collector_daemon.py`) with automated daily Parquet materialization.
- **Scalability**: Vectorized NumPy Silver/Gold reconstruction (**10.2× / 5.0× speedups** with multi-core parallelism) and DuckDB hash-equi joins (**11.80M rows fused in 39.7s**) with 100% bit-exact historical feature invariance.
- **Unit & Integration Test Suite**: **100% pass rate across all 111 unit and integration tests**.
- **Open Access**: Complete codebase, data pipelines, and documentation hosted at:  
  [`https://github.com/giangh14cqt/warsaw-public-transport-monitor`](https://github.com/giangh14cqt/warsaw-public-transport-monitor)

---

## Questions & Discussion
*Thank you!*

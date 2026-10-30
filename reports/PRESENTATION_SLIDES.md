# Warsaw Public Transport Telemetry & Delay Attribution
## Econometric Benchmarking & Explainable AI (xAI) Diagnostics

**Master's Thesis Presentation | University of Warsaw (UW)**  
**Author**: Giang Truong Do  
**Advisor & Department**: Faculty of Economic Sciences & Data Science  
**Date**: October 2026  

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
[Bronze Verbatim Store]                    [Raw Hive Parquet Partitions]
  ▼                                           │
[Silver Kinematic Trajectories]               │
  │ (83.4M GPS fixes, DST-safe, <=300s age)   │
  ▼                                           │
[Gold Reconstructed Stop Events] ─────────────┘
  │ (6.65M arrival/departure events across 259,313 trips)
  ▼
[Embedded Zero-Lock DuckDB Equi-Join Streaming Fusion]
   ├── IMGW-PIB Synoptic Weather (Warszawa-Okęcie WMO 12375, 1h/3h lags)
   ├── Citywide OSM Road Network (10,081 links, 4,179 signals, 99.52% match)
   ├── Polish School & Holiday Calendar (Mazowieckie academic schedule)
   └── Metro M1/M2 Station Buffers & Transfer Hub Centrality
  ▼
[Full-Dataset & Multi-Dimensional Stratified IPS Weighting]
   └── 5.61M adjacent stop-to-stop observations (100% range-invariant features)
  ▼
[Purged Temporal Block Validation (30m Embargo + Winsorization + Huber Loss)]
   ├── Econometric TWFE Baseline (Weighted Least Squares + Route-Clustered CRVE)
   ├── Non-Linear Gradient Boosters (LightGBM & CatBoost with Huber Loss δ=60s)
   └── xAI Suite (TreeSHAP, ALE Curves, 2D Interactions, DiCE Counterfactuals)
```

---

## Slide 3: Telemetry & Feature Engineering Scale

- **Observed Ground-Truth Tier (Track B Historian, Oct 18–27, 2026)**:
  - **Silver Trajectories**: **83,415,407 GPS fixes** deduplicated, DST-fold resolved, stale fixes filtered ($\le 300\text{s}$), with kinematic velocity & acceleration.
  - **Gold Stop Events**: **6,650,842 reconstructed stop arrival/departure events** across **259,313 trips** with sub-10s linear interpolation, single-vehicle trip enforcement, **90.7% retention of severely delayed trips ($>600\text{s}$)**, and 100% active calendar compliance.
- **Multi-Source Spatiotemporal Feature Mart (`feature_mart_observed.parquet`, 5,614,929 rows)**:
  - Operational: `prev_stop_delay`, `prev2_stop_delay`, date-partitioned deterministic `headway_deviation`, `trip_progress`.
  - Infrastructure (**99.52% exact OSM corridor match**): `lane_capacity`, `signalized_intersection_count`, `actual_path_length_meters`, `tortuosity_ratio`, `is_dedicated_right_of_way`.
  - Transfer Hubs: `interchange_line_count`, `is_metro_proximity` (300m buffer to 39 M1/M2 stations).
  - Decomposed Weather (`Warszawa-Okęcie WMO 12375`): `precipitation_mm`, `is_raining`, `rain_intensity_category`, `precip_lag_1h`, `precip_3h_cumulative`, `temperature_c`.
- **Multi-Month Scalability**: DuckDB hash-equi joins (5.61M rows fused in **15.0s**), memory downcasting (`downcast_dataframe`), and 100% bit-exact historical feature invariance under date-range expansion.

---

## Slide 4: Validation Strategy: Zero Data Leakage & Debiasing

### 1. Full-Dataset & Multi-Dimensional Stratified IPS Weighting
- **Full-Scale Evaluation**: Evaluated across all **5,614,929** fused records (`Train = 3,344,039`, `Val = 1,061,810`, `Test = 1,096,705`), alongside 12-cell orthogonal stratified sampling ($\text{Weather} \times \text{ROW} \times \text{Peak}$).
- **Normalized Inverse Probability of Sampling (IPS) Weights**:
  $$w_i = \frac{N_{\text{pop}, k} / N_{\text{pop}, \text{total}}}{N_{\text{sample}, k} / N_{\text{sample}, \text{total}}}$$
  Applied in Weighted Least Squares (WLS) TWFE regression and gradient boosting.

### 2. Boundary-Scoped Purged Temporal Block Splitting
- **Trip Boundary Purging**: Active vehicle trips crossing split boundaries within $\pm 4\text{h}$ (`Train–Val = 2,286`, `Val–Test = 928`) are completely dropped without discarding recurring weekday `trip_id`s on other service dates.
- **Embargo Buffering**: 30-minute blackout window between train, validation, and test sets.
- **In-Fold Winsorization & Huber Loss**: Features and training targets winsorized strictly on training folds; Huber loss ($\delta=60.0\text{s}$) prevents quadratic penalty blowups on extreme delay shocks.

---

## Slide 5: Model Benchmark Performance

### Hold-Out Test Results ($N_{\text{test}}=1{,}096{,}705$ Full Dataset, Target: $\Delta t_{\text{run}}$, Cleaned Layovers)

| Transit Cohort | Model Architecture | Test MAE | Test RMSE | Median AE | WAPE | Test $R^2$ | Dry MAE | Rain MAE | Peak MAE | Off-Peak MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pooled** *(All, $1.10\text{M}$)* | **TWFE Panel (WLS)** | 28.84s | 45.13s | 19.28s | 99.04% | +0.0261 | 28.76s | 30.32s | 32.14s | 26.25s |
| | **LightGBM (Huber)** | **26.73s** | **42.70s** | **17.24s** | **91.80%** | **+0.1279** | **26.67s** | **27.87s** | **29.89s** | **24.25s** |
| | **CatBoost (Huber)** | 27.56s | 43.61s | 17.93s | 94.65% | +0.0907 | 27.50s | 28.71s | 30.88s | 24.95s |
| **Urban Tram** *($284\text{k}$)* | **TWFE Panel (WLS)** | 25.04s | 35.69s | 17.26s | 100.05% | -0.0044 | 25.12s | 24.38s | 25.24s | 24.88s |
| | **LightGBM (Huber)** | **18.39s** | **27.58s** | **12.36s** | **73.48%** | **+0.4003** | **18.45s** | **17.94s** | **18.92s** | **17.96s** |
| | **CatBoost (Huber)** | 20.53s | 29.95s | 14.10s | 82.01% | +0.2927 | 20.60s | 19.90s | 21.01s | 20.14s |
| **Urban Bus** *($735\text{k}$)* | **TWFE Panel (WLS)** | 30.33s | 48.05s | 20.01s | 99.11% | +0.0274 | 30.13s | 34.97s | 34.79s | 26.91s |
| | **LightGBM (Huber)** | **28.43s** | **45.53s** | **18.15s** | **92.89%** | **+0.1268** | **28.25s** | **32.59s** | **32.62s** | **25.20s** |
| | **CatBoost (Huber)** | 29.19s | 46.48s | 18.69s | 95.37% | +0.0899 | 29.01s | 33.34s | 33.58s | 25.81s |
| **Suburban Bus** *($79\text{k}$)* | **TWFE Panel (WLS)** | 29.28s | 46.71s | 20.20s | 97.74% | +0.0321 | 29.17s | 31.91s | 32.78s | 26.29s |
| | **LightGBM (Huber)** | **25.61s** | **42.14s** | **17.52s** | **85.50%** | **+0.2121** | **25.54s** | **27.50s** | **28.78s** | **22.91s** |
| | **CatBoost (Huber)** | 27.00s | 43.96s | 18.48s | 90.12% | +0.1428 | 26.91s | 29.19s | 30.34s | 24.14s |

### Key Empirical Findings:
1. **Non-Linear Boost over Linear TWFE**: On first-differenced link running delay ($\Delta t_{\text{run}}$), LightGBM increases out-of-sample $R^2$ by **4.6x to 6.6x** over TWFE ($+0.1279$ vs $+0.0261$ pooled; $+0.4003$ on Urban Trams; $+0.2121$ on Suburban Buses).
2. **Infrastructure Isolation Advantage**: Urban trams on dedicated tracks achieve the lowest errors (**LightGBM MAE = 18.39s**, **MedAE = 12.36s**, **$R^2 = +0.4003$**)—confirming physical segregation insulates transit from traffic volatility.
3. **Peak & Weather Disparity**: Rush hour increases Urban Bus MAE by **+7.42s/stop** (25.20s $\to$ 32.62s) vs only **+0.96s/stop** on Trams; rain elevates Urban Bus MAE by **+4.34s/stop** (28.25s $\to$ 32.59s) and Suburban Bus MAE by **+1.96s/stop** (25.54s $\to$ 27.50s).

---

## Slide 6: Econometric Panel Specification (TWFE with WLS & CRVE)

### Model Specification
$$\Delta t_{\text{run}, ist} = \alpha_i + \lambda_t + \beta_1 \text{Signals}_{is} + \beta_2 \text{Lanes}_{is} + \beta_3 \text{Length}_{is} + \beta_4 \text{Progress}_{ist} + \epsilon_{ist}$$
- Estimated via **Weighted Least Squares (WLS)** with iterative multi-way fixed-effects demeaning (`route_id` $\times$ `hour_of_day`, $N = 3{,}344{,}039$ training rows) and route-clustered CRVE standard errors.
- **Variance Decomposition**:
  - Pooled TWFE: Overall $R^2 = 0.0278$, Within-$R^2 = 0.0100$ (Hold-out Test $R^2 = 0.0261$).
  - Suburban Bus TWFE: Overall $R^2 = 0.0362$, Within-$R^2 = 0.0165$ (Hold-out Test $R^2 = 0.0321$).
- **Non-Linear Advantage of Tree Boosters**: Linear TWFE assumes additive shocks across heterogeneous road links. Gradient boosting captures non-linear signal, tortuosity, and upstream recovery thresholds, reaching $R^2 = 0.4003$ on trams and $0.2121$ on suburban buses.

---

## Slide 7: Global Delay Attribution (TreeSHAP)

### Decomposing Delay Drivers Across Operational Domains
- **Fleet & Corridor Topology (`signalized_intersection_count`, `is_tram`, `actual_path_length_meters`, `tortuosity_ratio`, `segment_length_meters`)**: Primary structural determinants of inter-stop running time variance.
- **Upstream Delay & Recovery (`prev_stop_delay`, `headway_deviation`, `trip_progress`)**: Captures early-departure pacing, schedule slack absorption, upstream delay carryover, and vehicle bunching dynamics.
- **Exact Additivity**: Verified across 24 features ($E[f(X)] = +5.40\text{s}$ positive net link running delay drift, $\max |\Delta| = 1.35 \times 10^{-13}\text{s}$).

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
  - **Upstream Delay (`prev_stop_delay`)**: Dedicated ROW mitigates **4.03s per stop link (48.7% reduction)** of delay under upstream congestion shocks.
  - **Precipitation (`precipitation_mm`)**: Dedicated ROW mitigates **3.17s per stop link (36.8% reduction)** during rainfall.
  - **Signalized Intersections & Headway Bunching**: Dedicated ROW saves **3.02s/link (21.4% reduction)** across signalized corridors and **1.49s/link (9.6% reduction)** under headway bunching.

*Exhibit*: `reports/figures/xai/interaction_headway_deviation_vs_dedicated_row.png`

---

## Slide 10: Actionable Counterfactual Remedies (DiCE)

### Operational Remedies for Delayed Vehicles (50 Severe Incident Case Studies, Mean Delay 494.4s)

| Operational Remedy | Mean Delay Reduction | On-Time Restoration Rate ($\le 120$s) |
| :--- | :--- | :--- |
| **Dedicated ROW Upgrade ($ROW \to 1$)** | **+0.63s / stop link** saved | **0.0%** (single link cannot erase 8.2 min of prior delay) |
| **Dynamic Headway Regularization** | **+1.73s / stop link** saved | **0.0%** (prevents bunching escalation per stop) |
| **Transit Signal Priority ($TSP$)** | **+1.79s / stop link** saved | **0.0%** (cumulative corridor savings ~45s per 25-stop trip) |
| **Upstream Delay Absorption (Turnaround Recovery)** | **+307.66s / incident** saved | **34.0%** (absorbs accumulated backlog at turnaround/hubs) |
| **Joint Multimodal Remedy** | **+313.30s / incident** saved | **38.0%** (highest recovery + downstream link resilience) |

- **Key Policy Takeaway**: Link-level physical & signal upgrades (TSP + headway pacing + dedicated ROW) prevent incremental delay accumulation along a 25-stop corridor (~45–60s cumulative trip savings) and buffer against weather and signal shocks (21%–49% reduction), while turnaround schedule recovery absorption is essential to clear large pre-existing upstream backlogs.

*Exhibit*: `reports/figures/xai/counterfactual_policy_impact.png`, `reports/figures/xai/counterfactual_case_study.png`

---

## Slide 11: Policy Recommendations for Warsaw ZTM

1. **Implement Dynamic Headway Pacing**:
   - Shift dispatch optimization from static schedule adherence to active headway holding at major transfer hubs (e.g., Rondo Daszyńskiego, Dworzec Centralny) to prevent delay cascading.
2. **Target Dedicated ROW Upgrades**:
   - Prioritize bus lane allocation on mixed-traffic corridors with severe weather and peak sensitivity (Urban & Suburban feeder corridors, Puławska, Al. Jerozolimskie).
3. **Deploy Corridor-Wide Transit Signal Priority (TSP)**:
   - Target TSP green-extension at high-density signalized intersections ($\ge 5$ signals/link) to save ~1.8s per link (~45s per route).

---

## Slide 12: Engineering & Open Science Deliverables

- **Codebase**: Fully reproducible, modular Python package.
- **Production Daemons**: 10s Warsaw City API poller (`historian_daemon.py`) + GTFS-RT poller (`collector_daemon.py`).
- **Scalability**: DuckDB hash-equi joins (5.61M rows fused in 15.0s) and memory downcasting supporting 4+ months (~72M records) with 100% bit-exact historical feature invariance.
- **Unit & Integration Test Suite**: **100% pass rate across all unit and integration tests**.
- **Open Access**: Complete codebase, data pipelines, and documentation hosted at:  
  [`https://github.com/giangh14cqt/warsaw-public-transport-monitor`](https://github.com/giangh14cqt/warsaw-public-transport-monitor)

---

## Questions & Discussion
*Thank you!*

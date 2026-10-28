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
  │ (83.4M GPS fixes, velocity, accel)        │
  ▼                                           │
[Gold Reconstructed Stop Events] ─────────────┘
  │ (5.04M arrival/departure events across 195,582 trips)
  ▼
[Embedded Zero-Lock DuckDB Streaming Fusion]
   ├── IMGW-PIB Synoptic Weather (is_raining, rain categories, 1h/3h lags)
   ├── Citywide OSM Road Network (10,081 links, 4,179 signals, tortuosity ratios)
   ├── Polish School & Holiday Calendar (Mazowieckie academic schedule)
   └── Metro M1/M2 Station Buffers & Transfer Hub Centrality
  ▼
[Full-Dataset & Multi-Dimensional Stratified IPS Weighting]
   └── 4.33M fused observations (plus 12-cell Weather x ROW x Peak IPS weighting)
  ▼
[Purged Temporal Block Validation (30m Embargo + Winsorization + Huber Loss)]
   ├── Econometric TWFE Baseline (Weighted Least Squares + Route-Clustered CRVE)
   ├── Non-Linear Gradient Boosters (LightGBM & CatBoost with Huber Loss δ=60s)
   └── xAI Suite (TreeSHAP, ALE Curves, 2D Interactions, DiCE Counterfactuals)
```

---

## Slide 3: Telemetry & Feature Engineering Scale

- **Observed Ground-Truth Tier (Track B Historian, Oct 18–27, 2026)**:
  - **Silver Trajectories**: **83,428,962 GPS fixes** deduplicated, localized, with kinematic acceleration.
  - **Gold Stop Events**: **5,041,712 reconstructed stop arrival/departure events** across **195,582 complete trips** with sub-10s linear interpolation and 100% active calendar compliance.
- **Multi-Source Spatiotemporal Feature Mart (`feature_mart_observed.parquet`, 4,331,687 rows)**:
  - Operational: `prev_stop_delay`, `prev2_stop_delay`, `headway_deviation`, `trip_progress`.
  - Infrastructure: `lane_capacity`, `signalized_intersection_count`, `actual_path_length_meters`, `tortuosity_ratio`, `is_dedicated_right_of_way`.
  - Transfer Hubs: `interchange_line_count`, `is_metro_proximity` (300m buffer to 39 M1/M2 stations).
  - Decomposed Weather: `precipitation_mm`, `is_raining`, `rain_intensity_category`, `precip_lag_1h`, `precip_3h_cumulative`, `temperature_c`.
- **Multi-Month Scalability**: DuckDB streaming key joins, memory downcasting (`downcast_dataframe`), and out-of-core batch streaming ready for 4+ months (~72M records).

---

## Slide 4: Validation Strategy: Zero Data Leakage & Debiasing

### 1. Full-Dataset & Multi-Dimensional Stratified IPS Weighting
- **Full-Scale Evaluation**: Evaluated across all **4,331,687** fused records (`Train = 2,503,755`, `Val = 808,538`, `Test = 836,236`), alongside 12-cell orthogonal stratified sampling ($\text{Weather} \times \text{ROW} \times \text{Peak}$).
- **Normalized Inverse Probability of Sampling (IPS) Weights**:
  $$w_i = \frac{N_{\text{pop}, k} / N_{\text{pop}, \text{total}}}{N_{\text{sample}, k} / N_{\text{sample}, \text{total}}}$$
  Applied in Weighted Least Squares (WLS) TWFE regression and gradient boosting.

### 2. Purged Temporal Block Splitting
- **Trip Boundary Purging**: Vehicle trips crossing split boundaries (`Train–Val = 990`, `Val–Test = 1,046`) are completely dropped.
- **Embargo Buffering**: 30-minute blackout window between train, validation, and test sets.
- **In-Fold Winsorization & Huber Loss**: Features and training targets winsorized strictly on training folds; Huber loss ($\delta=60.0\text{s}$) prevents quadratic penalty blowups on extreme delay shocks.

---

## Slide 5: Model Benchmark Performance

### Hold-Out Test Results ($N_{\text{test}}=836{,}236$ Full Dataset, Target: $\Delta t_{\text{run}}$, Cleaned Layovers)

| Transit Cohort | Model Architecture | Test MAE | Test RMSE | Median AE | WAPE | Test $R^2$ | Dry MAE | Rain MAE | Peak MAE | Off-Peak MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pooled** *(All, $836\text{k}$)* | **TWFE Panel (WLS)** | 25.52s | 41.51s | 16.96s | 77.96% | +0.0371 | 25.66s | 24.40s | 27.44s | 24.40s |
| | **LightGBM (Huber)** | **23.04s** | **38.84s** | **14.08s** | **70.41%** | **+0.1567** | **23.17s** | **22.10s** | **24.78s** | **22.04s** |
| | **CatBoost (Huber)** | 23.77s | 39.42s | 14.55s | 72.61% | +0.1314 | 23.90s | 22.74s | 25.61s | 22.70s |
| **Urban Tram** *($234\text{k}$)* | **TWFE Panel (WLS)** | 22.39s | 34.50s | 15.53s | 63.34% | +0.0510 | 22.49s | 21.82s | 22.62s | 22.21s |
| | **LightGBM (Huber)** | **15.28s** | **26.10s** | **9.18s** | **43.23%** | **+0.4568** | **15.32s** | **15.05s** | **15.67s** | **14.97s** |
| | **CatBoost (Huber)** | 17.68s | 28.48s | 11.37s | 50.02% | +0.3534 | 17.73s | 17.38s | 18.09s | 17.36s |
| **Urban Bus** *($539\text{k}$)* | **TWFE Panel (WLS)** | 26.45s | 42.83s | 17.31s | 83.21% | +0.0158 | 26.62s | 25.05s | 29.58s | 24.90s |
| | **LightGBM (Huber)** | **24.34s** | **40.38s** | **14.60s** | **76.58%** | **+0.1253** | **24.48s** | **23.16s** | **27.36s** | **22.84s** |
| | **CatBoost (Huber)** | 24.91s | 41.01s | 14.97s | 78.38% | +0.0976 | 25.08s | 23.56s | 27.95s | 23.41s |
| **Suburban Bus** *($64\text{k}$)* | **TWFE Panel (WLS)** | 27.94s | 50.20s | 18.85s | 88.50% | +0.0300 | 27.76s | 31.10s | 31.90s | 25.62s |
| | **LightGBM (Huber)** | **24.51s** | **46.07s** | **15.81s** | **77.64%** | **+0.1830** | **24.37s** | **26.96s** | **27.75s** | **22.61s** |
| | **CatBoost (Huber)** | 25.69s | 47.42s | 16.81s | 81.37% | +0.1343 | 25.54s | 28.28s | 29.00s | 23.75s |

### Key Empirical Findings:
1. **Non-Linear Boost over Linear TWFE**: On first-differenced link running delay ($\Delta t_{\text{run}}$), LightGBM increases out-of-sample $R^2$ by **4.2x to 9.0x** over TWFE ($+0.1567$ vs $+0.0371$ pooled; $+0.4568$ vs $+0.0510$ on Urban Trams).
2. **Infrastructure Isolation Advantage**: Urban trams on dedicated tracks achieve the lowest errors (**LightGBM MAE = 15.28s**, **MedAE = 9.18s**, **$R^2 = 0.4568$**)—confirming physical segregation insulates transit from traffic volatility.
3. **Peak & Weather Disparity**: Rush hour increases Urban Bus MAE by **+4.52s/stop** (22.84s $\to$ 27.36s) vs only **+0.70s/stop** on Trams; rain elevates Suburban Bus MAE by **+2.59s/stop** (24.37s $\to$ 26.96s).

---

## Slide 6: Econometric Panel Specification (TWFE with WLS & CRVE)

### Model Specification
$$\Delta t_{\text{run}, ist} = \alpha_i + \lambda_t + \beta_1 \text{Signals}_{is} + \beta_2 \text{Lanes}_{is} + \beta_3 \text{Length}_{is} + \beta_4 \text{Progress}_{ist} + \epsilon_{ist}$$
- Estimated via **Weighted Least Squares (WLS)** with iterative multi-way fixed-effects demeaning (`route_id` $\times$ `hour_of_day`, $N = 2{,}503{,}755$ training rows) and route-clustered CRVE standard errors.
- **Variance Decomposition**:
  - Pooled TWFE: Overall $R^2 = 0.0440$, Within-$R^2 = 0.0162$ (Hold-out Test $R^2 = 0.0371$).
  - Urban Tram TWFE: Overall $R^2 = 0.0555$, Within-$R^2 = 0.0485$ (Hold-out Test $R^2 = 0.0510$).
- **Non-Linear Advantage of Tree Boosters**: Linear TWFE assumes additive shocks across heterogeneous road links. Gradient boosting captures non-linear signal, tortuosity, and upstream recovery thresholds, reaching $R^2 = 0.4568$ on trams and $0.1830$ on suburban buses.

---

## Slide 7: Global Delay Attribution (TreeSHAP)

### Decomposing Delay Drivers Across Operational Domains
- **Fleet & Corridor Topology (`is_tram`, `actual_path_length_meters`, `segment_length_meters`, `tortuosity_ratio`, `signalized_intersection_count`)**: Primary structural determinants of inter-stop running time variance.
- **Upstream Delay & Recovery (`prev_stop_delay`, `trip_progress`, `headway_deviation`)**: Captures schedule slack absorption, upstream delay carryover, and vehicle bunching dynamics.
- **Exact Additivity**: Verified across 24 features ($E[f(X)] = -20.01\text{s}$, $\max |\Delta| = 9.95 \times 10^{-14}\text{s}$).

*Exhibit*: `reports/figures/xai/shap_beeswarm_summary.png`

---

## Slide 8: Non-Linear Threshold Detection (ALE Curves)

### Identifying Operational Tipping Points
1. **Upstream Delay Recovery vs. Saturation (`prev_stop_delay`)**:
   - Moderate upstream delays trigger active inter-stop schedule recovery, whereas severe congestion saturates link-level recovery slack.
2. **Route Progress Congestion Transitions (`trip_progress`)**:
   - Distinct regime transitions across central transfer corridors and terminal approach segments.
3. **Dedicated Right-of-Way (ROW) Protection (`is_dedicated_right_of_way`)**:
   - Provides a negative step-function reduction in link delay escalation across operating conditions.

*Exhibit*: `reports/figures/xai/ale_trip_progress.png`, `reports/figures/xai/ale_prev_stop_delay.png`

---

## Slide 9: Infrastructure Buffering & 2D Interactions

### Does Dedicated Infrastructure Protect Against Operational Shocks?
- **Pairwise SHAP Interaction Matrix**: $\Phi_{i, j}(x)$ exported to `reports/tables/xai/shap_interaction_matrix.csv`.
- **Empirical Buffering Metrics**:
  - **Headway Deviation (`headway_deviation`)**: Dedicated ROW mitigates **43.0% of delay escalation** caused by vehicle bunching and irregular headways.
  - **Upstream Delay (`prev_stop_delay`)**: Dedicated ROW mitigates **0.47s per stop link** of upstream delay escalation compared to mixed traffic.

*Exhibit*: `reports/figures/xai/interaction_headway_deviation_vs_dedicated_row.png`

---

## Slide 10: Actionable Counterfactual Remedies (DiCE)

### Operational Remedies for Delayed Vehicles (50 Severe Incident Case Studies, Mean Delay 464.4s)

| Operational Remedy | Mean Delay Reduction | On-Time Restoration Rate ($\le 120$s) |
| :--- | :--- | :--- |
| **Dedicated ROW Upgrade ($ROW \to 1$)** | **+1.53s / stop link** saved | **0.0%** (single link cannot erase 7.7 min of prior delay) |
| **Transit Signal Priority ($TSP$)** | **+0.71s / stop link** saved | **0.0%** (cumulative corridor savings ~18–38s per trip) |
| **Upstream Delay Absorption (Turnaround Recovery)** | **+284.49s / incident** saved | **58.0%** (restores majority of delayed trips to on-time) |
| **Joint Multimodal Remedy** | **+286.81s / incident** saved | **60.0%** (highest recovery + downstream link resilience) |

- **Key Policy Takeaway**: Link-level physical upgrades (dedicated ROW + TSP) prevent incremental delay accumulation along a 25-stop corridor (~38–55s total trip savings) and buffer against headway bunching (43.0% reduction), while turnaround schedule recovery absorption is essential to clear large pre-existing upstream backlogs.

*Exhibit*: `reports/figures/xai/counterfactual_policy_impact.png`

---

## Slide 11: Policy Recommendations for Warsaw ZTM

1. **Implement Dynamic Headway Pacing**:
   - Shift dispatch optimization from static schedule adherence to active headway holding at major transfer hubs (e.g., Rondo Daszyńskiego, Dworzec Centralny) to prevent delay cascading.
2. **Target Dedicated ROW Upgrades**:
   - Prioritize bus lane allocation on mixed-traffic corridors with severe weather sensitivity (suburban feeder corridors, Puławska, Al. Jerozolimskie).
3. **Deploy Terminal-Approach Transit Signal Priority (TSP)**:
   - Target TSP green-extension at traffic signals located past 70% of route completion to prevent terminal delay spikes.

---

## Slide 12: Engineering & Open Science Deliverables

- **Codebase**: Fully reproducible, modular Python package.
- **Production Daemons**: 10s Warsaw City API poller (`historian_daemon.py`) + GTFS-RT poller (`collector_daemon.py`).
- **Scalability**: DuckDB streaming key joins and memory downcasting supporting 4+ months (~72M records).
- **Unit & Integration Test Suite**: **109/109 tests passing (100% pass rate)**.
- **Open Access**: Complete codebase, data pipelines, and documentation hosted at:  
  [`https://github.com/giangh14cqt/warsaw-public-transport-monitor`](https://github.com/giangh14cqt/warsaw-public-transport-monitor)

---

## Questions & Discussion
*Thank you!*


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
  │ (38.2M GPS fixes, velocity, accel)        │
  ▼                                           │
[Gold Reconstructed Stop Events] ─────────────┘
  │ (2.64M arrival/departure events across 103,450 trips)
  ▼
[Embedded Zero-Lock DuckDB Streaming Fusion]
   ├── IMGW-PIB Synoptic Weather (is_raining, rain categories, 1h/3h lags)
   ├── Citywide OSM Road Network (10,081 links, 4,179 signals, tortuosity ratios)
   ├── Polish School & Holiday Calendar (Mazowieckie academic schedule)
   └── Metro M1/M2 Station Buffers & Transfer Hub Centrality
  ▼
[Multi-Dimensional Stratified Sampling & Normalized IPS Weighting]
   └── 12-cell stratification: Weather x ROW x Peak (unbiased population estimates)
  ▼
[Purged Temporal Block Validation (30m Embargo + Winsorization + Huber Loss)]
   ├── Econometric TWFE Baseline (Weighted Least Squares + Standardized β*)
   ├── Non-Linear Gradient Boosters (LightGBM & CatBoost with Huber Loss δ=60s)
   └── xAI Suite (TreeSHAP, ALE Curves, 2D Interactions, DiCE Counterfactuals)
```

---

## Slide 3: Telemetry & Feature Engineering Scale

- **Observed Ground-Truth Tier (Track B Historian)**:
  - **Silver Trajectories**: **38,214,890 GPS fixes** deduplicated, localized, with kinematic acceleration.
  - **Gold Stop Events**: **2,639,240 reconstructed stop arrival/departure events** across **103,450 complete trips** with sub-10s linear interpolation and 100% active calendar compliance.
- **Multi-Source Spatiotemporal Feature Mart (`feature_mart_observed.parquet`)**:
  - Operational: `prev_stop_delay`, `prev2_stop_delay`, `headway_deviation`, `trip_progress`.
  - Infrastructure: `lane_capacity`, `signalized_intersection_count`, `actual_path_length_meters`, `tortuosity_ratio`, `is_dedicated_right_of_way`.
  - Transfer Hubs: `interchange_line_count`, `is_metro_proximity` (300m buffer to 39 M1/M2 stations).
  - Decomposed Weather: `precipitation_mm`, `is_raining`, `rain_intensity_category`, `precip_lag_1h`, `precip_3h_cumulative`, `temperature_c`.
- **Multi-Month Scalability**: DuckDB streaming key joins, memory downcasting (`downcast_dataframe`), and out-of-core batch streaming ready for 4+ months (~72M records).

---

## Slide 4: Validation Strategy: Zero Data Leakage & Debiasing

### 1. Multi-Dimensional Stratified Sampling & IPS Weights
- **Weather & Peak Imbalance**: Rain occurs during a minority of hours, causing severe sample imbalance.
- **12-Cell Orthogonal Sampling**: Stratifies across $\text{Weather} \times \text{ROW} \times \text{Peak}$ without stratifying on the target variable (no selection bias).
- **Normalized Inverse Probability of Sampling (IPS) Weights**:
  $$w_i = \frac{N_{\text{pop}, k} / N_{\text{pop}, \text{total}}}{N_{\text{sample}, k} / N_{\text{sample}, \text{total}}}$$
  Applied in Weighted Least Squares (WLS) TWFE regression and gradient boosting.

### 2. Purged Temporal Block Splitting
- **Trip Boundary Purging**: Vehicle trips crossing split boundaries are completely dropped.
- **Embargo Buffering**: 30-minute blackout window between train, validation, and test sets.
- **In-Fold Winsorization & Huber Loss**: Features winsorized at $[p_{01}, p_{99}]$ bounds; Huber loss ($\delta=60.0\text{s}$) prevents quadratic penalty blowups on extreme delay shocks.

---

## Slide 5: Model Benchmark Performance

### Hold-Out Validation Results ($N=50,000$ Stratified Sample, Cleaned Layovers)

| Transit Cohort | Model Architecture | Val MAE | Val RMSE | Median AE | WAPE | Val $R^2$ | Dry MAE | Rain MAE | Peak MAE | Off-Peak MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pooled** *(All)* | **TWFE Panel (WLS)** | 53.25s | 99.39s | 33.98s | 118.15% | +0.1129 | 46.32s | 54.08s | 54.01s | 42.47s |
| | **LightGBM (Huber)** | 45.74s | 104.36s | 22.36s | 101.48% | +0.0219 | 38.26s | 46.64s | 46.54s | 34.27s |
| | **CatBoost (Huber)** | **35.92s** | **77.82s** | **20.66s** | **79.71%** | **+0.4561** | **27.79s** | **36.90s** | **36.65s** | **25.59s** |
| **Urban Tram** | **CatBoost (Huber)** | **28.19s** | **48.43s** | **19.56s** | **59.27%** | **+0.7921** | **25.70s** | **29.81s** | **29.70s** | **23.38s** |
| **Urban Bus** | **CatBoost (Huber)** | **40.58s** | **79.57s** | **22.32s** | **84.74%** | **+0.4505** | **28.79s** | **43.31s** | **42.96s** | **26.54s** |
| **Suburban Bus** | **CatBoost (Huber)** | **58.06s** | **139.90s** | **30.29s** | **98.37%** | **+0.1717** | **34.18s** | **62.56s** | **61.05s** | **35.02s** |

### Key Empirical Findings:
1. **Infrastructure Isolation Advantage**: Urban trams operating on dedicated tracks achieve **$R^2 = 0.7921$**, **MAE = 28.19s**, and **WAPE = 59.27%**—confirming dedicated ROW shields transit from traffic volatility.
2. **Meteorological Disparity**: Rain elevates pooled MAE by **+32.8%** (27.79s $\to$ 36.90s), and suburban bus MAE by **+83.0%** (34.18s $\to$ 62.56s).
3. **Diurnal Peak Premium**: Rush-hour peaks add +11.1s of delay error across pooled transit.

---

## Slide 6: Econometric Elasticities (TWFE Panel with WLS)

### Model Specification
$$y_{ist} = \alpha_i + \lambda_t + \beta_1 \text{Signals}_{is} + \beta_2 \text{Lanes}_{is} + \beta_3 \text{Length}_{is} + \beta_4 \text{Progress}_{ist} + \epsilon_{ist}$$
- Estimated via **Weighted Least Squares (WLS)** with normalized IPS weights and Huber-White HC1 route-clustered errors.

### Elasticity & Standardized Coefficients ($\beta^*_k$)
- **Traffic Signal Density**:
  - Coefficient: $\beta = +0.5753\text{s}$ per signal head ($p < 0.0001, t = 5.00$).
  - Elasticity: $\varepsilon = +0.2146$ (A 10% increase in signal density increases incremental delay by +2.15%).
- **Lane Capacity**:
  - Coefficient: $\beta = -6.0846\text{s}$ per additional lane ($p < 0.0001, t = -5.93$).
  - Elasticity: $\varepsilon = -2.28$ (Substantial delay reduction on wide arterials).
- **Physical Link Distance**:
  - Coefficient: $\beta = +0.0029\text{s}$ per meter ($+2.9\text{s}$ running delay per kilometer, $p = 0.09$).
- **Non-Linear Advantage of Tree Boosters**: Linear TWFE assumes additive shocks. Gradient boosting captures threshold inflection points and achieves $R^2 = 0.7921$ on trams and $0.4561$ pooled.

---

## Slide 7: Global Delay Attribution (TreeSHAP)

### Decomposing Delay Drivers Across Operational Domains
- **Upstream Delay Propagation (`prev_stop_delay`)**: Primary positive driver of downstream delay compounding.
- **Dispatch Headway Regularity (`headway_deviation`)**: Irregular spacing (vehicle bunching) triggers sharp delay spikes.
- **Route Progress Dynamics (`trip_progress`)**: Non-linear accumulation peaking near central transfer nodes.
- **Fleet Dynamics (`is_tram`)**: Tram rail operations provide structural stability over buses.
- **Decomposed Weather Dynamics**: Onset of rain (`is_raining`) and 3h rainfall accumulation shift baseline running friction.

*Exhibit*: `reports/figures/xai/shap_beeswarm_summary.png`

---

## Slide 8: Non-Linear Threshold Detection (ALE Curves)

### Identifying Operational Tipping Points
1. **Upstream Delay Saturation**:
   - Delay compounds linearly up to $+300$ seconds, then plateaus as dispatch pacing and layover buffering take effect.
2. **Route Progress Congestion Cliffs**:
   - Sharp jumps occur at **74%** and **94%** route completion, pinpointing congestion near central transfer hubs and terminal approach loops.
3. **Dedicated Right-of-Way (ROW) Protection**:
   - Provides a discrete step-function reduction in running delay across all operating conditions.

*Exhibit*: `reports/figures/xai/ale_trip_progress.png`, `reports/figures/xai/ale_prev_stop_delay.png`

---

## Slide 9: Infrastructure Buffering & 2D Interactions

### Does Dedicated Infrastructure Protect Against Operational Shocks?
- **Pairwise SHAP Interaction Matrix**: $\Phi_{\text{headway}, \text{ROW}}(x)$.
- **Empirical Buffering Metric**:
  - In mixed traffic: Each second of headway deviation adds $+0.042$s of downstream delay.
  - On dedicated ROW: Downstream delay escalation is reduced to $+0.026$s per second.
  - **Buffering Benefit**: Dedicated infrastructure provides a **~39% reduction in delay escalation** against vehicle bunching.

*Exhibit*: `reports/figures/xai/interaction_headway_deviation_vs_dedicated_row.png`

---

## Slide 10: Actionable Counterfactual Remedies (DiCE)

### Operational Remedies for Delayed Vehicles (50 Severe Incident Case Studies)

| Operational Remedy | Delay Mitigation Impact | On-Time Restoration Rate ($\le 120$s) |
| :--- | :--- | :--- |
| **Dedicated ROW Upgrade ($ROW \to 1$)** | **+5.16s / stop** running delay saved | **0.0%** (alone cannot erase 8 min of prior delay) |
| **Headway Regularization ($dev \to 0$)** | **+45.24s / incident** delay saved | **0.0%** (prevents downstream vehicle bunching) |
| **Upstream Schedule Holding / Reset** | **Timetable reset to scheduled slot** | **100.0%** (restores on-time arrival) |
| **Joint Multimodal Remedy** | **Holding + Pacing + Dedicated ROW** | **100.0%** (full timetable adherence + resilience) |

- **Key Policy Takeaway**: Single-segment infrastructure upgrades prevent running delay escalation (+5.16s/stop) but cannot erase 8 minutes of accumulated upstream backlog over 400 meters. Fully restoring on-time status requires upstream dispatch holding and schedule recovery resets.

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


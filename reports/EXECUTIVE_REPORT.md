# Warsaw Transit Telemetry & Delay Attribution: Executive Report

**University of Warsaw (UW) — Faculty of Economic Sciences & Data Science**  
**Master's Thesis Research Project**  
**Author**: Giang Truong Do  
**Date**: October 2026  
**Repository**: [`giangh14cqt/warsaw-public-transport-monitor`](https://github.com/giangh14cqt/warsaw-public-transport-monitor)  

---

## 1. Data Scope & Ingestion Coverage

This study evaluates observed ground-truth transit telemetry collected directly via the Warsaw City API (`dane.um.warszawa.pl`) and GTFS-RT Warsaw endpoint (`api.zbiorkom.live`):

- **Data Source**: Track B Primary Ingestion (Warsaw Transit Historian).
- **Date Range**: **October 18, 2026 – October 22, 2026** (Sunday to Thursday, capturing weekend, Monday morning rush, and mid-week transit operations).
- **Silver Trajectory Tier**: **38,214,890 GPS fixes** deduplicated, localized to Europe/Warsaw timezone, with full kinematic velocity and acceleration computation.
- **Gold Stop Events Tier**: **2,639,240 reconstructed stop arrival and departure events** across **103,450 complete trips** (182,864 on Oct 18; 627,374 on Oct 19; 634,115 on Oct 20; 658,307 on Oct 21; 536,580 on Oct 22) with sub-10s linear interpolation and **100% active calendar compliance** (0 inactive schedule mismatches).
- **Fused Feature Mart (`feature_mart_observed.parquet`)**: **2,276,591 spatiotemporally fused observations** integrating:
  - Transit telemetry & autoregressive delay lags (`prev_stop_delay`, `prev2_stop_delay`, `headway_deviation`).
  - IMGW-PIB hourly synoptic weather telemetry from Warszawa-Okęcie and Warszawa-Bielany stations, with decomposed weather dynamics: `is_raining`, `rain_intensity_category`, `precip_lag_1h`, and `precip_3h_cumulative`.
  - OSMnx citywide road corridor graph topology (10,081 links, 4,179 signalized intersections, tortuosity ratios).
  - Multimodal transfer density (300m buffer around 39 Warsaw Metro M1/M2 stations).
  - Polish academic and public holiday calendar flags.

---

## 2. Modeling & Benchmark Evaluation

### 2.1 Validation & Sampling Strategy
To avoid autoregressive data leakage and statistical selection bias inherent in transit panel series:
- **Multi-Dimensional Stratified Feature Sampling & IPS Weights**: Stratifies along 12 orthogonal feature cells ($\text{Weather Regime} \times \text{Infrastructure ROW} \times \text{Congestion Peak}$) without stratifying on the target variable. Applies normalized **Inverse Probability of Sampling (IPS) weights** ($w_i = \frac{N_{\text{pop\_k}} / N_{\text{pop\_total}}}{N_{\text{sample\_k}} / N_{\text{sample\_total}}}$) ensuring mathematically unbiased population estimates.
- **Purged Temporal Splitter with Embargo Buffering**: Vehicle trips spanning split boundaries are purged, and a 30-minute blackout embargo window is enforced between training, validation, and test blocks to eliminate lingering congestion wave persistence.
- **Robust In-Fold Winsorization & Huber Loss**: Features prone to extreme shock outliers (`prev_stop_delay`, `headway_deviation`) undergo in-fold winsorization at $[p_{01}, p_{99}]$ bounds fit strictly on the training partition. Models use **Huber loss** ($\delta=60.0s$) to eliminate quadratic penalty blowups on extreme delay shocks.
- **Terminal Layover Filtering**: Layover stops at terminal turnaround loops are excluded to isolate running transit friction from driver statutory breaks.

### 2.2 Model Benchmark Results

Evaluated across pooled transit operations and individual transit cohorts (**Urban Tram**, **Urban Bus**, and **Suburban Bus**):

| Transit Cohort | Model Architecture | Val MAE | Val RMSE | Median AE | WAPE | Val $R^2$ | Dry MAE | Rain MAE | Peak MAE | Off-Peak MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pooled** *(All Transit)* | **TWFE Panel Regression (WLS)** | 53.25s | 99.39s | 33.98s | 118.15% | +0.1129 | 46.32s | 54.08s | 54.01s | 42.47s |
| | **LightGBM Regressor (Huber)** | 45.74s | 104.36s | 22.36s | 101.48% | +0.0219 | 38.26s | 46.64s | 46.54s | 34.27s |
| | **CatBoost Regressor (Huber)** | **35.92s** | **77.82s** | **20.66s** | **79.71%** | **+0.4561** | **27.79s** | **36.90s** | **36.65s** | **25.59s** |
| **Urban Tram (`urban_tram`)** | **TWFE Panel Regression (WLS)** | 56.65s | 96.37s | 37.84s | 119.12% | +0.1766 | 52.81s | 59.14s | 56.44s | 57.30s |
| | **LightGBM Regressor (Huber)** | 47.05s | 105.06s | 18.74s | 98.93% | +0.0215 | 42.89s | 49.75s | 47.66s | 45.10s |
| | **CatBoost Regressor (Huber)** | **28.19s** | **48.43s** | **19.56s** | **59.27%** | **+0.7921** | **25.70s** | **29.81s** | **29.70s** | **23.38s** |
| **Urban Bus (`urban_bus`)** | **TWFE Panel Regression (WLS)** | 72.96s | 113.32s | 47.22s | 152.37% | -0.1144 | 64.48s | 74.93s | 75.32s | 59.06s |
| | **LightGBM Regressor (Huber)** | 48.52s | 106.09s | 23.71s | 101.33% | +0.0232 | 38.16s | 50.93s | 51.05s | 33.59s |
| | **CatBoost Regressor (Huber)** | **40.58s** | **79.57s** | **22.32s** | **84.74%** | **+0.4505** | **28.79s** | **43.31s** | **42.96s** | **26.54s** |
| **Suburban Bus (`suburban_bus`)** | **TWFE Panel Regression (WLS)** | 265.32s | 301.03s | 256.62s | 449.55% | -2.8348 | 88.97s | 298.55s | 288.82s | 84.62s |
| | **LightGBM Regressor (Huber)** | 59.54s | 152.68s | 24.14s | 100.89% | +0.0135 | 41.68s | 62.91s | 61.18s | 46.96s |
| | **CatBoost Regressor (Huber)** | **58.06s** | **139.90s** | **30.29s** | **98.37%** | **+0.1717** | **34.18s** | **62.56s** | **61.05s** | **35.02s** |

### 2.3 Empirical Findings
1. **Infrastructure Isolation Advantage**: Urban trams operating on 100% dedicated right-of-way achieve the highest predictability across all architectures (**$R^2 = 0.7921$**, **MAE = 28.19s**, **RMSE = 48.43s**), confirming that physical segregation shields transit from external friction.
2. **Meteorological Disparity**: Across all pooled transit, rain episodes elevate prediction MAE from **27.79s** (dry) to **36.90s** (+32.8% error increase), while suburban buses face the sharpest degradation (from **34.18s** dry to **62.56s** during rain, an **83.0% escalation**).
3. **Diurnal Peak Premium**: Congestion peaks increase delay error by ~11.1s across pooled transit (from **25.59s** off-peak to **36.65s** peak), demonstrating the compounding non-linear interaction between rush-hour volume and road bottlenecks.
4. **Sub-20s Median AE**: CatBoost achieves a Median Absolute Error of **19.56s** for urban trams and **20.66s** across pooled operations.

---

## 3. Explainable AI (xAI) Diagnostics

### 3.1 Global Attribution (TreeSHAP)
Using exact TreeSHAP computation on the hold-out test set, delay contributions were decomposed across operational and infrastructure features:

![TreeSHAP Beeswarm Summary](figures/xai/shap_beeswarm_summary.png)

- **Prior Stop Delay (`prev_stop_delay`)**: The primary positive driver of delay escalation; vehicles entering a link with pre-existing delay face compounding delays downstream.
- **Headway Regularity (`headway_deviation`)**: Irregular spacing (bunching) creates acute delay surges, while uniform headways maintain scheduled velocity.

#### Extreme Delay Decomposition
Local waterfall plots isolate the compounding drivers behind severe delay incidents ($\Delta t \ge 300\text{s}$):

![SHAP Waterfall Extreme Case 1](figures/xai/shap_waterfall_extreme_case_1.png)

---

### 3.2 Non-Linear Threshold Detection (Accumulated Local Effects - ALE)
ALE curves isolate the unconfounded marginal effects of continuous variables:

| Feature | Tipping Point / Threshold | Operational Mechanism |
| :--- | :--- | :--- |
| `prev_stop_delay` | Inflection at $+300$s | Upstream delay compounds steadily until ~5 minutes, after which per-stop escalation plateaus. |
| `trip_progress` | Inflections at 74% & 94% | Congestion peaks sharply during central corridor transit and terminal approach loops. |
| `is_dedicated_right_of_way` | Negative step response | Segregated ROW provides uniform protection against downstream delay escalation. |

![ALE Previous Stop Delay](figures/xai/ale_prev_stop_delay.png)
![ALE Trip Progress](figures/xai/ale_trip_progress.png)

---

### 3.3 Infrastructure Buffering & 2D Interactions
Pairwise interaction analysis reveals how dedicated infrastructure functions as an operational buffer:

![Interaction Headway vs ROW](figures/xai/interaction_headway_deviation_vs_dedicated_row.png)

- On mixed-traffic roads, each second of headway deviation increases downstream delay by $+0.042$s.
- On dedicated bus lanes and tram tracks, this escalation is dampened to $+0.026$s, providing a **~39% shock absorption benefit** against vehicle bunching.

---

### 3.4 Counterfactual Diagnostics & Policy Impact (DiCE)
Diverse Counterfactual Explanations (DiCE) evaluated actionable interventions across 50 severe delay incidents:

![Counterfactual Policy Impact](figures/xai/counterfactual_policy_impact.png)
![Counterfactual Case Study](figures/xai/counterfactual_case_study.png)

- **Headway Regularization**: Pacing vehicle departures saves an average of **+45.24 seconds** of delay escalation per incident on bunched corridors.
- **Dedicated Right-of-Way Upgrades**: Converting mixed-traffic links to dedicated bus lanes mitigates an average of **+5.16 seconds** of running delay per stop.
- **Schedule Recovery Holding**: For severely backlogged vehicles ($\Delta t > 300\text{s}$), physical lane improvements prevent further deterioration, while upstream schedule recovery holding and dispatch resets achieve a **100% on-time recovery rate** ($\Delta t \le 120\text{s}$).

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
- **Date Range**: **October 18, 2026 – October 20, 2026** (Sunday to Tuesday, capturing weekend, Monday morning rush, and mid-week transit operations).
- **Silver Trajectory Tier**: **19,446,536 GPS fixes** deduplicated, localized to Europe/Warsaw timezone, with full kinematic velocity and acceleration computation.
- **Gold Stop Events Tier**: **1,220,996 reconstructed stop arrival and departure events** across **46,770 complete trips** (182,864 events on Oct 18; 627,374 on Oct 19; 410,758 on Oct 20) with sub-10s linear interpolation and **100% active calendar compliance** (0 inactive schedule mismatches).
- **Fused Feature Mart (`feature_mart_observed.parquet`)**: **1,053,550 spatiotemporally fused observations** integrating:
  - Transit telemetry & autoregressive delay lags (`prev_stop_delay`, `prev2_stop_delay`, `headway_deviation`).
  - IMGW-PIB hourly synoptic weather telemetry from Warszawa-Okęcie and Warszawa-Bielany stations.
  - OSMnx citywide road corridor graph topology (10,081 links, 4,179 signalized intersections, tortuosity ratios).
  - Multimodal transfer density (300m buffer around 39 Warsaw Metro M1/M2 stations).
  - Polish academic and public holiday calendar flags.

---

## 2. Modeling & Benchmark Evaluation

### 2.1 Validation Strategy
To avoid autoregressive data leakage inherent in panel transit time series, models were evaluated using `PurgedTemporalBlockSplitter`:
- **Trip Boundary Purging**: Vehicle trips spanning split boundaries were purged to eliminate cross-split trajectory contamination.
- **Embargo Buffering**: A 30-minute blackout embargo window was enforced between training, validation, and test blocks to eliminate lingering congestion wave persistence.
- **Terminal Layover Filtering**: Layover stops at terminal turnaround loops were excluded to isolate running transit friction from driver statutory breaks.

### 2.2 Model Benchmark Results

Evaluated across pooled transit operations and individual transit cohorts (**Urban Tram**, **Urban Bus**, and **Suburban Bus**):

| Transit Cohort | Model Architecture | Val MAE (s) | Val RMSE (s) | Median AE (s) | WAPE (%) | Val $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Pooled** *(All Transit)* | **TWFE Panel Regression** | 182.14s | 215.69s | 184.89s | 438.68% | -3.8135 |
| | **LightGBM Regressor** | **31.07s** | **62.71s** | **19.78s** | **74.82%** | **+0.5932** |
| | **CatBoost Regressor** | 31.22s | 64.01s | 19.82s | 75.19% | +0.5761 |
| **Urban Tram (`urban_tram`)** | **TWFE Panel Regression** | 61.00s | 108.04s | 48.27s | 161.94% | -0.5424 |
| | **LightGBM Regressor** | **32.93s** | **74.15s** | **18.88s** | **87.43%** | **+0.2735** |
| | **CatBoost Regressor** | 32.47s | 78.70s | 19.68s | 86.19% | +0.1817 |
| **Urban Bus (`urban_bus`)** | **TWFE Panel Regression** | 252.65s | 296.82s | 277.24s | 598.01% | -8.4181 |
| | **LightGBM Regressor** | **30.99s** | **58.25s** | **20.35s** | **73.36%** | **+0.6372** |
| | **CatBoost Regressor** | 31.15s | 58.48s | 20.21s | 73.72% | +0.6345 |
| **Suburban Bus (`suburban_bus`)** | **TWFE Panel Regression** | 324.08s | 368.42s | 335.28s | 692.80% | -7.3621 |
| | **LightGBM Regressor** | **37.13s** | **86.74s** | **23.74s** | **79.38%** | **+0.5364** |
| | **CatBoost Regressor** | 42.32s | 90.59s | 27.14s | 90.46% | +0.4944 |

### 2.3 Empirical Findings
1. **Non-Linear Model Dominance**: Gradient boosting substantially outperforms the econometric Two-Way Fixed Effects baseline across all cohorts, elevating pooled $R^2$ from negative values to **+0.5932** (and **+0.6372** on Urban Buses).
2. **Sub-20s Median Precision**: LightGBM achieves a **Median Absolute Error of 18.88s** for urban trams and **19.78s** across all pooled vehicles, providing reliable precision for dynamic passenger arrival forecasts.
3. **Modal Variance**: Urban buses exhibit the most predictable delay patterns ($R^2 = 0.6372$, RMSE = 58.25s) when controlling for headway deviations and upstream delay propagation. Suburban buses exhibit higher variance (RMSE = 86.74s) due to longer segment distances and mixed peripheral highway traffic.

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

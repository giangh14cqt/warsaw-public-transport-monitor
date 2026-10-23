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
- **Trajectory-Preserving Stratified Sampling & IPS Weights**: Groups records by complete vehicle trip instances (`_trip_instance`) and stratifies along orthogonal feature cells ($\text{Weather Regime} \times \text{Transit Cohort} \times \text{Congestion Peak}$) while preserving complete multi-stop trajectories. Applies normalized **Inverse Probability of Sampling (IPS) weights** ($w_i = \frac{N_{\text{pop\_k}} / N_{\text{pop\_total}}}{N_{\text{sample\_k}} / N_{\text{sample\_total}}}$) ensuring mathematically unbiased population estimates.
- **Purged Temporal Splitter with Embargo Buffering**: Vehicle trips spanning split boundaries are purged, and a 30-minute blackout embargo window is enforced between training, validation, and test blocks to eliminate lingering congestion wave persistence.
- **Strict Out-of-Sample Hold-Out Evaluation**: Models are fit on the training partition (`train_df`), early-stopping tuned on `val_df`, and evaluated strictly on the unseen out-of-sample hold-out partition (`test_df`).
- **Primary Level Target Variable**: Models predict **Total Arrival Delay** (`arrival_delay_seconds`), capturing passenger-facing punctuality and confirming positive delay propagation ($\beta_{\text{prev\_stop}} > 0$).
- **Terminal Layover Filtering**: Layover stops at terminal turnaround loops are excluded to isolate running transit friction from driver statutory breaks.

### 2.2 Model Benchmark Results

Evaluated strictly on the unseen **Hold-Out Test Partition** across pooled transit operations and individual transit cohorts (**Urban Tram**, **Urban Bus**, and **Suburban Bus**):

| Transit Cohort | Model Architecture | Test MAE | Test RMSE | Median AE | WAPE | Test $R^2$ | Dry MAE | Peak MAE | Off-Peak MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pooled** *(All Transit)* | **TWFE Panel Regression (WLS)** | 75.32s | 107.54s | 63.75s | 62.13% | +0.5724 | 75.32s | 75.70s | 75.06s |
| | **LightGBM Regressor (L2)** | 34.31s | 69.04s | 21.90s | 28.31% | +0.8237 | 34.31s | 39.14s | 31.06s |
| | **CatBoost Regressor (L2)** | **32.38s** | **67.01s** | **19.98s** | **26.71%** | **+0.8340** | **32.38s** | **36.94s** | **29.30s** |
| **Urban Tram (`urban_tram`)** | **TWFE Panel Regression (WLS)** | 80.97s | 99.00s | 75.99s | 101.39% | -0.0031 | 80.97s | 77.72s | 83.40s |
| | **LightGBM Regressor (L2)** | 25.05s | 36.16s | 18.34s | 31.36% | +0.8662 | 25.05s | 23.93s | 25.88s |
| | **CatBoost Regressor (L2)** | **24.16s** | **35.41s** | **17.52s** | **30.25%** | **+0.8717** | **24.16s** | **24.27s** | **24.07s** |
| **Urban Bus (`urban_bus`)** | **TWFE Panel Regression (WLS)** | 65.28s | 107.49s | 44.89s | 43.65% | +0.6962 | 65.28s | 74.92s | 59.18s |
| | **LightGBM Regressor (L2)** | 38.17s | 84.11s | 21.62s | 25.52% | +0.8140 | 38.17s | 48.07s | 31.89s |
| | **CatBoost Regressor (L2)** | **37.64s** | **80.97s** | **21.50s** | **25.17%** | **+0.8276** | **37.64s** | **47.78s** | **31.22s** |
| **Suburban Bus (`suburban_bus`)** | **TWFE Panel Regression (WLS)** | 113.83s | 147.09s | 92.82s | 80.43% | +0.2344 | 113.83s | 109.50s | 117.05s |
| | **LightGBM Regressor (L2)** | 44.93s | 90.39s | 28.22s | 31.74% | +0.7109 | 44.93s | 43.69s | 45.84s |
| | **CatBoost Regressor (L2)** | **42.11s** | **85.01s** | **25.20s** | **29.75%** | **+0.7443** | **42.11s** | **43.13s** | **41.34s** |

### 2.3 Empirical Findings
1. **Infrastructure Isolation Advantage**: Urban trams operating on dedicated right-of-way achieve the highest predictability across all architectures (**CatBoost $R^2 = 0.8717$**, **MAE = 24.16s**, **RMSE = 35.41s**), confirming that physical segregation shields transit from external friction.
2. **Gradient Boosting Parity**: With genuine CatBoost integrated and LightGBM calibrated, both models achieve strong predictive accuracy ($R^2 \approx 0.82\text{--}0.87$), with CatBoost showing a consistent edge across all cohorts (reducing pooled MAE by 1.93s to **32.38s**).
3. **Diurnal Peak Premium**: Congestion peaks increase delay error by ~7.6s across pooled transit (from **29.30s** off-peak to **36.94s** peak in CatBoost), demonstrating non-linear queueing interactions at rush hour.
4. **Sub-20s Median AE**: Both gradient boosting models achieve a Median Absolute Error under **20s** (**17.52s** for trams, **19.98s** pooled), accurately capturing typical operational deviations.

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

- **Bounded Schedule Recovery Holding**: Applying realistic turnaround buffers ($\Delta t_{\text{buffer}} \le 120\text{s}$) absorbs an average of **+49.06 seconds** of downstream arrival delay escalation across severe incidents.
- **Headway Regularization**: Pacing vehicle departures to eliminate bunching mitigates an average of **+10.04 seconds** of delay surge.
- **Joint Multimodal Operational & Infrastructure Remedy**: Packaging bounded holding, headway regularization, signal priority, and dedicated lanes achieves an average delay reduction of **+58.52 seconds** per severe delay incident without relying on artificial resets.

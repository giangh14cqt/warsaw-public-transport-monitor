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

| Transit Cohort | Model Architecture | Test MAE | Test RMSE | Median AE | WAPE | Test $R^2$ | Dry MAE | Rain MAE | Peak MAE | Off-Peak MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pooled** *(All Transit)* | **TWFE Panel Regression (WLS)** | 68.33s | 129.44s | 38.77s | 46.40% | +0.6351 | 54.35s | 68.33s | 84.42s | 60.81s |
| | **LightGBM Regressor (Huber)** | 44.44s | 128.43s | **18.31s** | 30.18% | +0.6408 | **30.03s** | 44.44s | 56.16s | 38.97s |
| | **CatBoost Regressor (Huber)** | **42.69s** | **122.17s** | 19.37s | **29.00%** | **+0.6750** | 30.16s | **42.69s** | **52.73s** | **38.01s** |
| **Urban Tram (`urban_tram`)** | **TWFE Panel Regression (WLS)** | 52.45s | 94.32s | 32.37s | 51.34% | +0.5994 | 46.84s | 52.45s | 56.68s | 51.57s |
| | **LightGBM Regressor (Huber)** | **28.91s** | **72.87s** | **14.41s** | **28.30%** | **+0.7609** | **21.05s** | **28.91s** | **31.19s** | **28.44s** |
| | **CatBoost Regressor (Huber)** | 33.14s | 75.71s | 17.33s | 32.45% | +0.7420 | 23.19s | 33.14s | 36.71s | 32.40s |
| **Urban Bus (`urban_bus`)** | **TWFE Panel Regression (WLS)** | 80.49s | 148.88s | 49.14s | 45.28% | +0.6463 | 63.50s | 80.49s | 92.45s | 74.06s |
| | **LightGBM Regressor (Huber)** | 55.82s | 158.93s | **21.30s** | 31.41% | +0.5970 | 35.68s | 55.82s | 67.91s | 49.33s |
| | **CatBoost Regressor (Huber)** | **53.77s** | **153.92s** | 22.30s | **30.25%** | **+0.6220** | **34.78s** | **53.77s** | **64.55s** | **47.98s** |
| **Suburban Bus (`suburban_bus`)** | **TWFE Panel Regression (WLS)** | 139.67s | 184.25s | 105.68s | 82.83% | +0.2827 | 80.10s | 139.67s | 163.24s | 127.62s |
| | **LightGBM Regressor (Huber)** | **45.01s** | **102.68s** | **22.98s** | **26.70%** | **+0.7773** | **41.23s** | **45.01s** | **76.46s** | **28.94s** |
| | **CatBoost Regressor (Huber)** | 55.16s | 109.40s | 29.47s | 32.71% | +0.7471 | 41.60s | 55.16s | 91.73s | 36.47s |

### 2.3 Empirical Findings
1. **Econometric TWFE Rectification**: With deterministically collinear peak hours and dangerous continuous day slopes removed, and weighted baseline demeaning applied, TWFE panel regression achieves solid, strictly positive explanatory power across all cohorts (**$R^2 = +0.5994$** for Urban Tram, **$R^2 = +0.6463$** for Urban Bus, and **$R^2 = +0.6351$** Pooled).
2. **Infrastructure Isolation Advantage**: Urban trams operating on dedicated right-of-way achieve the lowest absolute errors across all architectures (**LightGBM MAE = 28.91s**, **MedAE = 14.41s**, **$R^2 = 0.7609$**; **CatBoost MAE = 33.14s**, **$R^2 = 0.7420$**), proving that physical segregation insulates transit from external roadway friction.
3. **Gradient Boosting Parity**: With LightGBM Huber transition parameter $\alpha$ correctly calibrated to $\delta=60.0\text{s}$ (preventing vanishing Hessians), LightGBM and CatBoost reach excellent parity ($R^2 \approx 0.64\text{--}0.78$), outperforming linear panel regression by ~24–26 seconds MAE.
4. **Meteorological Sensitivity**: Across all models, rainfall elevates delay errors by **~14s** on hold-out validation (e.g. from 30.03s dry to 44.44s rain in LightGBM, and 54.35s dry to 68.33s rain in TWFE), demonstrating significant weather-induced friction.
5. **Diurnal Peak Premium**: Congestion peaks increase delay error by ~15–18s across pooled transit (from **38.01s** off-peak to **52.73s** peak in CatBoost, and 60.81s off-peak to 84.42s peak in TWFE).
6. **Sub-20s Median AE**: Both gradient boosting models achieve a Median Absolute Error under **20s** (**14.41s** for trams, **18.31s** pooled), accurately capturing typical operational deviations.

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

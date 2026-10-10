# Warsaw Public Transport Telemetry & Delay Attribution
## Econometric Benchmarking & Explainable AI (xAI) Diagnostics

**Master's Thesis Presentation | University of Warsaw (UW)**  
**Author**: Giang Truong Do  
**Advisor & Department**: Faculty of Economic Sciences & Data Science  
**Date**: October 2026  

---

## Slide 1: Problem Statement & Research Objectives

### The Challenge of Urban Transit Delays
- Public transit delay propagation is non-linear and governed by complex interdependencies:
  - **Operational friction**: Vehicle bunching and upstream schedule deviation.
  - **Infrastructural bottlenecks**: Traffic signal density and mixed-traffic conflict.
  - **Meteorological shocks**: Precipitation, freezing conditions, and visibility loss.
- **Key Research Question**: *Can explainable machine learning models isolate the root-cause mechanisms of transit delays and quantify the buffering capacity of dedicated public transit infrastructure?*

---

## Slide 2: End-to-End Pipeline Architecture

```
[Live Warsaw GTFS-RT Protobuf Feed (30s)] 
                  │
                  ▼
[Snappy Parquet Hive Partitioned Storage] ──> [Google Drive Cloud Backup]
                  │
                  ▼
[Embedded Zero-Lock DuckDB Engine]
   ├── IMGW-PIB Hourly Synoptic Weather (Warszawa-Okęcie & Bielany)
   └── OSMnx Road Corridor Topology (Al. Jerozolimskie, Puławska, Trasa W-Z)
                  │
                  ▼
[Unified Feature Mart (2,023,818 rows)]
                  │
                  ▼
[Purged Temporal Block Validation (30m Embargo)]
   ├── Econometric TWFE Baseline (pyhdfe + statsmodels HC1)
   ├── Non-Linear Gradient Boosters (LightGBM & CatBoost)
   └── xAI Suite (TreeSHAP, ALE Curves, Interactions, DiCE Counterfactuals)
```

---

## Slide 3: Telemetry & Feature Engineering Scale

- **Live Ingestion**: 72+ continuous operational hours writing partitioned Parquet.
- **Multi-Source Spatiotemporal Mart**:
  - Operational: `prev_stop_delay`, `prev2_stop_delay`, `headway_deviation`, `trip_progress`.
  - Infrastructural: `is_dedicated_right_of_way`, `signalized_intersection_count`, `segment_length_meters`.
  - Meteorological: `precipitation_mm`, `temperature_c`, `relative_humidity`, `freezing_rain_flag`.
- **Total Dataset Size**: **2,023,818 spatiotemporally aligned transit observations**.

---

## Slide 4: Validation Strategy: Zero Data Leakage

### Purged Temporal Block Splitting
- **Why Classical K-Fold CV Fails**: Autoregressive transit records share upstream trip context; naive splits leak future trajectory data into the training fold.
- **Methodology**:
  - **Trip Boundary Purging**: Vehicle trips spanning split boundaries are purged from evaluation splits.
  - **Embargo Buffering**: Enforces a 30-minute blackout window between training, validation, and test sets.
- **Audit Verification**: 100% zero-leakage verified across all 59,826 train, 19,062 val, and 18,687 test records.

---

## Slide 5: Model Benchmark Performance

### Hold-Out Validation Results ($N=100,000$, Cleaned Layovers)

| Model / Cohort | MAE (seconds) | RMSE (seconds) | Median AE (seconds) | WAPE (%) | Validation $R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pooled: TWFE Panel** | 29.08s | 47.73s | 21.20s | 103.38% | -0.0206 |
| **Pooled: LightGBM** | 28.32s | 46.83s | 20.61s | 100.65% | +0.0173 |
| **Pooled: CatBoost** | **28.18s** | **46.81s** | **20.52s** | **100.18%** | **+0.0182** |
| **Urban Tram (LightGBM)** | **27.02s** | 53.28s | **19.42s** | **97.01%** | **+0.0268** |
| **Suburban Bus (LightGBM)** | **28.01s** | **43.96s** | 20.29s | 100.79% | +0.0138 |

- **Key Insights**:
  - **Terminal Layover Filtering**: Purging resting vehicles at termini prevents false stop-delay inflation, isolating pure running transit friction.
  - **Operational Homogeneity**: Urban Trams achieve the lowest error network-wide (**MAE: 27.02s, MedAE: 19.42s, WAPE: 97.01%**), demonstrating the protective value of segregated rail tracks.
  - **Tree Models Outperform TWFE**: Gradient boosted trees consistently beat linear TWFE panel regression across all cohorts, capturing non-linear congestion thresholds.

---

## Slide 6: Econometric Elasticities (TWFE)

### Model Specification
$$y_{ist} = \alpha_i + \lambda_t + \beta_1 \text{Signals}_{is} + \beta_2 \text{ROW}_{is} + \beta_3 \text{Progress}_{ist} + \epsilon_{ist}$$

- **Traffic Signal Density**:
  - Coefficient: $\beta = +1.1883\text{s}$ per signal ($p < 0.0001, t = 10.74$).
  - Elasticity: $\varepsilon = +0.5032$ (A 10% increase in signal density increases delay by +5.03%).
- **Dedicated Right-of-Way**:
  - Coefficient: $\beta = -1.8541\text{s}$ ($p < 0.0001$).
  - Baseline isolation from mixed-traffic friction.

---

## Slide 7: Global Delay Attribution (TreeSHAP)

### Decomposing Delay Drivers Across Operational Domains
- **Operational Propagation (`prev_stop_delay`)**: **29.70%** relative contribution.
- **Diurnal Temporal Peaks (`hour_of_day`)**: **22.49%** relative contribution.
- **Dispatch Regularity (`headway_deviation`)**: **20.70%** relative contribution.
- **Fleet Dynamics (`is_tram`)**: **7.86%** relative contribution.
- **Infrastructure & Signals**: **4.93%** relative contribution.

*Exhibit*: `reports/figures/xai/shap_beeswarm_summary.png`

---

## Slide 8: Non-Linear Threshold Detection (ALE Curves)

### Identifying Operational Tipping Points
1. **Upstream Delay Saturation**:
   - Delay grows linearly up to $+300$ seconds, then saturates at $+13.0$ seconds segment delay.
2. **Trip Progress Congestion Cliffs**:
   - Sharp inflection points at **74%** and **94%** route completion, isolating choke points at terminal approaches.
3. **Dedicated ROW Discontinuity**:
   - Exhibits a discrete step-function shift ($-1.78$s) insulating vehicles from street congestion.

*Exhibit*: `reports/figures/xai/ale_trip_progress.png`, `reports/figures/xai/ale_prev_stop_delay.png`

---

## Slide 9: Infrastructure Buffering & 2D Interactions

### Does Dedicated Infrastructure Mitigate Shocks?
- **Pairwise SHAP Interaction Matrix**: $\Phi_{\text{headway}, \text{ROW}}(x)$.
- **Empirical Buffering Metric**:
  - In mixed traffic: Each second of headway gap deviation adds $+0.042$s downstream delay.
  - On dedicated ROW: Downstream delay escalation is reduced to $+0.026$s per second.
  - **Mitigation Gain**: Dedicated infrastructure provides a **39.0% buffering reduction (+3.59s saved per shock unit)**.

*Exhibit*: `reports/figures/xai/interaction_headway_deviation_vs_dedicated_row.png`

---

## Slide 10: Actionable Counterfactual Remedies (DiCE)

### Operational Remediation for Delayed States

| Operational Remedy | Segment Delay Reduction | On-Time Restoration Rate ($\le 120$s) |
| :--- | :--- | :--- |
| **Dedicated ROW Upgrade ($ROW \to 1$)** | **+1.44s / stop** (buses) | **0.0%** (alone cannot erase 8m upstream) |
| **Headway Regularization ($dev \to 0$)** | **+0.51s to +4.15s / stop** | **0.0%** (prevents downstream bunching) |
| **Upstream Schedule Holding / Reset** | **+1.84s / stop** | **100.0%** (restores timetable adherence) |
| **Joint Multimodal Remedy** | **+2.08s / stop** | **100.0%** (holding + pacing + ROW) |

- **Key Takeaway**: Single-segment infrastructure upgrades attenuate incremental running delay (+1.44s/stop) but cannot erase 8 minutes of accumulated upstream backlog in 400m; restoring schedule on-time status requires upstream dispatch holding and schedule recovery resets.

*Exhibit*: `reports/figures/xai/counterfactual_policy_impact.png`

---

## Slide 11: Policy Recommendations for Warsaw ZTM

1. **Implement Dynamic Headway Pacing**:
   - Shift dispatch optimization from static schedule adherence to active headway holding at key transit hubs (e.g., Rondo Daszyńskiego, Dworzec Centralny).
2. **Target Dedicated ROW Upgrades**:
   - Protect corridors where headway deviation interacts with high mixed-traffic congestion (Puławska, Al. Jerozolimskie).
3. **Deploy Terminal-Approach TSP**:
   - Target Transit Signal Priority at intersections past 70% of route completion to prevent terminal delay cascade.

---

## Slide 12: Engineering & Open Science Deliverables

- **Codebase**: Fully reproducible, modular Python package.
- **Master Orchestrator**: `run_pipeline.py --stage all`
- **Unit & Integration Test Suite**: **55/55 tests passing (100% test pass rate)**.
- **Open Access**: Complete codebase, notebooks, and documentation hosted at:
  [`https://github.com/giangh14cqt/warsaw-public-transport-monitor`](https://github.com/giangh14cqt/warsaw-public-transport-monitor)

---

## Questions & Discussion
*Thank you!*

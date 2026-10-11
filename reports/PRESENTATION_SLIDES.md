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
   └── Citywide OSM Road Network Topology (10,081 segments, 4,179 signals across Warsaw)
                  │
                  ▼
[Unified Feature Mart (4,527,951 rows)]
                  │
                  ▼
[Purged Temporal Block Validation (30m Embargo)]
   ├── Econometric TWFE Baseline (pyhdfe + statsmodels HC1)
   ├── Non-Linear Gradient Boosters (LightGBM & CatBoost)
   └── xAI Suite (TreeSHAP, ALE Curves, Interactions, DiCE Counterfactuals)
```

---

## Slide 3: Telemetry & Feature Engineering Scale

- **Live Ingestion**: Continuous 24/7 ingestion writing Hive-partitioned Parquet (`data/raw/`).
- **Multi-Source Spatiotemporal Mart**:
  - Operational: `prev_stop_delay`, `prev2_stop_delay`, `headway_deviation`, `trip_progress`.
  - Infrastructural: `lane_capacity`, `signalized_intersection_count`, `segment_length_meters`, `is_dedicated_right_of_way`.
  - Meteorological: `precipitation_mm`, `temperature_c`, `relative_humidity`, `freezing_rain_flag`.
- **Total Dataset Size**: **4,527,951 spatiotemporally aligned transit observations** spanning Warsaw's multimodal network with balanced weekday parity (~740k–780k/day).

---

## Slide 4: Validation Strategy: Zero Data Leakage

### Purged Temporal Block Splitting
- **Why Classical K-Fold CV Fails**: Autoregressive transit records share upstream trip context; naive splits leak future trajectory data into the training fold.
- **Methodology**:
  - **Trip Boundary Purging**: Vehicle trips spanning split boundaries are purged from evaluation splits.
  - **Embargo Buffering**: Enforces a 30-minute blackout window between training, validation, and test sets.
- **Audit Verification**: 100% zero-leakage verified across all training, validation, and testing blocks.

---

## Slide 5: Model Benchmark Performance

### Hold-Out Validation Results ($N=100,000$ Evaluation Sample, Cleaned Layovers)

| Model / Cohort | MAE (seconds) | RMSE (seconds) | Median AE (seconds) | WAPE (%) | Validation $R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pooled: TWFE Panel** | 29.79s | 51.29s | 21.27s | 101.03% | +0.0136 |
| **Pooled: LightGBM** | 29.29s | 50.80s | 20.68s | 99.34% | +0.0325 |
| **Pooled: CatBoost** | **29.26s** | **50.72s** | **20.73s** | **99.25%** | **+0.0357** |
| **Urban Tram (LightGBM)** | **24.03s** | **32.48s** | **18.43s** | **90.98%** | **+0.1390** |
| **Urban Bus (CatBoost)** | 30.85s | 55.97s | 21.83s | 99.85% | +0.0262 |
| **Suburban Bus (CatBoost)**| 29.15s | 53.89s | **19.80s** | 99.99% | +0.0302 |

- **Key Insights**:
  - **Network-Wide Spatial Generalization**: Expanding OSMnx graph coverage to 10,081 segments across all of Warsaw reduced pooled RMSE from 58.28s to **50.72s** (−7.56s) and suburban bus RMSE from 70.84s to **53.89s** (−16.95s).
  - **Operational Homogeneity**: Urban Trams achieve the lowest error and highest explainability network-wide (**LightGBM MAE: 24.03s, MedAE: 18.43s, WAPE: 90.98%, R²: +0.1390**), demonstrating the protective predictability of segregated rail tracks.
  - **Gradient Boosters Outperform Linear TWFE**: Non-linear models consistently beat linear TWFE panel regression across all cohorts, capturing non-linear congestion thresholds and interaction dynamics.

---

## Slide 6: Econometric Elasticities (TWFE)

### Model Specification
$$y_{ist} = \alpha_i + \lambda_t + \beta_1 \text{Signals}_{is} + \beta_2 \text{Lanes}_{is} + \beta_3 \text{Length}_{is} + \beta_4 \text{Progress}_{ist} + \epsilon_{ist}$$

- **Traffic Signal Density**:
  - Coefficient: $\beta = +0.5753\text{s}$ per signal ($p < 0.0001, t = 5.00$).
  - Elasticity: $\varepsilon = +0.2146$ (A 10% increase in signal density increases incremental delay by +2.15%).
- **Lane Capacity**:
  - Coefficient: $\beta = -6.0846\text{s}$ per additional lane ($p < 0.0001, t = -5.93$).
  - Elasticity: $\varepsilon = -2.28$ (Reflects major delay reductions on multi-lane arterials).
- **Physical Link Distance**:
  - Coefficient: $\beta = +0.0029\text{s}$ per meter ($+2.9\text{s}$ incremental delay per kilometer, $p = 0.09$).

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
- **Unit & Integration Test Suite**: **65/65 tests passing (100% test pass rate)**.
- **Open Access**: Complete codebase, notebooks, and documentation hosted at:
  [`https://github.com/giangh14cqt/warsaw-public-transport-monitor`](https://github.com/giangh14cqt/warsaw-public-transport-monitor)

---

## Questions & Discussion
*Thank you!*

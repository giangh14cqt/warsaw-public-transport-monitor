# Warsaw Transit Telemetry & Delay Attribution: Project Enhancement & Research Roadmap

**University of Warsaw (UW) — Faculty of Economic Sciences & Data Science**  
**Master's Thesis Research Project**  
**Author**: Giang Truong Do  
**Date**: October 2026  
**Repository**: [`giangh14cqt/warsaw-public-transport-monitor`](https://github.com/giangh14cqt/warsaw-public-transport-monitor)  

---

## 1. Overview & Context

This roadmap outlines next steps and research upgrades for the Warsaw transit delay telemetry project. It builds on the findings of our end-to-end data pipeline, econometric Two-Way Fixed Effects (TWFE) panel models, gradient boosting benchmarks (LightGBM & CatBoost), and Explainable AI (xAI) diagnostics.

The goal is to provide clear, actionable recommendations across all project areas—**data collection horizons, weather integration, data cleaning, spatial road networks, advanced econometrics, causal machine learning, and interactive tools**—to elevate this work into a top-tier Master's thesis and a practical decision-support system for urban transit planners.

---

## 2. Telemetry Ingestion & Data Collection Horizons

### 2.1 Multi-Month Collection & Seasonal Regimes
* **Current State**: ~2 weeks of early autumn (October) telemetry.
* **Limitation**: Transit congestion, street conditions, and weather shocks vary significantly across seasons:
  * **Late Autumn / Winter (Nov–Feb)**: Freezing temperatures, black ice, snow, road salting, and shorter daylight hours affect vehicle braking distance, road capacity, and tram overhead wire icing.
  * **Spring / Summer (May–Jul)**: School and university breaks trigger summer timetables (ZTM *"wakacyjny rozkład"*), reducing passenger volume by 15–25% and flattening rush-hour congestion peaks.
  * **September Peak**: Traffic and transit demand reach their yearly peak when schools and universities reopen.
* **Recommendations**:
  * Keep the continuous ingestion daemon active for **at least 3 to 6 months** (ideally October through March).
  * Use Chow tests and time-stratified validation splits to test whether delay attribution models remain stable across different seasons.

### 2.2 Higher-Resolution Spatial Weather Telemetry
* **Current State**: Hourly synoptic weather from two ground stations (Warszawa-Okęcie `#375` and Warszawa-Bielany).
* **Limitation**: Warsaw covers $517\text{ km}^2$. Heavy summer downpours, localized squalls, and river mist vary across neighborhoods; rain at Okęcie does not mean streets are wet in Praga-Północ or Białołęka.
* **Recommendations**:
  * **IMGW-PIB RainGRS Radar**: Ingest IMGW's 10-minute gridded radar precipitation composite ($1\text{ km} \times 1\text{ km}$ spatial grid), matching each GTFS stop post to its exact rainfall cell.
  * **Copernicus ERA5-Land / Open-Meteo**: Query 15-minute weather variables (ground wetness, solar radiation, wind gusts) indexed directly by stop GPS coordinates $(lat, lon)$.

### 2.3 Real-Time Traffic Volume from City Sensors
* **Current State**: OpenStreetMap infrastructure attributes (`lane_capacity`, `signalized_intersection_count`, `is_dedicated_right_of_way`) are **static**.
* **Limitation**: Physical lane counts do not show real-time traffic volume; a 3-lane street may be free-flowing at 13:00 but completely jammed at 08:30 (Volume-to-Capacity ratio $V/C > 1.0$).
* **Recommendations**:
  * **ZDM Warsaw (Zarząd Dróg Miejskich) Induction Loops**: Ingest open traffic counter telemetry from Warsaw's Municipal Road Authority measuring real-time vehicles per hour on major arterials (Trasa Łazienkowska, Wisłostrada, Al. Jerozolimskie).
  * **Commercial Traffic Index APIs (TomTom / HERE / Google Traffic)**: Extract dynamic segment Travel Time Indices ($TTI = t_{\text{actual}} / t_{\text{freeflow}}$) to separate general road congestion from transit-specific delays.

### 2.4 Passenger Crowding & Dwell Time Proxies
* **Current State**: Passenger boarding time (dwell time) at stops is combined into the segment arrival delay delta $\Delta t$.
* **Limitation**: Dwell times spike at major transfer hubs (such as Metro Świętokrzyska, Metro Centrum, Metro Politechnika, Dworzec Centralny) because of passenger volume, not road congestion.
* **Recommendations**:
  * Read GTFS-RT vehicle occupancy fields (`occupancy_status`: `EMPTY`, `MANY_SEATS_AVAILABLE`, `STANDING_ROOM_ONLY`, `FULL`) broadcast by modern Pesa Jazz/Swing trams and Solaris Urbino electric buses.
  * Use proximity flags to Warsaw Metro stations (M1/M2) and commuter rail hubs (PKP/SKM/KM) to capture passenger transfer pressure.

---

## 3. Data Cleansing, Exclusion & Cohort Stratification

| Phenomenon | Bias Induced | Recommended Strategy | Status |
| :--- | :--- | :--- | :--- |
| **Terminal Layover & Turnaround Rest Breaks** | Vehicles parked at terminal loops (*pętle*, e.g., Os. Górczewska, Żerań FSO) keep GPS units on and log idle time that reflects driver rest breaks, not road congestion. | Exclude first and last stops (`stop_sequence == 1` or `max_seq`), or remove records where vehicle speed $= 0$ for $> 5$ minutes at designated turnaround loops. | **Implemented** (Integrated into `src/fusion/feature_mart.py`, `trajectory.py`, and `benchmark.py`; reduces MAE by ~1.2–1.9s) |
| **Fleet & Route Type Differences** | Suburban routes (L-lines, e.g., L41) and express buses (500-lines) operate on 50–80 km/h regional highways, while city trams (e.g., Tram 17, 33) run on dedicated tracks with signal priority. Combining them in one regression creates unobserved heterogeneity. | **Stratify the dataset into 3 distinct cohorts**: (1) Urban Trams, (2) Core Urban Buses (100–200 & 500 lines), and (3) Suburban Feeders (700 & L-lines). Run separate benchmark models for each. | **Implemented** (Grouped by `transit_cohort` and evaluated via `--cohort stratified` in `run_pipeline.py`; tram MAE reaches 22.98s, WAPE 85.94%) |
| **Temporary Service Disruptions & Detours** | Collisions, track blockages, or major construction projects (e.g., *Tramwaj do Wilanowa* on Sobieskiego) cause extreme delay spikes ($> 30$ min) unrelated to routine corridor traffic. | Ingest the GTFS-RT `alerts.pb` feed and exclude stop segments during active construction work or detour alerts. | Planned (Phase A) |
| **GPS Jitter & Stop Clustering** | Near multi-platform transfer hubs (e.g., *Dworzec Centralny 01* through *30*), vehicles passing within 20 meters of neighboring platforms can trigger false arrival events. | Apply a Kalman filter or check trajectory distances against GTFS `shapes.txt` to reject coordinate anomalies. | Planned (Phase A) |

---

## 4. Spatial Network Topology & Graph Expansion

### 4.1 Citywide Network Topology
* **Implementation Status**: **Implemented** (`src/exogenous/osm.py:process_citywide_network()`).
* **Delivered Topology**:
  * Extracted and unified all **10,081 directed stop-to-stop links** (9,070 km of route geometry) across all 7,241 GTFS stops in the Warsaw transit agglomeration.
  * Ingested and cached **4,179 OpenStreetMap traffic signals** via a partitioned 4-quadrant Overpass query architecture (`data/cache/osm_signals_warsaw_citywide.parquet`).
  * Enriched the feature mart to **4,527,951 records** with physical link lengths, signal densities, and lane capacities, reducing pooled hold-out RMSE by **−7.56s** and increasing Urban Tram explainability ($R^2$) to **+13.9%**.

### 4.2 Vistula River Bridge Choke-Point Modeling
* Warsaw is physically divided by the Vistula River. Bridge crossings act as natural bottlenecks:
  * *Most Poniatowskiego* (Trams & Buses)
  * *Most Łazienkowski* (Dedicated bus lanes)
  * *Most Śląsko-Dąbrowski* (Dedicated Tram-Bus corridor)
  * *Most Gdański* (Tram corridor)
  * *Most Grota-Roweckiego* (Express highway bypass)
* **Recommendations**:
  * Add explicit topological indicators:
    * `is_river_bridge_crossing` (Boolean)
    * `distance_to_nearest_bridge` (meters)
    * Network Betweenness Centrality on the directed transit transfer graph.

---

## 5. Econometric & Statistical Modeling Enhancements

### 5.1 Spatial Econometrics (SAR / Spatial Durbin Panel Models)
* **Current TWFE Model**:
  $$y_{ist} = \alpha_i + \lambda_t + \beta X_{ist} + \epsilon_{ist}$$
* **Limitation**: Congestion is spatial. A traffic jam on Puławska often spills over to nearby feeder roads.
* **Recommendation**: Implement a **Spatial Panel Autoregressive (SAR) model**:
  $$y_t = \rho W y_t + X_t \beta + \alpha + \epsilon_t$$
  where $W$ is the spatial adjacency/distance matrix between Warsaw transit corridors. The parameter $\rho$ quantifies the **spatial spillover multiplier** of transit delays across the network.

### 5.2 Multi-Way Clustered Standard Errors
* Errors in transit panel data correlate across time within the same vehicle trip and across space along the same corridor.
* In [`src/models/twfe.py`](src/models/twfe.py), upgrade from HC1 to **Two-Way Cluster-Robust Standard Errors** clustered along `(route_id, date)` to ensure hypothesis tests remain robust against spatial-temporal autocorrelation.

### 5.3 Survival / Hazard Duration Modeling for Schedule Recovery
* Rather than predicting only arrival delay $\Delta t$, formulate the problem as **Time-to-Recovery Duration Analysis** (using Cox Proportional Hazards or Accelerated Failure Time models):
  $$h(t | X) = h_0(t) \exp(\beta_1 \text{ROW} + \beta_2 \text{HeadwayDev} + \beta_3 \text{Precipitation})$$
  This directly answers: *If a vehicle is delayed by 8 minutes, what is the probability it returns to on-time status within the next 5 stops under dedicated right-of-way versus mixed traffic?*

---

## 6. Causal Machine Learning (Beyond Predictive xAI)

* **Current State**: TreeSHAP and ALE quantify model feature attribution, which reflects correlation and predictive importance.
* **Causal Enhancement**: To provide firm policy advice to Warsaw ZTM on whether investing tens of millions of PLN in a dedicated bus lane will *cause* delay reductions:
  * **Double Machine Learning (DML / EconML)**:
    Estimate the **Partially Linear Treatment Effect**:
    $$Y = \theta(X) \cdot \text{Dedicated\_ROW} + g(X) + \epsilon$$
    $$D = m(X) + \eta$$
    where $g(X)$ and $m(X)$ are estimated via LightGBM, and $\theta(X)$ captures the **Heterogeneous Treatment Effect (HTE)** of dedicated transit infrastructure conditioned on traffic density and weather shocks.

---

## 7. Interactive Deliverables: Real-Time Policy Dashboard

To make the research findings clear and interactive for defenses and presentations:
* **Interactive Streamlit / Dash Web Application**:
  * Build a lightweight web UI on top of [`run_pipeline.py`](run_pipeline.py):
    1. **Route Selector**: Choose any Warsaw transit line (e.g., Tram 9, Bus 175, Bus 523).
    2. **Corridor Map**: Interactive Leaflet / MapLibre map displaying stop-level delays with color-coded ALE congestion inflection points.
    3. **Interactive Counterfactual Sandbox**: Slider allowing transit planners to simulate *"What if we reduce headway deviation by 50%?"* or *"What if we convert this segment to dedicated ROW?"* with real-time DiCE model inference.

---

## 8. Completed Phase: Top 3 High-Impact Feature Enhancements (October 2026)

Based on feasibility and econometric impact, the top 3 candidate features were engineered, validated, and fused into the production feature mart (**8,828,803 records**, expanded from the initial 4.53M dataset):

### 8.1 Implemented Feature Sets
1. **Polish School & Public Holiday Calendar** ([`src/core/polish_calendar.py`](src/core/polish_calendar.py)):
   - `is_school_day`: Exact Mazowieckie voivodeship academic calendar (Winter breaks *ferie zimowe*, summer breaks, Christmas/Easter breaks, Teacher's Day).
   - `is_public_holiday`: Polish statutory holidays computed via the Anonymous Gregorian Easter algorithm.
2. **Transit Transfer Hub Density & Metro Proximity** ([`src/exogenous/interchange.py`](src/exogenous/interchange.py)):
   - `interchange_line_count`: Total transit routes serving each stop post (7,241 GTFS stops mapped).
   - `is_metro_proximity`: 300-meter geodesic buffer around all 39 Warsaw Metro (M1 & M2) stations (600 stops flagged).
3. **GTFS Route Shape Micro-Topology & Tortuosity** ([`src/exogenous/osm.py`](src/exogenous/osm.py)):
   - `actual_path_length_meters`: Cumulative polyline length along GTFS shapes.
   - `tortuosity_ratio`: Ratio of actual route trajectory length to Euclidean displacement ($\ge 1.0$), capturing chicanes, roundabouts, and sharp curves across 10,081 corridor segments (mean tortuosity = 1.219).

### 8.2 Empirical Uplift Across Cohorts

| Cohort | Best Model | Baseline MAE | New MAE | Baseline RMSE | New RMSE | Baseline $R^2$ | **New $R^2$** | Relative $R^2$ Gain |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pooled Network** | LightGBM | 29.26s | **29.05s** | 50.72s | **50.17s** | +0.0357 | **+0.0450** | **+26.1%** |
| **Urban Trams** | LightGBM | 24.03s | **22.98s** | 35.83s | **35.33s** | +0.1390 | **+0.1837** | **+32.2%** |
| **Urban Buses** | LightGBM | 30.85s | **30.32s** | 55.97s | **52.16s** | +0.0262 | **+0.0342** | **+30.5%** |
| **Suburban Buses** | CatBoost | 29.15s | **29.24s** | 55.89s | **55.43s** | +0.0169 | **+0.0195** | **+15.4%** |

---

## 9. Prioritized Phased Implementation Plan

```text
Phase A: Data & Hygiene Refinements
├── Keep background daemon capturing winter telemetry continuously
├── [x] Exclude terminal turnaround idling records (stop_sequence == 1 or max_seq)
├── [x] Stratify evaluation by transit cohort (Trams vs Core Buses vs Suburban)
└── [x] Top 3 High-Impact Features: Polish Calendar, Metro/Hub Proximity, Shape Tortuosity

Phase B: Advanced Econometric & Spatial Upgrades
├── Add Two-Way Clustered Standard Errors (route_id x date) in TWFE
├── [x] Map-match full citywide stop pairs against OSMnx via GTFS shapes.txt (10,081 segments, 4,179 signals)
├── [x] Compute shape tortuosity ratio and polyline lengths across all corridor links
└── Attach Vistula river bridge bottleneck indicator flags

Phase C: Causal ML & Interactive Interface
├── Implement Double Machine Learning (DML) for Causal ROW Treatment Effects
├── Fit Hazard / Duration models for schedule recovery rates
└── Deploy Streamlit interactive policy sandbox for route-level simulation
```

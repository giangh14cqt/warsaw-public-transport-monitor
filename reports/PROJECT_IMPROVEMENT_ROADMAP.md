# Warsaw Transit Telemetry & Delay Attribution: Project Enhancement & Research Roadmap

**University of Warsaw (UW) — Faculty of Economic Sciences & Data Science**  
**Master's Thesis Research Project**  
**Author**: Giang Truong Do  
**Date**: October 2026  
**Repository**: [`giangh14cqt/warsaw-public-transport-monitor`](https://github.com/giangh14cqt/warsaw-public-transport-monitor)  

---

## 1. Overview & Context

This document captures the knowledge gained from designing, deploying, and validating the end-to-end Warsaw transit delay telemetry pipeline, econometric Two-Way Fixed Effects (TWFE) baseline, non-linear gradient boosting models (LightGBM & CatBoost), and explainable AI (xAI) diagnostics suite.

It outlines concrete, actionable recommendations across all dimensions of the project—**data collection horizons, exogenous telemetry integration, data cleansing strategies, spatial network topology, advanced econometric modeling, causal machine learning, and interactive tooling**—to elevate this research from a strong proof-of-concept into a top-tier academic thesis and production-grade urban transport decision system.

---

## 2. Telemetry Ingestion & Data Collection Horizons

### 2.1 Multi-Month Longitudinal Capture & Seasonal Regimes
* **Current State**: ~2 weeks of early Autumn (October) telemetry.
* **Limitation**: Transit congestion, street friction, and weather shocks exhibit severe seasonal dependencies:
  * **Late Autumn / Winter (Nov–Feb)**: Freezing temperatures, black ice, heavy snowfall, street salting, and rapid twilight drastically alter vehicle braking distances, road lane capacity, and tram pantograph icing.
  * **Spring / Summer (May–Jul)**: School and university summer breaks trigger structural timetable changes (*ZTM "wakacyjny rozkład"*), lowering demand by 15–25% and flattening morning/evening peak-hour curves.
  * **September Peak Shock**: The annual maximum congestion ceiling occurs when schools and universities reopen in September.
* **Recommendations**:
  * Keep the continuous ingestion daemon active for **at least 3 to 6 months** (ideally October $\to$ March).
  * Formally evaluate **structural seasonal breaks** via Chow tests and time-stratified validation splits to determine whether delay attribution models remain invariant across weather seasons.

### 2.2 High-Resolution Spatial Micro-Weather Telemetry
* **Current State**: Hourly synoptic observations from two ground stations (Warszawa-Okęcie `#375` and Warszawa-Bielany).
* **Limitation**: Warsaw spans $517\text{ km}^2$. Convective summer downpours, localized squalls, and freezing rain along river banks are spatially heterogeneous; a storm at Okęcie does not imply wet pavement in Praga-Północ or Białołęka.
* **Recommendations**:
  * **IMGW-PIB RainGRS Radar**: Ingest IMGW's 10-minute gridded radar-gauge precipitation composite ($1\text{ km} \times 1\text{ km}$ spatial grid), mapping each GTFS stop post to its localized rainfall cell.
  * **Copernicus ERA5-Land / Open-Meteo Reanalysis**: Query 15-minute meteorological variables (surface wetness, solar radiation, wind gusts) indexed directly by stop coordinates $(lat, lon)$.

### 2.3 Dynamic Mixed-Traffic Volume & City Sensor Feeds
* **Current State**: OpenStreetMap infrastructure attributes (`lane_capacity`, `signalized_intersection_count`, `is_dedicated_right_of_way`) are **static**.
* **Limitation**: Physical lane count does not reflect real-time vehicular volume; a 3-lane arterial is free-flowing at 13:00 PM but heavily oversaturated at 08:30 AM (Volume-to-Capacity ratio $V/C > 1.0$).
* **Recommendations**:
  * **ZDM Warsaw (Zarząd Dróg Miejskich) Induction Loops**: Ingest open traffic counter telemetry from Warsaw's Municipal Road Authority measuring real-time vehicles-per-hour across major arterials (Trasa Łazienkowska, Wisłostrada, Al. Jerozolimskie).
  * **Commercial Traffic Index APIs (TomTom / HERE / Google Traffic)**: Extract dynamic segment Travel Time Indices ($TTI = t_{\text{actual}} / t_{\text{freeflow}}$) to cleanly decouple mixed-traffic congestion from transit-specific dwelling delays.

### 2.4 Passenger Demand & Dwell Time Proxies
* **Current State**: Dwell time at stops is merged into segment arrival delay delta $\Delta t$.
* **Limitation**: Dwell time spikes at major multimodal transfer nodes (Metro Świętokrzyska, Metro Centrum, Metro Politechnika, Dworzec Centralny) due to high boarding volume, not traffic congestion.
* **Recommendations**:
  * Extract GTFS-RT vehicle occupancy fields (`occupancy_status`: `EMPTY`, `MANY_SEATS_AVAILABLE`, `STANDING_ROOM_ONLY`, `FULL`) where broadcast by modern Pesa Jazz/Swing trams and Solaris Urbino electric buses.
  * Attach proximity flags to Warsaw Metro stations (M1/M2) and railway hubs (PKP/SKM/KM) to capture exogenous boarding pressure.

---

## 3. Data Cleansing, Exclusion & Cohort Stratification

| Phenomenon | Bias Induced | Recommended Filtering Strategy |
| :--- | :--- | :--- |
| **Terminal Layover & Turnaround Drift** | Vehicles resting at termini (*pętle*, e.g., Os. Górczewska, Żerań FSO) keep GPS units active and accumulate artificial idle delays that are driver rest breaks, not street congestion. | Exclude first and last stops (`stop_sequence == 1` or `max_seq`), or filter records where vehicle speed $= 0$ for $> 5$ minutes at designated turnaround facilities. |
| **Fleet / Line Type Heterogeneity** | Suburban charter routes (L-lines, e.g., L41) and express buses (500-lines) operate on 50–80 km/h regional highways, while city trams (e.g., Tram 17, 33) operate on dedicated tracks with signal priority. Aggregating them into one regression introduces unobserved heterogeneity. | **Stratify the dataset into 3 distinct cohorts**: (1) Urban Trams, (2) Core Urban Buses (100–200 & 500 lines), and (3) Suburban Feeders (700 & L-lines). Run separate benchmark models for each. |
| **Temporary Service Disruptions & Detours** | Traffic collisions, track blockages, or major construction projects (e.g., *Tramwaj do Wilanowa* along Sobieskiego) produce extreme outlier delays ($> 30$ min) unrelated to routine corridor dynamics. | Ingest the GTFS-RT `alerts.pb` feed and exclude stop segments during active civil works / detour alerts. |
| **GPS Jitter & Stop Clustering** | Around mega-stops (e.g., *Dworzec Centralny 01* through *30*), vehicles passing within 20m of adjacent platforms trigger misassigned GTFS-RT arrivals. | Apply a Kalman filter or Hausdorff distance check against GTFS `shapes.txt` to reject coordinate anomalies. |

---

## 4. Spatial Network Topology & Graph Expansion

### 4.1 Citywide Map-Matching via GTFS `shapes.txt`
* **Current State**: OSMnx features are extracted for 4–6 selected transit test corridors.
* **Expansion**: Utilize Valhalla or OSRM map-matching to match all ~3,000 Warsaw stop pairs to OSM edges automatically, scaling infrastructure attribution across the entire ZTM network.

### 4.2 Vistula River Bridge Choke-Point Modeling
* Warsaw's transit network is physically partitioned by the Vistula river. River crossings represent systemic vulnerability choke points:
  * *Most Poniatowskiego* (Trams & Buses)
  * *Most Łazienkowski* (Bus lanes)
  * *Most Śląsko-Dąbrowski* (Dedicated Tram-Bus corridor)
  * *Most Gdański* (Tram corridor)
  * *Most Grota-Roweckiego* (Express bypass)
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
* **Limitation**: Congestion is spatial. A delay shock on Puławska spills over to adjacent feeder corridors.
* **Recommendation**: Implement a **Spatial Panel Autoregressive (SAR) model**:
  $$y_t = \rho W y_t + X_t \beta + \alpha + \epsilon_t$$
  where $W$ is the spatial adjacency/distance matrix between Warsaw transit corridors. The parameter $\rho$ quantifies the **spatial spillover multiplier** of transit delays across the network.

### 5.2 Multi-Way Clustered Standard Errors
* Autoregressive errors in transit panels correlate across time within the same vehicle trip and across space within the same corridor.
* In [`src/models/twfe.py`](file:///Users/truonggiangdo/Data/LearningMaterials/UW/Thesis/DelayTraffic/playground/src/models/twfe.py), upgrade from HC1 to **Two-Way Cluster-Robust Standard Errors** clustered along `(route_id, date)` to ensure hypothesis tests remain robust against spatial-temporal autocorrelation.

### 5.3 Survival / Hazard Duration Modeling for Schedule Recovery
* Rather than predicting only arrival delay $\Delta t$, formulate the problem as **Time-to-Recovery Duration Analysis** (Cox Proportional Hazards or Accelerated Failure Time model):
  $$h(t | X) = h_0(t) \exp(\beta_1 \text{ROW} + \beta_2 \text{HeadwayDev} + \beta_3 \text{Precipitation})$$
  This directly answers: *Given a vehicle is delayed by 8 minutes, what is the hazard rate of returning to on-time status within the next 5 stops under dedicated ROW vs. mixed traffic?*

---

## 6. Causal Machine Learning (Beyond Predictive xAI)

* **Current State**: TreeSHAP and ALE quantify model feature attribution, which reflects correlation and predictive importance.
* **Causal Enhancement**: To provide ironclad policy advice to Warsaw ZTM regarding whether investing tens of millions of PLN in a dedicated bus lane will *cause* delay reductions:
  * **Double Machine Learning (DML / EconML)**:
    Estimate the **Partially Linear Treatment Effect**:
    $$Y = \theta(X) \cdot \text{Dedicated\_ROW} + g(X) + \epsilon$$
    $$D = m(X) + \eta$$
    where $g(X)$ and $m(X)$ are estimated via LightGBM, and $\theta(X)$ captures the **Heterogeneous Treatment Effect (HTE)** of dedicated transit infrastructure conditioned on traffic density and weather shocks.

---

## 7. Interactive Deliverables: Real-Time Policy Dashboard

To make the thesis visually compelling and demonstrative during defenses and presentations:
* **Interactive Streamlit / Dash Web Application**:
  * Build a lightweight web UI on top of [`run_pipeline.py`](file:///Users/truonggiangdo/Data/LearningMaterials/UW/Thesis/DelayTraffic/playground/run_pipeline.py):
    1. **Route Selector**: Choose any Warsaw line (e.g., Tram 9, Bus 175, Bus 523).
    2. **Corridor Map**: Interactive Leaflet / MapLibre map displaying stop-level delays with color-coded ALE congestion inflections.
    3. **Interactive Counterfactual Sandbox**: Slider allowing transit planners to simulate *"What if we regularize headway deviation by 50%?"* or *"What if we convert this segment to dedicated ROW?"* with real-time DiCE model inference.

---

## 8. Prioritized Phased Implementation Plan

```
Phase A: Data & Hygiene Refinements (Weeks 1–2)
├── Keep background daemon capturing winter telemetry continuously
├── Exclude terminal turnaround idling records (stop_sequence == 1 or max_seq)
└── Stratify evaluation by transit cohort (Trams vs Core Buses vs Suburban)

Phase B: Advanced Econometric & Spatial Upgrades (Weeks 3–4)
├── Add Two-Way Clustered Standard Errors (route_id x date) in TWFE
├── Map-match full citywide stop pairs against OSMnx via GTFS shapes.txt
└── Attach Vistula river bridge bottleneck indicator flags

Phase C: Causal ML & Interactive Interface (Weeks 5–6)
├── Implement Double Machine Learning (DML) for Causal ROW Treatment Effects
├── Fit Hazard / Duration models for schedule recovery rates
└── Deploy Streamlit interactive policy sandbox for route-level simulation
```

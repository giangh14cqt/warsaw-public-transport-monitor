# Warsaw Transit Telemetry & Delay Attribution: Data Schema Reference

This document provides a comprehensive technical reference for all data schemas used across the Warsaw Transit Telemetry & Delay Attribution System, including ingestion feeds, exogenous auxiliary stores, and the unified modeling feature mart.

---

## 1. Data Flow & Master Analysis Table Overview

The telemetry pipeline processes transit observations through progressive refinement stages, culminating in the **Master Feature Mart**, which serves as the single source of truth for econometric regression and machine learning benchmarks.

### 1.1 Pipeline Data Flow

```mermaid
flowchart TD
    subgraph INGESTION["Stage 1-2 · Ingestion & Storage"]
        GTFSRT["🛰️ GTFS-RT Protobuf Feed\napi.zbiorkom.live (v6) · every 30s"]
        GTFS["📅 Static GTFS Timetable\ndata/gtfs_static/ + data/gtfs/"]
        RAW["🗄️ Raw Partitioned Telemetry\ndata/raw/year=YYYY/month=MM/day=DD/*.parquet\n24.4M records · 5,101 Parquet files"]
        GTFSRT -->|"decode protobuf\n+ schedule join"| RAW
        GTFS -->|"schedule reference"| RAW
    end

    subgraph EXOGENOUS["Stage 2 · Exogenous Sources"]
        WEATHER["🌦️ Synoptic Weather\nimgw_weather_hourly.parquet\n1,057 hours · IMGW-PIB"]
        OSM["🛣️ Road Topology\nosm_corridor_segments.parquet\n10,081 stop-to-stop links"]
        CAL["📆 Polish Calendar\ncalendar_lookup.parquet\n3,652 days"]
        HUB["🚉 Stop Interchanges\ntransit_stop_interchanges.parquet\n7,372 stops"]
    end

    subgraph FUSION["Stage 3 · Spatiotemporal Fusion"]
        direction TB
        SQL["⚙️ DuckDB Fusion SQL\ntrajectory lags · temporal join · spatial join"]
    end

    subgraph MART["🎯 Master Analysis Dataset"]
        FM["✅ feature_mart.parquet\n8,828,803 rows × 45 columns\nOct 4 – Oct 17, 2026"]
    end

    subgraph ANALYSIS["Stage 4-5 · Modeling & xAI"]
        TWFE["📐 Econometric TWFE\nPanel Regression"]
        GBM["🌳 LightGBM + CatBoost\nGradient Boosting"]
        XAI["🔍 xAI Suite\nTreeSHAP · ALE · DiCE"]
    end

    RAW -->|"vehicle trajectories\narrival delay Δt"| SQL
    WEATHER -->|"hourly timestamp join\non timestamp_bucket"| SQL
    OSM -->|"spatial join\non stop_id_prev + stop_id_curr"| SQL
    CAL -->|"date join\non calendar_date"| SQL
    HUB -->|"stop join\non stop_id"| SQL
    SQL --> FM
    FM --> TWFE
    FM --> GBM
    FM --> XAI

    style FM fill:#16a34a,color:#fff,stroke:#15803d,stroke-width:3px
    style MART fill:#dcfce7,stroke:#16a34a,stroke-width:2px
    style INGESTION fill:#eff6ff,stroke:#3b82f6
    style EXOGENOUS fill:#fff7ed,stroke:#f97316
    style FUSION fill:#faf5ff,stroke:#a855f7
    style ANALYSIS fill:#fefce8,stroke:#eab308
```

---

### 1.2 Table Relationships (Entity-Relationship Diagram)

```mermaid
erDiagram
    RAW_TELEMETRY {
        BIGINT feed_timestamp PK
        STRING trip_id FK
        STRING route_id FK
        STRING vehicle_id
        STRING stop_id FK
        INT32  stop_sequence
        STRING start_date
        INT32  arrival_delay_seconds
        INT32  departure_delay_seconds
        BIGINT rt_arrival_time
        BIGINT rt_departure_time
        STRING record_ingested_at
        BIGINT year
        BIGINT month
        STRING day
    }

    STATIC_GTFS_TRIPS {
        STRING trip_id PK
        STRING route_id FK
        STRING service_id
        STRING shape_id
    }

    STATIC_GTFS_ROUTES {
        STRING route_id PK
        STRING route_short_name
        STRING route_long_name
        INT    route_type
    }

    IMGW_WEATHER_HOURLY {
        STRING    station_id PK
        TIMESTAMP timestamp_bucket PK
        DOUBLE    temperature_c
        DOUBLE    precipitation_mm
        DOUBLE    wind_speed_ms
        DOUBLE    relative_humidity
        DOUBLE    pressure_hpa
        BOOLEAN   freezing_rain_flag
        DOUBLE    visibility_m
    }

    OSM_CORRIDOR_SEGMENTS {
        STRING  edge_id PK
        STRING  stop_id_prev FK
        STRING  stop_id_curr FK
        STRING  corridor_name
        DOUBLE  segment_length_meters
        DOUBLE  actual_path_length_meters
        DOUBLE  tortuosity_ratio
        BOOLEAN is_dedicated_right_of_way
        BIGINT  signalized_intersection_count
        BIGINT  lane_capacity
    }

    CALENDAR_LOOKUP {
        STRING calendar_date PK
        BIGINT is_school_day
        BIGINT is_public_holiday
        BIGINT is_holiday_eve
    }

    TRANSIT_STOP_INTERCHANGES {
        STRING stop_id PK
        STRING stop_name
        BIGINT interchange_line_count
        BIGINT is_metro_proximity
        DOUBLE min_dist_to_metro_m
        BIGINT is_major_transfer_hub
    }

    FEATURE_MART {
        STRING  trip_id
        STRING  route_id
        STRING  vehicle_id
        STRING  stop_id
        STRING  prev_stop_id
        INT     arrival_delay_seconds
        INT     departure_delay_seconds
        INT     prev_stop_delay
        INT     prev2_stop_delay
        DECIMAL delta_t_run
        DOUBLE  trip_progress
        INT     is_tram
        STRING  transit_cohort
        TSTZ    arrival_ts
        TSTZ    timestamp_bucket
        BIGINT  hour_of_day
        BIGINT  day_of_week
        INT     is_weekend
        INT     is_peak_hour
        BIGINT  is_school_day
        BIGINT  is_public_holiday
        BIGINT  headway_actual_s
        DOUBLE  headway_deviation
        BIGINT  interchange_line_count
        BIGINT  is_metro_proximity
        DOUBLE  temperature_c
        DOUBLE  precipitation_mm
        DOUBLE  wind_speed_ms
        BOOLEAN freezing_rain_flag
        STRING  corridor_name
        DOUBLE  segment_length_meters
        BOOLEAN is_dedicated_right_of_way
        BIGINT  signalized_intersection_count
        BIGINT  lane_capacity
        INT     is_study_corridor
    }

    RAW_TELEMETRY ||--o{ FEATURE_MART : "trajectory lag enrichment (trip_id + stop_sequence)"
    STATIC_GTFS_TRIPS ||--o{ RAW_TELEMETRY : "schedule join (trip_id)"
    STATIC_GTFS_ROUTES ||--o{ STATIC_GTFS_TRIPS : "route metadata (route_id)"
    IMGW_WEATHER_HOURLY ||--o{ FEATURE_MART : "hourly temporal join (timestamp_bucket)"
    OSM_CORRIDOR_SEGMENTS ||--o{ FEATURE_MART : "spatial join (stop_id_prev + stop_id_curr)"
    CALENDAR_LOOKUP ||--o{ FEATURE_MART : "date join (calendar_date)"
    TRANSIT_STOP_INTERCHANGES ||--o{ FEATURE_MART : "stop join (stop_id)"
```

---

### Table Usage at a Glance

| Dataset | File Path | Format / Size | Role in System | Analysis Status |
| :--- | :--- | :--- | :--- | :---: |
| **Unified Feature Mart** | `data/processed/feature_mart.parquet` | Parquet (~286 MB, 8.83M rows) | Master modeling matrix | **PRIMARY ANALYSIS** 🎯 |
| **Raw Telemetry Partitions** | `data/raw/year=*/month=*/day=*/*.parquet` | Parquet (~225 MB, 24.4M rows) | Ingestion archive | Ingestion / Stage 1-2 |
| **Static GTFS Schedules** | `data/gtfs_static/` & `data/gtfs/` | Text CSV / DuckDB (~163 MB) | Scheduled timetable reference | Reference Lookup |
| **Hourly Weather Telemetry** | `data/processed/imgw_weather_hourly.parquet` | Parquet (~28 KB, 1,057 hours) | Meteorological exogenous feed | Joined into Mart |
| **Corridor Road Topology** | `data/processed/osm_corridor_segments.parquet` | Parquet (~438 KB, 10,081 links) | Road & signal infrastructure | Joined into Mart |
| **Polish Calendar Lookup** | `data/processed/calendar_lookup.parquet` | Parquet (~11 KB, 3,652 days) | School days & national holidays | Joined into Mart |
| **Stop Interchanges & Metro** | `data/processed/transit_stop_interchanges.parquet` | Parquet (~150 KB, 7,372 stops) | Hub connectivity & transfer load | Joined into Mart |

---

## 2. Primary Analysis Dataset: Unified Feature Mart

* **File Path**: [`data/processed/feature_mart.parquet`](data/processed/feature_mart.parquet)
* **Status**: **ACTIVE TARGET FOR ALL EMPIRICAL ANALYSIS & BENCHMARKS**
* **Dimensions**: **8,828,803 rows × 45 columns** (compressed Apache Parquet with Snappy)
* **Observation Unit**: Single stop-to-stop transit vehicle arrival transition $(i, s, t)$

### Schema & Data Dictionary

| Column Name | Data Type | Nullable | Domain Taxonomy | Description & Example |
| :--- | :--- | :---: | :--- | :--- |
| `trip_id` | `VARCHAR` | No | Identifier | Unique GTFS trip identifier (e.g., `'100/1/TP-SZO'`) |
| `route_id` | `VARCHAR` | No | Identifier | Transit line number (e.g., `'17'`, `'503'`, `'709'`) |
| `vehicle_id` | `VARCHAR` | No | Identifier | Physical vehicle tracking code (e.g., `'1001'`, `'5402'`) |
| `stop_sequence` | `INTEGER` | No | Operational | Sequential stop index along the trip (1, 2, 3...) |
| `stop_id` | `VARCHAR` | No | Spatial Key | Current GTFS stop ID (e.g., `'700201'`) |
| `prev_stop_id` | `VARCHAR` | Yes | Spatial Key | Upstream stop ID traversed immediately prior |
| **`arrival_delay_seconds`** | `INTEGER` | No | **Target Variable** | **Primary regression target**: Observed minus scheduled arrival time ($\Delta t$, in seconds) |
| `departure_delay_seconds` | `INTEGER` | Yes | Target / Auxiliary | Observed minus scheduled departure time ($\Delta t_{dep}$, in seconds) |
| `prev_stop_delay` | `INTEGER` | Yes | Operational | Lag-1 delay at preceding stop ($\Delta t_{s-1}$, in seconds) |
| `prev2_stop_delay` | `INTEGER` | Yes | Operational | Lag-2 delay two stops prior ($\Delta t_{s-2}$, in seconds) |
| `delta_t_run` | `DECIMAL(11,1)` | Yes | Operational | Running delay component: $\Delta t_{arr, s} - \Delta t_{dep, s-1}$ (seconds) |
| `delta_t_dwell` | `INTEGER` | Yes | Operational | Dwell delay component: $\Delta t_{dep, s} - \Delta t_{arr, s}$ (seconds) |
| `observed_run_time_s` | `BIGINT` | Yes | Operational | Actual elapsed transit time between stops (seconds) |
| `trip_progress` | `DOUBLE` | No | Operational | Normalized route completion ratio $\in [0.0, 1.0]$ |
| `is_origin_stop` | `BOOLEAN` | No | Operational | Flag indicating trip terminal departure (`true` if first stop) |
| `is_terminal_stop` | `BOOLEAN` | No | Operational | Flag indicating route terminus (`true` if final stop) |
| `is_tram` | `INTEGER` | No | Fleet / Mode | Modal indicator: `1` for electric tram, `0` for bus |
| `transit_cohort` | `VARCHAR` | No | Operational | Classification: `'urban_tram'`, `'urban_bus'`, `'suburban_bus'`, `'other'` |
| `arrival_ts` | `TIMESTAMPTZ` | No | Temporal Key | Microsecond-precision timestamp of vehicle arrival |
| `timestamp_bucket` | `TIMESTAMPTZ` | No | Temporal Key | Hourly-truncated timestamp for weather joining (e.g., `2026-10-04 08:00:00+00`) |
| `hour_of_day` | `BIGINT` | No | Temporal | Local hour $\in [0, 23]$ |
| `day_of_week` | `BIGINT` | No | Temporal | Day index: $0 = \text{Monday}, \dots, 6 = \text{Sunday}$ |
| `is_weekend` | `INTEGER` | No | Temporal | Binary flag: `1` on Saturday/Sunday, `0` on weekdays |
| `is_peak_hour` | `INTEGER` | No | Temporal | Binary flag: `1` during rush hours (07:00–09:00, 16:00–18:00) |
| `is_morning_peak` | `INTEGER` | No | Temporal | Binary flag: `1` during morning rush hour (07:00–09:00) |
| `is_evening_peak` | `INTEGER` | No | Temporal | Binary flag: `1` during evening rush hour (16:00–18:00) |
| `is_school_day` | `BIGINT` | No | Calendar | Binary flag: `1` if Polish public schools are in session |
| `is_public_holiday` | `BIGINT` | No | Calendar | Binary flag: `1` on official Polish national holidays |
| `headway_actual_s` | `BIGINT` | Yes | Operational | Observed vehicle time interval following prior bus on same route (seconds) |
| `headway_deviation` | `DOUBLE` | Yes | Operational | Normalized headway deviation: $(h_{actual} - h_{sched}) / h_{sched}$ |
| `interchange_line_count` | `BIGINT` | No | Infrastructure | Number of distinct transit routes serving the stop (network centrality) |
| `is_metro_proximity` | `BIGINT` | No | Infrastructure | Binary flag: `1` if stop is within 250m walking radius of Metro M1/M2 |
| `temperature_c` | `DOUBLE` | No | Meteorological | Synoptic 2m ambient air temperature in Celsius |
| `precipitation_mm` | `DOUBLE` | No | Meteorological | Cumulative hourly liquid rainfall / liquid equivalent (mm) |
| `wind_speed_ms` | `DOUBLE` | No | Meteorological | 10m sustained horizontal wind speed (m/s) |
| `relative_humidity` | `DOUBLE` | No | Meteorological | Relative atmospheric humidity percentage $\in [0, 100]$ |
| `freezing_rain_flag` | `BOOLEAN` | No | Meteorological | Binary indicator for road icing / freezing drizzle |
| `corridor_name` | `VARCHAR` | Yes | Topology | Name of primary arterial corridor (e.g., `'Puławska'`, `'Trasa W-Z'`) |
| `segment_length_meters` | `DOUBLE` | No | Topology | Stop-to-stop road segment traversal length (meters) |
| `actual_path_length_meters`| `DOUBLE` | No | Topology | True roadway centerline path distance (meters) |
| `tortuosity_ratio` | `DOUBLE` | No | Topology | Route curve ratio: $\text{actual\_path\_length} / \text{euclidean\_distance}$ |
| `signalized_intersection_count` | `BIGINT` | No | Topology | Number of traffic signals along the road segment |
| `is_dedicated_right_of_way` | `BOOLEAN` | No | Topology | Physical right-of-way segregation: `true` for tram tracks / bus lanes |
| `lane_capacity` | `BIGINT` | No | Topology | Total mixed-traffic vehicular travel lanes on the corridor |
| `is_study_corridor` | `INTEGER` | No | Topology | Binary flag: `1` if segment lies on one of Warsaw's 4 focal study corridors |

---

## 3. Which Features Are Used in Specific Analyses?

The 45 columns of the feature mart are organized into distinct modeling subsets depending on the econometric or machine learning methodology applied:

### A. Econometric Baseline: Two-Way Fixed Effects (TWFE) Panel Regression
* **Objective**: Measure causal, ceteris paribus marginal effects of exogenous infrastructure and weather shocks.
* **Equation**:
  $$\text{arrival\_delay}_{i,s,t} = \alpha_i + \gamma_t + \beta_1 \text{precip}_t + \beta_2 \text{temp}_t + \beta_3 \text{signals}_s + \beta_4 \text{dedicated\_row}_s + \rho \text{prev\_delay}_{i,s-1,t} + \varepsilon_{i,s,t}$$
* **Dependent Variable ($Y$)**: `arrival_delay_seconds`
* **Entity Fixed Effects ($\alpha_i$)**: `route_id` (absorbing route-level unobserved heterogeneity)
* **Time Fixed Effects ($\gamma_t$)**: `hour_of_day` (absorbing citywide diurnal traffic demand)
* **Standard Error Clustering**: Clustered at `route_id` level using Huber-White HC1 covariance.
* **Regressors ($X$)**:
  - `precipitation_mm` (Rainfall elasticity)
  - `temperature_c` (Thermal shock)
  - `signalized_intersection_count` (Traffic signal friction)
  - `segment_length_meters` (Distance exposure)
  - `prev_stop_delay` (Autoregressive persistence)
  - *Note: Invariant spatial features like `is_dedicated_right_of_way` and `lane_capacity` are absorbed by entity fixed effects in pure within-route estimators and evaluated via between-cohort pooled regressions.*

### B. Machine Learning Benchmark: Non-Linear Gradient Boosting (LightGBM & CatBoost)
* **Objective**: Maximize out-of-sample predictive accuracy, capture non-linear thresholds, and detect multi-way feature interactions.
* **Feature Vector ($X$, 20 Active Features)**:
  1. `prev_stop_delay` (Operational: Lag-1 delay persistence)
  2. `signalized_intersection_count` (Infrastructure: Traffic lights)
  3. `segment_length_meters` (Infrastructure: Inter-stop distance)
  4. `actual_path_length_meters` (Infrastructure: Roadway travel distance)
  5. `tortuosity_ratio` (Infrastructure: Road winding / curve index)
  6. `interchange_line_count` (Infrastructure: Transfer hub connectivity)
  7. `is_metro_proximity` (Infrastructure: Proximity to Metro M1/M2)
  8. `is_dedicated_right_of_way` (Infrastructure: Physical right-of-way isolation)
  9. `trip_progress` (Operational: Schedule progress along trip route)
  10. `headway_deviation` (Operational: Dispatch bunching / gap irregularity)
  11. `hour_of_day` (Temporal: Hour of day, 0–23)
  12. `day_of_week` (Temporal: Day of week, 0–6)
  13. `is_peak_hour` (Temporal: Peak period indicator)
  14. `is_school_day` (Calendar: School operational day)
  15. `is_public_holiday` (Calendar: Statutory holiday)
  16. `is_tram` (Fleet: Rail vs. Rubber-tire vehicle mode)
  17. `lane_capacity` (Infrastructure: Corridor mixed-traffic lanes)
  18. `precipitation_mm` (Weather: Rainfall intensity)
  19. `temperature_c` (Weather: Temperature)
  20. `relative_humidity` (Weather: Air moisture)

### C. Explainable AI (xAI) Delay Attribution & Counterfactuals
* **TreeSHAP Attributions**: Computed across all 20 benchmark features to calculate global importance and local waterfall decompositions.
* **Accumulated Local Effects (ALE)**: 7 key continuous and discrete variables analyzed for tipping points:
  - `prev_stop_delay`, `trip_progress`, `is_dedicated_right_of_way`, `signalized_intersection_count`, `segment_length_meters`, `precipitation_mm`, `temperature_c`.
* **DiCE Counterfactual Interventions**:
  - **Actionable (Controllable by Transit Agency)**:
    - `prev_stop_delay` (Upstream schedule recovery holding)
    - `headway_deviation` (Dispatch interval regularization)
    - `is_dedicated_right_of_way` (Bus lane allocation / dedicated track conversion)
    - `signalized_intersection_count` (Transit Signal Priority green wave)
  - **Immutable (Fixed Environmental & Physical Constraints)**:
    - `hour_of_day`, `day_of_week`, `is_peak_hour`, `is_school_day`, `is_public_holiday`
    - `temperature_c`, `precipitation_mm`, `wind_speed_ms`, `relative_humidity`
    - `segment_length_meters`, `actual_path_length_meters`, `tortuosity_ratio`, `is_tram`

---

## 4. Ingestion & Auxiliary Source Schemas

### A. Raw Partitioned Transit Telemetry
* **Directory**: [`data/raw/year=YYYY/month=MM/day=DD/*.parquet`](data/raw/)
* **Format**: Hive-partitioned Apache Parquet (Snappy compression, strict PyArrow schema)
* **Frequency**: Written every 5 minutes / 5,000 records by continuous ingestion daemon.

| Field Name | PyArrow Type | Description |
| :--- | :--- | :--- |
| `feed_timestamp` | `INT64` | Unix epoch seconds when the GTFS-RT feed was produced |
| `trip_id` | `STRING` | Trip identifier corresponding to static timetable |
| `route_id` | `STRING` | Public line identifier (e.g., `'17'`, `'190'`) |
| `start_date` | `STRING` | Scheduled trip start date (`'YYYYMMDD'`) |
| `start_time` | `STRING` | Scheduled trip start time (`'HH:MM:SS'`) |
| `vehicle_id` | `STRING` | Unique physical vehicle tracker code |
| `stop_sequence` | `INT32` | Stop sequence integer |
| `stop_id` | `STRING` | Identifier of current stop |
| `arrival_delay_seconds` | `INT32` | Delay in seconds at arrival (negative = early, positive = late) |
| `departure_delay_seconds`| `INT32` | Delay in seconds at departure |
| `rt_arrival_time` | `INT64` | Real-time estimated arrival Unix timestamp |
| `rt_departure_time` | `INT64` | Real-time estimated departure Unix timestamp |
| `record_ingested_at` | `STRING` | UTC ISO-8601 ingestion audit timestamp |
| *Partition: `year`* | `INT64` | Hive partition year (e.g., `2026`) |
| *Partition: `month`* | `INT64` | Hive partition month (e.g., `10`) |
| *Partition: `day`* | `STRING` | Hive partition day (e.g., `'04'`) |

---

### B. Synoptic Weather Telemetry (IMGW-PIB)
* **File Path**: [`data/processed/imgw_weather_hourly.parquet`](data/processed/imgw_weather_hourly.parquet)
* **Source**: Official IMGW-PIB Synoptic Weather Stations (*Warszawa-Okęcie* WMO `12375` & *Warszawa-Bielany* `352200380`).

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `station_id` | `VARCHAR` | Official station identifier |
| `station_name` | `VARCHAR` | Weather station name (e.g., `'WARSZAWA-OKECIE'`) |
| `timestamp` | `TIMESTAMP` | Exact observation timestamp |
| `timestamp_bucket` | `TIMESTAMP` | Hourly-truncated timestamp for join linkage |
| `temperature_c` | `DOUBLE` | Air temperature at 2 meters altitude (°C) |
| `relative_humidity` | `DOUBLE` | Relative air humidity percentage (%) |
| `precipitation_mm` | `DOUBLE` | Total precipitation depth past hour (mm) |
| `wind_speed_ms` | `DOUBLE` | Mean horizontal wind speed (m/s) |
| `pressure_hpa` | `DOUBLE` | Atmospheric pressure reduced to sea level (hPa) |
| `freezing_rain_flag` | `BOOLEAN` | Indicator for freezing precipitation / icy road risk |
| `visibility_m` | `DOUBLE` | Horizontal atmospheric visibility (meters) |

---

### C. Corridor Road Topology & Infrastructure (OSMnx)
* **File Path**: [`data/processed/osm_corridor_segments.parquet`](data/processed/osm_corridor_segments.parquet)
* **Source**: OpenStreetMap road network extracted via OSMnx, clipped to Warsaw's core transit corridors.

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `edge_id` | `VARCHAR` | Unique corridor edge identifier (e.g., `'pulawska_edge_001'`) |
| `corridor_name` | `VARCHAR` | Radial corridor name (`'Puławska'`, `'Aleje Jerozolimskie'`, `'Trasa W-Z'`, `'Towarowa / Okopowa'`) |
| `stop_id_prev` | `VARCHAR` | Upstream transit stop ID |
| `stop_id_curr` | `VARCHAR` | Downstream transit stop ID |
| `stop_name_prev` | `VARCHAR` | Upstream stop official name |
| `stop_name_curr` | `VARCHAR` | Downstream stop official name |
| `segment_length_meters` | `DOUBLE` | Straight-line distance between stops (m) |
| `actual_path_length_meters`| `DOUBLE` | Actual road centerline traversal distance (m) |
| `tortuosity_ratio` | `DOUBLE` | Sinuosity / tortuosity ratio ($\ge 1.0$) |
| `is_dedicated_right_of_way`| `BOOLEAN`| Physical lane isolation (tram reservation or bus lane) |
| `signalized_intersection_count` | `BIGINT` | Number of traffic signal heads along the segment |
| `lane_capacity` | `BIGINT` | Number of general traffic travel lanes |

---

### D. Polish Calendar & Statutory Holidays
* **File Path**: [`data/processed/calendar_lookup.parquet`](data/processed/calendar_lookup.parquet)
* **Source**: Programmatically generated according to Polish statutory law (*Ustawa o dniach wolnych od pracy*) and the Ministry of National Education (*MEN*) school calendars.

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `calendar_date` | `VARCHAR` | Date string in ISO format (`'YYYY-MM-DD'`) |
| `is_school_day` | `BIGINT` | `1` if schools are in session; `0` during summer/winter breaks |
| `is_public_holiday` | `BIGINT` | `1` on statutory public holidays (e.g., All Saints' Day, Constitution Day) |
| `is_holiday_eve` | `BIGINT` | `1` on days preceding a statutory holiday |

---

### E. Transit Stop Interchanges & Metro Proximity
* **File Path**: [`data/processed/transit_stop_interchanges.parquet`](data/processed/transit_stop_interchanges.parquet)
* **Source**: Graph topology derived from Static GTFS stop coordinate registry and Warsaw Metro station portals (M1 / M2).

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `stop_id` | `VARCHAR` | GTFS stop identifier |
| `stop_name` | `VARCHAR` | Official stop name (e.g., `'Centrum 01'`, `'Metro Politechnika 03'`) |
| `interchange_line_count` | `BIGINT` | Total count of distinct bus and tram lines stopping at this platform |
| `is_metro_proximity` | `BIGINT` | `1` if Euclidean distance to nearest Metro entrance $\le 250\text{m}$ |
| `min_dist_to_metro_m` | `DOUBLE` | Exact walking/Euclidean distance to nearest Metro station portal (meters) |
| `is_major_transfer_hub` | `BIGINT` | `1` if stop serves $\ge 5$ distinct transit routes |

---

## 5. SQL Reference: Querying the Datasets with DuckDB

You can query all schemas directly without loading them into memory using [DuckDB](https://duckdb.org). Below are standard query patterns:

### Querying the Master Analysis Table
```sql
-- Query the master analysis dataset for peak-hour tram performance
SELECT
    route_id,
    COUNT(*) AS total_observations,
    ROUND(AVG(arrival_delay_seconds), 1) AS mean_delay_seconds,
    ROUND(MEDIAN(arrival_delay_seconds), 1) AS median_delay_seconds,
    ROUND(AVG(CASE WHEN is_dedicated_right_of_way THEN 1.0 ELSE 0.0 END) * 100, 1) AS pct_dedicated_row
FROM read_parquet('data/processed/feature_mart.parquet')
WHERE is_peak_hour = 1
  AND is_tram = 1
GROUP BY route_id
ORDER BY total_observations DESC
LIMIT 10;
```

### Querying the Raw Telemetry Partitions
```sql
-- Query raw ingestion partitions across all days and months
SELECT
    year,
    month,
    day,
    COUNT(*) AS records_ingested,
    COUNT(DISTINCT route_id) AS distinct_routes,
    COUNT(DISTINCT vehicle_id) AS active_vehicles
FROM read_parquet('data/raw/*/*/*/*.parquet')
GROUP BY year, month, day
ORDER BY year, month, day;
```

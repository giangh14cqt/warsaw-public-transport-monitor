# Feature Mart Fusion: Dual-Datasource Vectorized Engine

The **Feature Mart Fusion Engine** ([`src/fusion/feature_mart.py`](file:///Users/truonggiangdo/Data/LearningMaterials/UW/Thesis/DelayTraffic/playground/src/fusion/feature_mart.py)) is an out-of-core, vectorized analytical processor built on top of [DuckDB](https://duckdb.org/). It unites transit telemetry streams with meteorological, topological, and calendar tables into a unified analytical feature store.

---

## ⚙️ Dual-Datasource Fusion Architecture

```mermaid
flowchart TD
    subgraph DataSources ["Input Datasource Selection (--datasource {legacy,historian})"]
        direction TB
        DS_Leg["<b>Legacy Mode (datasource='legacy')</b><br/>• Scans <code>data/raw/year=*/.../*.parquet</code><br/>• GTFS-RT TripUpdate forecast delays<br/>• Filters invalid sequences & coordinates"]
        DS_Hist["<b>Historian Mode (datasource='historian')</b><br/>• Scans <code>data/historian/gold/stop_events/date=*/*.parquet</code><br/>• Observed actual stop arrivals & real dwells<br/>• Pairs live forecast error Δ (zbk_forecast_delay_s)"]
    end

    subgraph DuckDBEngine ["DuckDB In-Memory Vectorized Query Pipeline (src/fusion/feature_mart.py)"]
        direction TB
        BaseCTE["<b>1. Raw Telemetry Scan CTE</b><br/>Parse timestamps, cast types, filter null coordinates"]
        WindowCTE["<b>2. Trajectory Window CTE</b><br/><code>LAG(delay) OVER (PARTITION BY trip_id ORDER BY stop_sequence)</code><br/>Derive inter-stop running delay increment (Δt_run)"]
        WeatherCTE["<b>3. Synoptic Weather CTE</b><br/>Extract Okęcie & Bielany hourly metrics"]
        OSMCTE["<b>4. Corridor Road Network CTE</b><br/>Join maxspeed, lanes, highway category"]
        InterchangeCTE["<b>5. Multi-Modal Hubs CTE</b><br/>Join transfer counts & metro flags"]
        CalendarCTE["<b>6. Calendar & Trigonometric Time CTE</b><br/>Derive is_rush_hour, is_weekend, sin/cos hour"]
        CohortCTE["<b>7. Transit Cohort Classification CTE</b><br/>urban_tram (route < 40) | urban_bus (< 700) | suburban_bus (700-899)"]
        FilterCTE["<b>8. Terminal Stop Exclusion</b><br/>Exclude first stop (origin terminal has no upstream propagation)"]
    end

    subgraph OutputStores ["Analytical Mart Parquet Storage"]
        Out_Leg[("data/processed/feature_mart.parquet<br/><i>(Legacy GTFS-RT Mart)</i>")]
        Out_Hist[("data/historian/processed/feature_mart_observed.parquet<br/><i>(Historian Observed Ground-Truth Mart)</i>")]
    end

    DS_Leg --> BaseCTE
    DS_Hist --> BaseCTE

    BaseCTE --> WindowCTE
    WindowCTE --> CohortCTE
    WeatherCTE --> CohortCTE
    OSMCTE --> CohortCTE
    InterchangeCTE --> CohortCTE
    CalendarCTE --> CohortCTE

    CohortCTE --> FilterCTE
    FilterCTE -->|Legacy mode| Out_Leg
    FilterCTE -->|Historian mode| Out_Hist

    %% Styles
    classDef dsStyle fill:#2d3748,stroke:#4a5568,color:#edf2f7;
    classDef qStyle fill:#1a365d,stroke:#2b6cb0,color:#bee3f8;
    classDef oStyle fill:#744210,stroke:#ecc94b,color:#ffffb0;

    class DS_Leg,DS_Hist dsStyle;
    class BaseCTE,WindowCTE,WeatherCTE,OSMCTE,InterchangeCTE,CalendarCTE,CohortCTE,FilterCTE qStyle;
    class Out_Leg,Out_Hist oStyle;
```

---

## 📐 Mathematical Delay Decomposition

At each transit stop $i$ along trip $k$, the Feature Mart decomposes delay propagation using window functions:

### 1. Cumulative Arrival Delay
$$\text{delay}_i = t_{\text{actual\_arrival}, i} - t_{\text{scheduled\_arrival}, i}$$

### 2. Upstream Propagation Delay ($\text{delay}_{i-1}$)
$$\text{prev\_stop\_delay} = \text{LAG}(\text{delay}_i, 1) \text{ OVER } (\text{PARTITION BY } k \text{ ORDER BY } \text{sequence})$$

### 3. Inter-Stop Section Delay Delta ($\Delta t_{\text{run}}$)
The pure incremental delay incurred strictly between stop $i-1$ and stop $i$ (stripping upstream shock):
$$\Delta t_{\text{run}} = \text{delay}_i - \text{prev\_stop\_delay}$$

### 4. Dwell Time ($\Delta t_{\text{dwell}}$) - Historian Observed Only
The real physical door-open boarding/alighting duration:
$$\Delta t_{\text{dwell}} = t_{\text{actual\_departure}, i} - t_{\text{actual\_arrival}, i}$$

### 5. Operator Forecast Error ($\Delta t_{\text{error}}$) - Historian Observed Only
The deviation between operator GTFS-RT forecast and ground-truth arrival:
$$\text{forecast\_error\_s} = \text{delay}_{\text{forecast}, i} - \text{delay}_{\text{observed}, i}$$

---

## 🗃️ Feature Mart Schema & Column Dictionary

```mermaid
classDiagram
    class FeatureMartRecord {
        +String service_date
        +String trip_id
        +String route_id
        +String vehicle_number
        +String transit_cohort
        +Int32 stop_sequence
        +String stop_id
        +String prev_stop_id
        +Int64 timestamp_utc
        +Int32 arrival_delay_seconds [TARGET]
        +Int32 prev_stop_delay
        +Float32 delta_t_run
        +Int32 dwell_seconds [HISTORIAN]
        +Int32 zbk_forecast_delay_s [HISTORIAN]
        +Int32 forecast_error_s [HISTORIAN]
        +Float32 temperature_c
        +Float32 wind_speed_ms
        +Float32 precipitation_mm
        +Float32 visibility_m
        +Int32 osm_maxspeed
        +Int32 osm_lanes
        +Int32 osm_has_busway
        +Int32 is_major_interchange
        +Int32 transfer_routes_count
        +Int32 has_metro_transfer
        +Int32 is_rush_hour
        +Int32 is_weekend
        +Int32 is_holiday
        +Float32 hour_sin
        +Float32 hour_cos
    }
```

---

## ⚡ Performance & Scalability Benchmark

Due to DuckDB's columnar execution and parallel scan capabilities:
- **Legacy Ingestion Mart**: Processes **1,000,000+ telemetry rows in $< 4.5\,\text{s}$** with zero disk spill.
- **Historian Mart**: Fuses **77,000+ stop events into `feature_mart_observed.parquet` in $0.21\,\text{s}$**.
- **Memory Footprint**: Streaming out-of-core execution allows processing large datasets with minimal RAM consumption.

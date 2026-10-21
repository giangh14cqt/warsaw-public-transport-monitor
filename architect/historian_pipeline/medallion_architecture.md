# Historian Pipeline: Medallion Architecture Deep Dive

The **Medallion Data Architecture** (Bronze $\to$ Silver $\to$ Gold) progressively cleans, deduplicates, and elevates raw transit GPS telemetry into high-value physical stop events ready for econometric analysis and ML attribution.

---

## 🥇 Medallion Architecture Overview

```mermaid
%%{init: {
  'theme': 'dark',
  'themeVariables': {
    'darkMode': true,
    'background': '#0d1117',
    'mainBkg': '#161b22',
    'primaryColor': '#1f2937',
    'primaryTextColor': '#f0f6fc',
    'primaryBorderColor': '#58a6ff',
    'lineColor': '#8b949e',
    'secondaryColor': '#161b22',
    'tertiaryColor': '#21262d',
    'clusterBkg': '#161b22',
    'clusterBorder': '#30363d',
    'edgeLabelBackground': '#161b22'
  }
}}%%
flowchart TD
    subgraph BronzeTier ["1. Bronze Tier (Verbatim Storage)"]
        direction TB
        B_In["Raw HTTP Responses (UM & ZBK)"] --> B_Hash["SHA256 Payload Hash & Timestamp"]
        B_Hash --> B_Disk[("data/historian/bronze/<br/>source={src}/date={date}/<br/>hour=HH.tar.zst")]
        B_Note["Characteristics:<br/>• 100% loss-free verbatim payload<br/>• Hourly tar.zst bundles (3-5x smaller, 99.6% fewer files)<br/>• In-memory stream decompression (zero disk unpacking)<br/>• Preserves raw municipal quirks & SHA256 audit"]
    end

    subgraph SilverTier ["2. Silver Tier (Clean Trajectories)"]
        direction TB
        S_Read["Stream Bronze Payload from tar.zst"] --> S_Dedup["Deduplicate fixes by (vehicle, timestamp)"]
        S_Dedup --> S_Filter["Velocity Sanity Check (0 <= v <= 120 km/h)"]
        S_Filter --> S_Segment["Trajectory Segmentation (time gap > 5 min -> new trip)"]
        S_Segment --> S_Disk[("data/historian/silver/<br/>date=YYYY-MM-DD/<br/>fixes.parquet")]
        S_Note["Characteristics:<br/>• Standardized column schema<br/>• Snappy compressed Parquet<br/>• Continuous spatial polylines"]
    end

    subgraph GoldTier ["3. Gold Tier (Observed Stop Events)"]
        direction TB
        G_Read["Read Silver Trajectories + Static GTFS"] --> G_Buffer["30-Meter Spatial Buffer Intersection"]
        G_Buffer --> G_State["Stop Event State Machine (ENTER -> STOPPED -> EXIT)"]
        G_State --> G_Dwell["Measure Real Dwell Time (t_exit - t_enter)"]
        G_State --> G_Sched["Match to Static GTFS Scheduled Arrival"]
        G_State --> G_Error["Match Live ZBK Forecast -> Compute Error Δ"]
        G_Error --> G_Disk[("data/historian/gold/stop_events/<br/>date=YYYY-MM-DD/<br/>stop_events.parquet")]
        G_Note["Characteristics:<br/>• Physical ground truth<br/>• Complete trips only (starting stop to terminal stop)<br/>• High confidence stopped events<br/>• Ready for Feature Mart fusion"]
    end

    B_Disk --> S_Read
    S_Disk --> G_Read

    %% Subgraph Styling
    style BronzeTier fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc
    style SilverTier fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc
    style GoldTier fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc

    %% High-Contrast Node Styling
    classDef bStyle fill:#2c2210,stroke:#d29922,stroke-width:2px,color:#f0f6fc;
    classDef sStyle fill:#16243b,stroke:#58a6ff,stroke-width:2px,color:#f0f6fc;
    classDef gStyle fill:#382b14,stroke:#ecc94b,stroke-width:2px,color:#f0f6fc;

    linkStyle default stroke:#8b949e,stroke-width:1.5px;

    class B_In,B_Hash,B_Disk,B_Note bStyle;
    class S_Read,S_Dedup,S_Filter,S_Segment,S_Disk,S_Note sStyle;
    class G_Read,G_Buffer,G_State,G_Dwell,G_Sched,G_Error,G_Disk,G_Note gStyle;
```

---

## 🔄 Database & Spatial Matching Engine: Raw Data (Bronze) → Silver → Gold

Raw municipal GPS telemetry polled from `dane.um.warszawa.pl` is purely observational: it contains vehicle IDs, route numbers, brigade codes, local timestamps, and raw coordinates, but **zero transit schedule context** (no `trip_id`, no `stop_id`, no scheduled timetable arrival targets).

To convert unindexed telemetry into high-value physical stop events, the Historian executes a multi-stage **database matching and geometric referencing pipeline** powered by an embedded [DuckDB](https://duckdb.org/) schedule index.

```mermaid
%%{init: {
  'theme': 'dark',
  'themeVariables': {
    'darkMode': true,
    'background': '#0d1117',
    'mainBkg': '#161b22',
    'primaryColor': '#1f2937',
    'primaryTextColor': '#f0f6fc',
    'primaryBorderColor': '#58a6ff',
    'lineColor': '#8b949e',
    'secondaryColor': '#161b22',
    'tertiaryColor': '#21262d',
    'clusterBkg': '#161b22',
    'clusterBorder': '#30363d',
    'edgeLabelBackground': '#161b22'
  }
}}%%
flowchart TD
    subgraph BronzeInput ["1. Raw Telemetry Feed (Bronze Tier)"]
        RawGPS["Raw Municipal JSON Fixes<br/><code>{Lines, Brigade, VehicleNumber, Time, Lat, Lon}</code>"]
        RawZBK["Zbiorkom GTFS-RT Protobuf Feed<br/><code>{trip_id, stop_sequence, arrival/departure delay}</code>"]
    end

    subgraph StaticDB ["2. Static GTFS Schedule Index (DuckDB)"]
        DuckDBFile[("schedule_index.duckdb<br/><code>trips, stop_times, stops, shapes</code>")]
        TripIndex["Trip Operational Spans CTE<br/><code>(route_id, brigade) -> [trip_start, trip_end]</code>"]
        StopIndex["Stop Timetable Sequences CTE<br/><code>trip_id -> [stop_sequence, stop_id, stop_lat, stop_lon]</code>"]
        DuckDBFile --> TripIndex
        DuckDBFile --> StopIndex
    end

    subgraph SilverStage ["3. Silver Processing: Trajectory & Trip Matching"]
        TimeNorm["Local Time Normalization<br/><i>Europe/Warsaw -> UTC epoch with DST awareness</i>"]
        Kinematics["Kinematic Deduplication<br/><i>Sort by vehicle & time, collapse duplicate fixes</i><br/><i>Compute Δd, Δt, velocity, static flags</i>"]
        TripMatch["<b>DuckDB Trip-Span Range Matcher</b><br/><code>(s_start - 600s) <= fix_sec <= (s_end + 900s)</code><br/><i>Assigns matched_trip_id</i>"]
        SilverStore[("data/historian/silver/<br/>date=YYYY-MM-DD/<br/><b>trajectories.parquet</b>")]
        
        RawGPS --> TimeNorm --> Kinematics
        TripIndex --> TripMatch
        Kinematics --> TripMatch --> SilverStore
    end

    subgraph GoldStage ["4. Gold Processing: Spatial Buffers & Event Interpolation"]
        StopQuery["DuckDB Timetable Extraction<br/><i>Batch query scheduled stops for matched trips</i>"]
        WarsawProj["Warsaw Local Metric Projection<br/><i>Equirectangular 52.23°N, 21.01°E (<0.08% distortion)</i>"]
        BufferInter["25m Geofence Buffer Intersection<br/><i>Vectorized Euclidean distance d_k <= 25m</i>"]
        Sub10Inter["Sub-10s Boundary Crossing Interpolation<br/><i>Linear ratio crossing estimation (t_arr, t_dep)</i>"]
        ZBKCompanion["<b>ZBK Forecast Error Matcher</b><br/><i>Join live GTFS-RT forecast on (trip_id, stop_sequence)</i><br/><code>forecast_error_s = delay_actual - zbk_forecast</code>"]
        Completeness["Trip Completeness Filter<br/><i>Enforce start stop (s_min) and terminal stop (s_max)</i>"]
        GoldStore[("data/historian/gold/stop_events/<br/>date=YYYY-MM-DD/<br/><b>stop_events.parquet</b>")]

        SilverStore --> StopQuery
        StopIndex --> StopQuery
        StopQuery --> WarsawProj --> BufferInter --> Sub10Inter
        RawZBK --> ZBKCompanion
        Sub10Inter --> ZBKCompanion --> Completeness --> GoldStore
    end

    %% Subgraph Styling
    style BronzeInput fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc
    style StaticDB fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc
    style SilverStage fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc
    style GoldStage fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc

    %% Class Styling
    classDef bNode fill:#2c2210,stroke:#d29922,stroke-width:2px,color:#f0f6fc;
    classDef dbNode fill:#271d3b,stroke:#bc8cff,stroke-width:2px,color:#f0f6fc;
    classDef sNode fill:#16243b,stroke:#58a6ff,stroke-width:2px,color:#f0f6fc;
    classDef gNode fill:#382b14,stroke:#ecc94b,stroke-width:2px,color:#f0f6fc;

    linkStyle default stroke:#8b949e,stroke-width:1.5px;

    class RawGPS,RawZBK bNode;
    class DuckDBFile,TripIndex,StopIndex dbNode;
    class TimeNorm,Kinematics,TripMatch,SilverStore sNode;
    class StopQuery,WarsawProj,BufferInter,Sub10Inter,ZBKCompanion,Completeness,GoldStore gNode;
```

---

### 🔍 Step-by-Step Matching Mechanics

#### 1. Static Schedule DuckDB Indexing
During weekly GTFS synchronization, [`src/historian/static_gtfs.py`](file:///Users/truonggiangdo/Data/LearningMaterials/UW/Thesis/DelayTraffic/playground/src/historian/static_gtfs.py) builds `schedule_index.duckdb` directly from static GTFS text archives. Indexes are established on `trips(route_id, brigade, trip_id)` and `stop_times(trip_id, stop_sequence, stop_id)`.

#### 2. Raw to Silver: Brigade-to-Trip Database Matching
Raw fixes from `dane.um.warszawa.pl` supply the operational dispatch keys `(route_id, brigade)`. In [`src/historian/silver.py`](file:///Users/truonggiangdo/Data/LearningMaterials/UW/Thesis/DelayTraffic/playground/src/historian/silver.py), the processor executes an analytical group query against DuckDB:

```sql
SELECT
    t.trip_id,
    t.route_id,
    t.brigade,
    MIN(st.departure_time) AS trip_start,
    MAX(st.arrival_time) AS trip_end
FROM trips t
JOIN stop_times st ON t.trip_id = st.trip_id
WHERE t.brigade IS NOT NULL
GROUP BY t.trip_id, t.route_id, t.brigade;
```

Timetable strings are converted to seconds since midnight ($s_{\text{start}}, s_{\text{end}}$). Each raw GPS fix at timestamp $t_{\text{fix}}$ (in seconds of day) is matched to the active trip using an asymmetric operational time window:
$$(s_{\text{start}} - 600\,\text{s}) \le t_{\text{fix}} \le (s_{\text{end}} + 900\,\text{s})$$
- **$-600\,\text{s}$ (10 min pre-departure buffer)**: Captures early vehicle staging at origin stands.
- **$+900\,\text{s}$ (15 min post-arrival buffer)**: Accounts for cumulative congestion delays without misattributing pings to deadhead returns.
Fixes matching this criteria receive an authoritative `matched_trip_id`.

#### 3. Silver to Gold: Timetable Extraction & Local Metric Projection
In [`src/historian/gold.py`](file:///Users/truonggiangdo/Data/LearningMaterials/UW/Thesis/DelayTraffic/playground/src/historian/gold.py), all scheduled stops for matched trips are pulled via parameterized query:

```sql
SELECT
    st.trip_id,
    st.stop_sequence,
    st.stop_id,
    st.arrival_time,
    st.departure_time,
    s.stop_name,
    s.stop_lat,
    s.stop_lon
FROM stop_times st
JOIN stops s ON st.stop_id = s.stop_id
WHERE st.trip_id IN (?)
ORDER BY st.trip_id, st.stop_sequence;
```

To eliminate costly spherical trigonometry across millions of GPS fixes, coordinates are projected to a **local Euclidean metric plane** centered on the Warsaw centroid ($52.23^\circ\text{N}, 21.01^\circ\text{E}$) with $<0.08\%$ distortion:
$$x = (\lambda - 21.01^\circ) \times 68,325.49\,\text{m/deg}$$
$$y = (\phi - 52.23^\circ) \times 111,271.72\,\text{m/deg}$$

Vectorized Euclidean metric distance to stop post $(s_x, s_y)$:
$$d_k = \sqrt{(x_k - s_x)^2 + (y_k - s_y)^2}$$

#### 4. Sub-10s Stop Boundary Interpolation
Because the ingestion grid operates at 10-second intervals, vehicles often cross the stop geofence ($R = 25.0\,\text{m}$) between discrete GPS pings. Linear boundary interpolation calculates the exact continuous sub-10s crossing instant:

Given fix $(t_{\text{out}}, d_{\text{out}})$ outside the buffer and $(t_{\text{in}}, d_{\text{in}})$ inside:
$$\text{ratio} = \frac{d_{\text{out}} - R}{d_{\text{out}} - d_{\text{in}}}$$
$$t_{\text{boundary}} = t_{\text{out}} + \text{ratio} \cdot (t_{\text{in}} - t_{\text{out}})$$

- **Physical Arrival Epoch**: $t_{\text{actual\_arrival}} = \text{round}(t_{\text{boundary, entry}})$
- **Physical Departure Epoch**: $t_{\text{actual\_departure}} = \text{round}(t_{\text{boundary, exit}})$
- **Measured Dwell Duration**: $\Delta t_{\text{dwell}} = \max(0, t_{\text{actual\_departure}} - t_{\text{actual\_arrival}})$
- **Ground-Truth Delay**: $\text{arrival\_delay\_seconds} = t_{\text{actual\_arrival}} - t_{\text{scheduled\_arrival}}$

#### 5. Live GTFS-RT Companion Forecast Matching
The Historian concurrently ingests raw `zbk_rt` GTFS-RT Protobuf feeds from `api.zbiorkom.live`. Stop events are cross-referenced by composite key `(trip_id, stop_sequence)` to extract the operator's predicted delay at that exact moment:
$$\text{forecast\_error\_s} = \text{arrival\_delay\_seconds} - \text{zbk\_forecast\_delay\_s}$$
This isolates pure operator prediction error from underlying physical road congestion.

---

## 🚦 Stop Event Detection: Finite State Machine (FSM)

The Gold layer uses a spatial-temporal state machine to reliably detect vehicle stop events at designated transit posts:

```mermaid
%%{init: {
  'theme': 'dark',
  'themeVariables': {
    'darkMode': true,
    'background': '#0d1117',
    'lineColor': '#58a6ff',
    'textColor': '#f0f6fc',
    'labelColor': '#f0f6fc',
    'stateBorder': '#58a6ff',
    'stateBkg': '#161b22',
    'noteBkgColor': '#21262d',
    'noteBorderColor': '#30363d',
    'noteTextColor': '#f0f6fc'
  }
}}%%
stateDiagram-v2
    [*] --> TRANSIT: Vehicle navigating corridor (Distance to stop post > 25m)
    
    TRANSIT --> BUFFER_ENTER: Distance <= 25m
    
    BUFFER_ENTER --> STOPPED: Speed < 1.0 m/s for >= 3 seconds (DOORS OPEN / DWELLING)
    BUFFER_ENTER --> BUFFER_EXIT: Speed >= 1.0 m/s (Bypassed stop without stopping)
    
    STOPPED --> BUFFER_EXIT: Vehicle accelerates & exits 25m radius
    
    BUFFER_EXIT --> [*]: Emit Gold Stop Event Record
    
    note right of STOPPED
        actual_arrival_utc = sub-10s interpolated entry
        actual_departure_utc = sub-10s interpolated exit
        dwell_seconds = departure - arrival
        confidence = HIGH (<=12s bracket) | MEDIUM (<=25s) | LOW
    end note

    note right of BUFFER_EXIT
        If vehicle passed within 60m but outside 25m:
        event_type = "PASSED" (dwell = 0, confidence = MEDIUM)
        If vehicle never approached (>60m):
        Non-event omitted from Gold to avoid null bloat
    end note
```

### 🎯 Trip Completeness Invariant (Start-to-Terminal Filtering)

To guarantee statistically unbiased dwell times and valid autoregressive lag features for downstream modeling:
- **Starting Stop Requirement**: The trip's scheduled initial stop ($s_{\min} = \min(\text{stop\_sequence})$ in GTFS) must be observed (`STOPPED` or `PASSED`).
- **Terminal Stop Requirement**: The trip's scheduled final destination ($s_{\max} = \max(\text{stop\_sequence})$ in GTFS) must be observed (`STOPPED` or `PASSED`).
- **Incomplete Trip Exclusion**: Partial or truncated trips (e.g., trips clipped mid-route by ingestion startup or shutdown) are dropped from Gold storage. This guarantees downstream Feature Mart fusion operates exclusively on end-to-end trips with authentic origin-to-destination progression.

---

## 📊 Medallion Tier Data Schemas

### Silver: `trajectories.parquet`
Location: `data/historian/silver/date=YYYY-MM-DD/trajectories.parquet`

| Column | Type | Description |
| :--- | :--- | :--- |
| `service_date` | `STRING` | Operational service date (`"YYYY-MM-DD"`) |
| `vehicle_type` | `STRING` | Vehicle mode (`"bus"` or `"tram"`) |
| `vehicle_number` | `STRING` | Municipal fleet vehicle identifier (`"1001"`, `"3120"`) |
| `route_id` | `STRING` | Route identifier (`"190"`, `"4"`, `"509"`) |
| `brigade` | `STRING` | Operational shift/brigade number |
| `lat` | `FLOAT64` | WGS84 Latitude |
| `lon` | `FLOAT64` | WGS84 Longitude |
| `gps_timestamp_utc` | `INT64` | Normalized GPS fix epoch timestamp (UTC) |
| `gps_time_str` | `STRING` | Original municipal Warsaw local timestamp string |
| `tick_ts` | `INT64` | 10s wall-clock poller cycle epoch timestamp |
| `delta_t_s` | `FLOAT32` | Elapsed time from preceding vehicle fix ($s$) |
| `delta_d_m` | `FLOAT32` | Euclidean metric displacement from preceding fix ($m$) |
| `speed_mps` | `FLOAT32` | Computed kinematic velocity ($m/s$) |
| `is_static` | `BOOL` | True if vehicle is stationary ($v < 0.5\,\text{m/s}$ and $\Delta d < 5\,\text{m}$) |
| `n_ticks_seen` | `INT32` | Count of identical consecutive raw pings collapsed |
| `matched_trip_id` | `STRING` | Authoritative GTFS static trip identifier matched via DuckDB |

---

### Gold: `stop_events.parquet`
Location: `data/historian/gold/stop_events/date=YYYY-MM-DD/stop_events.parquet`

| Column | Type | Description |
| :--- | :--- | :--- |
| `service_date` | `STRING` | Service date (`"YYYY-MM-DD"`) |
| `trip_id` | `STRING` | Matched GTFS trip identifier |
| `route_id` | `STRING` | Route number |
| `brigade` | `STRING` | Shift/brigade number |
| `vehicle_number` | `STRING` | Fleet vehicle ID |
| `vehicle_type` | `STRING` | Vehicle mode (`"bus"` or `"tram"`) |
| `stop_sequence` | `INT32` | Ordered stop sequence along trip |
| `stop_id` | `STRING` | Warsaw ZTM stop post identifier (`"700201"`) |
| `stop_name` | `STRING` | Polish stop name (`"Dw. Centralny 01"`) |
| `scheduled_arrival_utc`| `INT64` | Timetable scheduled arrival epoch (UTC) |
| `scheduled_departure_utc`| `INT64` | Timetable scheduled departure epoch (UTC) |
| `actual_arrival_utc` | `INT64` | **Observed physical arrival epoch** via sub-10s interpolation |
| `actual_departure_utc`| `INT64` | **Observed physical departure epoch** via sub-10s interpolation |
| `arrival_delay_seconds`| `INT32` | Ground truth arrival delay ($s$) ($t_{\text{actual\_arr}} - t_{\text{sched\_arr}}$) |
| `departure_delay_seconds`| `INT32`| Ground truth departure delay ($s$) ($t_{\text{actual\_dep}} - t_{\text{sched\_dep}}$) |
| `dwell_seconds` | `INT32` | Measured dwell time at stop ($s$) ($\max(0, t_{\text{dep}} - t_{\text{arr}})$) |
| `zbk_forecast_delay_s`| `INT32` | Operator live GTFS-RT forecast delay ($s$) |
| `forecast_error_s` | `INT32` | **Forecast Error: $\text{Delay}_{\text{actual}} - \text{Delay}_{\text{forecast}}$** |
| `event_type` | `STRING` | `"STOPPED"` (served stop) vs `"PASSED"` (bypassed corridor) |
| `confidence` | `STRING` | `"HIGH"` ($\le 12\text{s}$ bracket), `"MEDIUM"` ($\le 25\text{s}$), `"LOW"` |
| `min_distance_to_stop_m`| `FLOAT32`| Distance of closest approach to stop pole ($m$) |


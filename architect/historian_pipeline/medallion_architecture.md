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
        B_Hash --> B_Disk[("data/historian/bronze/<br/>date=YYYY-MM-DD/<br/>hour=HH.jsonl.gz")]
        B_Note["Characteristics:<br/>• 100% loss-free verbatim payload<br/>• Append-only chunking<br/>• Preserves raw municipal quirks"]
    end

    subgraph SilverTier ["2. Silver Tier (Clean Trajectories)"]
        direction TB
        S_Read["Read Bronze JSONL Chunks"] --> S_Dedup["Deduplicate fixes by (vehicle, timestamp)"]
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
        G_Note["Characteristics:<br/>• Physical ground truth<br/>• High confidence stopped events<br/>• Ready for Feature Mart fusion"]
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
    [*] --> TRANSIT: Vehicle navigating corridor (Distance to stop post > 30m)
    
    TRANSIT --> BUFFER_ENTER: Distance <= 30m
    
    BUFFER_ENTER --> STOPPED: Speed < 1.0 m/s for >= 3 seconds (DOORS OPEN / DWELLING)
    BUFFER_ENTER --> BUFFER_EXIT: Speed >= 1.0 m/s (Bypassed stop without stopping)
    
    STOPPED --> BUFFER_EXIT: Vehicle accelerates & exits 30m radius
    
    BUFFER_EXIT --> [*]: Emit Gold Stop Event Record
    
    note right of STOPPED
        actual_arrival_utc = timestamp of first standstill
        actual_departure_utc = timestamp of last standstill
        dwell_seconds = departure - arrival
        confidence = HIGH
    end note

    note right of BUFFER_EXIT
        If vehicle passed within 60m but outside 25m:
        event_type = "PASSED" (dwell = 0, confidence = MEDIUM)
        If vehicle never approached (>60m):
        Non-event omitted from Gold to avoid null data bloat
    end note
```

---

## 📊 Medallion Tier Data Schemas

### Silver: `fixes.parquet`
| Column | Type | Description |
| :--- | :--- | :--- |
| `timestamp_utc` | `INT64` | Observation epoch timestamp |
| `vehicle_number` | `STRING` | Fleet vehicle ID (`"1001"`, `"3120"`) |
| `vehicle_type` | `STRING` | `"bus"` or `"tram"` |
| `line` | `STRING` | Route identifier (`"190"`, `"4"`, `"509"`) |
| `brigade` | `STRING` | Shift/brigade number |
| `lat` | `FLOAT64` | WGS84 Latitude |
| `lon` | `FLOAT64` | WGS84 Longitude |
| `speed_mps` | `FLOAT32` | Calculated velocity between consecutive fixes ($m/s$) |
| `delta_d_m` | `FLOAT32` | Displacement from preceding fix ($m$) |

---

### Gold: `stop_events.parquet`
| Column | Type | Description |
| :--- | :--- | :--- |
| `service_date` | `STRING` | Service date (`"YYYY-MM-DD"`) |
| `trip_id` | `STRING` | Matched GTFS trip identifier |
| `route_id` | `STRING` | Route number |
| `vehicle_number` | `STRING` | Fleet vehicle ID |
| `stop_sequence` | `INT32` | Stop position along the route |
| `stop_id` | `STRING` | ZTM stop post identifier (`"700201"`) |
| `stop_name` | `STRING` | Polish stop name (`"Dw. Centralny 01"`) |
| `scheduled_arrival_utc`| `INT64` | GTFS schedule timetable arrival epoch |
| `actual_arrival_utc` | `INT64` | **Observed physical arrival epoch** (0% nulls) |
| `actual_departure_utc`| `INT64` | **Observed physical departure epoch** (0% nulls) |
| `arrival_delay_seconds`| `INT32` | Ground truth arrival delay ($s$) (0% nulls) |
| `departure_delay_seconds`| `INT32`| Ground truth departure delay ($s$) (0% nulls) |
| `dwell_seconds` | `INT32` | Measured dwell time at stop ($s$) (0% nulls) |
| `zbk_forecast_delay_s`| `INT32` | Operator GTFS-RT forecast delay ($s$) |
| `forecast_error_s` | `INT32` | **Forecast Error: $\text{Delay}_{\text{forecast}} - \text{Delay}_{\text{observed}}$** |
| `event_type` | `STRING` | `"STOPPED"` (served stop) vs `"PASSED"` (bypassed corridor) |
| `confidence` | `STRING` | `"HIGH"` ($\le 12\text{s}$ bracket), `"MEDIUM"` ($\le 25\text{s}$), `"LOW"` |
| `min_distance_to_stop_m`| `FLOAT32`| Distance of closest approach to stop pole ($m$) |

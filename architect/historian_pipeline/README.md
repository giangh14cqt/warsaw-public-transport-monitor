# Historian Observed Actuals Ingestion Pipeline

The **Historian Pipeline** is a next-generation, high-frequency public transit telemetry engine designed to solve the critical limitation of legacy GTFS-RT feeds: **the absence of ground-truth physical stop arrival, dwell time, and departure events**.

Operating on a strict **10-second wall-clock boundary**, the Historian captures raw vehicle positions directly from the Warsaw Municipal API (`dane.um.warszawa.pl`), pairs them with synchronized GTFS-RT dispatch forecasts, and processes them through an end-to-end **Medallion Data Architecture** (Bronze $\to$ Silver $\to$ Gold).

---

## 🏛️ Historian Architecture Overview

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
    %% Sources
    subgraph Sources ["Ingestion Sources"]
        UM_API["Warsaw Open Data API (UM)<br/><code>dane.um.warszawa.pl</code><br/><i>Buses (type=1) & Trams (type=2)</i>"]
        ZBK_API["Zbiorkom GTFS-RT Companion<br/><code>api.zbiorkom.live</code><br/><i>TripUpdates (Forecast Delays)</i>"]
        GTFS_Static[("Static GTFS Schedule<br/><i>Stops, Shapes, StopTimes</i>")]
    end

    %% Daemon
    subgraph Daemon ["historian_daemon.py (Continuous 10s Service)"]
        direction TB
        ClockSync["Wall-Clock Aligner<br/><b>src/historian/clock.py</b><br/><i>sleep_until_next_boundary(10s)</i>"]
        
        subgraph MultiPoller ["src/historian/sources/"]
            AsyncFetch["Async Concurrent Poller<br/><code>aiohttp.ClientSession</code>"]
            UMFetcher["UM Telemetry Fetcher<br/><b>um_api.py</b>"]
            ZBKFetcher["GTFS-RT Fetcher<br/><b>zbiorkom.py</b>"]
        end

        BronzeEngine["Verbatim Bronze Writer<br/><b>src/historian/bronze.py</b><br/><i>JSONL chunking & payload hashing</i>"]
        Healthcheck["Healthchecks.io Ping<br/><i>5-min Heartbeat</i>"]
    end

    %% Medallion Pipeline
    subgraph Medallion ["Medallion Data Pipeline (Local Tier)"]
        direction TB
        BronzeTier[("Bronze Tier (Verbatim)<br/><code>data/historian/bronze/YYYY-MM-DD/*.jsonl.gz</code>")]
        SilverEngine["Silver Deduplication & Filtering<br/><b>src/historian/silver.py</b><br/><i>Velocity (0-120 km/h) & Trajectories</i>"]
        SilverTier[("Silver Tier (Clean Fixes)<br/><code>data/historian/silver/YYYY-MM-DD/fixes.parquet</code>")]
        GoldEngine["Gold Stop Event Detector<br/><b>src/historian/gold.py</b><br/><i>30m GIS buffer + Dwell + Error Δ</i>"]
        GoldTier[("Gold Tier (Stop Events)<br/><code>data/historian/gold/stop_events/date=YYYY-MM-DD/</code>")]
    end

    %% Remote Sync
    subgraph Offsite ["Cloud Archive Tier"]
        RcloneHist["rclone 6-Hour Cron Sync<br/><b>backup_historian_to_gdrive.sh</b>"]
        GDriveHist[("Google Drive Historian Archive<br/><code>${RCLONE_REMOTE}:WarsawDelayTelemetry/data/historian</code>")]
    end

    %% Connections
    ClockSync --> AsyncFetch
    AsyncFetch --> UMFetcher --> UM_API
    AsyncFetch --> ZBKFetcher --> ZBK_API
    
    UMFetcher --> BronzeEngine
    ZBKFetcher --> BronzeEngine
    BronzeEngine --> BronzeTier
    ClockSync --> Healthcheck

    BronzeTier --> SilverEngine
    SilverEngine --> SilverTier

    SilverTier --> GoldEngine
    GTFS_Static -.-> GoldEngine
    BronzeTier -.->|ZBK TripUpdates| GoldEngine
    GoldEngine --> GoldTier

    BronzeTier -.->|every 6h| RcloneHist
    SilverTier -.->|every 6h| RcloneHist
    GoldTier -.->|every 6h| RcloneHist
    RcloneHist -.-> GDriveHist

    %% Subgraph Styling
    style Sources fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc
    style Daemon fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc
    style MultiPoller fill:#0d1117,stroke:#58a6ff,stroke-width:1.5px,color:#58a6ff
    style Medallion fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc
    style Offsite fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc

    %% High-Contrast Node Styling
    classDef ext fill:#21262d,stroke:#8b949e,stroke-width:2px,color:#f0f6fc;
    classDef daemon fill:#16243b,stroke:#58a6ff,stroke-width:2px,color:#f0f6fc;
    classDef bronze fill:#2c2210,stroke:#d29922,stroke-width:2px,color:#f0f6fc;
    classDef silver fill:#1c2128,stroke:#cbd5e0,stroke-width:2px,color:#f0f6fc;
    classDef gold fill:#382b14,stroke:#ecc94b,stroke-width:2px,color:#f0f6fc;
    classDef cloud fill:#12281e,stroke:#3fb950,stroke-width:2px,color:#f0f6fc;

    linkStyle default stroke:#8b949e,stroke-width:1.5px;

    class UM_API,ZBK_API,GTFS_Static ext;
    class ClockSync,AsyncFetch,UMFetcher,ZBKFetcher,BronzeEngine,Healthcheck,SilverEngine,GoldEngine daemon;
    class BronzeTier bronze;
    class SilverTier silver;
    class GoldTier gold;
    class RcloneHist,GDriveHist cloud;
```

---

## ⏱️ Wall-Clock Synchronized Polling Cycle

Unlike standard background loops that use naive `time.sleep(10)`, the Historian uses precise **wall-clock interval synchronization** implemented in [`src/historian/clock.py`](file:///Users/truonggiangdo/Data/LearningMaterials/UW/Thesis/DelayTraffic/playground/src/historian/clock.py):

```mermaid
%%{init: {
  'theme': 'dark',
  'themeVariables': {
    'actorBkg': '#161b22',
    'actorBorder': '#58a6ff',
    'actorTextColor': '#f0f6fc',
    'actorLineColor': '#8b949e',
    'signalColor': '#58a6ff',
    'signalTextColor': '#f0f6fc',
    'noteBkgColor': '#21262d',
    'noteBorderColor': '#30363d',
    'noteTextColor': '#f0f6fc'
  }
}}%%
sequenceDiagram
    autonumber
    participant Clock as src/historian/clock.py
    participant Daemon as historian_daemon.py
    participant UM as dane.um.warszawa.pl
    participant ZBK as api.zbiorkom.live
    participant Bronze as src/historian/bronze.py
    participant FS as Local Bronze Storage

    Note over Clock: Target: Boundary at hh:mm:00, :10, :20, :30, :40, :50
    Clock->>Daemon: tick(epoch_target)
    
    par Concurrently query both endpoints
        Daemon->>UM: GET /api/action/busestrams_get/ (type=1 Buses)
        Daemon->>UM: GET /api/action/busestrams_get/ (type=2 Trams)
        Daemon->>ZBK: GET /warsaw/gtfsRealtime/default/tripUpdates (Forecasts)
    end

    UM-->>Daemon: JSON Bus Telemetry (~1,500 active vehicles)
    UM-->>Daemon: JSON Tram Telemetry (~400 active vehicles)
    ZBK-->>Daemon: Protobuf / JSON TripUpdates
    
    Daemon->>Bronze: write_chunk(source="um_api", payload)
    Daemon->>Bronze: write_chunk(source="zbk_rt", payload)
    Bronze->>FS: Append to data/historian/bronze/date=YYYY-MM-DD/hour=HH.jsonl.gz

    Note over Clock: Calculate elapsed processing time (e.g., 1.4s)
    Clock->>Clock: Sleep exactly (10.0 - 1.4) = 8.6s
    Clock->>Daemon: Next tick on next boundary
```

---

## 🎯 Ground-Truth Delay Attribution: Track B Forecast Error Matching

The Historian pipeline solves the delay attribution problem by explicitly pairing observed ground truth with live operator forecasts:

$$\text{Forecast Error} = \text{Delay}_{\text{forecast}} - \text{Delay}_{\text{observed}}$$

```mermaid
%%{init: {
  'theme': 'dark',
  'themeVariables': {
    'darkMode': true,
    'background': '#0d1117',
    'mainBkg': '#161b22',
    'clusterBkg': '#161b22',
    'clusterBorder': '#30363d',
    'primaryTextColor': '#f0f6fc'
  }
}}%%
graph TD
    subgraph ObservedActuals ["Observed Ground-Truth Arrival"]
        O1["GPS Trajectory (Silver)"] --> O2["Enter 30m Stop Buffer"]
        O2 --> O3["Vehicle Stops (speed < 1.0 m/s for >= 3s)"]
        O3 --> O4["<b>actual_arrival_utc</b> = timestamp"]
        O4 --> O5["<b>arrival_delay_seconds</b> = actual - scheduled"]
    end

    subgraph OperatorForecast ["ZTM Live GTFS-RT TripUpdate"]
        F1["tripUpdates feed"] --> F2["Target Stop ID & Stop Sequence"]
        F2 --> F3["<b>arrival.time</b> (Epoch) or <b>arrival.delay</b>"]
        F3 --> F4["<b>zbk_forecast_delay_s</b>"]
    end

    subgraph MetricAttribution ["Delay Attribution Engine (src/historian/gold.py)"]
        O5 --> Calc["<b>forecast_error_s</b> = forecast - actual"]
        F4 --> Calc
        Calc --> Attribution["Decompose error into:<br/>• IMGW Meteorological shock<br/>• OSM Corridor congestion<br/>• Inter-stop headway propagation"]
    end

    style ObservedActuals fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc
    style OperatorForecast fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc
    style MetricAttribution fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc

    classDef obs fill:#16243b,stroke:#58a6ff,stroke-width:2px,color:#f0f6fc;
    classDef fc fill:#2c2210,stroke:#d29922,stroke-width:2px,color:#f0f6fc;
    classDef att fill:#12281e,stroke:#3fb950,stroke-width:2px,color:#f0f6fc;

    linkStyle default stroke:#8b949e,stroke-width:1.5px;

    class O1,O2,O3,O4,O5 obs;
    class F1,F2,F3,F4 fc;
    class Calc,Attribution att;
```

---

## 📁 Key Module Index
- [`historian_daemon.py`](file:///Users/truonggiangdo/Data/LearningMaterials/UW/Thesis/DelayTraffic/playground/historian_daemon.py) - Master 10s daemon loop and signal orchestrator.
- [`src/historian/clock.py`](file:///Users/truonggiangdo/Data/LearningMaterials/UW/Thesis/DelayTraffic/playground/src/historian/clock.py) - Sub-second precision wall-clock boundary scheduler.
- [`src/historian/sources/um_api.py`](file:///Users/truonggiangdo/Data/LearningMaterials/UW/Thesis/DelayTraffic/playground/src/historian/sources/um_api.py) - Warsaw Open Data API client with vehicle type normalization.
- [`src/historian/sources/zbiorkom.py`](file:///Users/truonggiangdo/Data/LearningMaterials/UW/Thesis/DelayTraffic/playground/src/historian/sources/zbiorkom.py) - Companion GTFS-RT TripUpdates poller.
- [`src/historian/bronze.py`](file:///Users/truonggiangdo/Data/LearningMaterials/UW/Thesis/DelayTraffic/playground/src/historian/bronze.py) - Verbatim JSONL storage writer with gzip compression and SHA256 integrity.
- [`src/historian/silver.py`](file:///Users/truonggiangdo/Data/LearningMaterials/UW/Thesis/DelayTraffic/playground/src/historian/silver.py) - Trajectory cleaning, speed thresholding, and Parquet conversion.
- [`src/historian/gold.py`](file:///Users/truonggiangdo/Data/LearningMaterials/UW/Thesis/DelayTraffic/playground/src/historian/gold.py) - Spatial GIS buffer stop event detection, dwell time calculation, and forecast error pairing.

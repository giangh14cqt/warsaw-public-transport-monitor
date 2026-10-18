# Historian Observed Actuals Ingestion Pipeline

The **Historian Pipeline** is a next-generation, high-frequency public transit telemetry engine designed to solve the critical limitation of legacy GTFS-RT feeds: **the absence of ground-truth physical stop arrival, dwell time, and departure events**.

Operating on a strict **10-second wall-clock boundary**, the Historian captures raw vehicle positions directly from the Warsaw Municipal API (`dane.um.warszawa.pl`), pairs them with synchronized GTFS-RT dispatch forecasts, and processes them through an end-to-end **Medallion Data Architecture** (Bronze $\to$ Silver $\to$ Gold).

---

## 🏛️ Historian Architecture Overview

```mermaid
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

    %% Styles
    classDef ext fill:#2d3748,stroke:#4a5568,color:#edf2f7;
    classDef daemon fill:#1a365d,stroke:#2b6cb0,color:#bee3f8;
    classDef bronze fill:#744210,stroke:#d69e2e,color:#fefcbf;
    classDef silver fill:#4a5568,stroke:#cbd5e0,color:#ffffff;
    classDef gold fill:#744210,stroke:#ecc94b,color:#ffffb0;
    classDef cloud fill:#22543d,stroke:#38a169,color:#c6f6d5;

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

    classDef obs fill:#1a365d,stroke:#2b6cb0,color:#bee3f8;
    classDef fc fill:#744210,stroke:#d69e2e,color:#fefcbf;
    classDef att fill:#22543d,stroke:#38a169,color:#c6f6d5;

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

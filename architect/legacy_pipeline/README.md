# Legacy GTFS-RT Telemetry Ingestion Pipeline

The **Legacy Ingestion Pipeline** is a continuous 24/7 telemetry ingestion engine running in production on a remote self-hosted Linux server. It polls GTFS-Realtime Protobuf feeds every 30 seconds from `api.zbiorkom.live`, normalizes vehicle positions and delay estimates, buffers records in memory, and writes partitioned Parquet files on an hourly boundary.

---

## 🏛️ Pipeline Overview & Architecture

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
    subgraph External ["External Provider"]
        Endpoint["GTFS-RT Warsaw Endpoint<br/><code>https://api.zbiorkom.live/api6-open/warsaw/gtfsRealtime/default/...</code>"]
        StaticRepo["ZTM Static GTFS Repository<br/><code>https://mkuran.pl/gtfs/warsaw.zip</code>"]
    end

    subgraph DaemonProcess ["collector_daemon.py (Continuous Background Process)"]
        direction TB
        Timer["Loop Timer (30s Polling Cycle)"]
        
        subgraph FetcherSub ["src/ingestion/fetcher.py"]
            HTTPClient["Async/HTTP Request Client<br/><b>GTFSRealtimeFetcher</b>"]
            ProtoParser["Google Protobuf Parser<br/><code>FeedMessage.ParseFromString()</code>"]
            EntityFilter["Entity Classifier<br/><i>VehiclePositions | TripUpdates | Alerts</i>"]
        end

        subgraph StorageSub ["src/ingestion/storage.py"]
            MemBuffer["Thread-Safe In-Memory Buffer<br/><b>TelemetryStorageManager</b>"]
            HourDetector{"Hour Changed or<br/>Buffer Limit Reached?"}
            BatchWriter["PyArrow Parquet Batch Writer<br/><code>to_parquet(compression='snappy')</code>"]
        end

        subgraph StaticSub ["src/ingestion/gtfs_manager.py"]
            StaticArchiver["Weekly Static GTFS Archiver<br/><b>GTFSManager</b>"]
            HashCheck{"Archive Changed?"}
            ZipStore[("data/raw/static_gtfs/<br/>warsaw_YYYYWW.zip")]
        end

        Heartbeat["Healthchecks.io Ping<br/><i>15-min Dead Man's Switch</i>"]
    end

    subgraph LocalStorage ["Local File System Tier"]
        RawParquet[("data/raw/<br/>year=YYYY/<br/>month=MM/<br/>day=DD/<br/>hour=HH/*.parquet")]
    end

    subgraph CloudBackup ["Cloud Disaster Recovery"]
        Rclone["rclone Sync Job<br/><i>Daily 03:00 UTC via backup_to_gdrive.sh</i>"]
        GDrive[("Google Drive Backup<br/><code>${RCLONE_REMOTE}:WarsawDelayTelemetry/data/raw</code>")]
    end

    %% Wiring
    Endpoint -->|HTTP GET (30s)| HTTPClient
    StaticRepo -->|Weekly Poll| StaticArchiver
    StaticArchiver --> HashCheck
    HashCheck -->|New hash| ZipStore

    Timer --> HTTPClient
    HTTPClient --> ProtoParser
    ProtoParser --> EntityFilter
    EntityFilter -->|Normalized Dicts| MemBuffer
    
    MemBuffer --> HourDetector
    HourDetector -->|Yes| BatchWriter
    BatchWriter --> RawParquet
    
    Timer --> Heartbeat
    RawParquet -.->|Nightly Cron| Rclone
    Rclone -.-> GDrive

    %% Subgraph Styling for Dark Theme
    style External fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc
    style DaemonProcess fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc
    style FetcherSub fill:#0d1117,stroke:#58a6ff,stroke-width:1.5px,color:#58a6ff
    style StorageSub fill:#0d1117,stroke:#bc8cff,stroke-width:1.5px,color:#bc8cff
    style StaticSub fill:#0d1117,stroke:#3fb950,stroke-width:1.5px,color:#3fb950
    style LocalStorage fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc
    style CloudBackup fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc

    %% High-Contrast Node Styling
    classDef extNode fill:#21262d,stroke:#8b949e,stroke-width:2px,color:#f0f6fc;
    classDef daemonNode fill:#16243b,stroke:#58a6ff,stroke-width:2px,color:#f0f6fc;
    classDef storeNode fill:#2c2210,stroke:#d29922,stroke-width:2px,color:#f0f6fc;
    classDef cloudNode fill:#12281e,stroke:#3fb950,stroke-width:2px,color:#f0f6fc;

    linkStyle default stroke:#8b949e,stroke-width:1.5px;

    class Endpoint,StaticRepo extNode;
    class Timer,HTTPClient,ProtoParser,EntityFilter,MemBuffer,HourDetector,BatchWriter,StaticArchiver,HashCheck,Heartbeat daemonNode;
    class RawParquet,ZipStore storeNode;
    class Rclone,GDrive cloudNode;
```

---

## 🔄 Sequence Diagram: 30-Second Ingestion Loop

The interaction between the ingestion daemon, protobuf fetcher, in-memory buffer, and local storage is shown below:

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
    'labelBoxBkgColor': '#21262d',
    'labelBoxBorderColor': '#30363d',
    'labelTextColor': '#f0f6fc',
    'loopTextColor': '#f0f6fc',
    'noteBkgColor': '#21262d',
    'noteBorderColor': '#30363d',
    'noteTextColor': '#f0f6fc',
    'activationBkgColor': '#1f242c',
    'activationBorderColor': '#58a6ff'
  }
}}%%
sequenceDiagram
    autonumber
    participant D as collector_daemon.py
    participant F as fetcher.py (GTFSRealtimeFetcher)
    participant API as api.zbiorkom.live
    participant S as storage.py (TelemetryStorageManager)
    participant FS as Local Filesystem (data/raw/)
    participant HC as Healthchecks.io

    loop Every 30 Seconds
        D->>F: fetch_all_feeds()
        F->>API: GET /vehiclePositions,tripUpdates,serviceAlerts
        API-->>F: Binary Protobuf Payload (FeedMessage)
        F->>F: FeedMessage.ParseFromString()
        F->>F: Extract & Flatten entities (VP, TU, SA)
        F-->>D: Dictionary of normalized records
        D->>S: append_records(records, current_timestamp)
        S->>S: Store in memory deque
        
        alt Current Wall Clock Hour > Buffer Hour
            D->>S: flush_buffer(target_hour)
            S->>S: Convert to pyarrow.Table
            S->>FS: Write data/raw/year=YYYY/month=MM/day=DD/hour=HH/telemetry_batch.parquet
            S->>S: Reset buffer state
        end

        opt Every 15 Minutes
            D->>HC: GET ${HEARTBEAT_URL} (Success Ping)
            HC-->>D: 200 OK
        end
    end
```

---

## 📁 Directory & Partition Layout

The storage engine writes partitioned Parquet files following the standard Apache Hive convention:

```text
data/raw/
├── static_gtfs/
│   ├── warsaw_202615.zip          <- Weekly static GTFS snapshots
│   └── warsaw_202616.zip
└── year=2026/
    └── month=10/
        ├── day=17/
        │   ├── hour=00/
        │   │   └── telemetry_20261017_000000.parquet
        │   ├── hour=01/
        │   └── ...
        └── day=18/
            ├── hour=14/
            │   └── telemetry_20261018_140000.parquet
            └── hour=15/
```

---

## 🛡️ Reliability & Operational Protocols

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
graph LR
    subgraph Monitoring ["Dead Man's Switch (Healthchecks.io)"]
        HC1[" collector_daemon.py "] -->|"Ping every 15m"| HC2["Healthchecks.io"]
        HC2 -->|"Missed 2 pings (30m)"| HC3["Alert: Discord / Email / SMS"]
    end

    subgraph Recovery ["Disaster Recovery & Offsite Sync"]
        Cron["Cron: Daily 03:00 UTC"] --> Script["backup_to_gdrive.sh"]
        Script -->|"rclone copy --update"| Remote["Google Drive Remote"]
        Remote --> LocalPull["pull_from_gdrive.sh (Workstation Pull)"]
    end

    style Monitoring fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc
    style Recovery fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc

    classDef monNode fill:#33171a,stroke:#ff7b72,stroke-width:2px,color:#f0f6fc;
    classDef recNode fill:#12281e,stroke:#3fb950,stroke-width:2px,color:#f0f6fc;

    linkStyle default stroke:#8b949e,stroke-width:1.5px;

    class HC1,HC2,HC3 monNode;
    class Cron,Script,Remote,LocalPull recNode;

### Key Python Module References
- [`collector_daemon.py`](file:///Users/truonggiangdo/Data/LearningMaterials/UW/Thesis/DelayTraffic/playground/collector_daemon.py) - Master continuous loop and signal handlers (`SIGTERM`, `SIGINT`).
- [`src/ingestion/fetcher.py`](file:///Users/truonggiangdo/Data/LearningMaterials/UW/Thesis/DelayTraffic/playground/src/ingestion/fetcher.py) - HTTP request handling, retry exponential backoff, and GTFS-RT Protobuf parser.
- [`src/ingestion/storage.py`](file:///Users/truonggiangdo/Data/LearningMaterials/UW/Thesis/DelayTraffic/playground/src/ingestion/storage.py) - Deque buffer management and Parquet file serialization.
- [`src/ingestion/gtfs_manager.py`](file:///Users/truonggiangdo/Data/LearningMaterials/UW/Thesis/DelayTraffic/playground/src/ingestion/gtfs_manager.py) - Weekly MD5-verified static GTFS archiver.

# Warsaw Transit Telemetry & Delay Attribution System
# System Architecture & Technical Blueprint

This directory contains comprehensive architectural diagrams and technical specifications for the entire **Warsaw Public Transport Telemetry, Historian & Delay Attribution System**. All diagrams are written in standard [Mermaid.js](https://mermaid.js.org/) syntax and natively render within GitHub and compatible Markdown viewers.

---

## 🗺️ Master System Architecture

The following diagram illustrates the complete end-to-end data lifecycle: from external municipal and meteorological data feeds, through dual ingestion pipelines, storage and cloud synchronization, DuckDB-powered feature mart fusion, econometric modeling, and explainable AI attribution.

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
flowchart TB
    %% External Data Sources
    subgraph Sources ["1. External Ingestion Endpoints"]
        direction TB
        S1["ZTM / Zbiorkom GTFS-RT<br/><i>api.zbiorkom.live</i><br/>(VehiclePositions, TripUpdates, Alerts)"]
        S2["Warsaw Open Data UM<br/><i>dane.um.warszawa.pl</i><br/>(10s GPS Telemetry: Bus & Tram)"]
        S3["IMGW-PIB Synoptic Weather<br/><i>danepubliczne.imgw.pl</i><br/>(Hourly: Okęcie 375 & Bielany 370)"]
        S4["OpenStreetMap / OSMnx<br/><i>Overpass API</i><br/>(Corridor Topology & Lanes)"]
        S5["Static GTFS Schedule<br/><i>Weekly Warsaw Archive</i><br/>(Trips, Stops, StopTimes, Shapes)"]
    end

    %% Ingestion Pipelines
    subgraph Ingestion ["2. Ingestion Tier"]
        direction TB
        subgraph LegacyPipeline ["Legacy Telemetry Pipeline (collector_daemon.py)"]
            L1["GTFS-RT Poller (30s interval)"]
            L2["Memory Buffer & Schema Normalizer"]
            L3["Hourly Parquet Batch Writer"]
        end
        subgraph HistorianPipeline ["Historian Observed Actuals (historian_daemon.py)"]
            H1["10s Wall-Clock Poller (UM API)"]
            H2["Bronze Verbatim Store (JSONL / tar.zst)"]
            H3["Silver Trajectory Kinematics & DuckDB Trip Matcher"]
            H4["Gold Stop Event Reconstructor (25m buffer + sub-10s interpolation)"]
            H5["GTFS-RT Companion Matching (Error Δ)"]
        end
    end

    %% Storage & Remote Sync Tier
    subgraph StorageTier ["3. Storage & Cloud Synchronization Tier"]
        direction TB
        D_Raw[("data/raw/<br/>Hive Partitioned Parquet<br/><i>year=Y/month=M/day=D/hour=H/</i>")]
        D_Hist[("data/historian/<br/>Medallion Storage<br/><i>bronze/ | silver/ | gold/</i>")]
        D_Exo[("data/exogenous/<br/>Enriched Tables<br/><i>weather.parquet | osm.parquet</i>")]
        GDrive[("Google Drive Remote Backup<br/><i>rclone sync via daily/6h cron</i>")]
    end

    %% Data Processing & Mart Fusion
    subgraph ProcessingTier ["4. Data Processing & Fusion Engine (src/fusion/)"]
        direction TB
        MartAssembler["DuckDB Out-of-Core Mart Assembler<br/><b>FeatureMartAssembler</b>"]
        MartLegacy[("feature_mart.parquet<br/><i>Legacy GTFS-RT Telemetry</i>")]
        MartObserved[("feature_mart_observed.parquet<br/><i>Gold Observed Actuals + Error</i>")]
    end

    %% Modeling & Machine Learning
    subgraph ModelingTier ["5. Validation, Econometrics & GBDT Suite (src/models/)"]
        direction TB
        Splitter["Purged Temporal Block CV<br/><i>Adaptive Embargo Buffer (0-leakage)</i>"]
        TWFE["Econometric TWFE Baseline<br/><i>Two-Way Route & Time Fixed Effects</i>"]
        GBM["Gradient Boosted Trees Suite<br/><i>LightGBM & CatBoost</i>"]
        Stratification["Cohort Stratification<br/><i>Urban Bus | Urban Tram | Suburban Bus | Pooled</i>"]
    end

    %% xAI & Publication Diagnostics
    subgraph xAITier ["6. Explainable AI & Policy Diagnostics (src/xai/)"]
        direction TB
        SHAP["TreeSHAP Engine<br/><i>Global Feature Importance & Local Waterfalls</i>"]
        ALE["Accumulated Local Effects (ALE)<br/><i>1D & 2D Non-linear Threshold Surfaces</i>"]
        Interactions["Friedman's H-Statistic<br/><i>Second-Order Feature Interactions</i>"]
        DiCE["DiCE Counterfactuals<br/><i>Minimal Actionable Intervention Paths</i>"]
        PubPlots["Publication-Grade Visualizations<br/><i>academic styling (300 DPI)</i>"]
    end

    %% Flow Connections
    S1 --> L1
    L1 --> L2 --> L3 --> D_Raw
    S5 -.-> L1

    S2 --> H1
    S1 -.-> H5
    H1 --> H2 --> H3 --> H4
    H5 --> H4
    H4 --> D_Hist
    S5 -.->|schedule_index.duckdb| H3
    S5 -.->|stops timetable| H4

    S3 --> D_Exo
    S4 --> D_Exo

    D_Raw -->|datasource=legacy| MartAssembler
    D_Hist -->|datasource=historian| MartAssembler
    D_Exo --> MartAssembler

    D_Raw -.->|daily backup_to_gdrive.sh| GDrive
    D_Hist -.->|6h backup_historian_to_gdrive.sh| GDrive

    MartAssembler --> MartLegacy
    MartAssembler --> MartObserved

    MartLegacy --> Splitter
    MartObserved --> Splitter

    Splitter --> Stratification
    Stratification --> TWFE
    Stratification --> GBM

    GBM --> SHAP
    GBM --> ALE
    GBM --> Interactions
    GBM --> DiCE
    SHAP --> PubPlots
    ALE --> PubPlots
    Interactions --> PubPlots
    DiCE --> PubPlots

    %% Subgraph Styling for GitHub Dark Theme
    style Sources fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc
    style Ingestion fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc
    style LegacyPipeline fill:#0d1117,stroke:#1f6feb,stroke-width:1.5px,color:#58a6ff
    style HistorianPipeline fill:#0d1117,stroke:#238636,stroke-width:1.5px,color:#3fb950
    style StorageTier fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc
    style ProcessingTier fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc
    style ModelingTier fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc
    style xAITier fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc

    %% High-Contrast Node Styling
    classDef sourceNode fill:#21262d,stroke:#8b949e,stroke-width:2px,color:#f0f6fc;
    classDef legacyNode fill:#16243b,stroke:#58a6ff,stroke-width:2px,color:#f0f6fc;
    classDef histNode fill:#12281e,stroke:#3fb950,stroke-width:2px,color:#f0f6fc;
    classDef storageNode fill:#2c2210,stroke:#d29922,stroke-width:2px,color:#f0f6fc;
    classDef fusionNode fill:#271d3b,stroke:#bc8cff,stroke-width:2px,color:#f0f6fc;
    classDef modelNode fill:#33172b,stroke:#f472b6,stroke-width:2px,color:#f0f6fc;
    classDef xaiNode fill:#33171a,stroke:#ff7b72,stroke-width:2px,color:#f0f6fc;

    linkStyle default stroke:#8b949e,stroke-width:1.5px;

    class S1,S2,S3,S4,S5 sourceNode;
    class L1,L2,L3 legacyNode;
    class H1,H2,H3,H4,H5 histNode;
    class D_Raw,D_Hist,D_Exo,GDrive storageNode;
    class MartAssembler,MartLegacy,MartObserved fusionNode;
    class Splitter,TWFE,GBM,Stratification modelNode;
    class SHAP,ALE,Interactions,DiCE,PubPlots xaiNode;
```

---

## 🔄 End-to-End Database Matching Architecture

The system resolves raw telemetry into modeling matrices through two sequential, decoupled database matching pipelines:

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
flowchart LR
    subgraph MatchPhase1 ["Phase 1: Raw Data to Gold Stop Events"]
        direction TB
        P1_Bronze["Raw GPS Pings (Bronze)<br/><code>{route, brigade, lat, lon, time}</code>"]
        P1_DuckDB["DuckDB Schedule Index<br/><code>schedule_index.duckdb</code>"]
        P1_Silver["Silver Trajectories<br/><code>trajectories.parquet</code><br/><i>(route, brigade, time window match)</i>"]
        P1_Gold["Gold Stop Events<br/><code>stop_events.parquet</code><br/><i>(25m buffer + sub-10s crossing + ZBK error Δ)</i>"]

        P1_Bronze --> P1_Silver
        P1_DuckDB -.->|Trip spans| P1_Silver
        P1_Silver --> P1_Gold
        P1_DuckDB -.->|Stop timetable| P1_Gold
    end

    subgraph MatchPhase2 ["Phase 2: Gold Stop Events to Design Matrix ('Mat Matrix')"]
        direction TB
        P2_Gold["Gold Stop Events<br/><code>stop_events.parquet</code>"]
        P2_DuckDB["DuckDB Vectorized SQL Engine<br/><code>src/fusion/feature_mart.py</code>"]
        P2_Window["Window Lags & Deltas<br/><i>prev_stop_delay, Δt_run, headway_dev</i>"]
        P2_Exo["4-Way Exogenous Joins<br/><i>Weather, OSM Corridors, Calendar, Hubs</i>"]
        P2_Mart["Fused Feature Mart<br/><code>feature_mart_observed.parquet</code>"]
        P2_Matrix["Design Matrix [X | y]<br/><i>Target y = Δt_run, Matrix X in R^{N x P}</i><br/><i>Purged Temporal Block CV Splitting</i>"]

        P2_Gold --> P2_DuckDB
        P2_DuckDB --> P2_Window --> P2_Exo --> P2_Mart --> P2_Matrix
    end

    P1_Gold ==> P2_Gold

    style MatchPhase1 fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc
    style MatchPhase2 fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc

    classDef p1Style fill:#16243b,stroke:#58a6ff,stroke-width:2px,color:#f0f6fc;
    classDef p2Style fill:#271d3b,stroke:#bc8cff,stroke-width:2px,color:#f0f6fc;

    linkStyle default stroke:#8b949e,stroke-width:1.5px;

    class P1_Bronze,P1_DuckDB,P1_Silver,P1_Gold p1Style;
    class P2_Gold,P2_DuckDB,P2_Window,P2_Exo,P2_Mart,P2_Matrix p2Style;
```

1. **Raw Telemetry to Gold Stop Events**:
   - Matches raw `(route_id, brigade)` operational pings against `schedule_index.duckdb` within operational time windows $[s_{\text{start}} - 600\text{s}, s_{\text{end}} + 900\text{s}]$ to identify `matched_trip_id`.
   - Projects fixes onto Warsaw local metric plane ($<0.08\%$ distortion) and intersects with 25m stop buffers.
   - Computes continuous sub-10s boundary crossing timestamps, measured dwells, and live operator forecast error $\Delta$.
   - **Deep Dive**: [`architect/historian_pipeline/medallion_architecture.md`](file:///Users/truonggiangdo/Data/LearningMaterials/UW/Thesis/DelayTraffic/playground/architect/historian_pipeline/medallion_architecture.md#database-spatial-matching-engine-raw-data-bronze-silver-gold).

2. **Gold Stop Events to Modeling Design Matrix ("Mat Matrix")**:
   - Out-of-core DuckDB vectorized SQL query pipeline scanning `stop_events.parquet`.
   - Computes autoregressive lag features (`prev_stop_delay`, `prev2_stop_delay`, $\Delta t_{\text{run}}$, $\Delta t_{\text{dwell}}$, $\text{trip\_progress}$) partitioned by `trip_instance_key`.
   - Computes inter-vehicle headway deviations partitioned by `(route_id, stop_id)`.
   - Executes four concurrent exogenous joins (IMGW synoptic weather, OSMnx road corridor topology, Polish calendar, transit hubs).
   - Filters origin stops, terminal turnaround layovers (*pętle*), and extreme outliers.
   - Slices the resulting design matrix $[\mathbf{X} \mid \mathbf{y}]$ via purged temporal block cross-validation with an adaptive embargo buffer for econometric and machine learning estimation.
   - **Deep Dive**: [`architect/data_processing/feature_mart_fusion.md`](file:///Users/truonggiangdo/Data/LearningMaterials/UW/Thesis/DelayTraffic/playground/architect/data_processing/feature_mart_fusion.md#database-matching-feature-matrix-mat-matrix-assembly-gold-feature-mart-modeling-matrix).

---

## 📂 Architecture Directory Index

The architecture documentation is organized into three specialized segments:

| Segment | Path | Description |
| :--- | :--- | :--- |
| **1. Legacy Pipeline** | [`legacy_pipeline/`](file:///Users/truonggiangdo/Data/LearningMaterials/UW/Thesis/DelayTraffic/playground/architect/legacy_pipeline/) | GTFS-RT 30s protobuf ingestion, hourly partitioning, remote server daemon, and backup scripts. |
| **2. Historian Pipeline** | [`historian_pipeline/`](file:///Users/truonggiangdo/Data/LearningMaterials/UW/Thesis/DelayTraffic/playground/architect/historian_pipeline/) | 10s wall-clock aligned observed actuals, Medallion storage (Bronze/Silver/Gold), DuckDB static GTFS schedule matching, 25m geofence sub-10s interpolation, and GTFS-RT forecast error calculation. |
| **3. Data Processing & Modeling** | [`data_processing/`](file:///Users/truonggiangdo/Data/LearningMaterials/UW/Thesis/DelayTraffic/playground/architect/data_processing/) | Exogenous feature engineering (weather, OSM, hubs), DuckDB dual-datasource fusion, Database Matching (Gold to Mat Matrix), purged temporal validation, econometric TWFE, GBDTs, and xAI attribution. |

---

## ⚖️ Architecture Comparison: Legacy vs. Historian

| Feature / Metric | Legacy Pipeline (`data/raw/`) | Historian Pipeline (`data/historian/`) |
| :--- | :--- | :--- |
| **Primary Data Source** | `api.zbiorkom.live` (GTFS-RT Protobuf) | `dane.um.warszawa.pl` (UM Warsaw API REST) |
| **Polling Cadence** | 30 seconds (best effort) | 10 seconds (strict wall-clock synchronized) |
| **Primary Target Metric** | Schedule Deviation (Predicted arrival delay) | **Observed Ground-Truth Arrival/Departure Delay** |
| **Dwell Time Resolution** | Imputed / missing | Measured physical vehicle standstill ($\ge 3\,\text{s}$) |
| **Forecast Error ($\Delta$)** | N/A (predicts against static timetable) | $|\text{Delay}_{\text{forecast}} - \text{Delay}_{\text{observed}}|$ (evaluated against real arrival) |
| **Storage Architecture** | Hive Partitioned Parquet (`year/month/day/hour`) | Medallion Architecture (`bronze/jsonl` $\to$ `silver/parquet` $\to$ `gold/parquet`) |
| **Storage Volume / Day** | $\approx 200 - 300\,\text{MB/day}$ | $\approx 85 - 110\,\text{MB/day}$ (compact compressed Parquet) |
| **Sync Cadence (Google Drive)** | Daily at 03:00 UTC via `backup_to_gdrive.sh` | Every 6 hours at :30 via `backup_historian_to_gdrive.sh` |
| **Operational Process** | `collector_daemon.py` (Docker or systemd) | `historian_daemon.py` (Docker Compose or systemd) |

---

## 🛠️ Technology Stack & Engine Registry

```mermaid
mindmap
  root((Warsaw Transit Architecture))
    Ingestion & OS
      Docker Compose
      Python 3.12 Daemon
      Aiohttp & Requests
      Healthchecks.io Dead Man's Switch
      rclone Google Drive Backup
    High-Performance Storage
      Apache Parquet Snappy/ZSTD
      DuckDB Vectorized SQL Engine
      Medallion Tiering (Bronze / Silver / Gold)
    Exogenous Telemetry
      IMGW-PIB Synoptic Weather API
      OSMnx & NetworkX Road Topology
      Shapely & GeoPandas GIS Buffers
    Econometrics & ML
      Linearmodels (TWFE Fixed Effects)
      LightGBM & CatBoost Regressors
      Scikit-learn Validation & Metrics
    Explainable AI (xAI)
      TreeSHAP TreeExplainer
      PyALE Accumulated Local Effects
      DiCE Genetic Counterfactuals
      Matplotlib / Seaborn Publication Styling
```

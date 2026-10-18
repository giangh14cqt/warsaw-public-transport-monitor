# Exogenous Telemetry Enrichment Architecture

This document details the ingestion, spatial aggregation, and temporal indexing of exogenous features—meteorological variables, road corridor network topology, and transit interchange proximity—that explain non-recurrent transit delays.

---

## 🌐 Multi-Source Exogenous Ingestion Architecture

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
    subgraph WeatherSub ["1. Meteorological Telemetry (src/exogenous/imgw.py)"]
        direction TB
        IMGW_API["IMGW-PIB Synoptic Public API<br/><code>https://danepubliczne.imgw.pl/api/data/synop/id/...</code>"]
        Stations["Stations: Okęcie (375) & Bielany (370)"]
        W_Extract["Extract: Temp (°C), Wind Speed (m/s), Rain (mm), Pressure (hPa), Visibility (m)"]
        W_Fill["Forward-Fill Missing Hours (Max 2h)"]
        W_Out[("data/exogenous/weather.parquet")]

        IMGW_API --> Stations --> W_Extract --> W_Fill --> W_Out
    end

    subgraph OSMSub ["2. Road Corridor Network Topology (src/exogenous/osm.py)"]
        direction TB
        Overpass["Overpass API / OpenStreetMap"]
        OSM_Extract["OSMnx Road Graph (drive filter)"]
        OSM_Metrics["Compute Edge Attributes:<br/>• maxspeed (km/h)<br/>• lanes count<br/>• highway classification (primary, trunk, secondary)<br/>• busway / dedicated transit lane tag"]
        OSM_Out[("data/exogenous/osm_corridors.parquet")]

        Overpass --> OSM_Extract --> OSM_Metrics --> OSM_Out
    end

    subgraph HubSub ["3. Multi-Modal Transit Hubs (src/exogenous/interchange.py)"]
        direction TB
        StaticStops[("Static GTFS stops.txt")]
        Hub_Cluster["Spatial Clustering (DBSCAN / 150m radius)"]
        Hub_Metrics["Compute Hub Features:<br/>• transfer_lines_count<br/>• has_metro_transfer<br/>• has_rail_transfer<br/>• is_major_interchange"]
        Hub_Out[("data/exogenous/interchanges.parquet")]

        StaticStops --> Hub_Cluster --> Hub_Metrics --> Hub_Out
    end

    subgraph CalendarSub ["4. Calendar & Temporal Features (src/core/calendar.py)"]
        direction TB
        Epoch["Observation UTC Epoch"]
        CalEngine["Polish Holiday Calendar (holidays package)"]
        Cal_Metrics["Derive:<br/>• is_holiday (boolean)<br/>• is_weekend (boolean)<br/>• time_of_day_sin / cos<br/>• is_rush_hour (AM: 07-09, PM: 16-18)"]
    end

    %% Subgraph Styling
    style WeatherSub fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc
    style OSMSub fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc
    style HubSub fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc
    style CalendarSub fill:#161b22,stroke:#30363d,stroke-width:1.5px,color:#f0f6fc

    %% High-Contrast Node Styling
    classDef wStyle fill:#16243b,stroke:#58a6ff,stroke-width:2px,color:#f0f6fc;
    classDef oStyle fill:#12281e,stroke:#3fb950,stroke-width:2px,color:#f0f6fc;
    classDef hStyle fill:#2c2210,stroke:#d29922,stroke-width:2px,color:#f0f6fc;
    classDef cStyle fill:#271d3b,stroke:#bc8cff,stroke-width:2px,color:#f0f6fc;

    linkStyle default stroke:#8b949e,stroke-width:1.5px;

    class IMGW_API,Stations,W_Extract,W_Fill,W_Out wStyle;
    class Overpass,OSM_Extract,OSM_Metrics,OSM_Out oStyle;
    class StaticStops,Hub_Cluster,Hub_Metrics,Hub_Out hStyle;
    class Epoch,CalEngine,Cal_Metrics cStyle;
```

---

## ⏱️ Spatial-Temporal Feature Join Topology

During feature mart assembly in DuckDB, each transit stop observation is joined against exogenous tables using a hybrid **ASOF temporal join** and **stop-post attribute lookup**:

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
    participant FM as Feature Mart Assembler (DuckDB)
    participant T as Transit Telemetry (Legacy/Historian)
    participant W as weather.parquet (IMGW)
    participant O as osm_corridors.parquet (OSM)
    participant H as interchanges.parquet (Hubs)

    FM->>T: Scan transit stop observation (lat, lon, stop_id, timestamp_utc)
    FM->>W: Temporal ASOF Join ON date=date AND hour=hour
    Note over W: Match closest synoptic weather observation within preceding hour
    W-->>FM: temperature_c, wind_speed_ms, precipitation_mm, visibility_m
    
    FM->>O: Lookup ON stop_id / corridor_id
    Note over O: Extract roadway geometry: maxspeed, lanes, highway category
    O-->>FM: osm_maxspeed, osm_lanes, osm_is_primary
    
    FM->>H: Lookup ON stop_id
    Note over H: Extract transfer connectivity & multi-modal hub flags
    H-->>FM: is_major_interchange, transfer_routes_count, has_metro
```

---

## 📊 Exogenous Feature Definitions

| Feature Category | Column Name | Type | Unit / Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Meteorology** | `temperature_c` | `FLOAT32` | $-25.0 \dots +40.0\,^\circ\text{C}$ | Ambient synoptic air temperature |
| | `wind_speed_ms` | `FLOAT32` | $0.0 \dots 35.0\,\text{m/s}$ | Sustained wind velocity |
| | `precipitation_mm` | `FLOAT32` | $0.0 \dots 100.0\,\text{mm/h}$ | Hourly accumulated rainfall / melted snowfall |
| | `visibility_m` | `FLOAT32` | $50 \dots 10000\,\text{m}$ | Atmospheric horizontal optical visibility |
| | `air_pressure_hpa` | `FLOAT32` | $950 \dots 1050\,\text{hPa}$ | Atmospheric barometric surface pressure |
| **Corridor Network**| `osm_maxspeed` | `INT32` | $30 \dots 80\,\text{km/h}$ | Legal vehicular speed limit of adjacent corridor |
| | `osm_lanes` | `INT32` | $1 \dots 5$ | Number of road lanes along corridor edge |
| | `osm_has_busway` | `INT32` | $0 \text{ or } 1$ | Dedicated transit / bus lane flag (`busway=yes`) |
| **Transit Hub** | `is_major_interchange`| `INT32` | $0 \text{ or } 1$ | Flag for major transit interchange hub |
| | `transfer_routes_count`| `INT32` | $1 \dots 25$ | Number of intersecting unique transit routes |
| | `has_metro_transfer` | `INT32` | $0 \text{ or } 1$ | Metro M1 / M2 transfer connection |
| **Calendar/Time** | `is_weekend` | `INT32` | $0 \text{ or } 1$ | Saturday or Sunday indicator |
| | `is_holiday` | `INT32` | $0 \text{ or } 1$ | Polish statutory public holiday indicator |
| | `is_rush_hour` | `INT32` | $0 \text{ or } 1$ | Morning (07-09) or afternoon (16-18) peak window |
| | `hour_sin`, `hour_cos` | `FLOAT32` | $[-1.0, 1.0]$ | Cyclical trigonometric time-of-day encoding |

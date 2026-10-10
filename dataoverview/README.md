# Warsaw Public Transport Telemetry - Data Overview & Visualizations

This directory contains Jupyter notebooks for exploratory data analysis (EDA), visual diagnostics, and telemetry inspection of the Warsaw Transit Telemetry & Delay Attribution dataset.

---

## Notebooks Overview

### 1. [`01_transit_delay_overview.ipynb`](01_transit_delay_overview.ipynb)
**Focus**: *Live High-Frequency Transit Delay Telemetry*
- **DuckDB Parquet Scanning**: Vectorized queries scanning across 11.14M raw observations without loading excessive data into RAM.
- **Daily Ingestion Breakdown**: Ingestion volume across the 7-day collection window (Oct 4 – Oct 10, 2026).
- **Delay Distributions**: Statistical spread of arrival delay ($\Delta t$), kernel density estimations (KDE), and quantiles comparing **Trams vs. Buses**.
- **Diurnal Congestion Curve**: 24-hour delay profile highlighting Warsaw's morning peak (07:00–09:00) and evening peak (16:00–18:00).
- **Route Rankings**: Top 15 most delayed and top 15 most punctual routes in Warsaw.
- **Geospatial Delay Hotspots**: Spatial scatter map mapping delay intensity across physical transit stops in Warsaw.

### 2. [`02_exogenous_features_overview.ipynb`](02_exogenous_features_overview.ipynb)
**Focus**: *Meteorological Telemetry & Road Infrastructure Topology*
- **IMGW-PIB Synoptic Weather**: 169 hourly observations for Warsaw Okęcie across the October 4–10 study window (temperature, precipitation, wind speed, relative humidity, freezing rain).
- **OSMnx Study Corridors**: Physical topology of the 4 core transit test corridors (*Puławska*, *Al. Jerozolimskie*, *Trasa W-Z*, *Towarowa / Okopowa*).
- **Infrastructure Metrics**: Stop-to-stop spacing, traffic signal density (signals per km), mixed-traffic lane capacity, and dedicated right-of-way segregation.
- **Multi-Source Fusion Preview**: Demonstration of three-way join between transit delays, weather observations, and corridor topology.

---

## How to Run

1. Open the repository in your IDE (VS Code or JupyterLab).
2. Select the virtual environment kernel:
   ```bash
   .venv/bin/python
   ```
3. Run cells interactively.

### Tips for Large Data Handling
The raw telemetry contains over **11 million records** (225 MB Parquet). All queries utilize **DuckDB** with:
- **Sample Queries**: `USING SAMPLE 100000` for instant distribution plotting.
- **Anomaly Filters**: `WHERE ABS(arrival_delay_seconds) <= 1800` to isolate normal operational transit windows (eliminating terminal depot pullouts).
- **Partition Pruning**: Filter by `WHERE day = 5` or `WHERE route_id = '17'` to narrow down execution to sub-second runtimes.

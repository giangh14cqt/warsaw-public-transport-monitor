# Warsaw Public Transport Telemetry: Data Overview & Visualizations

This directory contains Jupyter notebooks for exploratory data analysis (EDA), visual diagnostics, and telemetry inspection of the Warsaw Transit Telemetry & Delay Attribution dataset.

---

## Notebooks Overview

### 1. [`01_transit_delay_overview.ipynb`](01_transit_delay_overview.ipynb)
**Focus**: *Live High-Frequency Transit Delay Telemetry*
- **Fast DuckDB Scanning**: Queries millions of raw observations in Parquet format quickly without running out of RAM.
- **Daily Ingestion Breakdown**: Tracks collection volume across the study window (October 4–10, 2026).
- **Delay Distributions**: Explores arrival delay ($\Delta t$) distributions, kernel density estimates (KDE), and quantiles comparing **Trams vs. Buses**.
- **Daily Rush-Hour Patterns**: Plots 24-hour delay trends across Warsaw's morning peak (07:00–09:00) and evening peak (16:00–18:00).
- **Route Rankings**: Identifies the 15 most delayed and 15 most punctual transit lines in Warsaw.
- **Geographic Delay Hotspots**: Visualizes delay hotspots across physical transit stops on an interactive map.

### 2. [`02_exogenous_features_overview.ipynb`](02_exogenous_features_overview.ipynb)
**Focus**: *Weather Observations & Road Network Topology*
- **IMGW-PIB Synoptic Weather**: Analyzes 169 hourly weather records from Warsaw Okęcie (temperature, rainfall, wind speed, relative humidity, and freezing rain).
- **OSMnx Transit Corridors**: Examines the road network geometry across 4 major radial corridors (*Puławska*, *Al. Jerozolimskie*, *Trasa W-Z*, *Towarowa / Okopowa*).
- **Infrastructure Metrics**: Measures stop spacing, traffic signal density (signals per kilometer), mixed-traffic lane count, and dedicated right-of-way segregation.
- **Multi-Source Data Fusion**: Demonstrates how transit delays, weather records, and road corridor geometries are joined together.

---

## How to Run

1. Open the repository in your code editor (such as VS Code or JupyterLab).
2. Select the virtual environment Python interpreter:
   ```bash
   .venv/bin/python
   ```
3. Run the notebook cells interactively.

### Tips for Querying Large Datasets
The raw telemetry contains over **24 million records** (~225 MB Parquet), and the processed feature mart contains **8.83 million records** (~286 MB Parquet). All notebook queries use **DuckDB** for maximum performance:
- **Fast Sampling**: Use `USING SAMPLE 100000` to preview distributions instantly.
- **Filter Outliers**: Use `WHERE ABS(arrival_delay_seconds) <= 1800` to focus on normal transit operations and remove vehicle depot movements.
- **Partition Filtering**: Filter by partition columns (such as `WHERE day = '05'` or `WHERE route_id = '17'`) so DuckDB reads only the needed files in milliseconds.
- **Data Schemas & Reference**: For full data dictionary and column-level definitions across all tables, see [DATA_SCHEMA.md](../DATA_SCHEMA.md).
- **SQL Workbench**: For ready-to-run DuckDB queries, see [`query.sql`](query.sql).

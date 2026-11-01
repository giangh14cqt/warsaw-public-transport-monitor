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

### 3. [`03_feature_mart_overview.ipynb`](03_feature_mart_overview.ipynb)
**Focus**: *Master Modeling Feature Mart Exploration & Econometric Analysis*
- **Dual-Track Comparative Inventory**: Audits both Track B Observed Ground Truth (5.61M rows, 54 columns) and Track A GTFS-RT Reference Baseline (8.83M rows, 45 columns).
- **Target Variable ($\Delta t$) Distribution & Decomposition**: Evaluates operational arrival delays within the $\pm 30$-minute operational envelope, segment running delay ($\Delta t_{run}$), and physical passenger dwell standstills.
- **Modal Buffer Effect**: Empirically demonstrates the punctuality advantage of 100% dedicated tram rails (median delay 35s, 87.4% punctual) over mixed-traffic buses (median delay 79s, 69.6% punctual).
- **Autoregressive Shock Transmission ($\rho$)**: Quantifies delay persistence ($\Delta t_s$ vs. $\Delta t_{s-1}$), proving that bus delays compound upstream ($\rho = 1.010 > 1.0$) while tram delays dampen ($\rho = 0.984 < 1.0$).
- **Exogenous Meteorological Shocks**: Measures the +32% rain delay penalty for road buses vs. +17% for segregated rail trams, and charts non-linear delay escalation under heavy rainfall (> 2.0 mm/h).
- **Corridor Bottlenecks & Right-of-Way**: Evaluates peak vs. off-peak congestion across Warsaw's 12 major transit arteries, contrasting fully mixed traffic (*Wisłostrada*) against dedicated right-of-way (*Towarowa / Okopowa*).
- **Network Centrality & Headway Irregularity**: Examines +22% longer dwell times at Metro M1/M2 interchange stops and the 5x delay spike under vehicle headway gapping.
- **GTFS-RT Forecast Fidelity**: Benchmarks public real-time delay predictions against observed physical ground truth (MAE $\approx 23$s, $r = 0.91$).
- **Multi-Collinearity & Readiness Matrix**: Generates the 14-feature correlation heatmap and descriptive statistics table for econometric TWFE and machine learning.

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

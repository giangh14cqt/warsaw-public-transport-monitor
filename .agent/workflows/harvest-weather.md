---
description: Harvest hourly meteorological telemetry from IMGW-PIB and update exogenous tables
---

# Weather Telemetry Harvesting Workflow (`/harvest-weather`)

This workflow pulls hourly synoptic station data from the IMGW-PIB portal for Warsaw stations (Okęcie and Bielany) and updates the local DuckDB weather mart.

## Step 1: Execute Weather Ingestion
```bash
python3 -m src.exogenous.imgw
```

## Step 2: Validate Hourly Coverage & Anomaly Checks
```bash
python3 -c "
import duckdb
con = duckdb.connect(':memory:')
df = con.execute('''
    SELECT 
        station_id,
        MIN(timestamp) AS earliest,
        MAX(timestamp) AS latest,
        COUNT(*) AS total_hours,
        ROUND(AVG(precipitation_mm), 2) AS avg_precip,
        ROUND(AVG(temperature_c), 1) AS avg_temp
    FROM 'data/processed/imgw_weather_hourly.parquet'
    GROUP BY station_id
''').fetchdf()
print(df)
"
```

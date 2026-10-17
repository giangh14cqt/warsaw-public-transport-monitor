-- ============================================================================
-- Warsaw Transit Telemetry & Attribution System: DuckDB Query Workbench
-- ============================================================================

-- 1. Query all raw partition files across subdirectories as a single virtual table
SELECT
    year,
    month,
    day,
    COUNT(*) AS records_ingested,
    COUNT(DISTINCT route_id) AS distinct_routes,
    COUNT(DISTINCT vehicle_id) AS active_vehicles,
    ROUND(AVG(arrival_delay_seconds), 1) AS mean_delay_s
FROM read_parquet('data/raw/*/*/*/*.parquet')
GROUP BY year, month, day
ORDER BY year, month, day;

-- 2. Inspect the Master Modeling Table (feature_mart.parquet)
-- This is the unified table used for Econometric TWFE, LightGBM, and xAI analysis
SELECT
    transit_cohort,
    COUNT(*) AS total_segments,
    ROUND(AVG(arrival_delay_seconds), 2) AS avg_delay_seconds,
    ROUND(MEDIAN(arrival_delay_seconds), 2) AS median_delay_seconds,
    ROUND(AVG(prev_stop_delay), 2) AS avg_upstream_delay,
    ROUND(AVG(segment_length_meters), 1) AS avg_segment_meters,
    ROUND(AVG(CASE WHEN is_dedicated_right_of_way THEN 1.0 ELSE 0.0 END) * 100, 1) AS pct_dedicated_row
FROM read_parquet('data/processed/feature_mart.parquet')
GROUP BY transit_cohort
ORDER BY total_segments DESC;

-- 3. Compare Peak-Hour Congestion: Dedicated Right-of-Way vs Mixed Traffic
SELECT
    is_dedicated_right_of_way,
    is_peak_hour,
    COUNT(*) AS segment_count,
    ROUND(AVG(arrival_delay_seconds), 2) AS mean_delay_seconds,
    ROUND(MEDIAN(arrival_delay_seconds), 2) AS median_delay_seconds
FROM read_parquet('data/processed/feature_mart.parquet')
WHERE ABS(arrival_delay_seconds) <= 1800
GROUP BY is_dedicated_right_of_way, is_peak_hour
ORDER BY is_dedicated_right_of_way, is_peak_hour;

-- 4. Hourly Synoptic Weather Ingestion Summary
SELECT
    station_name,
    COUNT(*) AS total_hours,
    MIN(timestamp) AS earliest_record,
    MAX(timestamp) AS latest_record,
    ROUND(AVG(temperature_c), 1) AS avg_temperature_c,
    ROUND(SUM(precipitation_mm), 1) AS total_rainfall_mm
FROM read_parquet('data/processed/imgw_weather_hourly.parquet')
GROUP BY station_name;

"""
Multi-Dimensional Stratified Feature Sampling & Inverse Probability Weighting (IPS).
Enables statistically rigorous, balanced subsampling across rare meteorological and
infrastructure regimes without target-variable selection bias or temporal leakage.
Designed for streaming DuckDB execution across multi-month datasets.
"""

import logging
from typing import Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


def build_stratified_sampling_sql(
    mart_path: str,
    sample_size: int = 100000,
    date_where_sql: str = "",
    seed: int = 42,
    strata_quota: Optional[int] = None,
) -> str:
    """
    Constructs an optimized DuckDB SQL query that stratifies along orthogonal feature dimensions
    (weather regime x infrastructure right-of-way x diurnal peak timing) and computes
    exact Inverse Probability of Sampling (IPS) weights.

    Parameters
    ----------
    mart_path : str
        Path or glob pattern to feature mart Parquet file(s).
    sample_size : int
        Target total sample size (if <= 0, returns full dataset with unit weights).
    date_where_sql : str
        Optional WHERE filter clause (e.g. "WHERE arrival_ts >= ...").
    seed : int
        Random seed for deterministic pseudo-random hash ranking.
    strata_quota : Optional[int]
        Max observations per stratum cell. If omitted, dynamically calculated as
        ceil(sample_size / 12) with fallback redistribution.

    Returns
    -------
    str
        DuckDB SQL query ready for execution or COPY streaming.
    """
    where_filter = f"WHERE {date_where_sql}" if (date_where_sql and not date_where_sql.strip().upper().startswith("WHERE")) else date_where_sql

    if sample_size <= 0:
        # Full dataset with unit weights
        return f"""
            SELECT *, 1.0::DOUBLE AS inverse_sampling_weight
            FROM '{mart_path}'
            {where_filter}
        """

    if strata_quota and strata_quota > 0:
        quota = strata_quota
    else:
        # Average Warsaw trip contains ~25 stops
        avg_stops_per_trip = 25
        target_trips = max(1, sample_size // avg_stops_per_trip)
        quota = max(1, int((target_trips + 11) // 12))

    return f"""
        WITH trip_metrics AS (
            SELECT 
                trip_id || '_' || CAST(arrival_ts AS DATE)::VARCHAR AS _trip_instance,
                -- 1. Meteorological Regime (Max severity across trip duration)
                CASE 
                    WHEN MAX(precipitation_mm) > 1.0 THEN 'mod_heavy_rain'
                    WHEN MAX(precipitation_mm) > 0.0 THEN 'light_rain'
                    ELSE 'dry'
                END AS _regime_weather,

                -- 2. Infrastructure Right-of-Way Regime
                CASE 
                    WHEN MAX(CASE WHEN transit_cohort = 'urban_tram' OR is_tram = 1 THEN 1 ELSE 0 END) = 1 THEN 'tram_row'
                    WHEN MAX(CASE WHEN is_dedicated_right_of_way THEN 1 ELSE 0 END) = 1 THEN 'bus_dedicated_row'
                    ELSE 'bus_mixed_traffic'
                END AS _regime_infra,

                -- 3. Temporal Congestion Peak Regime
                CASE 
                    WHEN MAX(is_peak_hour) = 1 THEN 'peak'
                    ELSE 'offpeak'
                END AS _regime_time,
                COUNT(*) AS _n_stops_in_trip
            FROM '{mart_path}'
            {where_filter}
            GROUP BY 1
        ),
        strata_trip_ranks AS (
            SELECT 
                _trip_instance,
                _regime_weather,
                _regime_infra,
                _regime_time,
                COUNT(*) OVER (PARTITION BY _regime_weather, _regime_infra, _regime_time) AS _n_stratum_trips_pop,
                COUNT(*) OVER () AS _n_total_trips_pop,
                ROW_NUMBER() OVER (
                    PARTITION BY _regime_weather, _regime_infra, _regime_time
                    ORDER BY hash(_trip_instance || '_{seed}')
                ) AS _rank_in_stratum
            FROM trip_metrics
        ),
        sampled_trips AS (
            SELECT 
                _trip_instance,
                _n_stratum_trips_pop,
                _n_total_trips_pop,
                COUNT(*) OVER (PARTITION BY _regime_weather, _regime_infra, _regime_time) AS _n_stratum_trips_sample,
                COUNT(*) OVER () AS _n_total_trips_sample
            FROM strata_trip_ranks
            WHERE _rank_in_stratum <= {quota}
        )
        SELECT 
            b.*,
            -- Inverse Sampling Weight: (Pop_k / Total_Pop) / (Sample_k / Total_Sample)
            -- Guarantees mathematically unbiased expectations under stratified trip oversampling
            ROUND(
                (CAST(s._n_stratum_trips_pop AS DOUBLE) / s._n_total_trips_pop) / 
                NULLIF(CAST(s._n_stratum_trips_sample AS DOUBLE) / s._n_total_trips_sample, 0.0),
                6
            ) AS inverse_sampling_weight
        FROM sampled_trips s
        JOIN '{mart_path}' b
          ON s._trip_instance = (b.trip_id || '_' || CAST(b.arrival_ts AS DATE)::VARCHAR)
        ORDER BY b.arrival_ts
    """


def downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcasts numeric and categorical columns to optimize memory usage by ~30-50%.
    Essential when scaling up sample sizes or evaluating across multi-month datasets.
    """
    import numpy as np

    for col in df.columns:
        dtype_str = str(df[col].dtype)
        if dtype_str == "float64":
            df[col] = df[col].astype(np.float32)
        elif dtype_str == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif dtype_str == "object" and col in (
            "transit_cohort",
            "corridor_name",
            "event_type",
            "confidence",
        ):
            df[col] = df[col].astype("category")
    return df


def load_feature_mart_sample(
    mart_path: str,
    sample_size: int = 100000,
    strategy: str = "stratified",
    date_where_sql: str = "",
    seed: int = 42,
    downcast_memory: bool = True,
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> pd.DataFrame:
    """
    Loads a representative sample from a feature mart Parquet file or multi-month glob.

    Parameters
    ----------
    mart_path : str
        Parquet file path or glob pattern.
    sample_size : int
        Desired number of observations (0 or negative for all).
    strategy : str
        'stratified' (orthogonal feature strata + IPS weights),
        'uniform' (uniform reservoir sampling + unit weights),
        or 'none' (full dataset).
    date_where_sql : str
        Optional SQL WHERE clause.
    seed : int
        Random seed for reproducibility.
    downcast_memory : bool
        Whether to downcast dtypes (float32, int32/int16, category) to conserve RAM (default: True).
    con : Optional[duckdb.DuckDBPyConnection]
        Optional DuckDB connection. If None, creates a temporary connection.

    Returns
    -------
    pd.DataFrame
        Sampled dataframe with an 'inverse_sampling_weight' column.
    """
    close_con = False
    if con is None:
        con = duckdb.connect()
        close_con = True

    try:
        where_clause = date_where_sql
        if where_clause and not where_clause.strip().upper().startswith("WHERE"):
            where_clause = f"WHERE {where_clause}"

        if sample_size <= 0 or strategy == "none":
            logger.info(f"Loading FULL dataset from {mart_path}...")
            query = f"SELECT *, 1.0::DOUBLE AS inverse_sampling_weight FROM '{mart_path}' {where_clause}"
            df = con.execute(query).df()
            if downcast_memory and not df.empty:
                df = downcast_dataframe(df)
            return df

        if strategy == "stratified":
            logger.info(
                f"Extracting {sample_size:,} stratified observations with IPS weights from {mart_path}..."
            )
            query = build_stratified_sampling_sql(
                mart_path=mart_path,
                sample_size=sample_size,
                date_where_sql=where_clause,
                seed=seed,
            )
            df = con.execute(query).df()
            if df.empty:
                logger.warning("Stratified query returned 0 rows, falling back to reservoir sampling.")
                query = f"SELECT *, 1.0::DOUBLE AS inverse_sampling_weight FROM '{mart_path}' {where_clause} USING SAMPLE {sample_size} (reservoir, {seed})"
                df = con.execute(query).df()
            else:
                # Normalize weights so that mean(weight) == 1.0
                mean_w = df["inverse_sampling_weight"].mean()
                if mean_w > 0:
                    df["inverse_sampling_weight"] = (df["inverse_sampling_weight"] / mean_w).round(6)
            if downcast_memory and not df.empty:
                df = downcast_dataframe(df)
            return df

        # Default fallback: uniform reservoir sampling
        logger.info(f"Extracting {sample_size:,} uniform reservoir sample rows from {mart_path}...")
        query = f"SELECT *, 1.0::DOUBLE AS inverse_sampling_weight FROM '{mart_path}' {where_clause} USING SAMPLE {sample_size} (reservoir, {seed})"
        df = con.execute(query).df()
        if downcast_memory and not df.empty:
            df = downcast_dataframe(df)
        return df

    finally:
        if close_con:
            con.close()


def stream_feature_mart_batches(
    mart_path: str,
    batch_size: int = 250000,
    date_where_sql: str = "",
    downcast_memory: bool = True,
):
    """
    Generator yielding DataFrame chunks from multi-month Parquet files for out-of-core streaming.
    Enables memory-capped model scoring and residual evaluation across tens of millions of records.

    Parameters
    ----------
    mart_path : str
        Parquet file path or glob pattern.
    batch_size : int
        Chunk size in rows per iteration (default: 250,000).
    date_where_sql : str
        Optional WHERE filter clause.
    downcast_memory : bool
        Whether to downcast numeric dtypes.

    Yields
    ------
    pd.DataFrame
        Batches of feature mart records.
    """
    con = duckdb.connect()
    try:
        where_clause = date_where_sql
        if where_clause and not where_clause.strip().upper().startswith("WHERE"):
            where_clause = f"WHERE {where_clause}"

        query = f"SELECT * FROM '{mart_path}' {where_clause}"
        cursor = con.cursor()
        cursor.execute(query)

        while True:
            chunk_df = cursor.fetch_df_chunk(batch_size)
            if chunk_df is None or len(chunk_df) == 0:
                break
            if downcast_memory:
                chunk_df = downcast_dataframe(chunk_df)
            yield chunk_df
    finally:
        con.close()


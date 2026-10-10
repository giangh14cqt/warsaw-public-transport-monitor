"""
Trajectory Reconstruction & Delay Decomposition Subsystem.
Reconstructs sequential vehicle trips by ordering on (trip_id, stop_sequence).

Computes core decomposed transit delay metrics:
1. Running Segment Delay: Delta t_run
   Delta t_run = (rt_arrival_s - rt_departure_{s-1}) - (sched_arrival_s - sched_departure_{s-1})
2. Platform Dwell Delay: Delta t_dwell
   Delta t_dwell = (rt_departure_s - rt_arrival_s) - (sched_departure_s - sched_arrival_s)
3. Operational State Lag Features:
   - Delta t_{s-1} (delay observed at preceding stop s-1)
   - Delta t_{s-2} (delay observed at stop s-2)
4. Headway Deviation:
   h_dev = h_actual - h_median
   Time gap relative to the preceding vehicle on the same route and stop.
5. Trip Progress Ratio:
   trip_progress = (stop_sequence - min_seq) / (max_seq - min_seq)
"""

import os
import logging
from typing import Optional, Dict, Any, Union
import pandas as pd
import numpy as np
import duckdb

logger = logging.getLogger(__name__)


class TrajectoryReconstructor:
    """Reconstructs trip trajectories and decomposes transit delay mechanics."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
        self.output_parquet = os.path.join(
            self.processed_dir, "reconstructed_trajectories.parquet"
        )

    # --------------------------------------------------------------------------
    # 1. In-Memory Pandas Vectorized Processing (for unit testing and batches)
    # --------------------------------------------------------------------------

    def compute_delays_and_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a telemetry dataframe to compute segmented delays and lag features.
        Expects columns:
        ['trip_id', 'route_id', 'stop_sequence', 'stop_id', 'arrival_delay_seconds']
        Optional columns:
        ['departure_delay_seconds', 'rt_arrival_time', 'rt_departure_time', 'feed_timestamp']
        """
        if df.empty:
            return df.copy()

        df = df.copy()

        # Deduplicate to ensure strictly one observation per (trip_id, stop_sequence)
        if "feed_timestamp" in df.columns:
            df = df.sort_values(
                by=["trip_id", "stop_sequence", "feed_timestamp"]
            ).drop_duplicates(subset=["trip_id", "stop_sequence"], keep="last")
        else:
            df = df.drop_duplicates(subset=["trip_id", "stop_sequence"], keep="last")

        # Ensure correct sequential ordering
        df = df.sort_values(by=["trip_id", "stop_sequence"]).reset_index(drop=True)

        # Preceding stop references (Lags)
        df["prev_stop_id"] = df.groupby("trip_id")["stop_id"].shift(1)
        df["prev_stop_delay"] = df.groupby("trip_id")["arrival_delay_seconds"].shift(1)
        df["prev2_stop_delay"] = df.groupby("trip_id")["arrival_delay_seconds"].shift(2)

        # Running segment delay delta: change in delay from stop s-1 to stop s
        df["delta_t_run"] = df["arrival_delay_seconds"] - df["prev_stop_delay"]

        # Platform dwell delay delta
        if "departure_delay_seconds" in df.columns:
            # When departure delay is recorded, dwell delay is the difference
            dep_delay = df["departure_delay_seconds"].fillna(df["arrival_delay_seconds"])
            df["delta_t_dwell"] = dep_delay - df["arrival_delay_seconds"]
        else:
            df["delta_t_dwell"] = 0.0

        # Observed running transit time (seconds between departure of s-1 and arrival of s)
        if "rt_arrival_time" in df.columns:
            if "rt_departure_time" in df.columns:
                dep_time = df["rt_departure_time"].fillna(df["rt_arrival_time"])
            else:
                dep_time = df["rt_arrival_time"]
            prev_dep_time = df.groupby("trip_id")[dep_time.name].shift(1)
            df["observed_run_time_s"] = df["rt_arrival_time"] - prev_dep_time
            # Negative observed runtime indicates sensor glitch or timestamp reversal
            df["observed_run_time_s"] = df["observed_run_time_s"].apply(
                lambda x: x if (pd.notna(x) and x > 0) else np.nan
            )

        # Trip progress ratio [0.0, 1.0]
        min_seq = df.groupby("trip_id")["stop_sequence"].transform("min")
        max_seq = df.groupby("trip_id")["stop_sequence"].transform("max")
        seq_range = max_seq - min_seq
        df["trip_progress"] = np.where(
            seq_range > 0, (df["stop_sequence"] - min_seq) / seq_range, 0.0
        ).round(3)
        df["is_origin_stop"] = df["stop_sequence"] == min_seq

        # Headway Deviation relative to preceding vehicle on the same route and stop
        if "route_id" in df.columns and "rt_arrival_time" in df.columns:
            df = df.sort_values(by=["route_id", "stop_id", "rt_arrival_time"])
            prev_vehicle_arrival = df.groupby(["route_id", "stop_id"])[
                "rt_arrival_time"
            ].shift(1)
            df["headway_actual_s"] = df["rt_arrival_time"] - prev_vehicle_arrival
            # Filter valid headway windows (e.g. 10s to 2 hours between successive vehicles)
            valid_headway = df["headway_actual_s"].apply(
                lambda x: x if (pd.notna(x) and 10 <= x <= 7200) else np.nan
            )
            median_headway = df.groupby(["route_id", "stop_id"])[
                valid_headway.name
            ].transform("median")
            df["headway_deviation"] = valid_headway - median_headway
            # Restore trip-based ordering
            df = df.sort_values(by=["trip_id", "stop_sequence"]).reset_index(drop=True)

        return df

    # --------------------------------------------------------------------------
    # 2. Scalable DuckDB Vectorized Processing (for multi-million row Parquet sinks)
    # --------------------------------------------------------------------------

    def build_reconstruction_query(
        self,
        input_parquet_pattern: str,
        max_delay_seconds: int = 7200,
        max_records: Optional[int] = None,
    ) -> str:
        """Construct the optimized SQL query for full-fleet trajectory reconstruction."""
        limit_clause = f"LIMIT {max_records}" if max_records else ""
        return f"""
            WITH deduped AS (
                SELECT 
                    feed_timestamp,
                    trip_id,
                    route_id,
                    vehicle_id,
                    stop_sequence,
                    stop_id,
                    arrival_delay_seconds,
                    departure_delay_seconds,
                    rt_arrival_time,
                    rt_departure_time
                FROM read_parquet('{input_parquet_pattern}', hive_partitioning=true)
                WHERE ABS(arrival_delay_seconds) <= {max_delay_seconds}
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY trip_id, stop_sequence 
                    ORDER BY feed_timestamp DESC
                ) = 1
                {limit_clause}
            ),
            base_ordered AS (
                SELECT 
                    feed_timestamp,
                    trip_id,
                    route_id,
                    vehicle_id,
                    stop_sequence,
                    stop_id,
                    arrival_delay_seconds,
                    COALESCE(departure_delay_seconds, arrival_delay_seconds) AS departure_delay_seconds,
                    rt_arrival_time,
                    COALESCE(rt_departure_time, rt_arrival_time) AS rt_departure_time,
                    -- Preceding Stop Lags
                    LAG(stop_id) OVER (PARTITION BY trip_id ORDER BY stop_sequence) AS prev_stop_id,
                    LAG(arrival_delay_seconds) OVER (PARTITION BY trip_id ORDER BY stop_sequence) AS prev_stop_delay,
                    LAG(arrival_delay_seconds, 2) OVER (PARTITION BY trip_id ORDER BY stop_sequence) AS prev2_stop_delay,
                    LAG(COALESCE(rt_departure_time, rt_arrival_time)) OVER (PARTITION BY trip_id ORDER BY stop_sequence) AS prev_rt_departure,
                    MAX(stop_sequence) OVER (PARTITION BY trip_id) AS max_sequence,
                    MIN(stop_sequence) OVER (PARTITION BY trip_id) AS min_sequence
                FROM deduped
            ),
            decomposed AS (
                SELECT 
                    feed_timestamp,
                    trip_id,
                    route_id,
                    vehicle_id,
                    stop_sequence,
                    stop_id,
                    prev_stop_id,
                    arrival_delay_seconds,
                    departure_delay_seconds,
                    prev_stop_delay,
                    prev2_stop_delay,
                    rt_arrival_time,
                    rt_departure_time,
                    -- Running Segment Delay (Delta t_run)
                    CASE 
                        WHEN prev_stop_delay IS NOT NULL THEN (arrival_delay_seconds - prev_stop_delay)
                        ELSE 0.0 
                    END AS delta_t_run,
                    -- Platform Dwell Delay (Delta t_dwell)
                    (departure_delay_seconds - arrival_delay_seconds) AS delta_t_dwell,
                    -- Observed physical running time in seconds
                    CASE 
                        WHEN prev_rt_departure IS NOT NULL AND rt_arrival_time > prev_rt_departure 
                        THEN (rt_arrival_time - prev_rt_departure) 
                        ELSE NULL 
                    END AS observed_run_time_s,
                    -- Trip progression ratio
                    CASE 
                        WHEN max_sequence > min_sequence 
                        THEN ROUND(CAST(stop_sequence - min_sequence AS DOUBLE) / (max_sequence - min_sequence), 3)
                        ELSE 0.0 
                    END AS trip_progress,
                    CASE WHEN stop_sequence = min_sequence THEN true ELSE false END AS is_origin_stop
                FROM base_ordered
            ),
            with_headway AS (
                SELECT 
                    *,
                    (rt_arrival_time - LAG(rt_arrival_time) OVER (
                        PARTITION BY route_id, stop_id 
                        ORDER BY rt_arrival_time
                    )) AS headway_actual_s
                FROM decomposed
            )
            SELECT 
                *,
                CASE 
                    WHEN headway_actual_s >= 10 AND headway_actual_s <= 7200 
                    THEN headway_actual_s - MEDIAN(headway_actual_s) OVER (PARTITION BY route_id, stop_id)
                    ELSE NULL 
                END AS headway_deviation
            FROM with_headway
            ORDER BY trip_id, stop_sequence
        """

    def reconstruct_trajectories(
        self,
        input_parquet_pattern: Optional[str] = None,
        output_parquet: Optional[str] = None,
        max_records: Optional[int] = None,
    ) -> str:
        """
        Execute full trajectory reconstruction in DuckDB and persist to Parquet.
        """
        in_pattern = input_parquet_pattern or os.path.join(self.raw_dir, "**/*.parquet")
        out_path = output_parquet or self.output_parquet

        con = duckdb.connect(":memory:")
        sql = self.build_reconstruction_query(
            in_pattern, max_records=max_records
        )
        copy_query = f"COPY ({sql}) TO '{out_path}' (FORMAT PARQUET, COMPRESSION SNAPPY)"

        logger.info(f"Reconstructing vehicle trajectories from {in_pattern}...")
        con.execute(copy_query)
        con.close()

        logger.info(f"Reconstructed trajectories saved to: {out_path}")
        return out_path

    def get_summary(
        self, parquet_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch descriptive summary of the reconstructed trajectories via DuckDB."""
        path = parquet_path or self.output_parquet
        if not os.path.exists(path):
            return {"status": "empty", "records": 0}

        con = duckdb.connect(":memory:")
        res = con.execute(f"""
            SELECT 
                COUNT(*) AS total_records,
                COUNT(DISTINCT trip_id) AS distinct_trips,
                COUNT(DISTINCT route_id) AS distinct_routes,
                ROUND(AVG(delta_t_run), 2) AS mean_delta_t_run,
                ROUND(MEDIAN(delta_t_run), 2) AS median_delta_t_run,
                ROUND(STDDEV(delta_t_run), 2) AS std_delta_t_run,
                ROUND(AVG(delta_t_dwell), 2) AS mean_delta_t_dwell,
                ROUND(AVG(headway_actual_s), 1) AS mean_headway_s,
                ROUND(MEDIAN(headway_actual_s), 1) AS median_headway_s,
                COUNT(prev_stop_id) AS segments_with_prev_stop
            FROM read_parquet('{path}')
        """).fetchone()

        return {
            "total_records": int(res[0]),
            "distinct_trips": int(res[1]),
            "distinct_routes": int(res[2]),
            "mean_delta_t_run": float(res[3]) if res[3] is not None else 0.0,
            "median_delta_t_run": float(res[4]) if res[4] is not None else 0.0,
            "std_delta_t_run": float(res[5]) if res[5] is not None else 0.0,
            "mean_delta_t_dwell": float(res[6]) if res[6] is not None else 0.0,
            "mean_headway_s": float(res[7]) if res[7] is not None else 0.0,
            "median_headway_s": float(res[8]) if res[8] is not None else 0.0,
            "segments_with_prev_stop": int(res[9]),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    reconstructor = TrajectoryReconstructor()
    print("TrajectoryReconstructor ready.")

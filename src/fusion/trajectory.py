"""
Trajectory Reconstruction & Delay Decomposition Subsystem.
Reconstructs sequential vehicle trips by ordering on (trip_id, stop_sequence).
Computes:
1. Running Segment Delay: Delta t_run
2. Platform Dwell Delay: Delta t_dwell
3. Lagged features: prev_stop_delay (Delta t_{s-1})
4. Operational Headway Deviation relative to the preceding vehicle on the same route.
"""

import logging
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TrajectoryReconstructor:
    """Reconstructs trip trajectories and decomposes transit delay mechanics."""

    def __init__(self):
        pass

    def compute_delays_and_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a telemetry dataframe to compute segmented delays and lag features.
        Expects columns:
        ['trip_id', 'route_id', 'stop_sequence', 'stop_id',
         'rt_arrival_time', 'rt_departure_time', 'arrival_delay_seconds']
        """
        if df.empty:
            return df

        # Ensure correct sequential ordering
        df = df.sort_values(by=["trip_id", "stop_sequence"]).copy()

        # Previous stop reference
        df["prev_stop_id"] = df.groupby("trip_id")["stop_id"].shift(1)
        df["prev_rt_departure"] = df.groupby("trip_id")["rt_departure_time"].shift(1)
        df["prev_stop_delay"] = df.groupby("trip_id")["arrival_delay_seconds"].shift(1)

        # Running segment observed transit time (seconds between departure of s-1 and arrival of s)
        df["observed_run_time_s"] = df["rt_arrival_time"] - df["prev_rt_departure"]

        # Running segment delay delta: change in delay from stop s-1 to stop s
        df["delta_t_run"] = df["arrival_delay_seconds"] - df["prev_stop_delay"]

        # Platform dwell delay delta
        if "departure_delay_seconds" in df.columns:
            df["delta_t_dwell"] = df["departure_delay_seconds"] - df["arrival_delay_seconds"]
        else:
            df["delta_t_dwell"] = 0

        # Calculate Headway Deviation relative to the preceding vehicle on the same route
        if "route_id" in df.columns and "rt_arrival_time" in df.columns:
            df = df.sort_values(by=["route_id", "stop_id", "rt_arrival_time"])
            df["prev_vehicle_arrival"] = df.groupby(["route_id", "stop_id"])["rt_arrival_time"].shift(1)
            df["headway_actual_s"] = df["rt_arrival_time"] - df["prev_vehicle_arrival"]
            # Rolling median headway approximation for headway deviation
            rolling_headway = df.groupby(["route_id", "stop_id"])["headway_actual_s"].transform("median")
            df["headway_deviation"] = df["headway_actual_s"] - rolling_headway

        return df

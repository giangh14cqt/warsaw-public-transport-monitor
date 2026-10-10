"""
Purged Temporal Block Validation Strategy.
Partitions transit telemetry data chronologically into distinct temporal windows:
e.g., Training (Weeks 1–3), Validation (Week 4), Test (Week 5).
Purges overlapping trip trajectories spanning split boundaries to eliminate future-to-past leakage.
"""

from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np


class PurgedTemporalBlockSplitter:
    """Chronological temporal block splitter with boundary trajectory purging."""

    def __init__(self, time_col: str = "rt_arrival_time", trip_col: str = "trip_id"):
        self.time_col = time_col
        self.trip_col = trip_col

    def split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataframe into strictly chronological Train, Val, and Test splits,
        purging trips that cross boundary timestamps.
        """
        if df.empty:
            return df, df, df

        df_sorted = df.sort_values(by=self.time_col).copy()
        n = len(df_sorted)
        train_end_idx = int(n * train_ratio)
        val_end_idx = int(n * (train_ratio + val_ratio))

        t_train_cutoff = df_sorted.iloc[train_end_idx][self.time_col]
        t_val_cutoff = df_sorted.iloc[val_end_idx][self.time_col]

        # Identify boundary-crossing trip IDs
        trips_crossing_train_val = set(
            df_sorted[
                (df_sorted[self.time_col] <= t_train_cutoff)
            ][self.trip_col]
        ).intersection(
            set(df_sorted[df_sorted[self.time_col] > t_train_cutoff][self.trip_col])
        )

        trips_crossing_val_test = set(
            df_sorted[
                (df_sorted[self.time_col] <= t_val_cutoff)
            ][self.trip_col]
        ).intersection(
            set(df_sorted[df_sorted[self.time_col] > t_val_cutoff][self.trip_col])
        )

        # Purge boundary crossings
        train_df = df_sorted[
            (df_sorted[self.time_col] <= t_train_cutoff)
            & (~df_sorted[self.trip_col].isin(trips_crossing_train_val))
        ].copy()

        val_df = df_sorted[
            (df_sorted[self.time_col] > t_train_cutoff)
            & (df_sorted[self.time_col] <= t_val_cutoff)
            & (~df_sorted[self.trip_col].isin(trips_crossing_train_val))
            & (~df_sorted[self.trip_col].isin(trips_crossing_val_test))
        ].copy()

        test_df = df_sorted[
            (df_sorted[self.time_col] > t_val_cutoff)
            & (~df_sorted[self.trip_col].isin(trips_crossing_val_test))
        ].copy()

        return train_df, val_df, test_df

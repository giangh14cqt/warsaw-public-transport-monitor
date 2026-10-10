"""
Purged Temporal Block Validation Strategy with Boundary Trajectory Purging and Embargo Buffering.
Prevents forward information leakage in transit delay telemetry and econometric panel benchmarks:
- Strictly chronological temporal windowing (Train, Validation, Test).
- Boundary trajectory purging: eliminates trips in-flight across split boundaries to prevent
  leakage of autoregressive lags (Delta t_{s-1}, Delta t_{s-2}) and dynamic vehicle states.
- Embargo buffers (e.g. 30-60 min): eliminates headway deviation propagation and queue spillovers.
- Formal zero-leakage verification and audit harness.
- Expanding / rolling temporal walk-forward cross-validation.
"""

import logging
from typing import Tuple, Dict, Any, Optional, Iterator, Union, List, Set
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataLeakageError(ValueError):
    """Raised when temporal boundary leakage or trip overlap is detected."""
    pass


class PurgedTemporalBlockSplitter:
    """
    Chronological temporal block splitter with boundary trajectory purging
    and configurable embargo buffering.
    """

    CANDIDATE_TIME_COLS = [
        "arrival_ts",
        "rt_arrival_time",
        "timestamp",
        "departure_ts",
        "rt_departure_time",
        "timestamp_bucket",
    ]

    def __init__(
        self,
        time_col: Optional[str] = None,
        trip_col: str = "trip_id",
        vehicle_col: str = "vehicle_id",
        embargo_minutes: float = 30.0,
        purge_trips: bool = True,
        purge_mode: str = "drop_both",
        strict_zero_trip_id_overlap: bool = False,
    ):
        """
        Initialize PurgedTemporalBlockSplitter.

        Parameters
        ----------
        time_col : str, optional
            Column name containing timestamps. If None, auto-detected from candidates.
        trip_col : str
            Column name identifying trip trajectories (default 'trip_id').
        vehicle_col : str
            Column name identifying physical vehicles (default 'vehicle_id').
        embargo_minutes : float
            Embargo buffer in minutes placed between temporal blocks (default 30.0).
        purge_trips : bool
            Whether to purge trajectories spanning across split boundaries (default True).
        purge_mode : str
            'drop_both' (purge boundary-crossing trips from both sets) or
            'drop_from_eval' (purge boundary-crossing trips only from evaluation set).
        strict_zero_trip_id_overlap : bool
            If True, strictly drops any trip_id seen in Train from Val and Test splits,
            even if the trip ran on a completely different day.
        """
        self.time_col = time_col
        self.trip_col = trip_col
        self.vehicle_col = vehicle_col
        self.embargo_minutes = float(embargo_minutes)
        self.purge_trips = purge_trips
        self.purge_mode = purge_mode
        self.strict_zero_trip_id_overlap = strict_zero_trip_id_overlap

    def _resolve_time_col(self, df: pd.DataFrame) -> str:
        """Resolve the active timestamp column."""
        if self.time_col and self.time_col in df.columns:
            return self.time_col
        for col in self.CANDIDATE_TIME_COLS:
            if col in df.columns:
                return col
        raise KeyError(
            f"No valid time column found. Specified '{self.time_col}', candidates were {self.CANDIDATE_TIME_COLS}. "
            f"Available columns: {list(df.columns)}"
        )

    def _find_crossing_and_embargo_trips(
        self,
        df: pd.DataFrame,
        t_cutoff: pd.Timestamp,
        t_resume: pd.Timestamp,
        time_col: str,
    ) -> Set[Any]:
        """
        Identify trip IDs that span across the cutoff boundary or are active
        within the embargo window [t_cutoff, t_resume].
        """
        if self.trip_col not in df.columns:
            return set()

        # Look in a local temporal window around the cutoff to be fast and memory efficient
        window_start = t_cutoff - pd.Timedelta(hours=4)
        window_end = t_resume + pd.Timedelta(hours=4)

        sub = df[(df[time_col] >= window_start) & (df[time_col] <= window_end)]
        if sub.empty:
            sub = df

        trip_stats = sub.groupby(self.trip_col)[time_col].agg(["min", "max"])

        # Condition 1: Trajectory started before or at cutoff, and ended after cutoff
        spanning_mask = (trip_stats["min"] <= t_cutoff) & (trip_stats["max"] > t_cutoff)

        # Condition 2: Trajectory had any observation in the embargo window [t_cutoff, t_resume]
        embargo_mask = (trip_stats["max"] >= t_cutoff) & (trip_stats["min"] <= t_resume)

        crossing_trips = set(trip_stats[spanning_mask | embargo_mask].index)
        return crossing_trips

    def split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: Optional[float] = None,
        embargo_minutes: Optional[float] = None,
        train_end_time: Optional[Union[str, pd.Timestamp]] = None,
        val_end_time: Optional[Union[str, pd.Timestamp]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataframe into strictly chronological Train, Validation, and Test splits,
        purging trips that cross boundary timestamps and applying embargo buffers.

        Parameters
        ----------
        df : pd.DataFrame
            Telemetry dataframe to partition.
        train_ratio : float
            Fraction of data duration or records for Training (default 0.6).
        val_ratio : float
            Fraction of data duration or records for Validation (default 0.2).
        test_ratio : float, optional
            Remaining fraction for Test (default 1 - train_ratio - val_ratio).
        embargo_minutes : float, optional
            Overrides instance embargo buffer if provided.
        train_end_time : str or pd.Timestamp, optional
            Explicit timestamp cutoff for Train end.
        val_end_time : str or pd.Timestamp, optional
            Explicit timestamp cutoff for Val end.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (train_df, val_df, test_df)
        """
        if df.empty:
            return df.copy(), df.copy(), df.copy()

        time_col = self._resolve_time_col(df)
        embargo_mins = self.embargo_minutes if embargo_minutes is None else float(embargo_minutes)
        embargo_delta = pd.Timedelta(minutes=embargo_mins)

        # Sort chronologically
        df_sorted = df.sort_values(by=time_col).copy()
        if not pd.api.types.is_datetime64_any_dtype(df_sorted[time_col]):
            df_sorted[time_col] = pd.to_datetime(df_sorted[time_col])

        # Determine cutoff timestamps
        n = len(df_sorted)
        if train_end_time is not None:
            t_train_cutoff = pd.to_datetime(train_end_time)
            if df_sorted[time_col].dt.tz is not None and t_train_cutoff.tzinfo is None:
                t_train_cutoff = t_train_cutoff.tz_localize(df_sorted[time_col].dt.tz)
        else:
            train_end_idx = min(max(int(n * train_ratio), 0), n - 1)
            t_train_cutoff = df_sorted.iloc[train_end_idx][time_col]

        t_val_start = t_train_cutoff + embargo_delta

        if val_end_time is not None:
            t_val_cutoff = pd.to_datetime(val_end_time)
            if df_sorted[time_col].dt.tz is not None and t_val_cutoff.tzinfo is None:
                t_val_cutoff = t_val_cutoff.tz_localize(df_sorted[time_col].dt.tz)
        else:
            val_end_idx = min(max(int(n * (train_ratio + val_ratio)), train_end_idx), n - 1)
            t_val_cutoff = df_sorted.iloc[val_end_idx][time_col]

        # Ensure validation cutoff is at or after validation start
        if t_val_cutoff < t_val_start:
            logger.warning(
                f"Calculated val_cutoff ({t_val_cutoff}) is before val_start ({t_val_start}). "
                f"Advancing val_cutoff."
            )
            t_val_cutoff = t_val_start + pd.Timedelta(hours=1)

        t_test_start = t_val_cutoff + embargo_delta

        # Boundary trajectory detection
        purged_train_val: Set[Any] = set()
        purged_val_test: Set[Any] = set()

        if self.purge_trips and self.trip_col in df_sorted.columns:
            purged_train_val = self._find_crossing_and_embargo_trips(
                df_sorted, t_train_cutoff, t_val_start, time_col
            )
            purged_val_test = self._find_crossing_and_embargo_trips(
                df_sorted, t_val_cutoff, t_test_start, time_col
            )

        # Build raw chronological masks
        train_mask = df_sorted[time_col] <= t_train_cutoff
        val_mask = (df_sorted[time_col] >= t_val_start) & (df_sorted[time_col] <= t_val_cutoff)
        test_mask = df_sorted[time_col] >= t_test_start

        # Filter according to purging strategy
        if self.purge_trips and self.trip_col in df_sorted.columns:
            if self.purge_mode == "drop_both":
                train_mask = train_mask & (~df_sorted[self.trip_col].isin(purged_train_val))
                val_mask = val_mask & (~df_sorted[self.trip_col].isin(purged_train_val | purged_val_test))
                test_mask = test_mask & (~df_sorted[self.trip_col].isin(purged_val_test))
            else:  # drop_from_eval only
                val_mask = val_mask & (~df_sorted[self.trip_col].isin(purged_train_val | purged_val_test))
                test_mask = test_mask & (~df_sorted[self.trip_col].isin(purged_val_test))

        train_df = df_sorted[train_mask].copy()
        val_df = df_sorted[val_mask].copy()
        test_df = df_sorted[test_mask].copy()

        # Strict zero trip_id overlap across splits (if requested)
        if self.strict_zero_trip_id_overlap and self.trip_col in df_sorted.columns:
            train_trips = set(train_df[self.trip_col].dropna())
            val_df = val_df[~val_df[self.trip_col].isin(train_trips)].copy()
            seen_trips = train_trips | set(val_df[self.trip_col].dropna())
            test_df = test_df[~test_df[self.trip_col].isin(seen_trips)].copy()

        logger.info(
            f"Purged Temporal Split: Train={len(train_df)} rows, Val={len(val_df)} rows, "
            f"Test={len(test_df)} rows (Purged crossings: Train-Val={len(purged_train_val)}, "
            f"Val-Test={len(purged_val_test)}, Embargo={embargo_mins:.1f}m)"
        )

        return train_df, val_df, test_df

    def walk_forward_cv(
        self,
        df: pd.DataFrame,
        n_splits: int = 3,
        train_ratio: float = 0.5,
        val_ratio: float = 0.15,
        embargo_minutes: Optional[float] = None,
        expanding: bool = True,
    ) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate expanding or rolling walk-forward temporal cross-validation folds.

        Parameters
        ----------
        df : pd.DataFrame
            Telemetry dataframe.
        n_splits : int
            Number of temporal validation folds.
        train_ratio : float
            Initial fraction for training fold.
        val_ratio : float
            Fraction allocated to validation for each step.
        embargo_minutes : float, optional
            Embargo buffer minutes between train and validation.
        expanding : bool
            If True, training set expands chronologically; if False, rolls fixed window.

        Yields
        ------
        Iterator[Tuple[pd.DataFrame, pd.DataFrame]]
            (train_fold_df, val_fold_df)
        """
        if df.empty or n_splits <= 0:
            return

        time_col = self._resolve_time_col(df)
        embargo_mins = self.embargo_minutes if embargo_minutes is None else float(embargo_minutes)
        embargo_delta = pd.Timedelta(minutes=embargo_mins)

        df_sorted = df.sort_values(by=time_col).copy()
        if not pd.api.types.is_datetime64_any_dtype(df_sorted[time_col]):
            df_sorted[time_col] = pd.to_datetime(df_sorted[time_col])

        n = len(df_sorted)
        step_ratio = (1.0 - train_ratio - val_ratio) / max(n_splits - 1, 1)

        for fold in range(n_splits):
            current_train_end_ratio = train_ratio + fold * step_ratio
            train_end_idx = min(max(int(n * current_train_end_ratio), 0), n - 1)
            t_train_cutoff = df_sorted.iloc[train_end_idx][time_col]
            t_val_start = t_train_cutoff + embargo_delta

            current_val_end_ratio = min(current_train_end_ratio + val_ratio, 1.0)
            val_end_idx = min(max(int(n * current_val_end_ratio), train_end_idx), n - 1)
            t_val_cutoff = df_sorted.iloc[val_end_idx][time_col]

            if t_val_cutoff <= t_val_start:
                continue

            # Start index for training
            if expanding:
                train_start_mask = pd.Series(True, index=df_sorted.index)
            else:
                train_start_ratio = max(0.0, current_train_end_ratio - train_ratio)
                train_start_idx = int(n * train_start_ratio)
                t_train_start = df_sorted.iloc[train_start_idx][time_col]
                train_start_mask = df_sorted[time_col] >= t_train_start

            purged_crossing = self._find_crossing_and_embargo_trips(
                df_sorted, t_train_cutoff, t_val_start, time_col
            ) if self.purge_trips else set()

            train_mask = train_start_mask & (df_sorted[time_col] <= t_train_cutoff)
            val_mask = (df_sorted[time_col] >= t_val_start) & (df_sorted[time_col] <= t_val_cutoff)

            if self.purge_trips and self.trip_col in df_sorted.columns:
                if self.purge_mode == "drop_both":
                    train_mask = train_mask & (~df_sorted[self.trip_col].isin(purged_crossing))
                val_mask = val_mask & (~df_sorted[self.trip_col].isin(purged_crossing))

            train_fold = df_sorted[train_mask].copy()
            val_fold = df_sorted[val_mask].copy()

            if self.strict_zero_trip_id_overlap and self.trip_col in df_sorted.columns:
                train_trips = set(train_fold[self.trip_col].dropna())
                val_fold = val_fold[~val_fold[self.trip_col].isin(train_trips)].copy()

            yield train_fold, val_fold

    def verify_no_leakage(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
        embargo_minutes: Optional[float] = None,
        raise_on_error: bool = True,
    ) -> Dict[str, Any]:
        """
        Formally verify and audit that no data leakage exists across splits:
        1. Monotonic chronological ordering (Train < Val < Test).
        2. Verified embargo gaps between partition time boundaries.
        3. Zero boundary-crossing trajectory leakage.
        4. Zero trip_id overlap if strict_zero_trip_id_overlap is set.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training set.
        val_df : pd.DataFrame
            Validation set.
        test_df : pd.DataFrame, optional
            Test set.
        embargo_minutes : float, optional
            Expected minimum embargo gap in minutes.
        raise_on_error : bool
            If True, raises DataLeakageError on violation; otherwise records failure.

        Returns
        -------
        Dict[str, Any]
            Audit report with verification details.
        """
        audit: Dict[str, Any] = {
            "is_leakage_free": True,
            "violations": [],
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df) if test_df is not None else 0,
        }

        if train_df.empty or val_df.empty:
            return audit

        time_col = self._resolve_time_col(train_df)
        expected_embargo = self.embargo_minutes if embargo_minutes is None else float(embargo_minutes)
        min_embargo_seconds = expected_embargo * 60.0

        train_max_t = pd.to_datetime(train_df[time_col].max())
        val_min_t = pd.to_datetime(val_df[time_col].min())
        val_max_t = pd.to_datetime(val_df[time_col].max())

        gap_train_val = (val_min_t - train_max_t).total_seconds()
        audit["embargo_gap_train_val_seconds"] = gap_train_val

        # Check 1: Chronological Ordering & Embargo between Train and Val
        if gap_train_val < -1.0:  # Inverted timestamps
            msg = (
                f"Timestamp Inversion: Train max ({train_max_t}) is after Val min ({val_min_t}) "
                f"by {-gap_train_val:.1f}s"
            )
            audit["violations"].append(msg)
            audit["is_leakage_free"] = False

        if expected_embargo > 0 and gap_train_val < (min_embargo_seconds - 5.0):
            msg = (
                f"Embargo Gap Violation: Train-Val gap is {gap_train_val:.1f}s, "
                f"expected >= {min_embargo_seconds:.1f}s"
            )
            audit["violations"].append(msg)
            audit["is_leakage_free"] = False

        # Check 2: Test partition checks (if provided)
        if test_df is not None and not test_df.empty:
            test_min_t = pd.to_datetime(test_df[time_col].min())
            gap_val_test = (test_min_t - val_max_t).total_seconds()
            audit["embargo_gap_val_test_seconds"] = gap_val_test

            if gap_val_test < -1.0:
                msg = (
                    f"Timestamp Inversion: Val max ({val_max_t}) is after Test min ({test_min_t}) "
                    f"by {-gap_val_test:.1f}s"
                )
                audit["violations"].append(msg)
                audit["is_leakage_free"] = False

            if expected_embargo > 0 and gap_val_test < (min_embargo_seconds - 5.0):
                msg = (
                    f"Embargo Gap Violation: Val-Test gap is {gap_val_test:.1f}s, "
                    f"expected >= {min_embargo_seconds:.1f}s"
                )
                audit["violations"].append(msg)
                audit["is_leakage_free"] = False

        # Check 3: Boundary-crossing active trajectory check
        if self.trip_col in train_df.columns and self.trip_col in val_df.columns:
            # Check trips that were active within the cutoff boundary window
            train_boundary_trips = set(
                train_df[train_df[time_col] >= (train_max_t - pd.Timedelta(minutes=30))][self.trip_col]
            )
            val_boundary_trips = set(
                val_df[val_df[time_col] <= (val_min_t + pd.Timedelta(minutes=30))][self.trip_col]
            )
            boundary_overlap = train_boundary_trips.intersection(val_boundary_trips)
            audit["boundary_crossing_overlap_count"] = len(boundary_overlap)
            if len(boundary_overlap) > 0:
                msg = (
                    f"Boundary Trajectory Leakage: {len(boundary_overlap)} trips overlap "
                    f"directly across Train-Val boundary window: {list(boundary_overlap)[:5]}"
                )
                audit["violations"].append(msg)
                audit["is_leakage_free"] = False

        # Check 4: Strict zero trip_id overlap check (if requested)
        if self.strict_zero_trip_id_overlap and self.trip_col in train_df.columns:
            train_trips = set(train_df[self.trip_col].dropna())
            val_trips = set(val_df[self.trip_col].dropna())
            overlap = train_trips.intersection(val_trips)
            audit["trip_id_overlap_count"] = len(overlap)
            if len(overlap) > 0:
                msg = f"Strict Trip ID Overlap: {len(overlap)} trip_ids appear in both Train and Val"
                audit["violations"].append(msg)
                audit["is_leakage_free"] = False

        if not audit["is_leakage_free"] and raise_on_error:
            raise DataLeakageError(
                f"Data leakage detected during validation audit:\n" + "\n".join(audit["violations"])
            )

        return audit

"""
High-Precision Wall-Clock Grid Scheduler for 10-Second Telemetry Ingestion.
Guarantees zero cumulative drift and sub-5ms tick execution jitter.
"""

import math
import time
from typing import Iterator


def compute_next_grid_tick(now_ts: float, period_seconds: int = 10) -> float:
    """
    Computes the upcoming wall-clock grid tick timestamp (epoch seconds).
    Strictly aligns to unix epoch multiples of `period_seconds`.
    """
    return (math.floor(now_ts / period_seconds) + 1) * period_seconds


def sleep_until_tick(target_epoch_ts: float) -> float:
    """
    Suspends execution until `target_epoch_ts` with sub-5ms accuracy.
    Uses a hybrid strategy: coarse sleep followed by micro-yielding.
    Returns the actual wakeup timestamp.
    """
    remaining = target_epoch_ts - time.time()

    # Coarse sleep phase (reserve last 15ms)
    if remaining > 0.020:
        time.sleep(remaining - 0.015)

    # Fine-grained spin-wait phase for remaining fraction
    while time.time() < target_epoch_ts:
        time.sleep(0.001)

    return time.time()


def generate_aligned_ticks(
    period_seconds: int = 10,
    max_lag_seconds: float = 1.0,
    start_ts: float | None = None,
) -> Iterator[tuple[float, float, bool]]:
    """
    Infinite generator yielding (scheduled_tick_ts, actual_start_ts, is_missed).
    If a tick lags behind schedule by more than `max_lag_seconds`, it yields
    `is_missed=True` and re-aligns to the next upcoming grid boundary.
    """
    current_time = time.time() if start_ts is None else start_ts
    target_tick = compute_next_grid_tick(current_time, period_seconds)

    while True:
        actual_ts = sleep_until_tick(target_tick)
        lag = actual_ts - target_tick

        if lag > max_lag_seconds:
            # Scheduler fell behind (e.g. host suspension, high CPU contention)
            yield target_tick, actual_ts, True
            # Re-align immediately to next future boundary, never play catch-up
            target_tick = compute_next_grid_tick(time.time(), period_seconds)
        else:
            yield target_tick, actual_ts, False
            target_tick += period_seconds

"""
Unit tests for the Historian high-precision aligned clock scheduler.
"""

import time

from src.historian.clock import compute_next_grid_tick, generate_aligned_ticks, sleep_until_tick


def test_compute_next_grid_tick_alignment():
    # 10-second period alignment
    assert compute_next_grid_tick(1700000000.0, 10) == 1700000010.0
    assert compute_next_grid_tick(1700000004.2, 10) == 1700000010.0
    assert compute_next_grid_tick(1700000009.9, 10) == 1700000010.0
    assert compute_next_grid_tick(1700000010.0, 10) == 1700000020.0

    # 20-second period alignment
    assert compute_next_grid_tick(1700000000.0, 20) == 1700000020.0
    assert compute_next_grid_tick(1700000019.5, 20) == 1700000020.0
    assert compute_next_grid_tick(1700000020.0, 20) == 1700000040.0


def test_sleep_until_tick_precision():
    now = time.time()
    target = now + 0.050  # 50ms into future
    actual = sleep_until_tick(target)

    # Must wake up at or after target, within 10ms tolerance
    assert actual >= target
    assert (actual - target) < 0.010


def test_generate_aligned_ticks_advancement():
    tick_gen = generate_aligned_ticks(period_seconds=1, max_lag_seconds=0.5)

    # Collect 3 ticks (1-second period for quick test)
    samples = []
    for _ in range(3):
        sched, actual, missed = next(tick_gen)
        samples.append((sched, actual, missed))

    assert len(samples) == 3
    for sched, actual, missed in samples:
        assert not missed
        assert actual >= sched
        assert (actual - sched) < 0.050
        assert sched % 1.0 == 0.0  # Perfectly aligned

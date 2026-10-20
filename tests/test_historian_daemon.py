"""
Unit tests for Warsaw Historian Ingestion Daemon error recovery and timeout resilience.
"""

import time
from unittest.mock import MagicMock, patch

from historian_daemon import WarsawHistorianDaemon
from src.historian.config import config


def test_daemon_timeout_graceful_handling(tmp_path):
    """
    Simulates a slow worker future that exceeds request_deadline_seconds.
    Verifies that execute_tick does not raise TimeoutError, marks the tick as TIMEOUT,
    and finishes cleanly.
    """
    daemon = WarsawHistorianDaemon(max_ticks=1)
    daemon.writer = MagicMock()
    daemon.writer.write_payload.return_value = ("mock_path", "mock_sha")

    # Mock city_client to hang for longer than deadline
    def slow_fetch(*args, **kwargs):
        time.sleep(0.5)
        return {"status": "OK", "records": [], "raw_bytes": b"[]"}

    daemon.city_client.fetch_vehicles = MagicMock(side_effect=slow_fetch)
    daemon.zbk_client.fetch_feed = MagicMock(return_value={"status": "OK", "n_trip_updates": 10})

    # Set deadline to very small value (50ms) to trigger timeout deterministically
    import dataclasses
    mock_config = dataclasses.replace(config, request_deadline_seconds=0.05)
    with patch("historian_daemon.config", mock_config):
        # 1729410000 is divisible by 30, so zbk is polled as well
        daemon.execute_tick(scheduled_ts=1729410000.0, actual_ts=1729410000.01, is_missed=False)

    # Daemon should record TIMEOUT for unfinished futures without crashing
    recorded_calls = daemon.writer.record_tick.call_args_list
    assert len(recorded_calls) >= 1
    timeout_records = [
        call.kwargs
        for call in recorded_calls
        if call.kwargs.get("status") == "TIMEOUT"
    ]
    assert len(timeout_records) > 0
    assert daemon.successful_ticks == 1


def test_daemon_in_flight_overlap_guard(tmp_path):
    """
    Verifies that if in_flight is True for a source, execute_tick records SKIPPED_OVERLAP
    and does not submit a duplicate future.
    """
    daemon = WarsawHistorianDaemon(max_ticks=1)
    daemon.writer = MagicMock()
    daemon.in_flight["um_bus"] = True

    daemon.execute_tick(scheduled_ts=1729410010.0, actual_ts=1729410010.01, is_missed=False)

    recorded_calls = daemon.writer.record_tick.call_args_list
    skipped = [
        call.kwargs
        for call in recorded_calls
        if call.kwargs.get("status") == "SKIPPED_OVERLAP" and call.kwargs.get("source") == "um_bus"
    ]
    assert len(skipped) == 1


def test_daemon_run_exception_shield(tmp_path):
    """
    Verifies that an unhandled exception inside execute_tick is caught in run()
    and does not terminate the process.
    """
    daemon = WarsawHistorianDaemon(max_ticks=2)
    daemon.writer = MagicMock()
    daemon.gtfs_versioner.sync_daily_gtfs = MagicMock()
    daemon._send_heartbeat = MagicMock()

    # Fail on first tick, succeed on second tick
    call_count = 0

    def mock_execute(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Simulated network/hardware fault")

    daemon.execute_tick = MagicMock(side_effect=mock_execute)

    # Mock generator to return 2 ticks immediately
    def mock_generator():
        yield 1729410000.0, 1729410000.0, False
        yield 1729410010.0, 1729410010.0, False

    with patch("historian_daemon.generate_aligned_ticks", return_value=mock_generator()):
        daemon.run()

    # Both ticks were reached despite tick 1 raising an exception
    assert call_count == 2
    # Check that error was recorded to audit writer
    recorded_calls = daemon.writer.record_tick.call_args_list
    exception_records = [
        call.kwargs
        for call in recorded_calls
        if call.kwargs.get("status") == "EXCEPTION"
    ]
    assert len(exception_records) == 1
    assert "Simulated network/hardware fault" in exception_records[0].get("error_message", "")

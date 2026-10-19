"""
Unit tests for Historian Gold stop event reconstruction and delay attribution.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.historian.gold import GoldStopEventReconstructor


def test_parse_hhmmss_to_utc():
    reconstructor = GoldStopEventReconstructor()
    # 08:30:00 on 2026-10-18 (Warsaw CEST is UTC+2) -> 06:30:00 UTC
    utc_epoch = reconstructor._parse_hhmmss_to_utc("2026-10-18", "08:30:00")
    # Verify matches datetime
    import datetime
    from zoneinfo import ZoneInfo
    expected = int(
        datetime.datetime(2026, 10, 18, 8, 30, 0, tzinfo=ZoneInfo("Europe/Warsaw"))
        .astimezone(datetime.timezone.utc)
        .timestamp()
    )
    assert utc_epoch == expected


def test_gold_stop_event_reconstruction(tmp_path: Path):
    silver_dir = tmp_path / "silver"
    gold_dir = tmp_path / "gold"
    silver_date_dir = silver_dir / "date=2026-10-18"
    silver_date_dir.mkdir(parents=True)

    # Synthetic vehicle approaching stop at (52.2300, 21.0000)
    # Stop is located at (52.2300, 21.0000)
    # Fix 1 at t=100s: 35m West (52.2300, 20.999488) -> outside 25m
    # Fix 2 at t=110s: 10m West (52.2300, 20.999853) -> inside 25m
    # Fix 3 at t=125s: 10m West (52.2300, 20.999853) -> still inside (dwelling)
    # Fix 4 at t=135s: 35m East (52.2300, 21.000512) -> outside 25m
    fixes = [
        {
            "service_date": "2026-10-18",
            "vehicle_type": "bus",
            "vehicle_number": "1001",
            "route_id": "100",
            "brigade": "1",
            "lat": 52.2300,
            "lon": 20.999488,
            "gps_timestamp_utc": 100,
            "gps_time_str": "2026-10-18 08:01:40",
            "tick_ts": 100,
            "delta_t_s": 10.0,
            "delta_d_m": 10.0,
            "speed_mps": 1.0,
            "is_static": False,
            "n_ticks_seen": 1,
            "matched_trip_id": "TRIP_1",
        },
        {
            "service_date": "2026-10-18",
            "vehicle_type": "bus",
            "vehicle_number": "1001",
            "route_id": "100",
            "brigade": "1",
            "lat": 52.2300,
            "lon": 20.999853,
            "gps_timestamp_utc": 110,
            "gps_time_str": "2026-10-18 08:01:50",
            "tick_ts": 110,
            "delta_t_s": 10.0,
            "delta_d_m": 25.0,
            "speed_mps": 2.5,
            "is_static": False,
            "n_ticks_seen": 1,
            "matched_trip_id": "TRIP_1",
        },
        {
            "service_date": "2026-10-18",
            "vehicle_type": "bus",
            "vehicle_number": "1001",
            "route_id": "100",
            "brigade": "1",
            "lat": 52.2300,
            "lon": 20.999853,
            "gps_timestamp_utc": 125,
            "gps_time_str": "2026-10-18 08:02:05",
            "tick_ts": 125,
            "delta_t_s": 15.0,
            "delta_d_m": 0.0,
            "speed_mps": 0.0,
            "is_static": True,
            "n_ticks_seen": 1,
            "matched_trip_id": "TRIP_1",
        },
        {
            "service_date": "2026-10-18",
            "vehicle_type": "bus",
            "vehicle_number": "1001",
            "route_id": "100",
            "brigade": "1",
            "lat": 52.2300,
            "lon": 21.000512,
            "gps_timestamp_utc": 135,
            "gps_time_str": "2026-10-18 08:02:15",
            "tick_ts": 135,
            "delta_t_s": 10.0,
            "delta_d_m": 45.0,
            "speed_mps": 4.5,
            "is_static": False,
            "n_ticks_seen": 1,
            "matched_trip_id": "TRIP_1",
        },
    ]

    silver_df = pd.DataFrame(fixes)
    silver_df.to_parquet(silver_date_dir / "trajectories.parquet")

    reconstructor = GoldStopEventReconstructor(
        silver_dir=silver_dir,
        gold_dir=gold_dir,
        stop_radius_meters=25.0,
    )

    # Mock DuckDB query for stop_times and stops
    mock_trip_stops = pd.DataFrame(
        [
            {
                "trip_id": "TRIP_1",
                "stop_sequence": 1,
                "stop_id": "STOP_A",
                "arrival_time": "08:01:45",
                "departure_time": "08:02:00",
                "stop_name": "Test Station",
                "stop_lat": 52.2300,
                "stop_lon": 21.0000,
            }
        ]
    )

    with patch.object(reconstructor, "_get_active_duckdb") as mock_db, \
         patch("duckdb.connect") as mock_conn:
        mock_db.return_value = "dummy.duckdb"
        mock_ctx = MagicMock()
        mock_ctx.execute.return_value.df.return_value = mock_trip_stops
        mock_conn.return_value = mock_ctx

        out_file = reconstructor.process_date("2026-10-18")

    assert out_file is not None
    assert out_file.exists()

    gold_df = pd.read_parquet(out_file)
    assert len(gold_df) == 1
    row = gold_df.iloc[0]

    assert row["event_type"] == "STOPPED"
    assert row["confidence"] == "HIGH"
    assert 100 < row["actual_arrival_utc"] < 110
    assert 125 <= row["actual_departure_utc"] <= 135
    assert row["dwell_seconds"] > 15


def test_gold_complete_trip_filtering(tmp_path: Path):
    """Verifies that incomplete trips (missing starting stop or terminal stop) are filtered out."""
    silver_dir = tmp_path / "silver"
    gold_dir = tmp_path / "gold"
    silver_date_dir = silver_dir / "date=2026-10-18"
    silver_date_dir.mkdir(parents=True)

    # Synthetic vehicle fixes:
    # TRIP_COMPLETE: visits stop 0 at t=100, stop 1 at t=200, stop 2 at t=300
    # TRIP_INCOMPLETE: visits stop 1 at t=200 (misses stop 0 and stop 2)
    fixes = [
        # TRIP_COMPLETE fixes
        {
            "service_date": "2026-10-18",
            "vehicle_type": "bus",
            "vehicle_number": "1001",
            "route_id": "100",
            "brigade": "1",
            "lat": 52.2000,
            "lon": 21.0000,
            "gps_timestamp_utc": 100,
            "gps_time_str": "2026-10-18 08:01:40",
            "tick_ts": 100,
            "delta_t_s": 10.0,
            "delta_d_m": 0.0,
            "speed_mps": 0.0,
            "is_static": True,
            "n_ticks_seen": 1,
            "matched_trip_id": "TRIP_COMPLETE",
        },
        {
            "service_date": "2026-10-18",
            "vehicle_type": "bus",
            "vehicle_number": "1001",
            "route_id": "100",
            "brigade": "1",
            "lat": 52.2100,
            "lon": 21.0100,
            "gps_timestamp_utc": 200,
            "gps_time_str": "2026-10-18 08:03:20",
            "tick_ts": 200,
            "delta_t_s": 10.0,
            "delta_d_m": 0.0,
            "speed_mps": 0.0,
            "is_static": True,
            "n_ticks_seen": 1,
            "matched_trip_id": "TRIP_COMPLETE",
        },
        {
            "service_date": "2026-10-18",
            "vehicle_type": "bus",
            "vehicle_number": "1001",
            "route_id": "100",
            "brigade": "1",
            "lat": 52.2200,
            "lon": 21.0200,
            "gps_timestamp_utc": 300,
            "gps_time_str": "2026-10-18 08:05:00",
            "tick_ts": 300,
            "delta_t_s": 10.0,
            "delta_d_m": 0.0,
            "speed_mps": 0.0,
            "is_static": True,
            "n_ticks_seen": 1,
            "matched_trip_id": "TRIP_COMPLETE",
        },
        # TRIP_INCOMPLETE fixes (only stop 1)
        {
            "service_date": "2026-10-18",
            "vehicle_type": "bus",
            "vehicle_number": "1002",
            "route_id": "100",
            "brigade": "2",
            "lat": 52.2100,
            "lon": 21.0100,
            "gps_timestamp_utc": 200,
            "gps_time_str": "2026-10-18 08:03:20",
            "tick_ts": 200,
            "delta_t_s": 10.0,
            "delta_d_m": 0.0,
            "speed_mps": 0.0,
            "is_static": True,
            "n_ticks_seen": 1,
            "matched_trip_id": "TRIP_INCOMPLETE",
        },
    ]

    silver_df = pd.DataFrame(fixes)
    silver_df.to_parquet(silver_date_dir / "trajectories.parquet")

    mock_trip_stops = pd.DataFrame(
        [
            # TRIP_COMPLETE stops: 0, 1, 2
            {
                "trip_id": "TRIP_COMPLETE",
                "stop_sequence": 0,
                "stop_id": "STOP_0",
                "arrival_time": "08:01:40",
                "departure_time": "08:01:40",
                "stop_name": "Start Station",
                "stop_lat": 52.2000,
                "stop_lon": 21.0000,
            },
            {
                "trip_id": "TRIP_COMPLETE",
                "stop_sequence": 1,
                "stop_id": "STOP_1",
                "arrival_time": "08:03:20",
                "departure_time": "08:03:20",
                "stop_name": "Mid Station",
                "stop_lat": 52.2100,
                "stop_lon": 21.0100,
            },
            {
                "trip_id": "TRIP_COMPLETE",
                "stop_sequence": 2,
                "stop_id": "STOP_2",
                "arrival_time": "08:05:00",
                "departure_time": "08:05:00",
                "stop_name": "End Station",
                "stop_lat": 52.2200,
                "stop_lon": 21.0200,
            },
            # TRIP_INCOMPLETE stops: 0, 1, 2
            {
                "trip_id": "TRIP_INCOMPLETE",
                "stop_sequence": 0,
                "stop_id": "STOP_0",
                "arrival_time": "08:01:40",
                "departure_time": "08:01:40",
                "stop_name": "Start Station",
                "stop_lat": 52.2000,
                "stop_lon": 21.0000,
            },
            {
                "trip_id": "TRIP_INCOMPLETE",
                "stop_sequence": 1,
                "stop_id": "STOP_1",
                "arrival_time": "08:03:20",
                "departure_time": "08:03:20",
                "stop_name": "Mid Station",
                "stop_lat": 52.2100,
                "stop_lon": 21.0100,
            },
            {
                "trip_id": "TRIP_INCOMPLETE",
                "stop_sequence": 2,
                "stop_id": "STOP_2",
                "arrival_time": "08:05:00",
                "departure_time": "08:05:00",
                "stop_name": "End Station",
                "stop_lat": 52.2200,
                "stop_lon": 21.0200,
            },
        ]
    )

    # Case 1: require_complete_trips=True (default) -> TRIP_INCOMPLETE dropped
    rec_strict = GoldStopEventReconstructor(
        silver_dir=silver_dir,
        gold_dir=gold_dir / "strict",
        stop_radius_meters=25.0,
        require_complete_trips=True,
    )
    with patch.object(rec_strict, "_get_active_duckdb") as mock_db, \
         patch("duckdb.connect") as mock_conn:
        mock_db.return_value = "dummy.duckdb"
        mock_ctx = MagicMock()
        mock_ctx.execute.return_value.df.return_value = mock_trip_stops
        mock_conn.return_value = mock_ctx

        out_strict = rec_strict.process_date("2026-10-18")

    assert out_strict is not None
    df_strict = pd.read_parquet(out_strict)
    unique_trips_strict = set(df_strict["trip_id"].unique())
    assert unique_trips_strict == {"TRIP_COMPLETE"}
    assert "TRIP_INCOMPLETE" not in unique_trips_strict

    # Case 2: require_complete_trips=False -> both trips kept
    rec_relaxed = GoldStopEventReconstructor(
        silver_dir=silver_dir,
        gold_dir=gold_dir / "relaxed",
        stop_radius_meters=25.0,
        require_complete_trips=False,
    )
    with patch.object(rec_relaxed, "_get_active_duckdb") as mock_db, \
         patch("duckdb.connect") as mock_conn:
        mock_db.return_value = "dummy.duckdb"
        mock_ctx = MagicMock()
        mock_ctx.execute.return_value.df.return_value = mock_trip_stops
        mock_conn.return_value = mock_ctx

        out_relaxed = rec_relaxed.process_date("2026-10-18")

    assert out_relaxed is not None
    df_relaxed = pd.read_parquet(out_relaxed)
    unique_trips_relaxed = set(df_relaxed["trip_id"].unique())
    assert unique_trips_relaxed == {"TRIP_COMPLETE", "TRIP_INCOMPLETE"}

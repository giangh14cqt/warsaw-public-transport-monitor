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

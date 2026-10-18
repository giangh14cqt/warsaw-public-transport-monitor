"""
Unit tests for Historian Silver trajectory processor.
"""



from src.historian.silver import SilverTrajectoryProcessor


def test_parse_warsaw_time_to_utc():
    processor = SilverTrajectoryProcessor()
    # Warsaw in October is UTC+2 (CEST)
    time_str = "2026-10-18 14:00:00"
    epoch = processor.parse_warsaw_time_to_utc(time_str, tick_ts=1792330000)

    # 14:00 Warsaw CEST = 12:00 UTC
    assert epoch == 1792324800


def test_deduplicate_and_compute_kinematics():
    processor = SilverTrajectoryProcessor()

    # Synthetic vehicle moving 100 meters in 10 seconds (10 m/s)
    raw_fixes = [
        {
            "service_date": "2026-10-18",
            "vehicle_type": "bus",
            "vehicle_number": "9999",
            "route_id": "100",
            "brigade": "1",
            "lat": 52.2300,
            "lon": 21.0000,
            "gps_timestamp_utc": 100,
            "gps_time_str": "2026-10-18 14:00:00",
            "tick_ts": 100,
        },
        # Repeated identical fix in next tick (stuttering GPS)
        {
            "service_date": "2026-10-18",
            "vehicle_type": "bus",
            "vehicle_number": "9999",
            "route_id": "100",
            "brigade": "1",
            "lat": 52.2300,
            "lon": 21.0000,
            "gps_timestamp_utc": 100,
            "gps_time_str": "2026-10-18 14:00:00",
            "tick_ts": 110,
        },
        # Moving fix at t=120s: ~136 meters East (0.002 deg lon at 52.23N is ~136.6m)
        {
            "service_date": "2026-10-18",
            "vehicle_type": "bus",
            "vehicle_number": "9999",
            "route_id": "100",
            "brigade": "1",
            "lat": 52.2300,
            "lon": 21.0020,
            "gps_timestamp_utc": 120,
            "gps_time_str": "2026-10-18 14:00:20",
            "tick_ts": 120,
        },
    ]

    df = processor.deduplicate_and_compute_kinematics(raw_fixes)

    # Check deduplication: 3 raw pings should collapse into 2 distinct trajectory points
    assert len(df) == 2
    assert df.iloc[0]["n_ticks_seen"] == 2
    assert df.iloc[0]["speed_mps"] == 0.0

    # Check second point kinematics
    assert df.iloc[1]["delta_t_s"] == 20.0
    assert df.iloc[1]["delta_d_m"] > 100.0
    assert 5.0 < df.iloc[1]["speed_mps"] < 10.0
    assert not df.iloc[1]["is_static"]

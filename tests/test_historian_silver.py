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


def test_match_trips_using_gtfs_calendar_awareness(tmp_path):
    from pathlib import Path
    import pandas as pd
    from src.historian.static_gtfs import HistorianGTFSVersioner

    gtfs_dir = tmp_path / "gtfs"
    snapshot_dir = gtfs_dir / "snap1"
    snapshot_dir.mkdir(parents=True)

    # Weekday trip at 12:24 vs Weekend trip at 12:27 for same route 503 brigade 7
    (snapshot_dir / "trips.txt").write_text(
        "trip_id,route_id,brigade,service_id,shape_id,direction_id\n"
        "TRIP_WEEKDAY,503,7,SERV_WEEKDAY,SH1,0\n"
        "TRIP_WEEKEND,503,7,SERV_WEEKEND,SH1,0\n"
    )
    (snapshot_dir / "stop_times.txt").write_text(
        "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
        "TRIP_WEEKDAY,STOP_1,1,12:24:00,12:24:00\n"
        "TRIP_WEEKDAY,STOP_2,2,13:10:00,13:10:00\n"
        "TRIP_WEEKEND,STOP_1,1,12:27:00,12:27:00\n"
        "TRIP_WEEKEND,STOP_2,2,13:12:00,13:12:00\n"
    )
    (snapshot_dir / "calendar_dates.txt").write_text(
        "service_id,date,exception_type\n"
        "SERV_WEEKDAY,20261019,1\n"
        "SERV_WEEKEND,20261018,1\n"
    )
    (snapshot_dir / "feed_info.txt").write_text(
        "feed_publisher_name,feed_publisher_url,feed_contact_email,feed_lang,feed_start_date,feed_end_date,feed_version\n"
        "Test,http://test.pl,test@test.pl,pl,20261018,20261025,1\n"
    )

    versioner = HistorianGTFSVersioner(gtfs_dir=gtfs_dir)
    versioner._build_duckdb_index(snapshot_dir)

    processor = SilverTrajectoryProcessor(silver_dir=tmp_path / "silver")
    processor.gtfs_versioner = versioner

    # Fix on Monday 2026-10-19 at 12:24:24
    monday_df = pd.DataFrame([{
        "route_id": "503",
        "brigade": "7",
        "gps_time_str": "2026-10-19 12:24:24",
    }])
    monday_matched = processor.match_trips_using_gtfs(monday_df, "2026-10-19")
    assert monday_matched.iloc[0]["matched_trip_id"] == "TRIP_WEEKDAY"

    # Fix on Sunday 2026-10-18 at 12:27:10
    sunday_df = pd.DataFrame([{
        "route_id": "503",
        "brigade": "7",
        "gps_time_str": "2026-10-18 12:27:10",
    }])
    sunday_matched = processor.match_trips_using_gtfs(sunday_df, "2026-10-18")
    assert sunday_matched.iloc[0]["matched_trip_id"] == "TRIP_WEEKEND"


"""
Unit tests for Historian Bronze Compaction and In-Memory Archive Readers.
"""

import gzip
import json
from pathlib import Path
from typing import List

import pytest

from src.historian.bronze import iter_bronze_ticks
from src.historian.compact import HourlyBronzeCompactor, compact_hour_directory, open_tar_zst_reader


@pytest.fixture
def temp_bronze_dir(tmp_path: Path) -> Path:
    bronze = tmp_path / "bronze"
    bronze.mkdir(parents=True, exist_ok=True)
    return bronze


def test_compact_hour_and_read_tar_zst(temp_bronze_dir: Path):
    source = "um_bus"
    date_str = "2026-10-18"
    hour_str = "10"

    hour_dir = temp_bronze_dir / f"source={source}" / f"date={date_str}" / f"hour={hour_str}"
    hour_dir.mkdir(parents=True, exist_ok=True)

    # Create 5 synthetic tick files
    created_payloads = {}
    for i in range(5):
        ts = 1792300000 + i * 10
        payload = [{"VehicleNumber": f"100{i}", "Lat": 52.23, "Lon": 21.01, "Time": "10:00:00"}]
        raw_json = json.dumps(payload).encode("utf-8")
        created_payloads[ts] = raw_json

        gz_file = hour_dir / f"tick_{ts}.json.gz"
        with gzip.open(gz_file, "wb") as f:
            f.write(raw_json)

    # Verify loose files exist
    assert len(list(hour_dir.glob("tick_*.gz"))) == 5

    # Run compaction
    archive_path = hour_dir.parent / f"hour={hour_str}.tar.zst"
    res = compact_hour_directory(
        hour_dir=hour_dir,
        output_archive=archive_path,
        compression_level=3,
        delete_source=True,
    )

    assert res["status"] == "COMPACTED"
    assert res["file_count"] == 5
    assert archive_path.exists()
    assert archive_path.stat().st_size > 0
    # Loose files and directory should be removed
    assert not hour_dir.exists()

    # Stream read using open_tar_zst_reader
    read_items = list(open_tar_zst_reader(archive_path))
    assert len(read_items) == 5

    for ts, raw_bytes, name in read_items:
        assert ts in created_payloads
        assert raw_bytes == created_payloads[ts]
        assert name == f"tick_{ts}.json"


def test_iter_bronze_ticks_dual_support(temp_bronze_dir: Path):
    source = "um_tram"
    date_str = "2026-10-19"

    # Hour 01: compacted archive
    h1_dir = temp_bronze_dir / f"source={source}" / f"date={date_str}" / "hour=01"
    h1_dir.mkdir(parents=True, exist_ok=True)
    ts1 = 1792390000
    p1 = b'{"line": "4"}'
    with gzip.open(h1_dir / f"tick_{ts1}.json.gz", "wb") as f:
        f.write(p1)

    archive_01 = h1_dir.parent / "hour=01.tar.zst"
    compact_hour_directory(h1_dir, archive_01, compression_level=3, delete_source=True)

    # Hour 02: uncompacted loose files
    h2_dir = temp_bronze_dir / f"source={source}" / f"date={date_str}" / "hour=02"
    h2_dir.mkdir(parents=True, exist_ok=True)
    ts2 = 1792393600
    p2 = b'{"line": "33"}'
    with gzip.open(h2_dir / f"tick_{ts2}.json.gz", "wb") as f:
        f.write(p2)

    # Test iter_bronze_ticks transparently reading both hours
    ticks = list(iter_bronze_ticks(temp_bronze_dir, source, date_str))
    assert len(ticks) == 2

    assert ticks[0][0] == ts1
    assert ticks[0][1] == p1
    assert ticks[1][0] == ts2
    assert ticks[1][1] == p2


def test_compactor_skips_current_hour(temp_bronze_dir: Path):
    import datetime

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    today_str = now_utc.strftime("%Y-%m-%d")
    current_hour_str = f"{now_utc.hour:02d}"

    hour_dir = temp_bronze_dir / "source=um_bus" / f"date={today_str}" / f"hour={current_hour_str}"
    hour_dir.mkdir(parents=True, exist_ok=True)
    gz_file = hour_dir / "tick_9999999999.json.gz"
    with gzip.open(gz_file, "wb") as f:
        f.write(b"[]")

    compactor = HourlyBronzeCompactor(bronze_dir=temp_bronze_dir)
    res = compactor.compact_date(date_str=today_str, only_past_hours=True)

    # Current hour must be skipped
    assert len(res) == 0
    assert hour_dir.exists()
    assert gz_file.exists()


def _make_zbk_pb_bytes(trip_id: str, stop_seq: int, delay_s: int) -> bytes:
    from google.transit import gtfs_realtime_pb2

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    feed.header.timestamp = 1792300000
    ent = feed.entity.add()
    ent.id = "1"
    ent.trip_update.trip.trip_id = trip_id
    stu = ent.trip_update.stop_time_update.add()
    stu.stop_sequence = stop_seq
    stu.arrival.delay = delay_s
    return feed.SerializeToString()


def test_compact_materializes_daily_parquet_and_incremental_idempotency(temp_bronze_dir: Path):
    import pandas as pd

    date_str = "2026-10-18"
    pq_dir = temp_bronze_dir.parent / "bronze_parquet"

    # Batch 1: Hour 08 for um_bus and zbk_rt
    h08_bus = temp_bronze_dir / "source=um_bus" / f"date={date_str}" / "hour=08"
    h08_bus.mkdir(parents=True, exist_ok=True)
    # 2026-10-18 10:00:00 Warsaw CEST (UTC+2) -> 08:00:00 UTC = 1792310400
    ts_08 = 1792310400
    bus_payload_08 = [
        {
            "VehicleNumber": "1001",
            "Lines": "503",
            "Brigade": "1",
            "Lat": 52.2300,
            "Lon": 21.0000,
            "Time": "2026-10-18 10:00:00",
        }
    ]
    with gzip.open(h08_bus / f"tick_{ts_08}.json.gz", "wb") as f:
        f.write(json.dumps(bus_payload_08).encode("utf-8"))

    h08_zbk = temp_bronze_dir / "source=zbk_rt" / f"date={date_str}" / "hour=08"
    h08_zbk.mkdir(parents=True, exist_ok=True)
    with gzip.open(h08_zbk / f"tick_{ts_08}.pb.gz", "wb") as f:
        f.write(_make_zbk_pb_bytes("TRIP_1", 1, 45))

    compactor = HourlyBronzeCompactor(
        bronze_dir=temp_bronze_dir, compression_level=3, bronze_parquet_dir=pq_dir
    )
    compactor.compact_date(date_str=date_str, only_past_hours=False)

    fixes_pq = pq_dir / f"date={date_str}" / "vehicle_fixes.parquet"
    zbk_pq = pq_dir / f"date={date_str}" / "zbk_forecasts.parquet"
    manifest_file = pq_dir / f"date={date_str}" / "_manifest.json"

    assert fixes_pq.exists()
    assert zbk_pq.exists()
    assert manifest_file.exists()

    df_fixes_1 = pd.read_parquet(fixes_pq)
    assert len(df_fixes_1) == 1
    assert df_fixes_1.iloc[0]["vehicle_number"] == "1001"

    df_zbk_1 = pd.read_parquet(zbk_pq)
    assert len(df_zbk_1) == 1
    assert df_zbk_1.iloc[0]["forecast_val"] == 45

    # Batch 2 (Incremental 6-hourly cron simulation): Add Hour 09 with updated forecast for TRIP_1 seq 1
    h09_bus = temp_bronze_dir / "source=um_bus" / f"date={date_str}" / "hour=09"
    h09_bus.mkdir(parents=True, exist_ok=True)
    ts_09 = 1792314000
    bus_payload_09 = [
        {
            "VehicleNumber": "1001",
            "Lines": "503",
            "Brigade": "1",
            "Lat": 52.2350,
            "Lon": 21.0050,
            "Time": "2026-10-18 11:00:00",
        }
    ]
    with gzip.open(h09_bus / f"tick_{ts_09}.json.gz", "wb") as f:
        f.write(json.dumps(bus_payload_09).encode("utf-8"))

    h09_zbk = temp_bronze_dir / "source=zbk_rt" / f"date={date_str}" / "hour=09"
    h09_zbk.mkdir(parents=True, exist_ok=True)
    with gzip.open(h09_zbk / f"tick_{ts_09}.pb.gz", "wb") as f:
        f.write(_make_zbk_pb_bytes("TRIP_1", 1, 90))

    compactor.compact_date(date_str=date_str, only_past_hours=False)
    # Re-run a second time to verify idempotency (zero duplicate rows)
    compactor.compact_date(date_str=date_str, only_past_hours=False)

    df_fixes_2 = pd.read_parquet(fixes_pq)
    assert len(df_fixes_2) == 2
    assert list(df_fixes_2["hour_utc"]) == ["08", "09"]

    df_zbk_2 = pd.read_parquet(zbk_pq)
    assert len(df_zbk_2) == 1
    assert df_zbk_2.iloc[0]["forecast_val"] == 90


def test_silver_and_gold_fast_path_vs_raw_fallback_equivalence(temp_bronze_dir: Path):
    import shutil
    import pandas as pd
    from src.historian.gold import GoldStopEventReconstructor
    from src.historian.silver import SilverTrajectoryProcessor

    date_str = "2026-10-18"
    pq_dir = temp_bronze_dir.parent / "bronze_parquet"
    silver_fast_dir = temp_bronze_dir.parent / "silver_fast"
    silver_raw_dir = temp_bronze_dir.parent / "silver_raw"

    h08_bus = temp_bronze_dir / "source=um_bus" / f"date={date_str}" / "hour=08"
    h08_bus.mkdir(parents=True, exist_ok=True)
    ts_1 = 1792310400
    ts_2 = 1792310410
    p1 = [
        {
            "VehicleNumber": "2001",
            "Lines": "175",
            "Brigade": "2",
            "Lat": 52.2300,
            "Lon": 21.0000,
            "Time": "2026-10-18 10:00:00",
        }
    ]
    p2 = [
        {
            "VehicleNumber": "2001",
            "Lines": "175",
            "Brigade": "2",
            "Lat": 52.2310,
            "Lon": 21.0010,
            "Time": "2026-10-18 10:00:10",
        }
    ]
    with gzip.open(h08_bus / f"tick_{ts_1}.json.gz", "wb") as f:
        f.write(json.dumps(p1).encode("utf-8"))
    with gzip.open(h08_bus / f"tick_{ts_2}.json.gz", "wb") as f:
        f.write(json.dumps(p2).encode("utf-8"))

    h08_zbk = temp_bronze_dir / "source=zbk_rt" / f"date={date_str}" / "hour=08"
    h08_zbk.mkdir(parents=True, exist_ok=True)
    with gzip.open(h08_zbk / f"tick_{ts_1}.pb.gz", "wb") as f:
        f.write(_make_zbk_pb_bytes("TRIP_A", 1, 30))

    compactor = HourlyBronzeCompactor(
        bronze_dir=temp_bronze_dir, compression_level=3, bronze_parquet_dir=pq_dir
    )
    compactor.compact_date(date_str=date_str, only_past_hours=False)

    # 1. Silver Fast Path vs Raw Fallback Path
    proc_fast = SilverTrajectoryProcessor(
        bronze_dir=temp_bronze_dir, silver_dir=silver_fast_dir, bronze_parquet_dir=pq_dir
    )
    proc_raw = SilverTrajectoryProcessor(
        bronze_dir=temp_bronze_dir, silver_dir=silver_raw_dir, bronze_parquet_dir=pq_dir
    )

    out_fast = proc_fast.process_date(date_str, force=True, use_raw_bronze=False)
    out_raw = proc_raw.process_date(date_str, force=True, use_raw_bronze=True)

    df_silver_fast = pd.read_parquet(out_fast)
    df_silver_raw = pd.read_parquet(out_raw)
    pd.testing.assert_frame_equal(df_silver_fast, df_silver_raw)

    # 2. Gold ZBK Forecast Fast Path vs Raw Fallback Path
    gold_rec = GoldStopEventReconstructor(
        silver_dir=silver_fast_dir,
        gold_dir=temp_bronze_dir.parent / "gold",
        bronze_dir=temp_bronze_dir,
        bronze_parquet_dir=pq_dir,
    )
    fc_fast = gold_rec.load_zbk_forecast_delays(date_str, use_raw_bronze=False)
    fc_raw = gold_rec.load_zbk_forecast_delays(date_str, use_raw_bronze=True)
    assert fc_fast == fc_raw == {("TRIP_A", 1): ("delay", 30)}

    # 3. Delete bronze_parquet/ and verify automatic fallback to cold .tar.zst
    shutil.rmtree(pq_dir)
    out_fallback = proc_fast.process_date(date_str, force=True, use_raw_bronze=False)
    df_silver_fallback = pd.read_parquet(out_fallback)
    pd.testing.assert_frame_equal(df_silver_fallback, df_silver_raw)
    assert gold_rec.load_zbk_forecast_delays(date_str, use_raw_bronze=False) == fc_raw


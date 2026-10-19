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

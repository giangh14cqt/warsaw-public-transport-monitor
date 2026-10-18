"""
Unit tests for Historian Bronze storage writer.
"""

import gzip
from pathlib import Path

import pyarrow.parquet as pq

from src.historian.bronze import BronzeStorageWriter


def test_write_payload_atomic_gzip(tmp_path: Path):
    writer = BronzeStorageWriter(base_dir=tmp_path)
    sample_data = b'{"result": [{"Lines": "17", "VehicleNumber": "1001"}]}'
    tick_ts = 1792330000

    file_path, sha = writer.write_payload("um_bus", tick_ts, sample_data, "json")

    assert Path(file_path).exists()
    assert Path(file_path).name == f"tick_{tick_ts}.json.gz"

    # Verify gzip contents match original raw bytes
    with gzip.open(file_path, "rb") as f:
        decompressed = f.read()
    assert decompressed == sample_data
    assert len(sha) == 64


def test_record_and_flush_ticks(tmp_path: Path):
    writer = BronzeStorageWriter(base_dir=tmp_path)
    writer.record_tick(
        tick_ts=1792330000,
        source="um_bus",
        status="OK",
        http_status=200,
        latency_ms=120.5,
        bytes_count=45000,
        payload_sha256="abc123sha",
        n_vehicles=1200,
        server_date="Sun, 18 Oct 2026 13:40:00 GMT",
        clock_offset_s=0.25,
    )
    writer.record_tick(
        tick_ts=1792330000,
        source="um_tram",
        status="OK",
        http_status=200,
        latency_ms=85.0,
        bytes_count=15000,
        payload_sha256="def456sha",
        n_vehicles=300,
    )

    flushed_file = writer.flush_ticks()
    assert flushed_file is not None
    assert Path(flushed_file).exists()

    # Verify Parquet table contains both records with proper schema
    table = pq.read_table(flushed_file)
    assert table.num_rows == 2
    assert "tick_ts" in table.column_names
    assert "source" in table.column_names

    sources = table.column("source").to_pylist()
    assert "um_bus" in sources
    assert "um_tram" in sources

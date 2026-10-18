"""
Unit tests for Historian Static GTFS Versioner and DuckDB indexing.
"""

import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import pyarrow.parquet as pq

from src.historian.static_gtfs import HistorianGTFSVersioner


def _create_mock_gtfs_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            "trips.txt",
            "trip_id,route_id,brigade,service_id,shape_id,direction_id\n"
            "T1,100,1,SERV1,SH1,0\n"
            "T2,100,2,SERV1,SH1,0\n",
        )
        z.writestr(
            "stop_times.txt",
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,S1,1,08:00:00,08:00:00\n"
            "T1,S2,2,08:05:00,08:05:00\n"
            "T2,S1,1,08:15:00,08:15:00\n",
        )
        z.writestr(
            "stops.txt",
            "stop_id,stop_name,stop_lat,stop_lon\n"
            "S1,Central Station,52.2300,21.0000\n"
            "S2,Old Town,52.2450,21.0100\n",
        )
        z.writestr(
            "shapes.txt",
            "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence\n"
            "SH1,52.2300,21.0000,0\n"
            "SH1,52.2450,21.0100,1\n",
        )
    return buffer.getvalue()


def test_sync_daily_gtfs_downloads_and_indexes(tmp_path: Path):
    mock_zip_bytes = _create_mock_gtfs_zip()
    versioner = HistorianGTFSVersioner(gtfs_dir=tmp_path)

    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = mock_zip_bytes
        mock_get.return_value = mock_resp

        unpack_dir = versioner.sync_daily_gtfs()

    assert unpack_dir is not None
    assert unpack_dir.exists()

    # Verify DuckDB index was built
    db_path = unpack_dir / "schedule_index.duckdb"
    assert db_path.exists()

    con = duckdb.connect(str(db_path), read_only=True)
    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
    assert "trips" in tables
    assert "stop_times" in tables
    assert "stops" in tables
    assert "shapes" in tables

    trips_count = con.execute("SELECT count(*) FROM trips").fetchone()[0]
    assert trips_count == 2
    stops_count = con.execute("SELECT count(*) FROM stops").fetchone()[0]
    assert stops_count == 2

    # Verify versions.parquet table was created
    versions_file = tmp_path / "versions.parquet"
    assert versions_file.exists()
    v_table = pq.read_table(versions_file)
    assert len(v_table) == 1

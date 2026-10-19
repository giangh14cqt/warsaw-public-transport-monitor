"""
Bronze Storage Engine for Warsaw Public Transport Historian.
Atomically archives verbatim compressed API responses and logs tick audit metrics.
"""

import datetime
import gzip
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.historian.config import paths

logger = logging.getLogger(__name__)

# Schema for tick audit log (capture_ticks.parquet)
CAPTURE_TICKS_SCHEMA = pa.schema(
    [
        ("tick_ts", pa.int64()),
        ("source", pa.string()),
        ("status", pa.string()),
        ("http_status", pa.int32()),
        ("latency_ms", pa.float64()),
        ("bytes", pa.int64()),
        ("payload_sha256", pa.string()),
        ("n_vehicles", pa.int32()),
        ("server_date", pa.string()),
        ("clock_offset_s", pa.float64()),
        ("error_message", pa.string()),
        ("recorded_at_utc", pa.string()),
    ]
)


class BronzeStorageWriter:
    """Manages atomic gzipped writes and tick audit logs for the raw Bronze data lake."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or paths.bronze_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._tick_buffer: List[Dict[str, Any]] = []

    def write_payload(
        self,
        source: str,
        tick_ts: int,
        raw_bytes: bytes,
        file_ext: str = "json",
    ) -> tuple[str, str]:
        """
        Atomically saves gzipped raw payload under:
        bronze/source={source}/date={YYYY-MM-DD}/hour={HH}/tick_{epoch}.{file_ext}.gz
        Returns (relative_file_path, payload_sha256).
        """
        if not raw_bytes:
            return "", ""

        # Compute SHA256 of uncompressed payload
        payload_sha256 = hashlib.sha256(raw_bytes).hexdigest()

        # Construct date/hour partition path using UTC tick timestamp
        tick_dt = datetime.datetime.fromtimestamp(tick_ts, datetime.timezone.utc)
        date_str = tick_dt.strftime("%Y-%m-%d")
        hour_str = tick_dt.strftime("%H")

        partition_dir = self.base_dir / f"source={source}" / f"date={date_str}" / f"hour={hour_str}"
        partition_dir.mkdir(parents=True, exist_ok=True)

        final_path = partition_dir / f"tick_{tick_ts}.{file_ext}.gz"
        tmp_path = partition_dir / f"tick_{tick_ts}.{file_ext}.gz.tmp"

        # Atomic write: compress to tmp file, then atomic rename
        with gzip.open(tmp_path, "wb", compresslevel=6) as f:
            f.write(raw_bytes)
        tmp_path.replace(final_path)

        return str(final_path), payload_sha256

    def record_tick(
        self,
        tick_ts: int,
        source: str,
        status: str,
        http_status: int = 200,
        latency_ms: float = 0.0,
        bytes_count: int = 0,
        payload_sha256: str = "",
        n_vehicles: int = 0,
        server_date: Optional[str] = None,
        clock_offset_s: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Buffers a tick audit record."""
        record = {
            "tick_ts": int(tick_ts),
            "source": str(source),
            "status": str(status),
            "http_status": int(http_status),
            "latency_ms": float(latency_ms),
            "bytes": int(bytes_count),
            "payload_sha256": str(payload_sha256 or ""),
            "n_vehicles": int(n_vehicles),
            "server_date": str(server_date or ""),
            "clock_offset_s": float(clock_offset_s) if clock_offset_s is not None else None,
            "error_message": str(error_message or ""),
            "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        self._tick_buffer.append(record)

    def flush_ticks(self) -> Optional[str]:
        """
        Appends buffered tick audit records into daily partitioned capture_ticks.parquet.
        Returns the updated file path, or None if buffer was empty.
        """
        if not self._tick_buffer:
            return None

        records_to_flush = self._tick_buffer
        self._tick_buffer = []

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        date_str = now_utc.strftime("%Y-%m-%d")
        ticks_dir = self.base_dir / "_ticks" / f"date={date_str}"
        ticks_dir.mkdir(parents=True, exist_ok=True)
        ticks_file = ticks_dir / "capture_ticks.parquet"

        new_df = pd.DataFrame(records_to_flush)

        # Cast to match schema explicitly
        new_df["tick_ts"] = new_df["tick_ts"].astype("int64")
        new_df["source"] = new_df["source"].astype("string")
        new_df["status"] = new_df["status"].astype("string")
        new_df["http_status"] = new_df["http_status"].astype("int32")
        new_df["latency_ms"] = new_df["latency_ms"].astype("float64")
        new_df["bytes"] = new_df["bytes"].astype("int64")
        new_df["payload_sha256"] = new_df["payload_sha256"].astype("string")
        new_df["n_vehicles"] = new_df["n_vehicles"].astype("int32")
        new_df["server_date"] = new_df["server_date"].astype("string")
        new_df["clock_offset_s"] = new_df["clock_offset_s"].astype("Float64")
        new_df["error_message"] = new_df["error_message"].astype("string")
        new_df["recorded_at_utc"] = new_df["recorded_at_utc"].astype("string")

        new_table = pa.Table.from_pandas(new_df, schema=CAPTURE_TICKS_SCHEMA, preserve_index=False)

        # Append to existing table if present
        if ticks_file.exists():
            try:
                existing_table = pq.read_table(ticks_file, schema=CAPTURE_TICKS_SCHEMA)
                combined_table = pa.concat_tables([existing_table, new_table])
            except Exception as e:
                logger.warning(f"Could not read existing capture_ticks table, overwriting: {e}")
                combined_table = new_table
        else:
            combined_table = new_table

        # Atomic write
        tmp_file = ticks_file.with_suffix(".parquet.tmp")
        pq.write_table(combined_table, tmp_file, compression="snappy")
        tmp_file.replace(ticks_file)

        logger.debug(f"Flushed {len(records_to_flush)} tick audit records to {ticks_file}")
        return str(ticks_file)


def iter_bronze_ticks(
    base_dir: Path,
    source: str,
    date_str: str,
    max_hours: Optional[List[str]] = None,
) -> Iterator[Tuple[int, bytes, str]]:
    """
    Chronologically streams raw payloads (tick_ts, raw_bytes, filename) for a given source and date.
    Transparently supports both compacted hourly .tar.zst archives and legacy loose tick_*.gz files.
    """
    from src.historian.compact import extract_tick_ts, open_tar_zst_reader

    date_dir = base_dir / f"source={source}" / f"date={date_str}"
    if not date_dir.exists():
        return

    # Discover all hours present (from either hour=HH.tar.zst archives or hour=HH/ directories)
    hours_set = set()
    for item in date_dir.iterdir():
        name = item.name
        if name.startswith("hour=") and not name.endswith(".tmp"):
            # Strip extensions if archive
            hour_part = name.split(".")[0].split("=")[-1]
            hours_set.add(hour_part)

    for hour_str in sorted(hours_set):
        if max_hours and hour_str not in max_hours:
            continue

        archive_path = date_dir / f"hour={hour_str}.tar.zst"
        hour_dir = date_dir / f"hour={hour_str}"

        # 1. Prefer compacted archive if present
        if archive_path.exists():
            try:
                for tick_ts, raw_bytes, member_name in open_tar_zst_reader(archive_path):
                    yield tick_ts, raw_bytes, member_name
                continue
            except Exception as e:
                logger.warning(f"Failed reading archive {archive_path}, checking for loose files: {e}")

        # 2. Fall back to loose tick_*.gz files in directory
        if hour_dir.is_dir():
            for gz_file in sorted(hour_dir.glob("tick_*.gz")):
                if gz_file.name.endswith(".tmp") or gz_file.name.startswith("."):
                    continue
                try:
                    tick_ts = extract_tick_ts(gz_file.name)
                    with gzip.open(gz_file, "rb") as gf:
                        raw_bytes = gf.read()
                    yield tick_ts, raw_bytes, gz_file.name
                except Exception as e:
                    logger.warning(f"Could not read {gz_file}: {e}")
                    continue


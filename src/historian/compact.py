"""
Compaction engine for Warsaw Public Transport Historian Bronze data lake.
Bundles thousands of high-frequency tick files into hourly compressed tar.zst archives,
drastically reducing file counts, speeding up cloud transfers, and saving storage.
"""

import datetime
import gzip
import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pd = None  # type: ignore[assignment]
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]

try:
    import zstandard as zstd
except ImportError:
    zstd = None

from src.historian.config import paths

logger = logging.getLogger(__name__)

VEHICLE_FIXES_PARQUET_SCHEMA = (
    pa.schema(
        [
            ("service_date", pa.string()),
            ("vehicle_type", pa.string()),
            ("vehicle_number", pa.string()),
            ("route_id", pa.string()),
            ("brigade", pa.string()),
            ("lat", pa.float64()),
            ("lon", pa.float64()),
            ("gps_timestamp_utc", pa.int64()),
            ("gps_time_str", pa.string()),
            ("tick_ts", pa.int64()),
            ("hour_utc", pa.string()),
        ]
    )
    if pa is not None
    else None
)

ZBK_FORECASTS_PARQUET_SCHEMA = (
    pa.schema(
        [
            ("service_date", pa.string()),
            ("trip_id", pa.string()),
            ("stop_sequence", pa.int32()),
            ("forecast_kind", pa.string()),
            ("forecast_val", pa.int64()),
            ("tick_ts", pa.int64()),
            ("hour_utc", pa.string()),
        ]
    )
    if pa is not None
    else None
)


def extract_tick_ts(filename: str) -> int:
    """Extracts integer unix epoch from filenames like tick_1792332560.json[.gz]."""
    base = os.path.basename(filename)
    if base.startswith("._"):
        base = base[2:]
    if base.startswith("tick_"):
        parts = base[5:].split(".")
        try:
            return int(parts[0])
        except ValueError:
            pass
    return 0


def open_tar_zst_reader(archive_path: Path) -> Iterator[Tuple[int, bytes, str]]:
    """
    Streams (tick_ts, raw_bytes, member_name) from a .tar.zst archive in memory.
    Uses Python zstandard if installed, otherwise falls back to system zstd CLI.
    """
    if zstd is not None:
        dctx = zstd.ZstdDecompressor()
        with open(archive_path, "rb") as fh:
            with dctx.stream_reader(fh, closefd=False) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    for member in tar:
                        if not member.isfile() or member.name.startswith("._") or member.name.startswith("."):
                            continue
                        f_mem = tar.extractfile(member)
                        if f_mem is not None:
                            raw = f_mem.read()
                            yield extract_tick_ts(member.name), raw, member.name
        return

    # Fallback to system zstd binary if zstandard module is not installed
    if shutil.which("zstd"):
        proc = subprocess.Popen(
            ["zstd", "-d", "-c", str(archive_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
                for member in tar:
                    if not member.isfile() or member.name.startswith("._") or member.name.startswith("."):
                        continue
                    f_mem = tar.extractfile(member)
                    if f_mem is not None:
                        raw = f_mem.read()
                        yield extract_tick_ts(member.name), raw, member.name
        finally:
            proc.stdout.close()
            proc.wait()
        return

    raise RuntimeError(
        f"Cannot read {archive_path}: 'zstandard' python package or 'zstd' command line tool is required."
    )


def compact_hour_directory(
    hour_dir: Path,
    output_archive: Path,
    compression_level: int = 15,
    delete_source: bool = True,
) -> Dict[str, Any]:
    """
    Compacts a loose hour directory (hour=HH/) containing tick_*.gz into a single
    verified hour=HH.tar.zst archive.
    """
    gz_files = sorted(
        [f for f in hour_dir.glob("tick_*.gz") if not f.name.endswith(".tmp") and not f.name.startswith(".")]
    )

    if not gz_files:
        # If directory is completely empty, clean it up
        try:
            if not any(hour_dir.iterdir()):
                hour_dir.rmdir()
        except Exception:
            pass
        return {"status": "SKIPPED_EMPTY", "file_count": 0}

    t0 = time.time()
    orig_gz_bytes = sum(f.stat().st_size for f in gz_files)

    output_archive.parent.mkdir(parents=True, exist_ok=True)
    tmp_archive = output_archive.with_suffix(".tar.zst.tmp")

    # In-memory checksum store for integrity verification
    expected_checksums: Dict[str, str] = {}

    if zstd is not None:
        cctx = zstd.ZstdCompressor(level=compression_level, threads=-1)
        with open(tmp_archive, "wb") as fh:
            with cctx.stream_writer(fh, closefd=False) as zst_out:
                with tarfile.open(fileobj=zst_out, mode="w|") as tar:
                    for gz_file in gz_files:
                        with gzip.open(gz_file, "rb") as gf:
                            raw_bytes = gf.read()

                        # Compute SHA256 of uncompressed payload
                        sha = hashlib.sha256(raw_bytes).hexdigest()

                        # Canonical member name: tick_12345.json or tick_12345.pb
                        member_name = gz_file.name
                        if member_name.endswith(".gz"):
                            member_name = member_name[:-3]
                        expected_checksums[member_name] = sha

                        ti = tarfile.TarInfo(name=member_name)
                        ti.size = len(raw_bytes)
                        ti.mtime = int(gz_file.stat().st_mtime)
                        tar.addfile(ti, io.BytesIO(raw_bytes))
    else:
        # Fallback using tar CLI and zstd CLI
        if not shutil.which("zstd") or not shutil.which("tar"):
            raise RuntimeError("Compaction requires 'zstandard' python package or system 'tar' and 'zstd'.")

        tmp_tar = output_archive.with_suffix(".tar.tmp")
        with tarfile.open(tmp_tar, mode="w") as tar:
            for gz_file in gz_files:
                with gzip.open(gz_file, "rb") as gf:
                    raw_bytes = gf.read()
                sha = hashlib.sha256(raw_bytes).hexdigest()
                member_name = gz_file.name[:-3] if gz_file.name.endswith(".gz") else gz_file.name
                expected_checksums[member_name] = sha

                ti = tarfile.TarInfo(name=member_name)
                ti.size = len(raw_bytes)
                ti.mtime = int(gz_file.stat().st_mtime)
                tar.addfile(ti, io.BytesIO(raw_bytes))

        # Compress tar with zstd
        subprocess.run(
            ["zstd", f"-{compression_level}", "-T0", "--rm", "-q", "-f", str(tmp_tar), "-o", str(tmp_archive)],
            check=True,
        )

    # Integrity verification: stream members from tmp_archive and verify SHA256
    verified_count = 0
    for tick_ts, raw_data, member_name in open_tar_zst_reader(tmp_archive):
        verified_count += 1
        actual_sha = hashlib.sha256(raw_data).hexdigest()
        expected_sha = expected_checksums.get(member_name)
        if expected_sha and actual_sha != expected_sha:
            tmp_archive.unlink(missing_ok=True)
            raise ValueError(
                f"Checksum mismatch in compacted bundle for {member_name}: expected {expected_sha}, got {actual_sha}"
            )

    if verified_count != len(gz_files):
        tmp_archive.unlink(missing_ok=True)
        raise ValueError(
            f"Member count mismatch in {tmp_archive}: expected {len(gz_files)}, read {verified_count}"
        )

    # Atomic rename into final destination
    tmp_archive.replace(output_archive)
    bundle_bytes = output_archive.stat().st_size
    elapsed = time.time() - t0

    # Safely delete loose source files after successful verification
    if delete_source:
        for gz_file in gz_files:
            try:
                gz_file.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to delete loose file {gz_file}: {e}")

        # Remove the directory if empty
        try:
            if not any(hour_dir.iterdir()):
                hour_dir.rmdir()
        except Exception:
            pass

    ratio = orig_gz_bytes / max(1, bundle_bytes)
    return {
        "status": "COMPACTED",
        "file_count": len(gz_files),
        "orig_gz_bytes": orig_gz_bytes,
        "bundle_bytes": bundle_bytes,
        "compression_ratio": round(ratio, 2),
        "elapsed_seconds": round(elapsed, 2),
        "archive_path": str(output_archive),
    }


def _extract_hour_vehicle_fixes_for_parquet(
    task_args: Tuple[str, str, str, str, str]
) -> List[Dict[str, Any]]:
    """Worker: extracts vehicle fixes for a single hour and tags hour_utc."""
    from src.historian.silver import _extract_single_hour_fixes

    _, _, _, _, hour_str = task_args
    fixes = _extract_single_hour_fixes(task_args)
    for r in fixes:
        r["hour_utc"] = hour_str
    return fixes


def _extract_hour_zbk_forecasts_for_parquet(
    task_args: Tuple[str, str, str]
) -> List[Dict[str, Any]]:
    """Worker: extracts latest GTFS-RT forecast per (trip_id, stop_sequence) for a single hour."""
    base_dir_str, date_str, hour_str = task_args
    base_dir = Path(base_dir_str)
    from google.transit import gtfs_realtime_pb2
    from src.historian.bronze import iter_bronze_ticks

    hour_forecasts: Dict[Tuple[str, int], Tuple[str, int, int]] = {}
    for tick_ts, content, _ in iter_bronze_ticks(
        base_dir=base_dir,
        source="zbk_rt",
        date_str=date_str,
        max_hours=[hour_str],
    ):
        try:
            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(content)

            for entity in feed.entity:
                if entity.HasField("trip_update"):
                    tu = entity.trip_update
                    trip_id = tu.trip.trip_id if tu.HasField("trip") else ""
                    if not trip_id:
                        continue
                    for stu in tu.stop_time_update:
                        seq = int(stu.stop_sequence)
                        if stu.HasField("arrival"):
                            if stu.arrival.HasField("delay"):
                                hour_forecasts[(trip_id, seq)] = ("delay", int(stu.arrival.delay), int(tick_ts))
                            elif stu.arrival.HasField("time"):
                                hour_forecasts[(trip_id, seq)] = ("time", int(stu.arrival.time), int(tick_ts))
                        elif stu.HasField("departure"):
                            if stu.departure.HasField("delay"):
                                hour_forecasts[(trip_id, seq)] = ("delay", int(stu.departure.delay), int(tick_ts))
                            elif stu.departure.HasField("time"):
                                hour_forecasts[(trip_id, seq)] = ("time", int(stu.departure.time), int(tick_ts))
        except Exception:
            continue

    rows: List[Dict[str, Any]] = []
    for (trip_id, seq), (kind, val, tick_ts) in hour_forecasts.items():
        rows.append(
            {
                "service_date": date_str,
                "trip_id": str(trip_id),
                "stop_sequence": int(seq),
                "forecast_kind": str(kind),
                "forecast_val": int(val),
                "tick_ts": int(tick_ts),
                "hour_utc": str(hour_str),
            }
        )
    return rows


class HourlyBronzeCompactor:
    """Manages scheduled and ad-hoc compaction of Historian Bronze directories and daily Parquet partitions."""

    def __init__(
        self,
        bronze_dir: Optional[Path] = None,
        compression_level: int = 15,
        bronze_parquet_dir: Optional[Path] = None,
    ):
        self.bronze_dir = bronze_dir or paths.bronze_dir
        if bronze_parquet_dir is not None:
            self.bronze_parquet_dir = bronze_parquet_dir
        elif bronze_dir is not None and bronze_dir != paths.bronze_dir:
            self.bronze_parquet_dir = bronze_dir.parent / "bronze_parquet"
        else:
            self.bronze_parquet_dir = paths.bronze_parquet_dir
        self.compression_level = compression_level

    def _discover_eligible_hours(
        self,
        source: str,
        date_str: str,
        only_past_hours: bool = True,
    ) -> List[str]:
        """Discovers all completed/compacted hours available for a given source and date."""
        src_date_dir = self.bronze_dir / f"source={source}" / f"date={date_str}"
        if not src_date_dir.exists():
            return []

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        today_utc_str = now_utc.strftime("%Y-%m-%d")
        current_hour_int = now_utc.hour

        hours_set = set()
        for item in src_date_dir.iterdir():
            name = item.name
            if not name.startswith("hour=") or name.endswith(".tmp"):
                continue
            hour_part = name.split(".")[0].split("=")[-1]
            try:
                hour_int = int(hour_part)
            except ValueError:
                continue

            if only_past_hours and date_str == today_utc_str and hour_int >= current_hour_int:
                continue

            if item.is_file() and name.endswith(".tar.zst"):
                hours_set.add(f"{hour_int:02d}")
            elif item.is_dir() and any(item.glob("tick_*.gz")):
                hours_set.add(f"{hour_int:02d}")

        return sorted(hours_set)

    def materialize_daily_parquet(
        self,
        date_str: str,
        only_past_hours: bool = True,
        force: bool = False,
        max_workers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Incrementally unpacks compacted Bronze hours into daily partitioned Parquet files:
          - bronze_parquet/date={date_str}/vehicle_fixes.parquet
          - bronze_parquet/date={date_str}/zbk_forecasts.parquet
        Preserves exact chronological ordering for 100% parity with raw .tar.zst reconstruction.
        """
        if pd is None or pa is None or pq is None:
            logger.warning(
                "Skipping daily Parquet materialization because 'pandas'/'pyarrow' are not installed in this Python interpreter."
            )
            return {"status": "SKIPPED_NO_PYARROW", "date": date_str}

        t0 = time.time()
        date_pq_dir = self.bronze_parquet_dir / f"date={date_str}"
        manifest_path = date_pq_dir / "_manifest.json"
        fixes_pq_path = date_pq_dir / "vehicle_fixes.parquet"
        zbk_pq_path = date_pq_dir / "zbk_forecasts.parquet"

        manifest: Dict[str, Any] = {
            "date": date_str,
            "vehicle_hours": {"um_bus": [], "um_tram": []},
            "zbk_hours": [],
        }
        if not force and manifest_path.exists():
            try:
                loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    manifest["vehicle_hours"]["um_bus"] = list(
                        loaded.get("vehicle_hours", {}).get("um_bus", [])
                    )
                    manifest["vehicle_hours"]["um_tram"] = list(
                        loaded.get("vehicle_hours", {}).get("um_tram", [])
                    )
                    manifest["zbk_hours"] = list(loaded.get("zbk_hours", []))
            except Exception as e:
                logger.warning(f"Could not read manifest {manifest_path}, rebuilding: {e}")
                force = True

        # 1. Determine which vehicle hours need unpacking
        veh_tasks: List[Tuple[str, str, str, str, str]] = []
        avail_veh_hours: Dict[str, List[str]] = {}
        for v_type, src in [("bus", "um_bus"), ("tram", "um_tram")]:
            avail = self._discover_eligible_hours(src, date_str, only_past_hours=only_past_hours)
            avail_veh_hours[src] = avail
            already_done = set() if (force or not fixes_pq_path.exists()) else set(manifest["vehicle_hours"].get(src, []))
            for h_str in avail:
                if h_str not in already_done:
                    veh_tasks.append((str(self.bronze_dir), src, v_type, date_str, h_str))

        # 2. Determine which zbk_rt hours need unpacking
        avail_zbk = self._discover_eligible_hours("zbk_rt", date_str, only_past_hours=only_past_hours)
        already_zbk = set() if (force or not zbk_pq_path.exists()) else set(manifest["zbk_hours"])
        zbk_tasks: List[Tuple[str, str, str]] = [
            (str(self.bronze_dir), date_str, h_str) for h_str in avail_zbk if h_str not in already_zbk
        ]

        if not veh_tasks and not zbk_tasks:
            return {
                "status": "UP_TO_DATE",
                "date": date_str,
                "new_vehicle_hours": 0,
                "new_zbk_hours": 0,
            }

        num_cores = os.cpu_count() or 4
        updated_any = False

        # Process vehicle fixes (um_bus + um_tram)
        if veh_tasks:
            workers = min(len(veh_tasks), max_workers or num_cores)
            new_fixes: List[Dict[str, Any]] = []
            if workers <= 1:
                for t in veh_tasks:
                    new_fixes.extend(_extract_hour_vehicle_fixes_for_parquet(t))
            else:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    for h_fixes in executor.map(_extract_hour_vehicle_fixes_for_parquet, veh_tasks):
                        new_fixes.extend(h_fixes)

            new_df = (
                pd.DataFrame(new_fixes)
                if new_fixes
                else pd.DataFrame(columns=[f.name for f in VEHICLE_FIXES_PARQUET_SCHEMA])
            )

            if not force and fixes_pq_path.exists():
                try:
                    existing_df = pq.read_table(fixes_pq_path).to_pandas()
                    if not existing_df.empty and not new_df.empty:
                        updated_pairs = {(t[2], t[4]) for t in veh_tasks}
                        keep_mask = ~existing_df.apply(
                            lambda r: (r["vehicle_type"], r["hour_utc"]) in updated_pairs, axis=1
                        )
                        existing_df = existing_df[keep_mask]
                    combined_df = (
                        pd.concat([existing_df, new_df], ignore_index=True)
                        if not existing_df.empty
                        else new_df
                    )
                except Exception as e:
                    logger.warning(f"Failed reading existing {fixes_pq_path}, overwriting: {e}")
                    combined_df = new_df
            else:
                combined_df = new_df

            if not combined_df.empty:
                # Stable sort by (vehicle_type, hour_utc) preserves exact within-hour record sequence
                combined_df.sort_values(
                    by=["vehicle_type", "hour_utc"],
                    kind="mergesort",
                    inplace=True,
                )
                combined_df.reset_index(drop=True, inplace=True)

                date_pq_dir.mkdir(parents=True, exist_ok=True)
                table = pa.Table.from_pandas(
                    combined_df, schema=VEHICLE_FIXES_PARQUET_SCHEMA, preserve_index=False
                )
                tmp_pq = fixes_pq_path.with_suffix(".parquet.tmp")
                pq.write_table(table, tmp_pq, compression="zstd")
                tmp_pq.replace(fixes_pq_path)
                updated_any = True

            for _, src, _, _, h_str in veh_tasks:
                if h_str not in manifest["vehicle_hours"][src]:
                    manifest["vehicle_hours"][src].append(h_str)
            for src in ("um_bus", "um_tram"):
                manifest["vehicle_hours"][src] = sorted(set(manifest["vehicle_hours"][src]))

        # Process zbk_rt forecasts
        if zbk_tasks:
            workers = min(len(zbk_tasks), max_workers or num_cores)
            new_zbk_rows: List[Dict[str, Any]] = []
            if workers <= 1:
                for t in zbk_tasks:
                    new_zbk_rows.extend(_extract_hour_zbk_forecasts_for_parquet(t))
            else:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    for h_rows in executor.map(_extract_hour_zbk_forecasts_for_parquet, zbk_tasks):
                        new_zbk_rows.extend(h_rows)

            new_zbk_df = (
                pd.DataFrame(new_zbk_rows)
                if new_zbk_rows
                else pd.DataFrame(columns=[f.name for f in ZBK_FORECASTS_PARQUET_SCHEMA])
            )

            if not force and zbk_pq_path.exists():
                try:
                    existing_zbk_df = pq.read_table(zbk_pq_path).to_pandas()
                    if not existing_zbk_df.empty and not new_zbk_df.empty:
                        updated_hours = {t[2] for t in zbk_tasks}
                        existing_zbk_df = existing_zbk_df[~existing_zbk_df["hour_utc"].isin(updated_hours)]
                    combined_zbk = (
                        pd.concat([existing_zbk_df, new_zbk_df], ignore_index=True)
                        if not existing_zbk_df.empty
                        else new_zbk_df
                    )
                except Exception as e:
                    logger.warning(f"Failed reading existing {zbk_pq_path}, overwriting: {e}")
                    combined_zbk = new_zbk_df
            else:
                combined_zbk = new_zbk_df

            if not combined_zbk.empty:
                # Sort by hour_utc and tick_ts, then deduplicate keeping the latest forecast per (trip_id, stop_sequence)
                combined_zbk.sort_values(
                    by=["hour_utc", "tick_ts"],
                    kind="mergesort",
                    inplace=True,
                )
                combined_zbk.drop_duplicates(
                    subset=["trip_id", "stop_sequence"],
                    keep="last",
                    inplace=True,
                )
                combined_zbk.reset_index(drop=True, inplace=True)

                date_pq_dir.mkdir(parents=True, exist_ok=True)
                zbk_table = pa.Table.from_pandas(
                    combined_zbk, schema=ZBK_FORECASTS_PARQUET_SCHEMA, preserve_index=False
                )
                tmp_zbk = zbk_pq_path.with_suffix(".parquet.tmp")
                pq.write_table(zbk_table, tmp_zbk, compression="zstd")
                tmp_zbk.replace(zbk_pq_path)
                updated_any = True

            for _, _, h_str in zbk_tasks:
                if h_str not in manifest["zbk_hours"]:
                    manifest["zbk_hours"].append(h_str)
            manifest["zbk_hours"] = sorted(set(manifest["zbk_hours"]))

        if updated_any or date_pq_dir.exists():
            date_pq_dir.mkdir(parents=True, exist_ok=True)
            tmp_manifest = manifest_path.with_suffix(".json.tmp")
            tmp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            tmp_manifest.replace(manifest_path)

        elapsed = time.time() - t0
        fixes_mb = (fixes_pq_path.stat().st_size / (1024 * 1024)) if fixes_pq_path.exists() else 0.0
        zbk_mb = (zbk_pq_path.stat().st_size / (1024 * 1024)) if zbk_pq_path.exists() else 0.0
        if updated_any:
            logger.info(
                f"Materialized daily Bronze Parquet for {date_str} in {elapsed:.2f}s: "
                f"vehicle_fixes={fixes_mb:.2f}MB (+{len(veh_tasks)}h), zbk_forecasts={zbk_mb:.2f}MB (+{len(zbk_tasks)}h)"
            )
        return {
            "status": "MATERIALIZED" if updated_any else "EMPTY",
            "date": date_str,
            "new_vehicle_hours": len(veh_tasks),
            "new_zbk_hours": len(zbk_tasks),
            "vehicle_fixes_bytes": fixes_pq_path.stat().st_size if fixes_pq_path.exists() else 0,
            "zbk_forecasts_bytes": zbk_pq_path.stat().st_size if zbk_pq_path.exists() else 0,
            "elapsed_seconds": round(elapsed, 2),
        }

    def compact_date(
        self,
        date_str: str,
        sources: Optional[List[str]] = None,
        only_past_hours: bool = True,
        delete_source: bool = True,
        materialize_parquet: bool = True,
        force_parquet: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Compacts all eligible hours for a specified date across given sources,
        and materializes the daily partitioned Parquet files.
        """
        target_sources = sources or ["um_bus", "um_tram", "zbk_rt"]
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        today_utc_str = now_utc.strftime("%Y-%m-%d")
        current_hour_int = now_utc.hour

        results: List[Dict[str, Any]] = []

        for src in target_sources:
            src_date_dir = self.bronze_dir / f"source={src}" / f"date={date_str}"
            if not src_date_dir.exists():
                continue

            # Find all loose hour directories
            for hour_dir in sorted(src_date_dir.glob("hour=*")):
                if not hour_dir.is_dir() or hour_dir.name.endswith(".tmp"):
                    continue

                try:
                    hour_int = int(hour_dir.name.split("=")[-1])
                except ValueError:
                    continue

                # Safety guard: if today, never touch current or future hours
                if only_past_hours and date_str == today_utc_str and hour_int >= current_hour_int:
                    logger.debug(f"Skipping active hour {hour_dir} (current UTC hour is {current_hour_int})")
                    continue

                archive_name = f"{hour_dir.name}.tar.zst"
                archive_path = src_date_dir / archive_name

                # If archive already exists and no loose files remain, skip
                loose_files = list(hour_dir.glob("tick_*.gz"))
                if archive_path.exists() and not loose_files:
                    continue

                logger.info(f"Compacting {hour_dir} ({len(loose_files)} files) -> {archive_path.name}...")
                try:
                    res = compact_hour_directory(
                        hour_dir=hour_dir,
                        output_archive=archive_path,
                        compression_level=self.compression_level,
                        delete_source=delete_source,
                    )
                    res["source"] = src
                    res["date"] = date_str
                    res["hour"] = hour_dir.name
                    results.append(res)
                    if res.get("status") == "COMPACTED":
                        logger.info(
                            f"Compacted {src} {date_str} {hour_dir.name}: "
                            f"{res['file_count']} files, {res['orig_gz_bytes'] / 1024 / 1024:.1f}MB -> "
                            f"{res['bundle_bytes'] / 1024 / 1024:.1f}MB ({res['compression_ratio']}x) "
                            f"in {res['elapsed_seconds']}s"
                        )
                except Exception as e:
                    logger.exception(f"Error compacting {hour_dir}: {e}")
                    results.append(
                        {
                            "status": "ERROR",
                            "source": src,
                            "date": date_str,
                            "hour": hour_dir.name,
                            "error": str(e),
                        }
                    )

        if materialize_parquet:
            try:
                self.materialize_daily_parquet(
                    date_str=date_str,
                    only_past_hours=only_past_hours,
                    force=force_parquet,
                )
            except Exception as e:
                logger.exception(f"Error materializing daily parquet for {date_str}: {e}")

        return results

    def compact_all(
        self,
        only_past_hours: bool = True,
        delete_source: bool = True,
        materialize_parquet: bool = True,
        force_parquet: bool = False,
    ) -> List[Dict[str, Any]]:
        """Scans all available dates across all sources and compacts eligible hours."""
        all_dates = set()
        for src_dir in self.bronze_dir.glob("source=*"):
            for d_dir in src_dir.glob("date=*"):
                all_dates.add(d_dir.name.split("=")[-1])

        results = []
        for d in sorted(all_dates):
            res = self.compact_date(
                date_str=d,
                only_past_hours=only_past_hours,
                delete_source=delete_source,
                materialize_parquet=materialize_parquet,
                force_parquet=force_parquet,
            )
            results.extend(res)
        return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Warsaw Transit Historian Bronze Compactor & Parquet Materializer")
    parser.add_argument("--date", type=str, default=None, help="Specific date to compact (YYYY-MM-DD)")
    parser.add_argument("--source", type=str, default=None, help="Specific source (um_bus, um_tram, zbk_rt)")
    parser.add_argument("--level", type=int, default=15, help="Zstandard compression level (1-22, default 15)")
    parser.add_argument(
        "--include-current-hour",
        action="store_true",
        help="Compact the current hour as well (caution: daemon may be writing)",
    )
    parser.add_argument(
        "--keep-loose-files",
        action="store_true",
        help="Do not delete loose source files after compaction",
    )
    parser.add_argument(
        "--backfill-parquet",
        action="store_true",
        help="Materialize daily bronze_parquet partitions from existing compacted .tar.zst archives",
    )
    parser.add_argument(
        "--force",
        "--force-parquet",
        dest="force_parquet",
        action="store_true",
        help="Force rebuild of daily bronze_parquet partitions from scratch",
    )
    parser.add_argument(
        "--skip-parquet",
        action="store_true",
        help="Skip daily bronze_parquet materialization (only compact .tar.zst)",
    )

    args = parser.parse_args()

    # If running on a Docker host where bare system python3 lacks pandas/pyarrow,
    # automatically delegate execution into a transient container using the historian image.
    if (pd is None or pa is None) and not args.skip_parquet and os.getenv("IN_DOCKER_COMPACT") != "1":
        if shutil.which("docker"):
            docker_img = ""
            for cname in ("warsaw_historian_daemon", "warsaw_telemetry_daemon"):
                try:
                    out = subprocess.check_output(
                        ["docker", "inspect", "--format={{.Config.Image}}", cname],
                        stderr=subprocess.DEVNULL,
                        text=True,
                    ).strip()
                    if out:
                        docker_img = out
                        break
                except Exception:
                    continue

            if docker_img:
                proj_root = str(paths.root)
                uid_gid = f"{os.getuid()}:{os.getgid()}"
                cmd = [
                    "docker",
                    "run",
                    "--rm",
                    "-e",
                    "IN_DOCKER_COMPACT=1",
                    "--user",
                    uid_gid,
                    "-v",
                    f"{proj_root}/src:/app/src",
                    "-v",
                    f"{proj_root}/data:/app/data",
                    "-w",
                    "/app",
                    docker_img,
                    "python3",
                    "-m",
                    "src.historian.compact",
                    *sys.argv[1:],
                ]
                logger.info(
                    f"Host python3 lacks pandas/pyarrow; delegating compaction & Parquet materialization to Docker image '{docker_img}'..."
                )
                os.execvp("docker", cmd)

    compactor = HourlyBronzeCompactor(compression_level=args.level)
    sources = [args.source] if args.source else None
    only_past = not args.include_current_hour
    delete_src = not args.keep_loose_files
    mat_pq = not args.skip_parquet

    if args.date:
        res = compactor.compact_date(
            date_str=args.date,
            sources=sources,
            only_past_hours=only_past,
            delete_source=delete_src,
            materialize_parquet=mat_pq,
            force_parquet=args.force_parquet,
        )
    else:
        res = compactor.compact_all(
            only_past_hours=only_past,
            delete_source=delete_src,
            materialize_parquet=mat_pq,
            force_parquet=args.force_parquet,
        )

    compacted = [r for r in res if r.get("status") == "COMPACTED"]
    logger.info(f"Finished. Total compacted hours: {len(compacted)}")

"""
Compaction engine for Warsaw Public Transport Historian Bronze data lake.
Bundles thousands of high-frequency tick files into hourly compressed tar.zst archives,
drastically reducing file counts, speeding up cloud transfers, and saving storage.
"""

import datetime
import gzip
import hashlib
import io
import logging
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    import zstandard as zstd
except ImportError:
    zstd = None

from src.historian.config import paths

logger = logging.getLogger(__name__)


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


class HourlyBronzeCompactor:
    """Manages scheduled and ad-hoc compaction of Historian Bronze directories."""

    def __init__(self, bronze_dir: Optional[Path] = None, compression_level: int = 15):
        self.bronze_dir = bronze_dir or paths.bronze_dir
        self.compression_level = compression_level

    def compact_date(
        self,
        date_str: str,
        sources: Optional[List[str]] = None,
        only_past_hours: bool = True,
        delete_source: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Compacts all eligible hours for a specified date across given sources.
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

        return results

    def compact_all(
        self,
        only_past_hours: bool = True,
        delete_source: bool = True,
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

    parser = argparse.ArgumentParser(description="Warsaw Transit Historian Bronze Compactor")
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

    args = parser.parse_args()

    compactor = HourlyBronzeCompactor(compression_level=args.level)
    sources = [args.source] if args.source else None
    only_past = not args.include_current_hour
    delete_src = not args.keep_loose_files

    if args.date:
        res = compactor.compact_date(
            date_str=args.date,
            sources=sources,
            only_past_hours=only_past,
            delete_source=delete_src,
        )
    else:
        res = compactor.compact_all(
            only_past_hours=only_past,
            delete_source=delete_src,
        )

    compacted = [r for r in res if r.get("status") == "COMPACTED"]
    logger.info(f"Finished. Total compacted hours: {len(compacted)}")

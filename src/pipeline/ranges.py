"""
Temporal Alignment & Multi-Source Data Range Inspector.

Audits available temporal coverage across all primary sources required for
Warsaw transit delay analytics and attribution:
1. Raw Transit Telemetry (Hive-partitioned Snappy Parquet in data/raw/)
2. IMGW Synoptic Weather Telemetry (data/processed/imgw_weather_hourly.parquet)
3. Static GTFS Schedule Archives (data/gtfs/)
4. OSMnx Road Corridor Topology (data/processed/osm_corridor_segments.parquet)
5. Assembled Feature Mart (data/processed/feature_mart.parquet)

Computes the intersecting common analytical window, flags data gaps or rebuild needs,
and provides actionable CLI recommendations and SQL filter clauses.
"""

import glob
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd

from src.core.config import paths

logger = logging.getLogger(__name__)

WARSAW_TZ = "Europe/Warsaw"


def validate_date_range(
    start_date: Optional[str] = None, end_date: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Validate and normalize start_date and end_date strings.
    Accepts 'YYYY-MM-DD' or ISO-like 'YYYY-MM-DD HH:MM:SS'.
    """
    if start_date is not None:
        start_date = start_date.strip()
        try:
            pd.to_datetime(start_date)
        except Exception as e:
            raise ValueError(
                f"Invalid start_date '{start_date}'. Must be YYYY-MM-DD or YYYY-MM-DD HH:MM:SS."
            ) from e

    if end_date is not None:
        end_date = end_date.strip()
        try:
            pd.to_datetime(end_date)
        except Exception as e:
            raise ValueError(
                f"Invalid end_date '{end_date}'. Must be YYYY-MM-DD or YYYY-MM-DD HH:MM:SS."
            ) from e

    if start_date and end_date:
        t_start = pd.to_datetime(start_date)
        t_end = pd.to_datetime(end_date)
        if t_start > t_end:
            raise ValueError(
                f"Invalid date range: start_date ({start_date}) cannot be after end_date ({end_date})."
            )

    return start_date, end_date


def build_date_filter_clause(
    column_expr: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    is_epoch_seconds: bool = False,
    timezone_name: str = WARSAW_TZ,
) -> str:
    """
    Build a DuckDB WHERE clause fragment filtering a timestamp or epoch column
    within start_date and end_date according to Warsaw local time.

    Args:
        column_expr: Column name or SQL expression (e.g. 'feed_timestamp' or 'arrival_ts').
        start_date: Start date string ('YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS').
        end_date: End date string ('YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS').
        is_epoch_seconds: True if column_expr is a unix epoch in seconds (e.g. feed_timestamp).
        timezone_name: Timezone name for date alignment (default Europe/Warsaw).

    Returns:
        SQL expression string suitable for inclusion in a WHERE clause (e.g. '1=1 AND ...').
    """
    start_date, end_date = validate_date_range(start_date, end_date)
    clauses: List[str] = []

    if is_epoch_seconds:
        ts_expr = f"to_timestamp({column_expr}) AT TIME ZONE '{timezone_name}'"
    else:
        # Check if already timestamp with time zone or timestamp
        ts_expr = f"{column_expr} AT TIME ZONE '{timezone_name}'"

    if start_date:
        if len(start_date) == 10:
            clauses.append(f"CAST({ts_expr} AS DATE) >= '{start_date}'::DATE")
        else:
            clauses.append(f"{ts_expr} >= '{start_date}'::TIMESTAMP")

    if end_date:
        if len(end_date) == 10:
            clauses.append(f"CAST({ts_expr} AS DATE) <= '{end_date}'::DATE")
        else:
            clauses.append(f"{ts_expr} <= '{end_date}'::TIMESTAMP")

    return " AND ".join(clauses) if clauses else "1=1"


def inspect_available_ranges(data_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Audit and inspect the available temporal ranges across all data sources.

    Returns a structured dictionary containing metadata, time bounds, record counts,
    and recommended overlapping study windows.
    """
    base_data_dir = data_dir or str(paths.data_dir)
    raw_pattern = os.path.join(base_data_dir, "raw", "**", "*.parquet")
    historian_pattern = os.path.join(base_data_dir, "historian", "gold", "stop_events", "date=*", "*.parquet")
    weather_path = os.path.join(base_data_dir, "processed", "imgw_weather_hourly.parquet")
    gtfs_dir = os.path.join(base_data_dir, "gtfs")
    osm_path = os.path.join(base_data_dir, "processed", "osm_corridor_segments.parquet")
    mart_path = os.path.join(base_data_dir, "processed", "feature_mart.parquet")
    mart_observed_path = os.path.join(base_data_dir, "historian", "processed", "feature_mart_observed.parquet")

    con = duckdb.connect(":memory:")

    report: Dict[str, Any] = {
        "timestamp_checked": datetime.now(timezone.utc).isoformat(),
        "sources": {},
        "common_overlap": {},
        "recommendations": {},
    }

    # 1. Raw Telemetry
    raw_files = glob.glob(raw_pattern, recursive=True)
    if raw_files:
        try:
            raw_stats = con.execute(f"""
                SELECT
                    to_timestamp(MIN(feed_timestamp)) AT TIME ZONE '{WARSAW_TZ}' as min_local,
                    to_timestamp(MAX(feed_timestamp)) AT TIME ZONE '{WARSAW_TZ}' as max_local,
                    to_timestamp(MIN(feed_timestamp)) AT TIME ZONE 'UTC' as min_utc,
                    to_timestamp(MAX(feed_timestamp)) AT TIME ZONE 'UTC' as max_utc,
                    COUNT(*) as total_rows
                FROM read_parquet('{raw_pattern}')
            """).fetchone()

            part_rows = con.execute(f"""
                SELECT DISTINCT printf('%04d-%02d-%02d', year, month, CAST(day AS INTEGER)) as p_date
                FROM read_parquet('{raw_pattern}')
                ORDER BY p_date
            """).fetchall()
            part_dates = [r[0] for r in part_rows if r[0] is not None]

            min_dt_local = str(raw_stats[0]) if raw_stats and raw_stats[0] else None
            max_dt_local = str(raw_stats[1]) if raw_stats and raw_stats[1] else None
            total_rows = int(raw_stats[4]) if raw_stats and raw_stats[4] is not None else 0

            report["sources"]["raw_telemetry"] = {
                "status": "AVAILABLE",
                "total_files": len(raw_files),
                "total_records": total_rows,
                "start_datetime_local": min_dt_local,
                "end_datetime_local": max_dt_local,
                "start_date": part_dates[0] if part_dates else None,
                "end_date": part_dates[-1] if part_dates else None,
                "partition_dates": part_dates,
                "path_pattern": raw_pattern,
            }
        except Exception as e:
            logger.warning(f"Error querying raw parquet: {e}")
            report["sources"]["raw_telemetry"] = {
                "status": "ERROR",
                "error": str(e),
                "total_files": len(raw_files),
            }
    else:
        report["sources"]["raw_telemetry"] = {
            "status": "MISSING",
            "total_files": 0,
            "total_records": 0,
            "guidance": "Run ./pull_from_gdrive.sh to pull daily raw partitions from Google Drive.",
        }

    # 1b. Historian Observed Ground-Truth Stop Events
    gold_files = glob.glob(historian_pattern)
    if gold_files:
        try:
            gold_stats = con.execute(f"""
                SELECT
                    to_timestamp(MIN(actual_arrival_utc)) AT TIME ZONE '{WARSAW_TZ}' as min_local,
                    to_timestamp(MAX(actual_arrival_utc)) AT TIME ZONE '{WARSAW_TZ}' as max_local,
                    COUNT(*) as total_rows,
                    COUNT(DISTINCT trip_id) as n_trips,
                    COUNT(DISTINCT route_id) as n_routes
                FROM read_parquet('{historian_pattern}')
            """).fetchone()

            part_dates = sorted(list({
                os.path.basename(os.path.dirname(f)).replace("date=", "")
                for f in gold_files if "date=" in f
            }))

            min_g_local = str(gold_stats[0]) if gold_stats and gold_stats[0] else None
            max_g_local = str(gold_stats[1]) if gold_stats and gold_stats[1] else None
            g_rows = int(gold_stats[2]) if gold_stats and gold_stats[2] is not None else 0

            report["sources"]["historian_gold"] = {
                "status": "AVAILABLE",
                "total_files": len(gold_files),
                "total_records": g_rows,
                "start_datetime_local": min_g_local,
                "end_datetime_local": max_g_local,
                "start_date": min_g_local[:10] if min_g_local else None,
                "end_date": max_g_local[:10] if max_g_local else None,
                "distinct_trips": int(gold_stats[3]) if gold_stats else 0,
                "distinct_routes": int(gold_stats[4]) if gold_stats else 0,
                "partition_dates": part_dates,
                "path_pattern": historian_pattern,
            }
        except Exception as e:
            logger.warning(f"Error querying historian gold parquet: {e}")
            report["sources"]["historian_gold"] = {
                "status": "ERROR",
                "error": str(e),
                "total_files": len(gold_files),
            }
    else:
        report["sources"]["historian_gold"] = {
            "status": "MISSING",
            "total_files": 0,
            "total_records": 0,
            "guidance": "Run ./pull_historian_from_gdrive.sh to sync Historian data from Google Drive.",
        }

    # 2. Weather Telemetry
    if os.path.exists(weather_path) and os.path.getsize(weather_path) > 0:
        try:
            w_stats = con.execute(f"""
                SELECT
                    MIN(timestamp) as min_ts,
                    MAX(timestamp) as max_ts,
                    COUNT(*) as total_rows,
                    COUNT(DISTINCT station_id) as n_stations
                FROM read_parquet('{weather_path}')
            """).fetchone()

            stations = [
                r[0]
                for r in con.execute(f"""
                SELECT DISTINCT station_name FROM read_parquet('{weather_path}') WHERE station_name IS NOT NULL
            """).fetchall()
            ]

            min_w_ts = str(w_stats[0]) if w_stats and w_stats[0] else None
            max_w_ts = str(w_stats[1]) if w_stats and w_stats[1] else None
            w_rows = int(w_stats[2]) if w_stats and w_stats[2] is not None else 0

            report["sources"]["weather"] = {
                "status": "AVAILABLE",
                "path": weather_path,
                "total_records": w_rows,
                "start_datetime": min_w_ts,
                "end_datetime": max_w_ts,
                "start_date": min_w_ts[:10] if min_w_ts else None,
                "end_date": max_w_ts[:10] if max_w_ts else None,
                "stations": stations,
            }
        except Exception as e:
            report["sources"]["weather"] = {
                "status": "ERROR",
                "error": str(e),
                "path": weather_path,
            }
    else:
        report["sources"]["weather"] = {
            "status": "MISSING",
            "path": weather_path,
            "guidance": "Run 'python run_pipeline.py --stage exogenous --force-rebuild' to harvest weather.",
        }

    # 3. Static GTFS Schedule
    gtfs_dirs = sorted([d for d in glob.glob(os.path.join(gtfs_dir, "*")) if os.path.isdir(d)])
    gtfs_dates: List[str] = []
    feed_versions: List[str] = []
    for d in gtfs_dirs:
        cal = os.path.join(d, "calendar_dates.txt")
        feed_info = os.path.join(d, "feed_info.txt")
        if os.path.exists(cal):
            try:
                df_cal = pd.read_csv(cal, dtype={"date": str})
                gtfs_dates.extend([str(df_cal["date"].min()), str(df_cal["date"].max())])
            except Exception:
                pass
        if os.path.exists(feed_info):
            try:
                df_info = pd.read_csv(feed_info)
                if "feed_version" in df_info.columns:
                    feed_versions.extend(df_info["feed_version"].astype(str).tolist())
            except Exception:
                pass

    if gtfs_dates:
        gtfs_dates_clean = [d for d in gtfs_dates if len(d) == 8 and d.isdigit()]
        if gtfs_dates_clean:
            min_g = min(gtfs_dates_clean)
            max_g = max(gtfs_dates_clean)
            min_g_fmt = f"{min_g[:4]}-{min_g[4:6]}-{min_g[6:8]}"
            max_g_fmt = f"{max_g[:4]}-{max_g[4:6]}-{max_g[6:8]}"
        else:
            min_g_fmt, max_g_fmt = None, None

        report["sources"]["static_gtfs"] = {
            "status": "AVAILABLE",
            "directory": gtfs_dir,
            "archive_packages": [os.path.basename(d) for d in gtfs_dirs],
            "start_date": min_g_fmt,
            "end_date": max_g_fmt,
            "feed_versions": feed_versions,
        }
    else:
        report["sources"]["static_gtfs"] = {
            "status": "MISSING",
            "directory": gtfs_dir,
            "guidance": "Run 'python run_pipeline.py --stage preflight' to download and sync static GTFS.",
        }

    # 4. OSMnx Corridor Topology
    if os.path.exists(osm_path) and os.path.getsize(osm_path) > 0:
        try:
            osm_stats = con.execute(f"""
                SELECT COUNT(*), COUNT(DISTINCT corridor_name)
                FROM read_parquet('{osm_path}')
            """).fetchone()
            corridors = [
                r[0]
                for r in con.execute(f"""
                SELECT DISTINCT corridor_name FROM read_parquet('{osm_path}') WHERE corridor_name IS NOT NULL
            """).fetchall()
            ]
            report["sources"]["osm_corridors"] = {
                "status": "AVAILABLE",
                "path": osm_path,
                "total_segments": int(osm_stats[0]) if osm_stats else 0,
                "distinct_corridors": int(osm_stats[1]) if osm_stats else 0,
                "corridor_names": corridors,
            }
        except Exception as e:
            report["sources"]["osm_corridors"] = {
                "status": "ERROR",
                "error": str(e),
                "path": osm_path,
            }
    else:
        report["sources"]["osm_corridors"] = {
            "status": "MISSING",
            "path": osm_path,
            "guidance": "Run 'python run_pipeline.py --stage exogenous' to extract corridor network topology.",
        }

    # 5. Feature Mart
    if os.path.exists(mart_path) and os.path.getsize(mart_path) > 0:
        try:
            m_stats = con.execute(f"""
                SELECT
                    MIN(arrival_ts) AT TIME ZONE '{WARSAW_TZ}' as min_local,
                    MAX(arrival_ts) AT TIME ZONE '{WARSAW_TZ}' as max_local,
                    COUNT(*) as total_rows,
                    COUNT(DISTINCT trip_id) as n_trips,
                    COUNT(DISTINCT route_id) as n_routes
                FROM read_parquet('{mart_path}')
            """).fetchone()

            min_m_dt = str(m_stats[0]) if m_stats and m_stats[0] else None
            max_m_dt = str(m_stats[1]) if m_stats and m_stats[1] else None
            m_rows = int(m_stats[2]) if m_stats and m_stats[2] is not None else 0

            report["sources"]["feature_mart"] = {
                "status": "AVAILABLE",
                "path": mart_path,
                "total_records": m_rows,
                "distinct_trips": int(m_stats[3]) if m_stats else 0,
                "distinct_routes": int(m_stats[4]) if m_stats else 0,
                "start_datetime_local": min_m_dt,
                "end_datetime_local": max_m_dt,
                "start_date": min_m_dt[:10] if min_m_dt else None,
                "end_date": max_m_dt[:10] if max_m_dt else None,
            }
        except Exception as e:
            report["sources"]["feature_mart"] = {
                "status": "ERROR",
                "error": str(e),
                "path": mart_path,
            }
    else:
        report["sources"]["feature_mart"] = {
            "status": "NOT_BUILT",
            "path": mart_path,
            "guidance": "Run 'python run_pipeline.py --stage fusion --datasource legacy' to assemble legacy feature mart.",
        }

    # 5b. Feature Mart Observed (Historian Ground Truth)
    if os.path.exists(mart_observed_path) and os.path.getsize(mart_observed_path) > 0:
        try:
            mo_stats = con.execute(f"""
                SELECT
                    MIN(arrival_ts) AT TIME ZONE '{WARSAW_TZ}' as min_local,
                    MAX(arrival_ts) AT TIME ZONE '{WARSAW_TZ}' as max_local,
                    COUNT(*) as total_rows,
                    COUNT(DISTINCT trip_id) as n_trips,
                    COUNT(DISTINCT route_id) as n_routes
                FROM read_parquet('{mart_observed_path}')
            """).fetchone()

            min_mo_dt = str(mo_stats[0]) if mo_stats and mo_stats[0] else None
            max_mo_dt = str(mo_stats[1]) if mo_stats and mo_stats[1] else None
            mo_rows = int(mo_stats[2]) if mo_stats and mo_stats[2] is not None else 0

            report["sources"]["feature_mart_observed"] = {
                "status": "AVAILABLE",
                "path": mart_observed_path,
                "total_records": mo_rows,
                "distinct_trips": int(mo_stats[3]) if mo_stats else 0,
                "distinct_routes": int(mo_stats[4]) if mo_stats else 0,
                "start_datetime_local": min_mo_dt,
                "end_datetime_local": max_mo_dt,
                "start_date": min_mo_dt[:10] if min_mo_dt else None,
                "end_date": max_mo_dt[:10] if max_mo_dt else None,
            }
        except Exception as e:
            report["sources"]["feature_mart_observed"] = {
                "status": "ERROR",
                "error": str(e),
                "path": mart_observed_path,
            }
    else:
        report["sources"]["feature_mart_observed"] = {
            "status": "NOT_BUILT",
            "path": mart_observed_path,
            "guidance": "Run 'python run_pipeline.py --stage fusion' to assemble observed feature mart.",
        }

    con.close()

    # 6. Compute Intersecting Analytical Window
    raw_info = report["sources"].get("raw_telemetry", {})
    weather_info = report["sources"].get("weather", {})
    gtfs_info = report["sources"].get("static_gtfs", {})

    raw_start = raw_info.get("start_date")
    raw_end = raw_info.get("end_date")
    weather_start = weather_info.get("start_date")
    weather_end = weather_info.get("end_date")
    gtfs_start = gtfs_info.get("start_date")
    gtfs_end = gtfs_info.get("end_date")

    starts = [d for d in [raw_start, weather_start, gtfs_start] if d is not None]
    ends = [d for d in [raw_end, weather_end, gtfs_end] if d is not None]

    if starts and ends:
        common_start = max(starts)
        common_end = min(ends)
    else:
        common_start, common_end = None, None

    # Check if raw extends beyond weather
    weather_lag_days = 0
    if raw_end and weather_end and raw_end > weather_end:
        try:
            d_raw = datetime.strptime(raw_end, "%Y-%m-%d")
            d_w = datetime.strptime(weather_end, "%Y-%m-%d")
            weather_lag_days = (d_raw - d_w).days
        except Exception:
            pass

    report["common_overlap"] = {
        "start_date": common_start,
        "end_date": common_end,
        "is_ready_for_analysis": (
            common_start is not None
            and common_end is not None
            and common_start <= common_end
            and report["sources"].get("osm_corridors", {}).get("status") == "AVAILABLE"
        ),
        "weather_lags_raw": weather_lag_days > 0,
        "weather_lag_days": weather_lag_days,
    }

    # Actionable recommendations
    if report["common_overlap"]["is_ready_for_analysis"]:
        # Find complete weekdays if available
        rec_start = common_start
        rec_end = common_end
        report["recommendations"] = {
            "full_range_command": f"python run_pipeline.py --stage all --start-date {common_start} --end-date {common_end}",
            "benchmark_command": f"python run_pipeline.py --stage benchmark --start-date {common_start} --end-date {common_end}",
            "recommended_start_date": rec_start,
            "recommended_end_date": rec_end,
        }
        if weather_lag_days > 0:
            report["recommendations"]["rebuild_weather_command"] = (
                f"python run_pipeline.py --stage exogenous --force-rebuild --start-date {common_start} --end-date {raw_end}"
            )
    else:
        report["recommendations"] = {
            "setup_command": "python run_pipeline.py --stage preflight && python run_pipeline.py --stage exogenous"
        }

    return report


def format_range_report(
    report: Optional[Dict[str, Any]] = None, data_dir: Optional[str] = None
) -> str:
    """Format the range inspection report into an elegant, publication-quality ASCII report."""
    if report is None:
        report = inspect_available_ranges(data_dir=data_dir)

    lines: List[str] = []
    sep = "=" * 80
    subsep = "-" * 80

    lines.append("\n" + sep)
    lines.append(" WARSAW TRANSIT TELEMETRY: DATA SOURCE RANGE & COHORT COVERAGE AUDIT ".center(80))
    lines.append(sep)

    # 1. Sources Table
    lines.append("\n[1] DATA SOURCE TEMPORAL AVAILABILITY:")
    lines.append(
        f"{'Source':<26} | {'Status':<11} | {'Start Date/Time':<20} | {'End Date/Time':<20} | {'Records/Files'}"
    )
    lines.append(subsep)

    sources = report.get("sources", {})

    # Historian Gold (Primary Ground Truth)
    hg = sources.get("historian_gold", {})
    hg_status = hg.get("status", "N/A")
    hg_start = hg.get("start_date", "-") or "-"
    hg_end = hg.get("end_date", "-") or "-"
    hg_rec = f"{hg.get('total_records', 0):,} stop events"
    lines.append(
        f"{'Historian Ground Truth':<26} | {hg_status:<11} | {hg_start:<20} | {hg_end:<20} | {hg_rec}"
    )

    # Observed Feature Mart (Primary)
    mo = sources.get("feature_mart_observed", {})
    mo_status = mo.get("status", "N/A")
    mo_start = mo.get("start_date", "-") or "-"
    mo_end = mo.get("end_date", "-") or "-"
    mo_rec = f"{mo.get('total_records', 0):,} fused obs"
    lines.append(
        f"{'Observed Feature Mart':<26} | {mo_status:<11} | {mo_start:<20} | {mo_end:<20} | {mo_rec}"
    )

    # Raw (Legacy Reference)
    raw = sources.get("raw_telemetry", {})
    raw_status = raw.get("status", "N/A")
    raw_start = raw.get("start_date", "-") or "-"
    raw_end = raw.get("end_date", "-") or "-"
    raw_rec = f"{raw.get('total_records', 0):,} rec ({raw.get('total_files', 0)} files)"
    lines.append(
        f"{'Raw Transit Telemetry (Leg)':<26} | {raw_status:<11} | {raw_start:<20} | {raw_end:<20} | {raw_rec}"
    )

    # Weather
    w = sources.get("weather", {})
    w_status = w.get("status", "N/A")
    w_start = w.get("start_date", "-") or "-"
    w_end = w.get("end_date", "-") or "-"
    w_rec = f"{w.get('total_records', 0):,} hours ({len(w.get('stations', []))} stations)"
    lines.append(
        f"{'IMGW Synoptic Weather':<26} | {w_status:<11} | {w_start:<20} | {w_end:<20} | {w_rec}"
    )

    # GTFS
    g = sources.get("static_gtfs", {})
    g_status = g.get("status", "N/A")
    g_start = g.get("start_date", "-") or "-"
    g_end = g.get("end_date", "-") or "-"
    g_rec = f"{len(g.get('archive_packages', []))} packages"
    lines.append(
        f"{'Static GTFS Schedule':<26} | {g_status:<11} | {g_start:<20} | {g_end:<20} | {g_rec}"
    )

    # OSM
    o = sources.get("osm_corridors", {})
    o_status = o.get("status", "N/A")
    o_rec = f"{o.get('total_segments', 0)} segments ({o.get('distinct_corridors', 0)} corridors)"
    lines.append(
        f"{'OSMnx Road Corridors':<26} | {o_status:<11} | {'(Static Network)':<20} | {'(Static Network)':<20} | {o_rec}"
    )

    # Legacy Feature Mart
    m = sources.get("feature_mart", {})
    m_status = m.get("status", "N/A")
    m_start = m.get("start_date", "-") or "-"
    m_end = m.get("end_date", "-") or "-"
    m_rec = f"{m.get('total_records', 0):,} fused obs"
    lines.append(
        f"{'Legacy Feature Mart':<26} | {m_status:<11} | {m_start:<20} | {m_end:<20} | {m_rec}"
    )

    # 2. Detailed Raw Telemetry Partitions
    lines.append("\n[2] RAW TELEMETRY PARTITIONS (DAILY PARQUET BATCHES):")
    p_dates = raw.get("partition_dates", [])
    if p_dates:
        chunks = [p_dates[i : i + 6] for i in range(0, len(p_dates), 6)]
        for chunk in chunks:
            lines.append("  " + " | ".join(chunk))
        if raw.get("start_datetime_local") and raw.get("end_datetime_local"):
            lines.append(
                f"  Exact Time Bounds (Warsaw Local): {raw.get('start_datetime_local')} -> {raw.get('end_datetime_local')}"
            )
    else:
        lines.append("  No raw partitions discovered in data/raw/.")

    # 3. Common Overlap & Analysis Window
    overlap = report.get("common_overlap", {})
    c_start = overlap.get("start_date")
    c_end = overlap.get("end_date")
    is_ready = overlap.get("is_ready_for_analysis", False)

    lines.append("\n[3] INTERSECTING ANALYTICAL COHORT WINDOW:")
    if is_ready:
        lines.append(f"  Maximum Concurrent Window:   {c_start} to {c_end}")
        if overlap.get("weather_lags_raw"):
            lines.append(
                f"  [!] Note: Raw telemetry extends to {raw.get('end_date')}, but IMGW weather stops at {w.get('end_date')}."
            )
            lines.append(
                f"            To include {raw.get('end_date')}, rebuild weather with: {report.get('recommendations', {}).get('rebuild_weather_command')}"
            )
    else:
        lines.append(
            "  [!] Common overlap could not be formed. One or more core sources are missing or disjoint."
        )

    # 4. Actionable Pipeline Commands
    lines.append("\n[4] ACTIONABLE PIPELINE RUN COMMANDS:")
    recs = report.get("recommendations", {})
    if "full_range_command" in recs:
        lines.append("  1. Run full pipeline on maximum available data range:")
        lines.append(f"     {recs['full_range_command']}")
        lines.append("  2. Select custom sub-window (e.g. workweek only):")
        lines.append(
            "     python run_pipeline.py --stage all --start-date 2026-10-05 --end-date 2026-10-09"
        )
        lines.append("  3. Benchmark models on selected sub-window:")
        lines.append(f"     {recs.get('benchmark_command')}")
    else:
        lines.append("  Initialize missing sources:")
        lines.append(f"     {recs.get('setup_command')}")

    lines.append("\n" + sep + "\n")
    return "\n".join(lines)


def print_range_report(
    report: Optional[Dict[str, Any]] = None, data_dir: Optional[str] = None
) -> None:
    """Print the formatted range audit directly to standard output."""
    output = format_range_report(report=report, data_dir=data_dir)
    print(output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print_range_report()

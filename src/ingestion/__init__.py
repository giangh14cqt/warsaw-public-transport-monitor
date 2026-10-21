"""
Warsaw Public Transport Ingestion Subsystem (Track A: Frozen Reference Baseline).

STATUS: FROZEN / COMPARATIVE BENCHMARK BASELINE
This module handles real-time GTFS-RT Protobuf polling, static timetable downloads,
deduplication, and Parquet storage under data/raw/.

It is intentionally maintained and frozen for:
1. Academic benchmarking: Comparing agency-estimated GTFS-RT delays against
   true observed ground truth from the Historian engine (Track B).
2. Historical access: Querying telemetry collected prior to the Historian launch.
3. Server stability: Preserving 100% backward compatibility for continuous remote daemons.

For active ground-truth delay analysis, use the Historian engine (src/historian/).
"""

from src.ingestion.fetcher import (
    CriticalSchemaMismatchError,
    GTFSRTFetcher,
    NetworkOutageError,
    RateLimitCircuitBreakerError,
    TransientHttpError,
)
from src.ingestion.gtfs_manager import GTFSManager
from src.ingestion.storage import DuckDBCatalog, ParquetPartitionStorage

__all__ = [
    "GTFSRTFetcher",
    "CriticalSchemaMismatchError",
    "RateLimitCircuitBreakerError",
    "NetworkOutageError",
    "TransientHttpError",
    "GTFSManager",
    "ParquetPartitionStorage",
    "DuckDBCatalog",
]

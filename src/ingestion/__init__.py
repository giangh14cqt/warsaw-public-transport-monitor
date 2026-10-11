"""
Warsaw Public Transport Ingestion Subsystem.
Handles GTFS-RT Protobuf polling, static timetable downloads, deduplication, and Parquet storage.
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

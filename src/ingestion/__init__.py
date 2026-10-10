"""
Warsaw Public Transport Ingestion Subsystem.
Handles GTFS-RT Protobuf polling, static timetable downloads, deduplication, and Parquet storage.
"""

from src.ingestion.fetcher import (
    GTFSRTFetcher,
    CriticalSchemaMismatchError,
    RateLimitCircuitBreakerError,
    NetworkOutageError,
    TransientHttpError,
)
from src.ingestion.gtfs_manager import GTFSManager
from src.ingestion.storage import ParquetPartitionStorage, DuckDBCatalog

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

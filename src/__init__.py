"""
Warsaw Transit Delay Attribution & Ingestion Engine Package.
Supports both legacy root imports and modern domain packages:
- src.ingestion
- src.exogenous
- src.fusion
- src.models
- src.xai
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

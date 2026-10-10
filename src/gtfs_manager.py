"""
Backwards-compatible proxy for src.ingestion.gtfs_manager.
Ensures zero-downtime remote deployments when repository code is updated.
"""
from src.ingestion.gtfs_manager import *  # noqa: F401, F403

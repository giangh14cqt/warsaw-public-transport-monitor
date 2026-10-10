"""
Backwards-compatible proxy for src.ingestion.fetcher.
Ensures zero-downtime remote deployments when repository code is updated.
"""
from src.ingestion.fetcher import *  # noqa: F401, F403

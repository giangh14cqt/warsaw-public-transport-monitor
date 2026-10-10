"""
Backwards-compatible proxy for src.ingestion.storage.
Ensures zero-downtime remote deployments when repository code is updated.
"""
from src.ingestion.storage import *  # noqa: F401, F403

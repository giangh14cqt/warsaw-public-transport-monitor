"""
Backwards-compatible proxy for src.ingestion.db.
Ensures zero-downtime remote deployments when repository code is updated.
"""
from src.ingestion.db import *  # noqa: F401, F403

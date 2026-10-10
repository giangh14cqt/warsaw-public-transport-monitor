"""
Backwards-compatible proxy for src.ingestion.parser.
Ensures zero-downtime remote deployments when repository code is updated.
"""
from src.ingestion.parser import *  # noqa: F401, F403

"""
Warsaw Transit Telemetry & Attribution System: Core Module.
Provides centralized configuration, project paths, feature registries, and domain constants.
"""

from src.core.config import DataPaths, FeatureRegistry, IngestionConfig, features, paths
from src.core.constants import DEFAULT_CORRIDORS, IMGW_STATIONS, TRANSIT_COHORTS
from src.core.stop_mapper import build_canonical_stop_mapping

__all__ = [
    "DataPaths",
    "FeatureRegistry",
    "IngestionConfig",
    "paths",
    "features",
    "DEFAULT_CORRIDORS",
    "TRANSIT_COHORTS",
    "IMGW_STATIONS",
    "build_canonical_stop_mapping",
]

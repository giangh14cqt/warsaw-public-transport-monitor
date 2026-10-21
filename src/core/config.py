"""
Warsaw Transit Telemetry & Attribution System: Centralized Configuration.
Provides immutable dataclasses for project paths, ingestion parameters, and feature registries.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# Anchor project root dynamically relative to this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class DataPaths:
    """Canonical path registry for datasets, archives, logs, and artifacts."""

    root: Path = PROJECT_ROOT
    data_dir: Path = root / "data"
    raw_dir: Path = data_dir / "raw"
    raw_pattern: str = "data/raw/**/*.parquet"
    gtfs_dir: Path = data_dir / "gtfs"
    processed_dir: Path = data_dir / "processed"

    # Historian Storage & Processed Datasets
    historian_dir: Path = data_dir / "historian"
    historian_gold_pattern: str = "data/historian/gold/stop_events/date=*/stop_events.parquet"
    feature_mart_observed: Path = data_dir / "historian" / "processed" / "feature_mart_observed.parquet"

    # Processed Datasets
    feature_mart: Path = processed_dir / "feature_mart.parquet"
    weather_hourly: Path = processed_dir / "imgw_weather_hourly.parquet"
    osm_corridors: Path = processed_dir / "osm_corridor_segments.parquet"
    calendar_lookup: Path = processed_dir / "calendar_lookup.parquet"
    stop_interchanges: Path = processed_dir / "transit_stop_interchanges.parquet"

    # Output Artifacts & Exhibits
    reports_dir: Path = root / "reports"
    figures_xai: Path = reports_dir / "figures" / "xai"
    tables_dir: Path = reports_dir / "tables"
    logs_dir: Path = root / "logs"

    def get_feature_mart_str(self, datasource: str = "historian") -> str:
        if datasource == "legacy":
            return str(self.feature_mart)
        return str(self.feature_mart_observed)

    def get_raw_pattern(self) -> str:
        return self.raw_pattern


@dataclass(frozen=True)
class IngestionConfig:
    """Runtime thresholds and remote endpoints for telemetry collection."""

    feed_url: str = os.getenv(
        "WARSAW_GTFS_RT_URL",
        os.getenv(
            "FEED_URL",
            "https://api.zbiorkom.live/api6-open/warsaw/gtfsRealtime/default/vehiclePositions,tripUpdates,serviceAlerts",
        ),
    )
    static_gtfs_url: str = os.getenv(
        "WARSAW_STATIC_GTFS_URL",
        os.getenv(
            "STATIC_GTFS_URL",
            "https://api.zbiorkom.live/api6-open/warsaw/gtfs/default",
        ),
    )
    poll_interval_seconds: int = 30
    flush_interval_seconds: int = 300  # 5 minutes
    flush_record_threshold: int = 5000  # 5,000 records
    health_report_interval_seconds: int = 900  # 15 minutes
    weekly_check_interval_seconds: int = 86400  # 24 hours


@dataclass(frozen=True)
class FeatureRegistry:
    """Canonical taxonomy of engineered transit, infrastructure, temporal, and weather features."""

    operational: List[str] = field(
        default_factory=lambda: [
            "prev_stop_delay",
            "prev2_stop_delay",
            "trip_progress",
            "headway_deviation",
            "headway_actual_s",
            "is_origin_stop",
            "is_tram",
        ]
    )

    infrastructure: List[str] = field(
        default_factory=lambda: [
            "signalized_intersection_count",
            "segment_length_meters",
            "actual_path_length_meters",
            "tortuosity_ratio",
            "interchange_line_count",
            "is_metro_proximity",
            "is_dedicated_right_of_way",
            "lane_capacity",
            "is_study_corridor",
        ]
    )

    temporal: List[str] = field(
        default_factory=lambda: [
            "hour_of_day",
            "day_of_week",
            "is_peak_hour",
            "is_morning_peak",
            "is_evening_peak",
            "is_weekend",
            "is_school_day",
            "is_public_holiday",
        ]
    )

    meteorological: List[str] = field(
        default_factory=lambda: [
            "precipitation_mm",
            "temperature_c",
            "relative_humidity",
            "wind_speed_ms",
            "freezing_rain_flag",
        ]
    )

    actionable: List[str] = field(
        default_factory=lambda: [
            "prev_stop_delay",
            "headway_deviation",
            "is_dedicated_right_of_way",
            "signalized_intersection_count",
        ]
    )

    immutable: List[str] = field(
        default_factory=lambda: [
            "hour_of_day",
            "day_of_week",
            "is_peak_hour",
            "is_school_day",
            "is_public_holiday",
            "segment_length_meters",
            "actual_path_length_meters",
            "tortuosity_ratio",
            "interchange_line_count",
            "is_metro_proximity",
            "lane_capacity",
            "is_tram",
            "precipitation_mm",
            "temperature_c",
            "relative_humidity",
            "wind_speed_ms",
            "freezing_rain_flag",
        ]
    )

    benchmark_features: List[str] = field(
        default_factory=lambda: [
            "prev_stop_delay",
            "signalized_intersection_count",
            "segment_length_meters",
            "actual_path_length_meters",
            "tortuosity_ratio",
            "interchange_line_count",
            "is_metro_proximity",
            "is_dedicated_right_of_way",
            "trip_progress",
            "headway_deviation",
            "hour_of_day",
            "day_of_week",
            "is_peak_hour",
            "is_school_day",
            "is_public_holiday",
            "is_tram",
            "lane_capacity",
            "precipitation_mm",
            "temperature_c",
        ]
    )

    feature_groups: Dict[str, str] = field(
        default_factory=lambda: {
            # Operational features
            "prev_stop_delay": "Operational Delay Propagation",
            "prev2_stop_delay": "Operational Delay Propagation",
            "trip_progress": "Operational Schedule Profile",
            "headway_deviation": "Operational Dispatch Regularity",
            "headway_actual_s": "Operational Dispatch Regularity",
            "is_origin_stop": "Operational Schedule Profile",
            "is_tram": "Operational Fleet Type",
            # Infrastructure & Roadway Topology
            "signalized_intersection_count": "Infrastructure Bottlenecks",
            "segment_length_meters": "Infrastructure Geometry",
            "actual_path_length_meters": "Infrastructure Geometry",
            "tortuosity_ratio": "Infrastructure Geometry",
            "interchange_line_count": "Infrastructure Bottlenecks",
            "is_metro_proximity": "Infrastructure Bottlenecks",
            "is_dedicated_right_of_way": "Infrastructure Right-of-Way",
            "lane_capacity": "Infrastructure Geometry",
            "is_study_corridor": "Infrastructure Corridor",
            # Temporal Dynamics
            "hour_of_day": "Temporal Congestion Peaks",
            "day_of_week": "Temporal Day-to-Day",
            "is_peak_hour": "Temporal Congestion Peaks",
            "is_morning_peak": "Temporal Congestion Peaks",
            "is_evening_peak": "Temporal Congestion Peaks",
            "is_weekend": "Temporal Day-to-Day",
            "is_school_day": "Temporal Day-to-Day",
            "is_public_holiday": "Temporal Day-to-Day",
            # Meteorological Telemetry
            "precipitation_mm": "Meteorological Conditions",
            "temperature_c": "Meteorological Conditions",
            "relative_humidity": "Meteorological Conditions",
            "wind_speed_ms": "Meteorological Conditions",
            "freezing_rain_flag": "Meteorological Conditions",
        }
    )


# Canonical singletons for easy import across modules
paths = DataPaths()
features = FeatureRegistry()
ingestion_config = IngestionConfig()

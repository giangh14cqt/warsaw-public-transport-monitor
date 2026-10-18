"""
Warsaw Observed-Delay Historian: Configuration and Settings.
Immutable dataclasses defining paths, endpoints, and physical tolerances.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WARSAW_TZ = ZoneInfo("Europe/Warsaw")


@dataclass(frozen=True)
class HistorianPaths:
    """Canonical path registry for Historian storage tiers."""

    root: Path = PROJECT_ROOT
    base_dir: Path = root / os.getenv("HISTORIAN_DATA_DIR", "data/historian")

    # Storage tiers
    bronze_dir: Path = field(init=False)
    silver_dir: Path = field(init=False)
    gold_dir: Path = field(init=False)
    gtfs_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "bronze_dir", self.base_dir / "bronze")
        object.__setattr__(self, "silver_dir", self.base_dir / "silver")
        object.__setattr__(self, "gold_dir", self.base_dir / "gold")
        object.__setattr__(self, "gtfs_dir", self.base_dir / "gtfs")
        object.__setattr__(self, "processed_dir", self.base_dir / "processed")

    def ensure_directories(self) -> None:
        """Create storage root directories if they do not exist."""
        for path in (self.bronze_dir, self.silver_dir, self.gold_dir, self.gtfs_dir, self.processed_dir):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class HistorianConfig:
    """Runtime configuration for high-frequency capture and stop-event matching."""

    # 1. Endpoints & Authentication
    # New official endpoint on dane.um.warszawa.pl (replaces legacy api.um.warszawa.pl)
    um_api_url: str = os.getenv(
        "UM_WARSAW_API_URL",
        "https://dane.um.warszawa.pl/api/action/get_ztm_lokalizacja_pojazdow",
    )
    um_api_key: str = field(
        default_factory=lambda: os.getenv("UM_WARSAW_API_KEY", "").strip()
    )
    # Legacy fallback endpoint
    um_legacy_url: str = "https://api.um.warszawa.pl/api/action/busestrams_get/"
    um_resource_id: str = "f2e5503e-927d-4ad3-9500-4ab9e55deb59"

    zbk_feed_url: str = field(
        default_factory=lambda: os.getenv(
            "WARSAW_GTFS_RT_URL",
            os.getenv(
                "FEED_URL",
                "https://api.zbiorkom.live/api6-open/warsaw/gtfsRealtime/default/vehiclePositions,tripUpdates,serviceAlerts",
            ),
        )
    )
    zbk_static_gtfs_url: str = field(
        default_factory=lambda: os.getenv(
            "WARSAW_STATIC_GTFS_URL",
            os.getenv(
                "STATIC_GTFS_URL",
                "https://api.zbiorkom.live/api6-open/warsaw/gtfs/default",
            ),
        )
    )

    # 2. Ingestion Cadence & Deadlines
    poll_period_seconds: int = int(os.getenv("HISTORIAN_POLL_PERIOD_S", "10"))
    zbk_poll_period_seconds: int = int(os.getenv("HISTORIAN_ZBK_POLL_PERIOD_S", "30"))  # Native 30s cadence for zbiorkom
    request_deadline_seconds: float = 8.0
    max_start_lag_seconds: float = 1.0

    # 3. Observability
    heartbeat_url: str = field(
        default_factory=lambda: os.getenv("HISTORIAN_HEARTBEAT_URL", "").strip()
    )
    health_interval_seconds: int = int(os.getenv("HISTORIAN_HEALTH_INTERVAL_S", "300"))  # 5 minutes

    # 4. Physical Reconstruction Tolerances
    stop_zone_entry_meters: float = 25.0
    stop_zone_exit_meters: float = 25.0
    lateral_tolerance_meters: float = 60.0
    max_stale_fix_seconds: float = 300.0  # 5 minutes
    max_gap_seconds: float = 90.0

    # Maximum plausible kinematic speeds (m/s)
    v_max_bus_ms: float = 25.0   # ~90 km/h
    v_max_tram_ms: float = 20.0  # ~72 km/h

    # Paths
    paths: HistorianPaths = field(default_factory=HistorianPaths)


# Singleton instance for convenient import
config = HistorianConfig()
paths = config.paths

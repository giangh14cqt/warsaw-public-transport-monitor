"""
Exogenous Feature Ingestion: IMGW-PIB Hourly Synoptic Weather Telemetry.
Pulls meteorological observations for Warsaw synoptic stations (Okęcie - ID 12375 / Bielany).
Stores observations in DuckDB / Parquet for downstream spatial-temporal fusion.
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import requests
import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

IMGW_LIVE_SYNOP_URL = "https://danepubliczne.imgw.pl/api/data/synop"
WARSAW_OKECIE_STATION_ID = "12375"  # WMO ID for Warszawa-Okęcie


class IMGWWeatherHarvester:
    """Collects and standardizes hourly meteorological telemetry from IMGW-PIB."""

    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.output_parquet = os.path.join(self.data_dir, "imgw_weather_hourly.parquet")

    def fetch_live_synop(self, station_id: str = WARSAW_OKECIE_STATION_ID) -> Optional[Dict[str, Any]]:
        """Fetch current synoptic observation from IMGW API."""
        try:
            url = f"{IMGW_LIVE_SYNOP_URL}/id/{station_id}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if "id_stacji" in data:
                    return self._clean_record(data)
            logger.warning(f"IMGW station {station_id} returned status {resp.status_code}")
        except Exception as e:
            logger.error(f"Failed to fetch IMGW observation: {e}")
        return None

    def fetch_all_warsaw_stations(self) -> List[Dict[str, Any]]:
        """Fetch all Warsaw synoptic stations from live synop feed."""
        records = []
        try:
            resp = requests.get(IMGW_LIVE_SYNOP_URL, timeout=10)
            if resp.status_code == 200:
                for item in resp.json():
                    station_name = item.get("stacja", "").lower()
                    if "warszawa" in station_name:
                        cleaned = self._clean_record(item)
                        if cleaned:
                            records.append(cleaned)
        except Exception as e:
            logger.error(f"Failed to fetch synop feed: {e}")
        return records

    def _clean_record(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Clean and typecast raw IMGW payload into typed schema."""
        try:
            date_str = raw.get("data_pomiaru", "")
            hour_str = raw.get("godzina_pomiaru", "").zfill(2)
            timestamp_str = f"{date_str} {hour_str}:00:00"
            ts = pd.to_datetime(timestamp_str)

            temp = float(raw["temperatura"]) if raw.get("temperatura") is not None else None
            humidity = float(raw["wilgotnosc_wzgledna"]) if raw.get("wilgotnosc_wzgledna") is not None else None
            precip = float(raw["suma_opadu"]) if raw.get("suma_opadu") is not None else 0.0
            wind_speed = float(raw["predkosc_wiatru"]) if raw.get("predkosc_wiatru") is not None else None
            pressure = float(raw["cisnienie"]) if raw.get("cisnienie") is not None else None

            # Derived freezing rain flag
            freezing_rain = bool(precip > 0.0 and temp is not None and temp <= 0.0)

            return {
                "station_id": str(raw.get("id_stacji")),
                "station_name": str(raw.get("stacja")),
                "timestamp": ts,
                "timestamp_bucket": ts.floor("h"),
                "temperature_c": temp,
                "relative_humidity": humidity,
                "precipitation_mm": precip,
                "wind_speed_ms": wind_speed,
                "pressure_hpa": pressure,
                "freezing_rain_flag": freezing_rain,
            }
        except Exception as e:
            logger.warning(f"Error parsing IMGW record {raw}: {e}")
            return None

    def upsert_to_parquet(self, records: List[Dict[str, Any]]) -> int:
        """Upsert records into the local weather mart Parquet store."""
        if not records:
            return 0
        new_df = pd.DataFrame(records)

        if os.path.exists(self.output_parquet):
            existing_df = pd.read_parquet(self.output_parquet)
            combined = pd.concat([existing_df, new_df]).drop_duplicates(
                subset=["station_id", "timestamp_bucket"], keep="last"
            )
        else:
            combined = new_df

        combined.sort_values(by=["station_id", "timestamp"], inplace=True)
        combined.to_parquet(self.output_parquet, index=False, compression="snappy")
        logger.info(f"Persisted {len(combined)} weather observations to {self.output_parquet}")
        return len(records)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    harvester = IMGWWeatherHarvester()
    warsaw_obs = harvester.fetch_all_warsaw_stations()
    count = harvester.upsert_to_parquet(warsaw_obs)
    print(f"Successfully processed {count} IMGW observations.")

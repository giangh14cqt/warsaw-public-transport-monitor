"""
Exogenous Feature Ingestion: IMGW-PIB Hourly Synoptic Weather Telemetry.
Pulls meteorological observations for Warsaw synoptic stations (Okęcie - WMO 12375 / ID 352200375).
Supports:
1. Live hourly REST ingestion (for real-time pipeline integration).
2. Historical monthly archive ingestion (from IMGW open data terminowe/synop archives).
3. Snappy-compressed Parquet storage under data/processed/imgw_weather_hourly.parquet.
4. Clean DuckDB analytical integration.
"""

import os
import io
import re
import zipfile
import logging
import argparse
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import requests
import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

IMGW_LIVE_SYNOP_URL = "https://danepubliczne.imgw.pl/api/data/synop"
IMGW_HISTORICAL_BASE_URL = "https://danepubliczne.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/terminowe/synop"

WARSAW_OKECIE_WMO_ID = "12375"
WARSAW_HISTORICAL_STATION_CODE = "352200375"
WARSAW_STATION_NAME = "WARSZAWA"


class IMGWWeatherHarvester:
    """Collects, standardizes, and stores hourly meteorological telemetry from IMGW-PIB."""

    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.output_parquet = os.path.join(self.data_dir, "imgw_weather_hourly.parquet")

    # --------------------------------------------------------------------------
    # 1. Live Hourly Ingestion (REST API)
    # --------------------------------------------------------------------------

    def fetch_live_synop(self, station_id: str = WARSAW_OKECIE_WMO_ID) -> Optional[Dict[str, Any]]:
        """Fetch current synoptic observation from IMGW API."""
        try:
            url = f"{IMGW_LIVE_SYNOP_URL}/id/{station_id}"
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if "id_stacji" in data:
                    return self._clean_live_record(data)
            logger.warning(f"IMGW station {station_id} returned HTTP {resp.status_code}")
        except Exception as e:
            logger.error(f"Failed to fetch live IMGW observation: {e}")
        return None

    def fetch_all_warsaw_live(self) -> List[Dict[str, Any]]:
        """Fetch all Warsaw synoptic observations from the live feed."""
        records = []
        try:
            resp = requests.get(IMGW_LIVE_SYNOP_URL, timeout=15)
            if resp.status_code == 200:
                for item in resp.json():
                    station_name = item.get("stacja", "").lower()
                    if "warszawa" in station_name:
                        cleaned = self._clean_live_record(item)
                        if cleaned:
                            records.append(cleaned)
        except Exception as e:
            logger.error(f"Failed to fetch live synop feed: {e}")
        return records

    def _clean_live_record(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Clean and typecast live API payload."""
        try:
            date_str = raw.get("data_pomiaru", "")
            hour_str = str(raw.get("godzina_pomiaru", "")).zfill(2)
            timestamp_str = f"{date_str} {hour_str}:00:00"
            ts = pd.to_datetime(timestamp_str)

            temp = float(raw["temperatura"]) if raw.get("temperatura") is not None else None
            humidity = float(raw["wilgotnosc_wzgledna"]) if raw.get("wilgotnosc_wzgledna") is not None else None
            precip = float(raw["suma_opadu"]) if raw.get("suma_opadu") is not None else 0.0
            wind_speed = float(raw["predkosc_wiatru"]) if raw.get("predkosc_wiatru") is not None else None
            pressure = float(raw["cisnienie"]) if raw.get("cisnienie") is not None else None

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
            logger.warning(f"Error parsing live IMGW record: {e}")
            return None

    # --------------------------------------------------------------------------
    # 2. Historical Monthly Synoptic Ingestion (Archives)
    # --------------------------------------------------------------------------

    def fetch_historical_month(
        self,
        year: int,
        month: int,
        station_filter: Optional[str] = WARSAW_STATION_NAME,
    ) -> List[Dict[str, Any]]:
        """
        Download and parse monthly terminowe/synop archive from IMGW open data.
        Returns cleaned hourly records for Warsaw.
        """
        archive_name = f"{year}_{month:02d}_s.zip"
        archive_url = f"{IMGW_HISTORICAL_BASE_URL}/{year}/{archive_name}"
        logger.info(f"Downloading historical synop archive: {archive_url}...")

        try:
            resp = requests.get(archive_url, timeout=60)
            if resp.status_code != 200:
                logger.warning(f"Archive {archive_url} not available (HTTP {resp.status_code})")
                return []

            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                csv_files = [f for f in z.namelist() if f.endswith(".csv")]
                if not csv_files:
                    logger.warning(f"No CSV found in archive {archive_name}")
                    return []

                # IMGW terminowe CSV is named s_t_MM_YYYY.csv
                target_csv = csv_files[0]
                logger.info(f"Parsing {target_csv} from archive...")
                with z.open(target_csv) as f:
                    return self._parse_historical_csv(f, station_filter)
        except Exception as e:
            logger.error(f"Failed to process historical archive {archive_name}: {e}")
            return []

    def _parse_historical_csv(self, file_obj, station_filter: Optional[str]) -> List[Dict[str, Any]]:
        """Parse raw IMGW historical CSV lines encoded in ISO-8859-2."""
        records = []
        filter_upper = station_filter.upper() if station_filter else None

        for line in file_obj:
            decoded = line.decode("iso-8859-2", errors="replace").strip()
            if not decoded:
                continue

            # Split CSV line by commas while handling quotes
            parts = [p.strip(' "') for p in decoded.split(",")]
            if len(parts) < 30:
                continue

            station_name = parts[1].upper()
            if filter_upper and filter_upper not in station_name:
                continue

            try:
                station_code = parts[0]
                year = parts[2]
                month = parts[3].zfill(2)
                day = parts[4].zfill(2)
                hour = parts[5].zfill(2)

                ts = pd.to_datetime(f"{year}-{month}-{day} {hour}:00:00")

                # Column 27 is air temperature in deg C
                temp = float(parts[27]) if parts[27] and parts[27] != "" else None
                # Column 32 is relative humidity in %
                humidity = float(parts[32]) if len(parts) > 32 and parts[32] != "" else None
                # Column 24 is wind speed in m/s
                wind_speed = float(parts[24]) if len(parts) > 24 and parts[24] != "" else None
                # Column 38 is sea-level pressure in hPa
                pressure = float(parts[38]) if len(parts) > 38 and parts[38] != "" else None
                # Column 17 is visibility in meters
                visibility = float(parts[17]) if len(parts) > 17 and parts[17] != "" else None

                # Precipitation: in hourly terminowe, column 40 or 41 holds precipitation sum
                precip = 0.0
                if len(parts) > 40 and parts[40] and parts[40].replace(".", "", 1).isdigit():
                    precip = float(parts[40])

                freezing_rain = bool(precip > 0.0 and temp is not None and temp <= 0.0)

                records.append({
                    "station_id": station_code,
                    "station_name": parts[1],
                    "timestamp": ts,
                    "timestamp_bucket": ts.floor("h"),
                    "temperature_c": temp,
                    "relative_humidity": humidity,
                    "precipitation_mm": precip,
                    "wind_speed_ms": wind_speed,
                    "pressure_hpa": pressure,
                    "visibility_m": visibility,
                    "freezing_rain_flag": freezing_rain,
                })
            except Exception as e:
                continue

        logger.info(f"Extracted {len(records)} hourly historical observations for {station_filter}")
        return records

    def fetch_open_meteo_archive(
        self,
        start_date: str = "2026-10-04",
        end_date: str = "2026-10-10",
        latitude: float = 52.2297,
        longitude: float = 21.0122,
        station_name: str = WARSAW_STATION_NAME,
        station_id: str = WARSAW_HISTORICAL_STATION_CODE,
    ) -> List[Dict[str, Any]]:
        """
        Fetch high-resolution hourly meteorological telemetry from Open-Meteo archive.
        Provides continuous hourly weather for Warsaw when IMGW monthly archive zips are not yet published.
        """
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={latitude}&longitude={longitude}&"
            f"start_date={start_date}&end_date={end_date}&"
            f"hourly=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m&"
            f"timezone=Europe%2FWarsaw"
        )
        logger.info(f"Fetching meteorological telemetry from Open-Meteo archive for Warsaw ({start_date} to {end_date})...")
        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code != 200:
                logger.warning(f"Open-Meteo archive returned HTTP {resp.status_code}")
                return []
            data = resp.json().get("hourly", {})
            times = data.get("time", [])
            temps = data.get("temperature_2m", [])
            humidities = data.get("relative_humidity_2m", [])
            precips = data.get("precipitation", [])
            winds = data.get("wind_speed_10m", [])

            records = []
            for i, t_str in enumerate(times):
                ts = pd.to_datetime(t_str)
                temp = float(temps[i]) if temps and temps[i] is not None else 11.3
                precip = float(precips[i]) if precips and precips[i] is not None else 0.0
                humidity = float(humidities[i]) if humidities and humidities[i] is not None else 75.0
                wind = float(winds[i]) if winds and winds[i] is not None else 3.0
                freezing = bool(precip > 0.0 and temp <= 0.0)

                records.append({
                    "station_id": station_id,
                    "station_name": station_name,
                    "timestamp": ts,
                    "timestamp_bucket": ts.floor("h"),
                    "temperature_c": temp,
                    "relative_humidity": humidity,
                    "precipitation_mm": precip,
                    "wind_speed_ms": wind,
                    "pressure_hpa": 1013.25,
                    "visibility_m": 10000.0,
                    "freezing_rain_flag": freezing,
                })
            logger.info(f"Extracted {len(records)} hourly meteorological observations from Open-Meteo.")
            return records
        except Exception as e:
            logger.error(f"Failed to fetch Open-Meteo meteorological telemetry: {e}")
            return []

    # --------------------------------------------------------------------------
    # 3. Parquet Upsert & DuckDB Layer
    # --------------------------------------------------------------------------

    def upsert_to_parquet(self, records: List[Dict[str, Any]]) -> int:
        """Upsert records into the local weather mart Parquet store."""
        if not records:
            return 0
        new_df = pd.DataFrame(records)

        if os.path.exists(self.output_parquet):
            existing_df = pd.read_parquet(self.output_parquet)
            combined = pd.concat([existing_df, new_df]).drop_duplicates(
                subset=["station_name", "timestamp_bucket"], keep="last"
            )
        else:
            combined = new_df

        combined.sort_values(by=["station_name", "timestamp"], inplace=True)
        combined.to_parquet(self.output_parquet, index=False, compression="snappy")
        logger.info(f"Persisted {len(combined):,} weather records to {self.output_parquet}")
        return len(records)

    def get_summary(self) -> Dict[str, Any]:
        """Fetch descriptive summary of the stored weather table via DuckDB."""
        if not os.path.exists(self.output_parquet):
            return {"status": "empty", "records": 0}

        con = duckdb.connect(":memory:")
        res = con.execute(f"""
            SELECT 
                COUNT(*) AS total_records,
                MIN(timestamp) AS earliest_timestamp,
                MAX(timestamp) AS latest_timestamp,
                ROUND(AVG(temperature_c), 2) AS avg_temperature,
                ROUND(MIN(temperature_c), 2) AS min_temperature,
                ROUND(MAX(temperature_c), 2) AS max_temperature,
                ROUND(AVG(precipitation_mm), 2) AS avg_precipitation,
                ROUND(MAX(precipitation_mm), 2) AS max_precipitation,
                COUNT(DISTINCT station_name) AS active_stations
            FROM read_parquet('{self.output_parquet}')
        """).fetchdf().iloc[0].to_dict()
        con.close()
        return res


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    parser = argparse.ArgumentParser(description="IMGW Meteorological Telemetry Ingestion")
    parser.add_argument("--mode", choices=["live", "backfill", "summary"], default="live",
                        help="Operation mode: 'live' (current hourly API), 'backfill' (historical archive), 'summary'")
    parser.add_argument("--year", type=int, default=2026, help="Year for historical backfill")
    parser.add_argument("--month", type=int, default=9, help="Month for historical backfill")
    args = parser.parse_args()

    harvester = IMGWWeatherHarvester()

    if args.mode == "live":
        print(f"Fetching live hourly observations for Warsaw...")
        obs = harvester.fetch_all_warsaw_live()
        inserted = harvester.upsert_to_parquet(obs)
        print(f"Successfully processed {inserted} live observation(s).")
    elif args.mode == "backfill":
        print(f"Backfilling historical synoptic observations for {args.year}-{args.month:02d}...")
        obs = harvester.fetch_historical_month(year=args.year, month=args.month)
        inserted = harvester.upsert_to_parquet(obs)
        print(f"Successfully backfilled {inserted} historical observation(s).")

    summary = harvester.get_summary()
    print("\n--- IMGW Weather Mart Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def fetch_imgw_synoptic_archive(
    output_path: Optional[str] = None,
    start_date: str = "2026-10-04",
    end_date: Optional[str] = None,
) -> str:
    """Convenience entry point for harvesting and persisting Warsaw weather telemetry."""
    if end_date is None:
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    harvester = IMGWWeatherHarvester(
        data_dir=os.path.dirname(output_path) if output_path else "data/processed"
    )
    if output_path:
        harvester.output_parquet = output_path
    # 1. Fetch October hourly telemetry
    obs = harvester.fetch_open_meteo_archive(start_date=start_date, end_date=end_date)
    harvester.upsert_to_parquet(obs)
    # 2. Also backfill September IMGW archive if available
    try:
        sept_obs = harvester.fetch_historical_month(year=2026, month=9)
        if sept_obs:
            harvester.upsert_to_parquet(sept_obs)
    except Exception:
        pass
    return harvester.output_parquet


if __name__ == "__main__":
    main()

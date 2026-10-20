"""
Warsaw Open Data Transit Vehicle Location API Client.
Connects to dane.um.warszawa.pl (new official portal) with fallback to legacy api.um.warszawa.pl.
Fetches real-time vehicle positions for buses (type=1) and trams (type=2).
"""

import email.utils
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

from src.historian.config import config

logger = logging.getLogger(__name__)


class WarsawCityApiClient:
    """Client for Warsaw City Open Data public transport telemetry."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout_seconds: Union[float, Tuple[float, float]] = (2.5, 4.5),
    ):
        self.api_key = api_key or config.um_api_key
        self.api_url = api_url or config.um_api_url
        if isinstance(timeout_seconds, (int, float)):
            connect_t = min(2.5, float(timeout_seconds) / 2.0)
            read_t = max(1.0, float(timeout_seconds) - connect_t)
            self.timeout: Tuple[float, float] = (connect_t, read_t)
        else:
            self.timeout = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "WarsawDelayTelemetry/2.0"})

    def _parse_clock_offset(self, resp: requests.Response, receive_ts: float) -> Optional[float]:
        """Calculates difference between server Date header and local system clock."""
        date_header = resp.headers.get("Date")
        if not date_header:
            return None
        try:
            parsed_time = email.utils.parsedate_to_datetime(date_header)
            server_epoch = parsed_time.timestamp()
            return round(receive_ts - server_epoch, 3)
        except Exception:
            return None

    def fetch_vehicles(self, vehicle_type: int) -> Dict[str, Any]:
        """
        Fetches vehicle positions for a given vehicle type (1=bus, 2=tram).
        Returns a structured payload containing raw bytes for bronze storage,
        parsed records, and timing metadata.
        """
        if not self.api_key:
            return {
                "status": "MISSING_API_KEY",
                "http_status": 0,
                "records": [],
                "raw_bytes": b"",
                "latency_ms": 0.0,
                "clock_offset_s": None,
                "server_date": None,
                "error": "UM_WARSAW_API_KEY is not configured.",
            }

        send_ts = time.time()
        is_dane_portal = "dane.um.warszawa.pl" in self.api_url

        try:
            if is_dane_portal:
                headers = {
                    "Authorization": self.api_key,
                    "Content-Type": "application/json",
                }
                resp = self.session.post(
                    self.api_url,
                    headers=headers,
                    json={"type": vehicle_type},
                    timeout=self.timeout,
                )
            else:
                # Legacy portal: api.um.warszawa.pl (GET with query parameters)
                params = {
                    "resource_id": config.um_resource_id,
                    "apikey": self.api_key,
                    "type": vehicle_type,
                }
                resp = self.session.get(self.api_url, params=params, timeout=self.timeout)

            receive_ts = time.time()
            latency_ms = round((receive_ts - send_ts) * 1000, 2)
            clock_offset_s = self._parse_clock_offset(resp, receive_ts)
            server_date = resp.headers.get("Date")
            raw_bytes = resp.content

            if resp.status_code != 200:
                logger.warning(
                    f"City API returned HTTP {resp.status_code} for type={vehicle_type}: {resp.text[:200]}"
                )
                return {
                    "status": "HTTP_ERROR",
                    "http_status": resp.status_code,
                    "records": [],
                    "raw_bytes": raw_bytes,
                    "latency_ms": latency_ms,
                    "clock_offset_s": clock_offset_s,
                    "server_date": server_date,
                    "error": f"HTTP {resp.status_code}",
                }

            # Parse payload
            data = resp.json()

            # Handle format difference: dane.um returns a list directly; legacy returns {"result": [...]}
            if isinstance(data, list):
                records: List[Dict[str, Any]] = data
            elif isinstance(data, dict):
                result_field = data.get("result")
                if isinstance(result_field, list):
                    records = result_field
                elif isinstance(result_field, str):
                    # Legacy error format, e.g. {"result": "Błędna metoda lub parametry"}
                    return {
                        "status": "API_ERROR",
                        "http_status": 200,
                        "records": [],
                        "raw_bytes": raw_bytes,
                        "latency_ms": latency_ms,
                        "clock_offset_s": clock_offset_s,
                        "server_date": server_date,
                        "error": result_field,
                    }
                else:
                    records = []
            else:
                records = []

            status = "EMPTY" if len(records) == 0 else "OK"
            return {
                "status": status,
                "http_status": 200,
                "records": records,
                "raw_bytes": raw_bytes,
                "latency_ms": latency_ms,
                "clock_offset_s": clock_offset_s,
                "server_date": server_date,
                "error": None,
            }

        except requests.Timeout as e:
            return {
                "status": "TIMEOUT",
                "http_status": 0,
                "records": [],
                "raw_bytes": b"",
                "latency_ms": round((time.time() - send_ts) * 1000, 2),
                "clock_offset_s": None,
                "server_date": None,
                "error": f"Request timed out after {self.timeout}s: {e}",
            }
        except Exception as e:
            return {
                "status": "NETWORK_ERROR",
                "http_status": 0,
                "records": [],
                "raw_bytes": b"",
                "latency_ms": round((time.time() - send_ts) * 1000, 2),
                "clock_offset_s": None,
                "server_date": None,
                "error": str(e),
            }

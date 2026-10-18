"""
zbiorkom GTFS-RT Telemetry Connector for Warsaw Public Transport Historian.
Fetches real-time protobuf feeds and extracts a filtered bronze subset:
active VehiclePositions, current-trip TripUpdates, and Service Alerts.
"""

import email.utils
import logging
import time
from typing import Any, Dict, Optional

import requests
from google.protobuf.message import DecodeError
from google.transit import gtfs_realtime_pb2

from src.historian.config import config

logger = logging.getLogger(__name__)


class ZbiorkomClient:
    """Connects to zbiorkom GTFS-RT and produces compact bronze subsets for cross-validation."""

    def __init__(self, feed_url: Optional[str] = None, timeout_seconds: float = 7.0):
        self.feed_url = feed_url or config.zbk_feed_url
        self.timeout = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "WarsawDelayTelemetry/2.0"})

    def _parse_clock_offset(self, resp: requests.Response, receive_ts: float) -> Optional[float]:
        date_header = resp.headers.get("Date")
        if not date_header:
            return None
        try:
            parsed_time = email.utils.parsedate_to_datetime(date_header)
            return round(receive_ts - parsed_time.timestamp(), 3)
        except Exception:
            return None

    def fetch_feed(self) -> Dict[str, Any]:
        """
        Fetches the GTFS-RT protobuf feed, filters to active vehicles and alerts,
        and serializes a compact protobuf payload for bronze storage.
        """
        send_ts = time.time()
        try:
            resp = self.session.get(self.feed_url, timeout=self.timeout)
            receive_ts = time.time()
            latency_ms = round((receive_ts - send_ts) * 1000, 2)
            clock_offset_s = self._parse_clock_offset(resp, receive_ts)
            server_date = resp.headers.get("Date")

            if resp.status_code != 200:
                return {
                    "status": "HTTP_ERROR",
                    "http_status": resp.status_code,
                    "raw_bytes": resp.content,
                    "latency_ms": latency_ms,
                    "n_vehicles": 0,
                    "n_trip_updates": 0,
                    "feed_timestamp": None,
                    "clock_offset_s": clock_offset_s,
                    "server_date": server_date,
                    "error": f"HTTP {resp.status_code}",
                }

            # Parse Protobuf FeedMessage
            full_feed = gtfs_realtime_pb2.FeedMessage()
            full_feed.ParseFromString(resp.content)

            feed_ts = int(full_feed.header.timestamp) if full_feed.header.HasField("timestamp") else int(receive_ts)

            # Filter to Bronze Subset:
            # 1. Collect all VehiclePosition entities and their active trip_ids
            active_vp_trips = set()
            active_vp_vehicles = set()
            vp_entities = []

            for entity in full_feed.entity:
                if entity.HasField("vehicle"):
                    vp_entities.append(entity)
                    v = entity.vehicle
                    if v.HasField("trip") and v.trip.trip_id:
                        active_vp_trips.add(v.trip.trip_id)
                    if v.HasField("vehicle") and v.vehicle.id:
                        active_vp_vehicles.add(v.vehicle.id)

            # 2. Collect current-trip TripUpdates matching active vehicles
            tu_entities = []
            for entity in full_feed.entity:
                if entity.HasField("trip_update"):
                    tu = entity.trip_update
                    trip_id = tu.trip.trip_id if tu.HasField("trip") else ""
                    vehicle_id = tu.vehicle.id if (tu.HasField("vehicle") and tu.vehicle.HasField("id")) else ""
                    # Keep if it matches an active vehicle position
                    if trip_id in active_vp_trips or vehicle_id in active_vp_vehicles:
                        tu_entities.append(entity)

            # 3. Collect all Service Alerts
            alert_entities = [e for e in full_feed.entity if e.HasField("alert")]

            # Construct compact filtered FeedMessage
            compact_feed = gtfs_realtime_pb2.FeedMessage()
            compact_feed.header.CopyFrom(full_feed.header)
            compact_feed.entity.extend(vp_entities)
            compact_feed.entity.extend(tu_entities)
            compact_feed.entity.extend(alert_entities)

            compact_bytes = compact_feed.SerializeToString()

            return {
                "status": "OK",
                "http_status": 200,
                "raw_bytes": compact_bytes,
                "latency_ms": latency_ms,
                "n_vehicles": len(vp_entities),
                "n_trip_updates": len(tu_entities),
                "feed_timestamp": feed_ts,
                "clock_offset_s": clock_offset_s,
                "server_date": server_date,
                "error": None,
            }

        except DecodeError as e:
            logger.error(f"Protobuf DecodeError from {self.feed_url}: {e}")
            return {
                "status": "DECODE_ERROR",
                "http_status": 200,
                "raw_bytes": b"",
                "latency_ms": round((time.time() - send_ts) * 1000, 2),
                "n_vehicles": 0,
                "n_trip_updates": 0,
                "feed_timestamp": None,
                "clock_offset_s": None,
                "server_date": None,
                "error": f"Protobuf decode error: {e}",
            }
        except requests.Timeout as e:
            return {
                "status": "TIMEOUT",
                "http_status": 0,
                "raw_bytes": b"",
                "latency_ms": round((time.time() - send_ts) * 1000, 2),
                "n_vehicles": 0,
                "n_trip_updates": 0,
                "feed_timestamp": None,
                "clock_offset_s": None,
                "server_date": None,
                "error": f"Request timed out: {e}",
            }
        except Exception as e:
            return {
                "status": "NETWORK_ERROR",
                "http_status": 0,
                "raw_bytes": b"",
                "latency_ms": round((time.time() - send_ts) * 1000, 2),
                "n_vehicles": 0,
                "n_trip_updates": 0,
                "feed_timestamp": None,
                "clock_offset_s": None,
                "server_date": None,
                "error": str(e),
            }

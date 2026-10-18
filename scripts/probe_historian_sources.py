#!/usr/bin/env python3
"""
Phase 0 Probe: Warsaw Public Transport Telemetry Diagnostic Tool.
Tests connectivity to dane.um.warszawa.pl and zbiorkom GTFS-RT.
Validates latency, response schemas, and join rates against static GTFS.
"""

import os
import sys
import time
from pathlib import Path

import duckdb
import requests
from google.transit import gtfs_realtime_pb2

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env if present
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                k, v = stripped.split("=", 1)
                if k.strip() not in os.environ:
                    os.environ[k.strip()] = v.strip()

from src.historian.config import config  # noqa: E402
from src.historian.sources.um_api import WarsawCityApiClient  # noqa: E402


def run_probe():
    print("=" * 70)
    print("WARSAW OBSERVED-DELAY HISTORIAN — PHASE 0 DIAGNOSTIC PROBE")
    print("=" * 70)

    # 1. Inspect configured API Key
    api_key = config.um_api_key
    if not api_key:
        print("[!] No UM_WARSAW_API_KEY configured in environment or .env.")
        print("    Please obtain an API key from https://dane.um.warszawa.pl/key-api")
        print("    and add UM_WARSAW_API_KEY=<token> to your .env file.\n")
        return

    print(f"[*] City API Endpoint: {config.um_api_url}")
    print(f"[*] API Key Token: {api_key[:16]}... (Length: {len(api_key)})")

    client = WarsawCityApiClient(api_key=api_key)

    # 2. Test City API - Buses (type=1)
    print("\n--- 1. Querying City API: Buses (type=1) ---")
    bus_res = client.fetch_vehicles(vehicle_type=1)
    print(f"Status: {bus_res['status']} | HTTP {bus_res['http_status']}")
    print(f"Latency: {bus_res['latency_ms']:.1f} ms | Payload Size: {len(bus_res['raw_bytes']):,} bytes")
    print(f"Buses Captured: {len(bus_res['records']):,}")
    print(f"Clock Offset vs Server Date: {bus_res['clock_offset_s']} s")
    if bus_res["records"]:
        print("Sample Bus Record:", bus_res["records"][0])

    # 3. Test City API - Trams (type=2)
    print("\n--- 2. Querying City API: Trams (type=2) ---")
    tram_res = client.fetch_vehicles(vehicle_type=2)
    print(f"Status: {tram_res['status']} | HTTP {tram_res['http_status']}")
    print(f"Latency: {tram_res['latency_ms']:.1f} ms | Payload Size: {len(tram_res['raw_bytes']):,} bytes")
    print(f"Trams Captured: {len(tram_res['records']):,}")
    if tram_res["records"]:
        print("Sample Tram Record:", tram_res["records"][0])

    all_um_records = bus_res["records"] + tram_res["records"]

    # 4. Test zbiorkom GTFS-RT Protobuf Feed
    print(f"\n--- 3. Querying zbiorkom GTFS-RT ({config.zbk_feed_url[:50]}...) ---")
    try:
        t_zbk0 = time.time()
        zbk_resp = requests.get(
            config.zbk_feed_url,
            timeout=10,
            headers={"User-Agent": "WarsawDelayTelemetry/2.0"},
        )
        t_zbk1 = time.time()
        print(f"HTTP Status: {zbk_resp.status_code} | Latency: {(t_zbk1 - t_zbk0)*1000:.1f} ms | Bytes: {len(zbk_resp.content):,}")

        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(zbk_resp.content)
        vp_count = sum(1 for e in feed.entity if e.HasField("vehicle"))
        tu_count = sum(1 for e in feed.entity if e.HasField("trip_update"))
        print(f"Entities: {len(feed.entity):,} (VehiclePositions: {vp_count:,}, TripUpdates: {tu_count:,})")
        print(f"Feed Header Timestamp: {feed.header.timestamp}")

        # Check cross-source identity alignment (City VehicleNumber vs zbiorkom vehicle.label)
        zbk_labels = {
            e.vehicle.vehicle.label: e.vehicle.trip.route_id
            for e in feed.entity
            if e.HasField("vehicle") and e.vehicle.vehicle.HasField("label")
        }
        um_numbers = {r.get("VehicleNumber"): r.get("Lines") for r in all_um_records if r.get("VehicleNumber")}

        common_vehicles = set(zbk_labels.keys()).intersection(set(um_numbers.keys()))
        print(f"Active Vehicles in City API: {len(um_numbers):,}")
        print(f"Active Vehicles in zbiorkom VP: {len(zbk_labels):,}")
        print(f"Common Physical Vehicle Codes: {len(common_vehicles):,}")

        if common_vehicles:
            route_matches = sum(1 for v in common_vehicles if zbk_labels[v] == str(um_numbers[v]))
            print(f"Route Agreement for Common Vehicles: {route_matches:,} / {len(common_vehicles):,} ({route_matches / len(common_vehicles)*100:.1f}%)")

    except Exception as e:
        print(f"[!] Error querying zbiorkom feed: {e}")

    # 5. Check Static GTFS Schedule Alignment
    print("\n--- 4. Checking Static GTFS Schedule Alignment ---")
    static_duckdb = Path("data/gtfs/2026_10_18/schedule_index.duckdb")
    if not static_duckdb.exists():
        # Fall back to checking any schedule_index.duckdb
        candidates = list(Path("data/gtfs").glob("**/schedule_index.duckdb"))
        static_duckdb = candidates[0] if candidates else None

    if static_duckdb and static_duckdb.exists():
        print(f"Found Static GTFS Index: {static_duckdb}")
        con = duckdb.connect(str(static_duckdb), read_only=True)

        # Check brigade & line matches
        um_pairs = set()
        for r in all_um_records:
            line = str(r.get("Lines", "")).strip()
            brigade = str(r.get("Brigade", "")).strip().lstrip("0")
            if line and brigade:
                um_pairs.add((line, brigade))

        print(f"Distinct Live (Line, Brigade) pairs observed: {len(um_pairs):,}")

        # Query static trips table
        db_pairs = con.execute("""
            SELECT DISTINCT route_id, ltrim(brigade, '0') AS clean_brigade
            FROM trips
            WHERE brigade IS NOT NULL
        """).fetchall()
        db_set = set((str(r[0]), str(r[1])) for r in db_pairs)

        matched = um_pairs.intersection(db_set)
        match_rate = len(matched) / max(1, len(um_pairs)) * 100
        print(f"Matching (Line, Brigade) pairs in static GTFS: {len(matched):,} / {len(um_pairs):,} ({match_rate:.1f}%)")
        con.close()
    else:
        print("[!] No static GTFS schedule_index.duckdb found to test join rate.")

    print("\n" + "=" * 70)
    print("PHASE 0 DIAGNOSTIC PROBE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_probe()

"""
Unit tests for zbiorkom GTFS-RT client.
"""

from unittest.mock import MagicMock, patch

from google.transit import gtfs_realtime_pb2

from src.historian.sources.zbiorkom import ZbiorkomClient


@patch("requests.Session.get")
def test_zbiorkom_client_filter_subset(mock_get):
    # Construct a synthetic protobuf feed with:
    # - 1 VehiclePosition (trip_id: "T1", vehicle_id: "V1")
    # - 1 TripUpdate for T1 (active trip)
    # - 1 TripUpdate for T2 (future trip without vehicle, should be filtered out)
    # - 1 Alert
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    feed.header.timestamp = 1792330000

    # Entity 1: VP
    e1 = feed.entity.add()
    e1.id = "vp1"
    e1.vehicle.trip.trip_id = "T1"
    e1.vehicle.vehicle.id = "V1"
    e1.vehicle.position.latitude = 52.2
    e1.vehicle.position.longitude = 21.0

    # Entity 2: Matching TU
    e2 = feed.entity.add()
    e2.id = "tu1"
    e2.trip_update.trip.trip_id = "T1"
    e2.trip_update.vehicle.id = "V1"
    stu = e2.trip_update.stop_time_update.add()
    stu.stop_sequence = 5

    # Entity 3: Unmatched Future TU
    e3 = feed.entity.add()
    e3.id = "tu2"
    e3.trip_update.trip.trip_id = "T2"

    # Entity 4: Alert
    e4 = feed.entity.add()
    e4.id = "alt1"
    e4.alert.header_text.translation.add().text = "Detour on Line 17"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"Date": "Sun, 18 Oct 2026 13:40:00 GMT"}
    mock_resp.content = feed.SerializeToString()
    mock_get.return_value = mock_resp

    client = ZbiorkomClient()
    res = client.fetch_feed()

    assert res["status"] == "OK"
    assert res["http_status"] == 200
    assert res["n_vehicles"] == 1
    assert res["n_trip_updates"] == 1  # tu2 was filtered out!

    # Verify deserialization of returned raw_bytes
    filtered_feed = gtfs_realtime_pb2.FeedMessage()
    filtered_feed.ParseFromString(res["raw_bytes"])
    assert len(filtered_feed.entity) == 3  # VP, TU1, Alert (TU2 excluded)

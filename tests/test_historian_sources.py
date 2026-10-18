"""
Unit tests for Warsaw City Open Data API client.
"""

from unittest.mock import MagicMock, patch

from src.historian.sources.um_api import WarsawCityApiClient


def test_missing_api_key():
    client = WarsawCityApiClient(api_key="")
    res = client.fetch_vehicles(vehicle_type=1)
    assert res["status"] == "MISSING_API_KEY"
    assert len(res["records"]) == 0


@patch("requests.Session.post")
def test_dane_portal_success_list(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"Date": "Sun, 18 Oct 2026 13:40:00 GMT"}
    mock_resp.content = b'[{"Lines": "17", "Brigade": "1", "VehicleNumber": "1001", "Lat": 52.2, "Lon": 21.0, "Time": "2026-10-18 15:40:00"}]'
    mock_resp.json.return_value = [
        {"Lines": "17", "Brigade": "1", "VehicleNumber": "1001", "Lat": 52.2, "Lon": 21.0, "Time": "2026-10-18 15:40:00"}
    ]
    mock_post.return_value = mock_resp

    client = WarsawCityApiClient(
        api_key="mock_jwt_token",
        api_url="https://dane.um.warszawa.pl/api/action/get_ztm_lokalizacja_pojazdow",
    )
    res = client.fetch_vehicles(vehicle_type=2)

    assert res["status"] == "OK"
    assert res["http_status"] == 200
    assert len(res["records"]) == 1
    assert res["records"][0]["Lines"] == "17"
    assert res["records"][0]["VehicleNumber"] == "1001"
    assert res["clock_offset_s"] is not None


@patch("requests.Session.get")
def test_legacy_portal_success_dict(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {}
    mock_resp.content = b'{"result": [{"Lines": "503", "Brigade": "2", "VehicleNumber": "2001"}]}'
    mock_resp.json.return_value = {"result": [{"Lines": "503", "Brigade": "2", "VehicleNumber": "2001"}]}
    mock_get.return_value = mock_resp

    client = WarsawCityApiClient(
        api_key="mock_uuid_key",
        api_url="https://api.um.warszawa.pl/api/action/busestrams_get/",
    )
    res = client.fetch_vehicles(vehicle_type=1)

    assert res["status"] == "OK"
    assert len(res["records"]) == 1
    assert res["records"][0]["Lines"] == "503"


@patch("requests.Session.get")
def test_legacy_portal_error_string(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {}
    mock_resp.content = b'{"result": "B\xc5\x82\xc4\x99dna metoda lub parametry wywo\xc5\x82ania"}'
    mock_resp.json.return_value = {"result": "Błędna metoda lub parametry wywołania"}
    mock_get.return_value = mock_resp

    client = WarsawCityApiClient(
        api_key="mock_uuid_key",
        api_url="https://api.um.warszawa.pl/api/action/busestrams_get/",
    )
    res = client.fetch_vehicles(vehicle_type=1)

    assert res["status"] == "API_ERROR"
    assert res["error"] == "Błędna metoda lub parametry wywołania"
    assert len(res["records"]) == 0

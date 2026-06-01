"""
test_v2.py — V2 test suite.

All weather tests use MockWeatherClient — no real HTTP calls needed.
Run: pytest tests/test_v2.py -v
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.weather_client import (
    AIRPORT_COORDS,
    MockWeatherClient,
)
from src.route_map import FlightRoute, generate_route_risk_map


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def clear_client() -> MockWeatherClient:
    return MockWeatherClient(storm=False)


@pytest.fixture
def storm_client() -> MockWeatherClient:
    return MockWeatherClient(storm=True)


@pytest.fixture
def api_client_clear():
    from src.realtime_api import create_app
    app = create_app(weather_client=MockWeatherClient(storm=False))
    return TestClient(app)


@pytest.fixture
def api_client_storm():
    from src.realtime_api import create_app
    app = create_app(weather_client=MockWeatherClient(storm=True))
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. AIRPORT_COORDS dict
# ---------------------------------------------------------------------------


class TestAirportCoords:
    def test_has_at_least_20_airports(self):
        assert len(AIRPORT_COORDS) >= 20

    def test_atl_present(self):
        assert "ATL" in AIRPORT_COORDS

    def test_lax_present(self):
        assert "LAX" in AIRPORT_COORDS

    def test_jfk_present(self):
        assert "JFK" in AIRPORT_COORDS

    def test_all_entries_have_lat_lon(self):
        for code, coords in AIRPORT_COORDS.items():
            assert "lat" in coords, f"{code} missing lat"
            assert "lon" in coords, f"{code} missing lon"

    def test_all_lats_continental_range(self):
        # US airports: lat roughly 18–72, lon roughly -180 to -65
        for code, coords in AIRPORT_COORDS.items():
            assert 15.0 <= coords["lat"] <= 75.0, f"{code} lat out of range"

    def test_all_lons_us_range(self):
        for code, coords in AIRPORT_COORDS.items():
            assert -180.0 <= coords["lon"] <= -60.0, f"{code} lon out of range"

    def test_required_20_airports_present(self):
        required = {
            "ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO",
            "SEA", "LAS", "MCO", "EWR", "MIA", "PHX", "IAH",
            "BOS", "MSP", "DTW", "FLL", "CLT", "SLC",
        }
        missing = required - AIRPORT_COORDS.keys()
        assert not missing, f"Missing airports: {missing}"


# ---------------------------------------------------------------------------
# 2. MockWeatherClient — airport weather
# ---------------------------------------------------------------------------


class TestMockWeatherClientClear:
    def test_returns_dict(self, clear_client):
        result = clear_client.get_airport_weather(33.64, -84.43)
        assert isinstance(result, dict)

    def test_has_required_keys(self, clear_client):
        result = clear_client.get_airport_weather(33.64, -84.43)
        required = {
            "temperature_c", "wind_speed_knots", "precipitation_mm_hr",
            "visibility_km", "cloud_cover_pct", "weather_code",
        }
        assert required.issubset(result.keys())

    def test_clear_weather_is_deterministic(self, clear_client):
        r1 = clear_client.get_airport_weather(33.64, -84.43)
        r2 = clear_client.get_airport_weather(33.64, -84.43)
        assert r1 == r2

    def test_clear_weather_low_wind(self, clear_client):
        w = clear_client.get_airport_weather(33.64, -84.43)
        assert w["wind_speed_knots"] <= 25

    def test_clear_weather_good_visibility(self, clear_client):
        w = clear_client.get_airport_weather(33.64, -84.43)
        assert w["visibility_km"] >= 4.828  # >= 3 sm

    def test_clear_weather_no_precipitation(self, clear_client):
        w = clear_client.get_airport_weather(33.64, -84.43)
        assert w["precipitation_mm_hr"] <= 0.1


class TestMockWeatherClientStorm:
    def test_storm_weather_high_wind(self, storm_client):
        w = storm_client.get_airport_weather(33.64, -84.43)
        assert w["wind_speed_knots"] > 25

    def test_storm_weather_low_visibility(self, storm_client):
        w = storm_client.get_airport_weather(33.64, -84.43)
        assert w["visibility_km"] < 4.828

    def test_storm_weather_has_precipitation(self, storm_client):
        w = storm_client.get_airport_weather(33.64, -84.43)
        assert w["precipitation_mm_hr"] > 0.1

    def test_storm_weather_code_is_thunderstorm(self, storm_client):
        w = storm_client.get_airport_weather(33.64, -84.43)
        assert w["weather_code"] in {80, 81, 82, 95, 96, 97, 98, 99}


# ---------------------------------------------------------------------------
# 3. Weather risk scores
# ---------------------------------------------------------------------------


class TestWeatherRiskScore:
    def test_clear_weather_score_is_zero(self, clear_client):
        w = clear_client.get_airport_weather(33.64, -84.43)
        score = clear_client.get_weather_risk_score(w)
        assert score == 0.0

    def test_storm_score_above_0_5(self, storm_client):
        w = storm_client.get_airport_weather(33.64, -84.43)
        score = storm_client.get_weather_risk_score(w)
        assert score > 0.5

    def test_score_capped_at_1_0(self, storm_client):
        w = storm_client.get_airport_weather(0.0, 0.0)
        score = storm_client.get_weather_risk_score(w)
        assert score <= 1.0

    def test_score_is_float(self, clear_client):
        w = clear_client.get_airport_weather(33.64, -84.43)
        score = clear_client.get_weather_risk_score(w)
        assert isinstance(score, float)

    def test_thunderstorm_code_adds_risk(self):
        client = MockWeatherClient()
        weather = {
            "temperature_c": 15.0,
            "wind_speed_knots": 10.0,
            "precipitation_mm_hr": 0.0,
            "visibility_km": 16.0,
            "cloud_cover_pct": 20.0,
            "weather_code": 95,  # thunderstorm
        }
        score = client.get_weather_risk_score(weather)
        assert score >= 0.40

    def test_high_wind_adds_risk(self):
        client = MockWeatherClient()
        weather = {
            "temperature_c": 15.0,
            "wind_speed_knots": 30.0,
            "precipitation_mm_hr": 0.0,
            "visibility_km": 16.0,
            "cloud_cover_pct": 20.0,
            "weather_code": 0,
        }
        score = client.get_weather_risk_score(weather)
        assert score >= 0.30

    def test_low_visibility_adds_risk(self):
        client = MockWeatherClient()
        weather = {
            "temperature_c": 15.0,
            "wind_speed_knots": 5.0,
            "precipitation_mm_hr": 0.0,
            "visibility_km": 2.0,  # below 3 sm threshold
            "cloud_cover_pct": 20.0,
            "weather_code": 0,
        }
        score = client.get_weather_risk_score(weather)
        assert score >= 0.30

    def test_score_in_range_for_various_inputs(self):
        client = MockWeatherClient()
        test_cases = [
            {"temperature_c": 10, "wind_speed_knots": 5, "precipitation_mm_hr": 0, "visibility_km": 20, "cloud_cover_pct": 10, "weather_code": 0},
            {"temperature_c": 10, "wind_speed_knots": 50, "precipitation_mm_hr": 10, "visibility_km": 0.5, "cloud_cover_pct": 100, "weather_code": 99},
        ]
        for w in test_cases:
            score = client.get_weather_risk_score(w)
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 4. get_route_weather
# ---------------------------------------------------------------------------


class TestGetRouteWeather:
    def test_returns_dict_with_expected_keys(self, clear_client):
        result = clear_client.get_route_weather(33.64, -84.43, 33.94, -118.41)
        assert "origin_weather" in result
        assert "dest_weather" in result
        assert "max_risk_score" in result

    def test_origin_weather_has_required_keys(self, clear_client):
        result = clear_client.get_route_weather(33.64, -84.43, 33.94, -118.41)
        required = {"temperature_c", "wind_speed_knots", "precipitation_mm_hr", "visibility_km", "cloud_cover_pct", "weather_code"}
        assert required.issubset(result["origin_weather"].keys())

    def test_dest_weather_has_required_keys(self, clear_client):
        result = clear_client.get_route_weather(33.64, -84.43, 33.94, -118.41)
        required = {"temperature_c", "wind_speed_knots", "precipitation_mm_hr", "visibility_km", "cloud_cover_pct", "weather_code"}
        assert required.issubset(result["dest_weather"].keys())

    def test_max_risk_score_in_range(self, clear_client):
        result = clear_client.get_route_weather(33.64, -84.43, 33.94, -118.41)
        assert 0.0 <= result["max_risk_score"] <= 1.0

    def test_storm_route_has_high_risk(self, storm_client):
        result = storm_client.get_route_weather(33.64, -84.43, 33.94, -118.41)
        assert result["max_risk_score"] > 0.5

    def test_clear_route_has_zero_risk(self, clear_client):
        result = clear_client.get_route_weather(33.64, -84.43, 33.94, -118.41)
        assert result["max_risk_score"] == 0.0


# ---------------------------------------------------------------------------
# 5. FastAPI endpoints
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_ok(self, api_client_clear):
        resp = api_client_clear.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_has_model_loaded_key(self, api_client_clear):
        resp = api_client_clear.get("/health")
        assert "model_loaded" in resp.json()


class TestAirportWeatherEndpoint:
    def test_atl_returns_200(self, api_client_clear):
        resp = api_client_clear.get("/airport-weather/ATL")
        assert resp.status_code == 200

    def test_atl_response_has_weather_dict(self, api_client_clear):
        resp = api_client_clear.get("/airport-weather/ATL")
        data = resp.json()
        assert "weather" in data

    def test_atl_weather_has_required_keys(self, api_client_clear):
        resp = api_client_clear.get("/airport-weather/ATL")
        weather = resp.json()["weather"]
        required = {"temperature_c", "wind_speed_knots", "visibility_km", "cloud_cover_pct"}
        assert required.issubset(weather.keys())

    def test_risk_score_in_response(self, api_client_clear):
        resp = api_client_clear.get("/airport-weather/ATL")
        data = resp.json()
        assert 0.0 <= data["risk_score"] <= 1.0

    def test_unknown_airport_returns_404(self, api_client_clear):
        resp = api_client_clear.get("/airport-weather/ZZZ")
        assert resp.status_code == 404

    def test_lowercase_code_works(self, api_client_clear):
        resp = api_client_clear.get("/airport-weather/atl")
        assert resp.status_code == 200


class TestPredictDelayEndpoint:
    def _payload(self, origin="ATL", dest="LAX"):
        return {
            "origin": origin,
            "destination": dest,
            "scheduled_departure": "2026-06-01T08:00:00",
            "airline": "DL",
            "aircraft_type": "Boeing 737",
        }

    def test_returns_200(self, api_client_clear):
        resp = api_client_clear.post("/predict-delay", json=self._payload())
        assert resp.status_code == 200

    def test_delay_probability_in_range(self, api_client_clear):
        resp = api_client_clear.post("/predict-delay", json=self._payload())
        prob = resp.json()["delay_probability"]
        assert 0.0 <= prob <= 1.0

    def test_expected_delay_minutes_non_negative(self, api_client_clear):
        resp = api_client_clear.post("/predict-delay", json=self._payload())
        assert resp.json()["expected_delay_minutes"] >= 0.0

    def test_weather_risk_score_in_range(self, api_client_clear):
        resp = api_client_clear.post("/predict-delay", json=self._payload())
        risk = resp.json()["weather_risk_score"]
        assert 0.0 <= risk <= 1.0

    def test_response_has_explanation(self, api_client_clear):
        resp = api_client_clear.post("/predict-delay", json=self._payload())
        assert "explanation" in resp.json()
        assert len(resp.json()["explanation"]) > 0

    def test_response_has_weather_summary(self, api_client_clear):
        resp = api_client_clear.post("/predict-delay", json=self._payload())
        assert "weather_summary" in resp.json()

    def test_response_has_confidence(self, api_client_clear):
        resp = api_client_clear.post("/predict-delay", json=self._payload())
        assert resp.json()["confidence"] in {"high", "low"}

    def test_storm_increases_probability(self, api_client_clear, api_client_storm):
        clear_resp = api_client_clear.post("/predict-delay", json=self._payload())
        storm_resp = api_client_storm.post("/predict-delay", json=self._payload())
        clear_prob = clear_resp.json()["delay_probability"]
        storm_prob = storm_resp.json()["delay_probability"]
        assert storm_prob >= clear_prob

    def test_unknown_origin_returns_404(self, api_client_clear):
        resp = api_client_clear.post("/predict-delay", json=self._payload(origin="ZZZ"))
        assert resp.status_code == 404

    def test_unknown_dest_returns_404(self, api_client_clear):
        resp = api_client_clear.post("/predict-delay", json=self._payload(dest="ZZZ"))
        assert resp.status_code == 404

    def test_delay_probability_is_float(self, api_client_clear):
        resp = api_client_clear.post("/predict-delay", json=self._payload())
        assert isinstance(resp.json()["delay_probability"], float)


# ---------------------------------------------------------------------------
# 6. Route risk map
# ---------------------------------------------------------------------------


class TestRouteRiskMap:
    @pytest.fixture
    def sample_flights(self):
        return [
            FlightRoute(
                origin="ATL",
                destination="LAX",
                delay_probability=0.25,
                weather_risk_score=0.10,
                airline="DL",
                flight_number="DL100",
            ),
            FlightRoute(
                origin="JFK",
                destination="ORD",
                delay_probability=0.65,
                weather_risk_score=0.70,
                airline="AA",
                flight_number="AA200",
            ),
            FlightRoute(
                origin="DEN",
                destination="SFO",
                delay_probability=0.45,
                weather_risk_score=0.40,
                airline="UA",
                flight_number="UA300",
            ),
        ]

    def test_returns_string(self, sample_flights):
        html = generate_route_risk_map(sample_flights)
        assert isinstance(html, str)

    def test_html_contains_folium_or_leaflet(self, sample_flights):
        html = generate_route_risk_map(sample_flights)
        lower = html.lower()
        assert "folium" in lower or "leaflet" in lower

    def test_html_contains_map_element(self, sample_flights):
        html = generate_route_risk_map(sample_flights)
        assert "map" in html.lower()

    def test_html_is_nonempty(self, sample_flights):
        html = generate_route_risk_map(sample_flights)
        assert len(html) > 500

    def test_empty_flights_still_returns_html(self):
        html = generate_route_risk_map([])
        assert isinstance(html, str)
        assert len(html) > 100

    def test_unknown_airports_silently_skipped(self):
        flights = [
            FlightRoute("ZZZ", "YYY", delay_probability=0.5, weather_risk_score=0.3)
        ]
        html = generate_route_risk_map(flights)
        assert isinstance(html, str)

    def test_high_risk_route_included(self, sample_flights):
        html = generate_route_risk_map(sample_flights)
        # High-risk color should appear
        assert "#f85149" in html  # red

    def test_low_risk_route_included(self, sample_flights):
        html = generate_route_risk_map(sample_flights)
        assert "#2ea043" in html  # green

"""
weather_client.py — Live weather integration via Open-Meteo API (free, no key).

WeatherClient   : fetches real hourly weather data using httpx.
MockWeatherClient: returns deterministic fake data for offline testing.
"""

from __future__ import annotations

from typing import Optional

import httpx

# ---------------------------------------------------------------------------
# Airport coordinate registry — top 50 US airports
# ---------------------------------------------------------------------------

AIRPORT_COORDS: dict[str, dict[str, float]] = {
    "ATL": {"lat": 33.6407, "lon": -84.4277},
    "LAX": {"lat": 33.9425, "lon": -118.4081},
    "ORD": {"lat": 41.9742, "lon": -87.9073},
    "DFW": {"lat": 32.8998, "lon": -97.0403},
    "DEN": {"lat": 39.8561, "lon": -104.6737},
    "JFK": {"lat": 40.6413, "lon": -73.7781},
    "SFO": {"lat": 37.6213, "lon": -122.3790},
    "SEA": {"lat": 47.4502, "lon": -122.3088},
    "LAS": {"lat": 36.0840, "lon": -115.1537},
    "MCO": {"lat": 28.4312, "lon": -81.3081},
    "EWR": {"lat": 40.6895, "lon": -74.1745},
    "MIA": {"lat": 25.7959, "lon": -80.2870},
    "PHX": {"lat": 33.4373, "lon": -112.0078},
    "IAH": {"lat": 29.9902, "lon": -95.3368},
    "BOS": {"lat": 42.3656, "lon": -71.0096},
    "MSP": {"lat": 44.8820, "lon": -93.2218},
    "DTW": {"lat": 42.2162, "lon": -83.3554},
    "FLL": {"lat": 26.0742, "lon": -80.1506},
    "CLT": {"lat": 35.2140, "lon": -80.9431},
    "SLC": {"lat": 40.7899, "lon": -111.9791},
    "SAN": {"lat": 32.7336, "lon": -117.1897},
    "BWI": {"lat": 39.1754, "lon": -76.6683},
    "MDW": {"lat": 41.7868, "lon": -87.7522},
    "TPA": {"lat": 27.9755, "lon": -82.5332},
    "IAD": {"lat": 38.9531, "lon": -77.4565},
    "PDX": {"lat": 45.5898, "lon": -122.5951},
    "HOU": {"lat": 29.6454, "lon": -95.2789},
    "HNL": {"lat": 21.3245, "lon": -157.9251},
    "SJC": {"lat": 37.3626, "lon": -121.9290},
    "OAK": {"lat": 37.7213, "lon": -122.2208},
    "SMF": {"lat": 38.6954, "lon": -121.5908},
    "AUS": {"lat": 30.1975, "lon": -97.6664},
    "SAT": {"lat": 29.5337, "lon": -98.4698},
    "RSW": {"lat": 26.5362, "lon": -81.7552},
    "BNA": {"lat": 36.1245, "lon": -86.6782},
    "STL": {"lat": 38.7487, "lon": -90.3700},
    "MCI": {"lat": 39.2976, "lon": -94.7139},
    "RDU": {"lat": 35.8801, "lon": -78.7880},
    "PHL": {"lat": 39.8744, "lon": -75.2424},
    "DCA": {"lat": 38.8521, "lon": -77.0377},
    "LGA": {"lat": 40.7773, "lon": -73.8726},
    "MEM": {"lat": 35.0424, "lon": -89.9767},
    "CLE": {"lat": 41.4117, "lon": -81.8498},
    "PIT": {"lat": 40.4915, "lon": -80.2329},
    "CVG": {"lat": 39.0488, "lon": -84.6678},
    "IND": {"lat": 39.7173, "lon": -86.2944},
    "CMH": {"lat": 39.9980, "lon": -82.8919},
    "OMA": {"lat": 41.3032, "lon": -95.8941},
    "ABQ": {"lat": 35.0402, "lon": -106.6090},
    "ANC": {"lat": 61.1743, "lon": -149.9963},
}


# ---------------------------------------------------------------------------
# Thunderstorm WMO weather codes (WW codes 95–99, 80–82 heavy showers)
# ---------------------------------------------------------------------------

_THUNDERSTORM_CODES = frozenset([80, 81, 82, 95, 96, 97, 98, 99])


# ---------------------------------------------------------------------------
# WeatherClient — real HTTP
# ---------------------------------------------------------------------------

class WeatherClient:
    """Fetch live weather from Open-Meteo (free, no API key)."""

    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    TIMEOUT = 10.0  # seconds

    def get_airport_weather(self, lat: float, lon: float) -> dict:
        """
        Fetch current conditions at (lat, lon).

        Returns a dict with keys:
            temperature_c, wind_speed_knots, precipitation_mm_hr,
            visibility_km, cloud_cover_pct, weather_code
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join([
                "temperature_2m",
                "wind_speed_10m",
                "precipitation",
                "visibility",
                "cloud_cover",
                "weather_code",
            ]),
            "wind_speed_unit": "kn",  # knots
            "forecast_days": 1,
        }
        with httpx.Client(timeout=self.TIMEOUT) as client:
            resp = client.get(self.BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        hourly = data.get("hourly", {})

        # Take first hour (most current forecast)
        def _first(key: str, fallback=0.0):
            vals = hourly.get(key, [])
            return vals[0] if vals else fallback

        raw_vis = _first("visibility", 10000)  # metres from API
        visibility_km = float(raw_vis) / 1000.0

        return {
            "temperature_c": float(_first("temperature_2m", 15.0)),
            "wind_speed_knots": float(_first("wind_speed_10m", 5.0)),
            "precipitation_mm_hr": float(_first("precipitation", 0.0)),
            "visibility_km": visibility_km,
            "cloud_cover_pct": float(_first("cloud_cover", 20.0)),
            "weather_code": int(_first("weather_code", 0)),
        }

    def get_weather_risk_score(self, weather: dict) -> float:
        """
        Compute a risk score in [0.0, 1.0] from a weather dict.

        Contributions:
            Wind speed > 25 kn       → +0.30
            Visibility < 3 miles     → +0.30  (3 mi ≈ 4.83 km)
            Precipitation > 0.1 mm/h → +0.20
            Cloud cover > 80%        → +0.10
            Thunderstorm code        → +0.40
        """
        score = 0.0

        if weather.get("wind_speed_knots", 0) > 25:
            score += 0.30

        # 3 statute miles ≈ 4.828 km
        if weather.get("visibility_km", 16.0) < 4.828:
            score += 0.30

        if weather.get("precipitation_mm_hr", 0) > 0.1:
            score += 0.20

        if weather.get("cloud_cover_pct", 0) > 80:
            score += 0.10

        if int(weather.get("weather_code", 0)) in _THUNDERSTORM_CODES:
            score += 0.40

        return round(min(score, 1.0), 3)

    def get_route_weather(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
    ) -> dict:
        """
        Fetch weather at both route endpoints.

        Returns:
            origin_weather   : dict from get_airport_weather
            dest_weather     : dict from get_airport_weather
            max_risk_score   : float — max of origin/dest risk scores
        """
        origin_weather = self.get_airport_weather(origin_lat, origin_lon)
        dest_weather = self.get_airport_weather(dest_lat, dest_lon)

        origin_risk = self.get_weather_risk_score(origin_weather)
        dest_risk = self.get_weather_risk_score(dest_weather)

        return {
            "origin_weather": origin_weather,
            "dest_weather": dest_weather,
            "max_risk_score": round(max(origin_risk, dest_risk), 3),
        }


# ---------------------------------------------------------------------------
# MockWeatherClient — deterministic, offline-safe
# ---------------------------------------------------------------------------

_CLEAR_WEATHER = {
    "temperature_c": 18.0,
    "wind_speed_knots": 10.0,
    "precipitation_mm_hr": 0.0,
    "visibility_km": 16.0,
    "cloud_cover_pct": 20.0,
    "weather_code": 0,
}

_STORM_WEATHER = {
    "temperature_c": 14.0,
    "wind_speed_knots": 35.0,
    "precipitation_mm_hr": 5.0,
    "visibility_km": 1.5,
    "cloud_cover_pct": 95.0,
    "weather_code": 95,
}


class MockWeatherClient(WeatherClient):
    """
    Drop-in replacement for WeatherClient that never makes HTTP calls.

    By default returns clear-weather data.  Pass storm=True in the
    constructor to always return thunderstorm data.
    """

    def __init__(self, storm: bool = False) -> None:
        self._storm = storm

    def get_airport_weather(self, lat: float, lon: float) -> dict:  # type: ignore[override]
        return dict(_STORM_WEATHER if self._storm else _CLEAR_WEATHER)

    def get_weather_risk_score(self, weather: dict) -> float:
        return super().get_weather_risk_score(weather)

    def get_route_weather(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
    ) -> dict:
        origin_weather = self.get_airport_weather(origin_lat, origin_lon)
        dest_weather = self.get_airport_weather(dest_lat, dest_lon)
        origin_risk = self.get_weather_risk_score(origin_weather)
        dest_risk = self.get_weather_risk_score(dest_weather)
        return {
            "origin_weather": origin_weather,
            "dest_weather": dest_weather,
            "max_risk_score": round(max(origin_risk, dest_risk), 3),
        }

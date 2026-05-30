"""
realtime_api.py — V2 FastAPI endpoint: live weather + ML delay prediction.

Usage:
    uvicorn src.realtime_api:app --reload --port 8001
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.weather_client import AIRPORT_COORDS, WeatherClient, MockWeatherClient

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class FlightRequest(BaseModel):
    origin: str = Field(..., description="IATA origin airport code (e.g. ATL)")
    destination: str = Field(..., description="IATA destination airport code (e.g. LAX)")
    scheduled_departure: str = Field(
        ..., description="ISO-8601 departure datetime, e.g. 2026-06-01T08:00:00"
    )
    airline: str = Field(default="UA", description="Airline code")
    aircraft_type: str = Field(default="Boeing 737", description="Aircraft type")


class DelayPrediction(BaseModel):
    origin: str
    destination: str
    delay_probability: float = Field(..., ge=0.0, le=1.0)
    expected_delay_minutes: float = Field(..., ge=0.0)
    weather_risk_score: float = Field(..., ge=0.0, le=1.0)
    weather_summary: str
    confidence: str
    explanation: str


# ---------------------------------------------------------------------------
# Lazy model loader — returns None if models not available
# ---------------------------------------------------------------------------

_ml_model = None


def _try_load_model():
    """Load the V1 Random Forest model if available; silently skip if not."""
    global _ml_model
    if _ml_model is not None:
        return _ml_model
    try:
        from intelligence.delay_predictor import load_models
        _ml_model = load_models()
    except Exception:  # noqa: BLE001
        _ml_model = None
    return _ml_model


def _model_delay_probability(
    dep_hour: int,
    day_of_week: int,
    origin_weather_score: float,
    dest_weather_score: float,
    distance_mi: float,
    aircraft_type: str,
    route_congestion_score: float,
) -> Optional[float]:
    """Run the V1 ML model. Returns None if model not available."""
    try:
        from intelligence.delay_predictor import predict_delay
        pred = predict_delay(
            dep_hour=dep_hour,
            day_of_week=day_of_week,
            origin_weather_score=origin_weather_score,
            dest_weather_score=dest_weather_score,
            distance_mi=distance_mi,
            aircraft_type=aircraft_type,
            route_congestion_score=route_congestion_score,
        )
        return pred.delay_probability
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# App factory — accepts an injected weather client (for testing)
# ---------------------------------------------------------------------------

def create_app(weather_client: Optional[WeatherClient] = None) -> FastAPI:
    """
    Create and return the FastAPI app.

    Parameters
    ----------
    weather_client : WeatherClient or subclass
        Injected client. Defaults to WeatherClient() (live HTTP).
        Pass MockWeatherClient() in tests to avoid real network calls.
    """
    _weather_client: WeatherClient = weather_client or WeatherClient()

    app = FastAPI(
        title="Flight Ops Intelligence API",
        description="V2 — real-time delay prediction combining ML + live Open-Meteo weather.",
        version="2.0.0",
    )

    @app.get("/health")
    async def health():
        model = _try_load_model()
        return {"status": "ok", "model_loaded": model is not None}

    @app.get("/airport-weather/{iata_code}")
    async def get_airport_weather(iata_code: str) -> dict:
        """Return current weather for an airport by IATA code."""
        code = iata_code.upper()
        if code not in AIRPORT_COORDS:
            raise HTTPException(
                status_code=404,
                detail=f"Airport '{code}' not found. Supported: {sorted(AIRPORT_COORDS)}",
            )
        coords = AIRPORT_COORDS[code]
        weather = _weather_client.get_airport_weather(coords["lat"], coords["lon"])
        risk = _weather_client.get_weather_risk_score(weather)
        return {
            "iata": code,
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "weather": weather,
            "risk_score": risk,
        }

    @app.post("/predict-delay", response_model=DelayPrediction)
    async def predict_delay_endpoint(flight: FlightRequest) -> DelayPrediction:
        """
        Predict flight delay by combining ML model output with live weather.

        final_probability = 0.7 × model_prob + 0.3 × weather_risk
        If ML model is unavailable, weather_risk alone drives the estimate.
        """
        origin = flight.origin.upper()
        dest = flight.destination.upper()

        if origin not in AIRPORT_COORDS:
            raise HTTPException(status_code=404, detail=f"Origin airport '{origin}' not found.")
        if dest not in AIRPORT_COORDS:
            raise HTTPException(status_code=404, detail=f"Destination airport '{dest}' not found.")

        # Parse departure time
        try:
            dep_dt = datetime.fromisoformat(flight.scheduled_departure)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail="scheduled_departure must be ISO-8601, e.g. 2026-06-01T08:00:00",
            )

        orig_coords = AIRPORT_COORDS[origin]
        dest_coords = AIRPORT_COORDS[dest]

        # Fetch route weather
        route = _weather_client.get_route_weather(
            orig_coords["lat"], orig_coords["lon"],
            dest_coords["lat"], dest_coords["lon"],
        )
        weather_risk = route["max_risk_score"]
        orig_weather = route["origin_weather"]
        dest_weather = route["dest_weather"]

        # Try ML model
        from geopy.distance import geodesic
        distance_mi = geodesic(
            (orig_coords["lat"], orig_coords["lon"]),
            (dest_coords["lat"], dest_coords["lon"]),
        ).miles

        # Crude hub-based congestion heuristic (mirrors V1 FeatureBuilder)
        hubs = {"ORD", "ATL", "DFW", "JFK", "LAX"}
        peak_hours = set(range(7, 10)) | set(range(16, 20))
        congestion = 0.0
        if origin in hubs:
            congestion += 0.3
        if dest in hubs:
            congestion += 0.2
        if dep_dt.hour in peak_hours:
            congestion += 0.3
        congestion = min(congestion, 1.0)

        model_prob = _model_delay_probability(
            dep_hour=dep_dt.hour,
            day_of_week=dep_dt.weekday(),
            origin_weather_score=weather_risk,
            dest_weather_score=_weather_client.get_weather_risk_score(dest_weather),
            distance_mi=distance_mi,
            aircraft_type=flight.aircraft_type,
            route_congestion_score=congestion,
        )

        if model_prob is not None:
            final_prob = round(0.7 * model_prob + 0.3 * weather_risk, 3)
            confidence = "high"
            explanation = (
                f"ML model probability {model_prob:.2f} (weight 0.70) combined with "
                f"weather risk {weather_risk:.2f} (weight 0.30) → {final_prob:.2f}."
            )
        else:
            final_prob = round(weather_risk, 3)
            confidence = "low"
            explanation = (
                f"ML model unavailable — delay probability driven entirely by "
                f"live weather risk score {weather_risk:.2f}."
            )

        # Expected delay heuristic: probability × 90 min max delay
        expected_delay = round(final_prob * 90.0, 1)

        # Weather summary
        wind = orig_weather.get("wind_speed_knots", 0)
        vis = orig_weather.get("visibility_km", 16)
        precip = orig_weather.get("precipitation_mm_hr", 0)
        weather_summary = (
            f"Origin: wind {wind:.0f} kn, vis {vis:.1f} km, precip {precip:.1f} mm/h. "
            f"Route max risk: {weather_risk:.2f}."
        )

        return DelayPrediction(
            origin=origin,
            destination=dest,
            delay_probability=final_prob,
            expected_delay_minutes=expected_delay,
            weather_risk_score=weather_risk,
            weather_summary=weather_summary,
            confidence=confidence,
            explanation=explanation,
        )

    return app


# ---------------------------------------------------------------------------
# Default app instance (uses live WeatherClient)
# ---------------------------------------------------------------------------

app = create_app()

"""
main.py — FastAPI application for Flight Ops Intelligence.
Exposes endpoints for flights, weather, delay predictions, and route analysis.
"""

from __future__ import annotations

import logging
import os
import re
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from api.database import get_db, init_db
from api.models import Flight as FlightModel, WeatherReport as WeatherModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiting — 60 requests/minute per IP on the prediction endpoint.
# Uses slowapi (Starlette-native wrapper around limits).
# ---------------------------------------------------------------------------

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

_limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

# IATA codes are exactly 3 uppercase ASCII letters.
_IATA_RE = re.compile(r"^[A-Z]{3}$")
# Aircraft type allowlist — must match the known training set values.
_ALLOWED_AIRCRAFT = frozenset(["Boeing 737", "Boeing 777", "Airbus A320", "Airbus A321"])
# Flight-ID format: airline (2 chars) + digits.
_FLIGHT_ID_RE = re.compile(r"^[A-Z0-9]{2,10}$")


def _validate_iata(code: str, field_name: str = "airport") -> str:
    """Raise HTTP 422 if code is not a valid 3-letter IATA code."""
    code = code.strip().upper()
    if not _IATA_RE.match(code):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid {field_name} code '{code}'. Must be exactly 3 uppercase letters.",
        )
    return code


def _validate_aircraft_type(aircraft_type: str) -> str:
    """Raise HTTP 422 if aircraft type is not in the known allowlist."""
    if aircraft_type not in _ALLOWED_AIRCRAFT:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown aircraft type '{aircraft_type}'. "
                f"Must be one of: {sorted(_ALLOWED_AIRCRAFT)}"
            ),
        )
    return aircraft_type


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

# CORS: restrict origins to configured list; falls back to localhost-only for
# development.  Set ALLOWED_ORIGINS env var (comma-separated) in production.
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS: List[str] = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()]
    if _raw_origins
    else ["http://localhost:3000", "http://localhost:8501", "http://127.0.0.1:8501"]
)

app = FastAPI(
    title="Flight Ops Intelligence API",
    description=(
        "ML-powered aviation analytics platform. "
        "Provides delay predictions, weather impact scores, "
        "route risk analysis, and ATC sector load data."
    ),
    version="1.1.0",
    # Disable interactive docs in production via env var
    docs_url="/docs" if os.environ.get("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.environ.get("ENVIRONMENT") != "production" else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
    allow_credentials=False,
)

# Wire rate-limiter state and its 429 error handler.
app.state.limiter = _limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ---------------------------------------------------------------------------
# Global exception handler — suppress internal details from API consumers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def _generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again later."},
    )


@app.on_event("startup")
def startup() -> None:
    init_db()
    _seed_database_if_empty()


def _seed_database_if_empty() -> None:
    """Seed the SQLite database from generated CSV on first start."""
    from api.database import SessionLocal
    from intelligence.flight_data import load_flights, AIRPORTS
    from intelligence.weather_engine import get_engine

    db = SessionLocal()
    try:
        if db.query(FlightModel).count() > 0:
            return  # already seeded

        print("[api] Seeding database from flight data…")
        df = load_flights()
        for _, row in df.iterrows():
            flight = FlightModel(
                flight_id=row["flight_id"],
                airline=row["airline"],
                flight_number=row["flight_number"],
                aircraft_type=row["aircraft_type"],
                origin=row["origin"],
                destination=row["destination"],
                distance_mi=float(row["distance_mi"]),
                scheduled_departure=datetime.fromisoformat(row["scheduled_departure"]),
                scheduled_arrival=datetime.fromisoformat(row["scheduled_arrival"]),
                actual_departure=datetime.fromisoformat(row["actual_departure"]),
                actual_arrival=datetime.fromisoformat(row["actual_arrival"]),
                delay_minutes=float(row["delay_minutes"]),
                delay_cause=str(row["delay_cause"]),
                is_delayed=bool(row["is_delayed"]),
                dep_hour=int(row["dep_hour"]),
                day_of_week=int(row["day_of_week"]),
                month=int(row["month"]),
            )
            db.add(flight)
        db.commit()
        print(f"[api] Seeded {df.shape[0]} flights")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class FlightOut(BaseModel):
    flight_id: str
    airline: str
    flight_number: str
    aircraft_type: str
    origin: str
    destination: str
    distance_mi: float
    delay_minutes: float
    delay_cause: str
    is_delayed: bool
    dep_hour: int

    class Config:
        from_attributes = True


class WeatherOut(BaseModel):
    airport: str
    conditions: str
    visibility_sm: float
    ceiling_ft: int
    wind_speed_kts: float
    weather_impact_score: float


class DelayPredictionOut(BaseModel):
    flight_id: Optional[str] = None
    predicted_delay_minutes: float
    delay_probability: float
    confidence_interval: List[float]
    feature_importances: dict
    risk_label: str


class RouteRiskOut(BaseModel):
    origin: str
    destination: str
    total_distance_mi: float
    risk_score: float
    max_weather_impact: float
    mean_weather_impact: float
    weather_impacted_waypoints: int


class SectorLoadOut(BaseModel):
    sector_loads: dict    # {sector_label: {hour: count}}
    peak_hour: int
    peak_sector: str
    overloaded_count: int


class StatsOut(BaseModel):
    total_flights: int
    delayed_flights: int
    delay_rate: float
    mean_delay_minutes: float
    top_delay_routes: List[dict]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return """
    <html><head><title>Flight Ops Intelligence</title></head>
    <body style="background:#0d1117;color:#e6edf3;font-family:monospace;padding:40px">
    <h1>✈ Flight Ops Intelligence API</h1>
    <p>ML-powered aviation analytics & delay prediction.</p>
    <ul>
      <li><a href="/docs" style="color:#58a6ff">/docs</a> — Swagger UI</li>
      <li><a href="/redoc" style="color:#58a6ff">/redoc</a> — ReDoc</li>
      <li><a href="/flights" style="color:#58a6ff">/flights</a> — All flights</li>
      <li><a href="/stats" style="color:#58a6ff">/stats</a> — Summary statistics</li>
      <li><a href="/weather/JFK" style="color:#58a6ff">/weather/{airport}</a> — Airport weather</li>
      <li><a href="/predict?origin=JFK&destination=LAX&dep_hour=8" style="color:#58a6ff">/predict</a> — Delay prediction</li>
    </ul>
    </body></html>
    """


@app.get("/flights", response_model=List[FlightOut])
def list_flights(
    origin: Optional[str] = Query(None, description="Filter by origin IATA (3 letters)", max_length=3),
    destination: Optional[str] = Query(None, description="Filter by destination IATA (3 letters)", max_length=3),
    is_delayed: Optional[bool] = Query(None, description="Filter delayed flights"),
    limit: int = Query(50, ge=1, le=200, description="Max results"),
    db: Session = Depends(get_db),
) -> List[FlightOut]:
    """List flights with optional filters."""
    q = db.query(FlightModel)
    if origin:
        q = q.filter(FlightModel.origin == _validate_iata(origin, "origin"))
    if destination:
        q = q.filter(FlightModel.destination == _validate_iata(destination, "destination"))
    if is_delayed is not None:
        q = q.filter(FlightModel.is_delayed == is_delayed)
    return q.limit(limit).all()


@app.get("/flights/{flight_id}", response_model=FlightOut)
def get_flight(flight_id: str, db: Session = Depends(get_db)) -> FlightOut:
    """Get a single flight by ID."""
    # Sanitize flight_id: only allow alphanumeric + dash, max 20 chars.
    if not re.match(r"^[A-Z0-9\-]{2,20}$", flight_id.upper()):
        raise HTTPException(status_code=422, detail="Invalid flight_id format.")
    flight = db.query(FlightModel).filter(FlightModel.flight_id == flight_id.upper()).first()
    if not flight:
        raise HTTPException(status_code=404, detail="Flight not found.")
    return flight


@app.get("/weather/{airport}", response_model=WeatherOut)
def get_airport_weather(airport: str, hour: int = Query(9, ge=0, le=23)) -> WeatherOut:
    """Get current synthetic weather for an airport."""
    from intelligence.weather_engine import get_weather, CLIMATE_PROFILES
    airport = _validate_iata(airport, "airport")
    if airport not in CLIMATE_PROFILES:
        raise HTTPException(status_code=404, detail="Airport not found.")
    ts = datetime(2024, 1, 15, hour, 0)
    rpt = get_weather(airport, ts)
    return WeatherOut(
        airport=rpt.airport,
        conditions=rpt.conditions,
        visibility_sm=rpt.visibility_sm,
        ceiling_ft=rpt.ceiling_ft,
        wind_speed_kts=rpt.wind_speed_kts,
        weather_impact_score=rpt.weather_impact_score,
    )


@app.get("/predict", response_model=DelayPredictionOut)
@_limiter.limit("30/minute")
def predict_delay(
    request: Request,
    origin: str = Query(..., description="Origin IATA code (3 letters)", max_length=3),
    destination: str = Query(..., description="Destination IATA code (3 letters)", max_length=3),
    dep_hour: int = Query(9, ge=0, le=23),
    day_of_week: int = Query(0, ge=0, le=6),
    aircraft_type: str = Query("Boeing 737", max_length=32),
    distance_mi: Optional[float] = Query(None, ge=1.0, le=15000.0),
) -> DelayPredictionOut:
    """Predict departure delay for a flight."""
    from intelligence.delay_predictor import predict_delay as _predict
    from intelligence.weather_engine import get_weather_impact_score
    from intelligence.flight_data import AIRPORTS
    import math

    origin = _validate_iata(origin, "origin")
    dest = _validate_iata(destination, "destination")
    _validate_aircraft_type(aircraft_type)

    ts = datetime(2024, 1, 15, dep_hour, 0)
    origin_score = get_weather_impact_score(origin, ts) if origin in AIRPORTS else 0.3
    dest_score = get_weather_impact_score(dest, ts) if dest in AIRPORTS else 0.3

    if distance_mi is None:
        if origin in AIRPORTS and dest in AIRPORTS:
            import numpy as np
            o, d = AIRPORTS[origin], AIRPORTS[dest]
            lat1, lon1 = math.radians(o["lat"]), math.radians(o["lon"])
            lat2, lon2 = math.radians(d["lat"]), math.radians(d["lon"])
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
            distance_mi = 3958.8 * 2 * math.asin(math.sqrt(a))
        else:
            distance_mi = 1500.0

    hubs = {"ORD", "ATL", "DFW", "JFK", "LAX"}
    peak = set(range(7, 10)) | set(range(16, 20))
    congestion = min(1.0, (0.3 if origin in hubs else 0) + (0.2 if dest in hubs else 0) + (0.3 if dep_hour in peak else 0))

    pred = _predict(
        dep_hour=dep_hour,
        day_of_week=day_of_week,
        origin_weather_score=origin_score,
        dest_weather_score=dest_score,
        distance_mi=distance_mi,
        aircraft_type=aircraft_type,
        route_congestion_score=congestion,
    )

    risk_label = (
        "HIGH" if pred.delay_probability > 0.6 else
        "MODERATE" if pred.delay_probability > 0.35 else
        "LOW"
    )

    return DelayPredictionOut(
        predicted_delay_minutes=pred.predicted_delay_minutes,
        delay_probability=pred.delay_probability,
        confidence_interval=list(pred.confidence_interval),
        feature_importances=pred.feature_importances,
        risk_label=risk_label,
    )


@app.get("/route-risk", response_model=RouteRiskOut)
@_limiter.limit("30/minute")
def route_risk(
    request: Request,
    origin: str = Query(..., description="Origin IATA code (3 letters)", max_length=3),
    destination: str = Query(..., description="Destination IATA code (3 letters)", max_length=3),
    dep_hour: int = Query(9, ge=0, le=23),
) -> RouteRiskOut:
    """Compute weather risk score along a route."""
    from intelligence.route_analyzer import compute_route_risk_score
    from intelligence.flight_data import AIRPORTS

    origin = _validate_iata(origin, "origin")
    destination = _validate_iata(destination, "destination")
    for code, name in [(origin, "origin"), (destination, "destination")]:
        if code not in AIRPORTS:
            raise HTTPException(status_code=404, detail=f"Unknown {name} airport.")

    ts = datetime(2024, 1, 15, dep_hour, 0)
    analysis = compute_route_risk_score(origin, destination, ts)
    return RouteRiskOut(
        origin=analysis.origin,
        destination=analysis.destination,
        total_distance_mi=analysis.total_distance_mi,
        risk_score=analysis.risk_score,
        max_weather_impact=analysis.max_weather_impact,
        mean_weather_impact=analysis.mean_weather_impact,
        weather_impacted_waypoints=analysis.weather_impacted_waypoints,
    )


@app.get("/sector-load", response_model=SectorLoadOut)
def sector_load() -> SectorLoadOut:
    """Get ATC sector load for a 24-hour window."""
    from intelligence.atc_simulator import compute_sector_load, identify_overloaded_sectors

    df = compute_sector_load()
    overloaded = identify_overloaded_sectors(df)

    sector_dict = {}
    for sector in df.index:
        sector_dict[str(sector)] = {str(h): int(df.loc[sector, h]) for h in df.columns}

    flat = df.values
    peak_idx = flat.argmax()
    peak_sector_idx = peak_idx // 24
    peak_hour = peak_idx % 24

    return SectorLoadOut(
        sector_loads=sector_dict,
        peak_hour=int(peak_hour),
        peak_sector=str(df.index[peak_sector_idx]),
        overloaded_count=len(overloaded),
    )


@app.get("/stats", response_model=StatsOut)
def stats(db: Session = Depends(get_db)) -> StatsOut:
    """Summary statistics for all flights."""
    from sqlalchemy import func, desc

    total = db.query(FlightModel).count()
    delayed = db.query(FlightModel).filter(FlightModel.is_delayed == True).count()

    mean_delay_row = db.query(func.avg(FlightModel.delay_minutes)).scalar() or 0.0

    # Top delay routes
    routes = (
        db.query(
            FlightModel.origin,
            FlightModel.destination,
            func.avg(FlightModel.delay_minutes).label("avg_delay"),
            func.count(FlightModel.id).label("flight_count"),
        )
        .group_by(FlightModel.origin, FlightModel.destination)
        .order_by(desc("avg_delay"))
        .limit(5)
        .all()
    )

    return StatsOut(
        total_flights=total,
        delayed_flights=delayed,
        delay_rate=round(delayed / total if total > 0 else 0, 3),
        mean_delay_minutes=round(float(mean_delay_row), 1),
        top_delay_routes=[
            {
                "route": f"{r.origin}→{r.destination}",
                "avg_delay": round(float(r.avg_delay), 1),
                "flight_count": r.flight_count,
            }
            for r in routes
        ],
    )


@app.get("/map", response_class=HTMLResponse)
def serve_map() -> str:
    """Serve the pre-generated Folium flight map (static file, not user-controlled)."""
    # Resolve against the project root so path traversal via symlink or relative
    # segments cannot escape the expected directory.
    base = Path(__file__).parent.parent.resolve()
    map_path = (base / "maps" / "flight_map.html").resolve()

    # Confirm the resolved path is still inside the expected directory.
    if not str(map_path).startswith(str(base / "maps")):
        raise HTTPException(status_code=400, detail="Invalid map path.")

    if not map_path.exists():
        return (
            "<html><body>"
            "<p>Map not yet generated. Run: python -m intelligence.map_generator</p>"
            "</body></html>"
        )
    return map_path.read_text(encoding="utf-8")

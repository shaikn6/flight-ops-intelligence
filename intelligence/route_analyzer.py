"""
route_analyzer.py — Great-circle route analysis with weather intersection.
Uses geopy for geodesic calculations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from geopy.distance import geodesic

from intelligence.flight_data import AIRPORTS
from intelligence.weather_engine import WeatherEngine, get_engine

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Waypoint:
    lat: float
    lon: float
    distance_from_origin_mi: float
    weather_impact_score: float = 0.0
    nearest_airport: Optional[str] = None
    dist_to_nearest_apt_mi: float = float("inf")


@dataclass
class RouteAnalysis:
    origin: str
    destination: str
    departure_time: datetime
    total_distance_mi: float
    waypoints: List[Waypoint]
    max_weather_impact: float
    mean_weather_impact: float
    weather_impacted_waypoints: int
    risk_score: float          # 0-1 aggregate route risk


# ---------------------------------------------------------------------------
# Great-circle waypoints
# ---------------------------------------------------------------------------

def _interpolate_great_circle(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    n: int,
) -> List[Tuple[float, float]]:
    """
    Compute n evenly-spaced waypoints along the great-circle arc
    from (lat1, lon1) to (lat2, lon2).

    Uses spherical linear interpolation (slerp) on unit vectors.
    """
    def to_xyz(lat_deg: float, lon_deg: float) -> np.ndarray:
        lat, lon = math.radians(lat_deg), math.radians(lon_deg)
        return np.array([
            math.cos(lat) * math.cos(lon),
            math.cos(lat) * math.sin(lon),
            math.sin(lat),
        ])

    def from_xyz(v: np.ndarray) -> Tuple[float, float]:
        lat = math.degrees(math.asin(float(np.clip(v[2], -1, 1))))
        lon = math.degrees(math.atan2(v[1], v[0]))
        return lat, lon

    v1 = to_xyz(lat1, lon1)
    v2 = to_xyz(lat2, lon2)

    dot = float(np.clip(np.dot(v1, v2), -1, 1))
    omega = math.acos(dot)

    points: List[Tuple[float, float]] = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0.0
        if omega < 1e-10:
            # Points are nearly coincident
            points.append(from_xyz(v1))
        else:
            sin_omega = math.sin(omega)
            v = (math.sin((1 - t) * omega) / sin_omega) * v1 + \
                (math.sin(t * omega) / sin_omega) * v2
            v = v / np.linalg.norm(v)
            points.append(from_xyz(v))

    return points


def great_circle_route(
    origin: str,
    dest: str,
    n_waypoints: int = 20,
) -> List[Tuple[float, float]]:
    """
    Generate lat/lon waypoints along the great-circle arc.

    Parameters
    ----------
    origin, dest : IATA codes
    n_waypoints : number of points including endpoints

    Returns
    -------
    List of (lat, lon) tuples
    """
    orig = AIRPORTS[origin]
    dst = AIRPORTS[dest]
    return _interpolate_great_circle(
        orig["lat"], orig["lon"],
        dst["lat"], dst["lon"],
        n_waypoints,
    )


# ---------------------------------------------------------------------------
# Weather intersection
# ---------------------------------------------------------------------------

def check_weather_intersection(
    waypoints: List[Tuple[float, float]],
    weather_engine: Optional[WeatherEngine] = None,
    timestamp: Optional[datetime] = None,
    impact_threshold: float = 0.4,
) -> List[Waypoint]:
    """
    Annotate each waypoint with weather impact from the nearest airport.

    Parameters
    ----------
    waypoints      : List of (lat, lon) tuples
    weather_engine : WeatherEngine instance (uses singleton if None)
    timestamp      : Evaluation time (uses 2024-01-01 07:00 if None)
    impact_threshold : Score above which waypoint is 'impacted'

    Returns
    -------
    List of Waypoint objects with weather_impact_score populated
    """
    if weather_engine is None:
        weather_engine = get_engine()
    if timestamp is None:
        timestamp = datetime(2024, 1, 1, 7, 0)

    apt_positions = {
        code: (info["lat"], info["lon"])
        for code, info in AIRPORTS.items()
    }

    result: List[Waypoint] = []
    cumulative_dist = 0.0

    for i, (lat, lon) in enumerate(waypoints):
        # Cumulative distance from origin
        if i > 0:
            prev_lat, prev_lon = waypoints[i - 1]
            seg_dist = geodesic((prev_lat, prev_lon), (lat, lon)).miles
            cumulative_dist += seg_dist

        # Find nearest airport
        nearest_apt = None
        min_dist = float("inf")
        for code, (alat, alon) in apt_positions.items():
            d = geodesic((lat, lon), (alat, alon)).miles
            if d < min_dist:
                min_dist = d
                nearest_apt = code

        # Weather impact from nearest airport
        if nearest_apt and min_dist < 300:
            impact = weather_engine.get_weather_impact_score(nearest_apt, timestamp)
        else:
            impact = 0.0

        result.append(Waypoint(
            lat=round(lat, 4),
            lon=round(lon, 4),
            distance_from_origin_mi=round(cumulative_dist, 1),
            weather_impact_score=round(impact, 3),
            nearest_airport=nearest_apt,
            dist_to_nearest_apt_mi=round(min_dist, 1),
        ))

    return result


# ---------------------------------------------------------------------------
# Route risk scoring
# ---------------------------------------------------------------------------

def compute_route_risk_score(
    origin: str,
    dest: str,
    departure_time: datetime,
    n_waypoints: int = 20,
    weather_engine: Optional[WeatherEngine] = None,
) -> RouteAnalysis:
    """
    Compute aggregate weather risk score for a route.

    Risk score formula:
        0.5 × (max impact along route)
        + 0.3 × (mean impact)
        + 0.2 × (fraction of waypoints with impact > 0.4)
    """
    if weather_engine is None:
        weather_engine = get_engine()

    orig_info = AIRPORTS[origin]
    dest_info = AIRPORTS[dest]
    total_dist = geodesic(
        (orig_info["lat"], orig_info["lon"]),
        (dest_info["lat"], dest_info["lon"]),
    ).miles

    raw_wps = great_circle_route(origin, dest, n_waypoints)
    annotated = check_weather_intersection(
        raw_wps, weather_engine, departure_time
    )

    scores = [wp.weather_impact_score for wp in annotated]
    max_impact = max(scores) if scores else 0.0
    mean_impact = float(np.mean(scores)) if scores else 0.0
    impacted_count = sum(1 for s in scores if s > 0.4)
    impacted_frac = impacted_count / len(scores) if scores else 0.0

    risk = 0.5 * max_impact + 0.3 * mean_impact + 0.2 * impacted_frac

    return RouteAnalysis(
        origin=origin,
        destination=dest,
        departure_time=departure_time,
        total_distance_mi=round(total_dist, 1),
        waypoints=annotated,
        max_weather_impact=round(max_impact, 3),
        mean_weather_impact=round(mean_impact, 3),
        weather_impacted_waypoints=impacted_count,
        risk_score=round(float(np.clip(risk, 0, 1)), 3),
    )


def analyze_all_routes(flights_df: pd.DataFrame) -> pd.DataFrame:
    """Compute route risk scores for all flights in the DataFrame."""
    engine = get_engine()
    records = []
    for _, row in flights_df.iterrows():
        try:
            ts = datetime.fromisoformat(row["scheduled_departure"])
            analysis = compute_route_risk_score(
                row["origin"], row["destination"], ts, weather_engine=engine
            )
            records.append({
                "flight_id": row["flight_id"],
                "origin": row["origin"],
                "destination": row["destination"],
                "route_risk_score": analysis.risk_score,
                "max_weather_impact": analysis.max_weather_impact,
                "mean_weather_impact": analysis.mean_weather_impact,
                "weather_impacted_waypoints": analysis.weather_impacted_waypoints,
            })
        except (KeyError, ValueError):
            continue
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ts = datetime(2024, 1, 1, 8, 0)

    routes = [("JFK", "LAX"), ("ORD", "MIA"), ("SFO", "SEA"), ("DFW", "BOS")]
    for origin, dest in routes:
        analysis = compute_route_risk_score(origin, dest, ts)
        print(
            f"{origin}→{dest}: dist={analysis.total_distance_mi:.0f}mi "
            f"risk={analysis.risk_score:.3f} "
            f"max_impact={analysis.max_weather_impact:.3f} "
            f"impacted_wps={analysis.weather_impacted_waypoints}"
        )

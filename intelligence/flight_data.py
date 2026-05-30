"""
flight_data.py — Synthetic FAA-style flight data generator.
Generates 500 realistic domestic flights across 10 major US airports.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Airport metadata
# ---------------------------------------------------------------------------

AIRPORTS = {
    "JFK": {"name": "John F. Kennedy International", "lat": 40.6413, "lon": -73.7781, "city": "New York"},
    "LAX": {"name": "Los Angeles International", "lat": 33.9425, "lon": -118.4081, "city": "Los Angeles"},
    "ORD": {"name": "O'Hare International", "lat": 41.9742, "lon": -87.9073, "city": "Chicago"},
    "DFW": {"name": "Dallas/Fort Worth International", "lat": 32.8998, "lon": -97.0403, "city": "Dallas"},
    "ATL": {"name": "Hartsfield-Jackson Atlanta International", "lat": 33.6407, "lon": -84.4277, "city": "Atlanta"},
    "SFO": {"name": "San Francisco International", "lat": 37.6213, "lon": -122.3790, "city": "San Francisco"},
    "SEA": {"name": "Seattle-Tacoma International", "lat": 47.4502, "lon": -122.3088, "city": "Seattle"},
    "MIA": {"name": "Miami International", "lat": 25.7959, "lon": -80.2870, "city": "Miami"},
    "BOS": {"name": "Logan International", "lat": 42.3656, "lon": -71.0096, "city": "Boston"},
    "DEN": {"name": "Denver International", "lat": 39.8561, "lon": -104.6737, "city": "Denver"},
}

AIRCRAFT_TYPES = ["Boeing 737", "Boeing 777", "Airbus A320", "Airbus A321"]

DELAY_CAUSES = {
    "weather": 0.40,
    "carrier": 0.30,
    "nas": 0.20,
    "security": 0.05,
    "late_aircraft": 0.05,
}

AIRLINES = ["AA", "UA", "DL", "WN", "AS", "B6", "NK", "F9"]

BASE_DATE = datetime(2024, 1, 1, 0, 0, 0)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _haversine_mi(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in miles."""
    R = 3958.8
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def _expected_flight_time(distance_mi: float) -> int:
    """Estimate flight time in minutes from distance."""
    airspeed_mph = 520
    taxi_and_approach = 40
    return int(distance_mi / airspeed_mph * 60 + taxi_and_approach)


def _sample_delay(delay_probability: float, rng: np.random.Generator) -> float:
    """Sample delay minutes given a per-flight delay probability."""
    if rng.random() > delay_probability:
        return 0.0
    # Delay minutes follow a log-normal distribution (many small, few large)
    delay = rng.lognormal(mean=3.0, sigma=0.8)
    return float(np.clip(delay, 1, 180))


def _sample_delay_cause(rng: np.random.Generator) -> str:
    causes = list(DELAY_CAUSES.keys())
    weights = list(DELAY_CAUSES.values())
    return rng.choice(causes, p=weights)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_flights(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate n synthetic domestic flights.

    Returns a DataFrame with columns:
        flight_id, airline, flight_number, aircraft_type,
        origin, destination, distance_mi,
        scheduled_departure, scheduled_arrival,
        actual_departure, actual_arrival,
        delay_minutes, delay_cause, is_delayed,
        dep_hour, day_of_week, month
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    iata_codes = list(AIRPORTS.keys())
    records = []

    for i in range(n):
        # Pick origin / destination (no self-loops)
        origin, dest = rng.choice(iata_codes, size=2, replace=False)
        orig_info = AIRPORTS[origin]
        dest_info = AIRPORTS[dest]

        distance_mi = _haversine_mi(
            orig_info["lat"], orig_info["lon"],
            dest_info["lat"], dest_info["lon"],
        )
        flight_time_min = _expected_flight_time(distance_mi)

        # Scheduled departure: random time across 30-day window
        day_offset = int(rng.integers(0, 30))
        dep_hour = int(rng.integers(5, 23))
        dep_minute = int(rng.choice([0, 15, 30, 45]))
        sched_dep = BASE_DATE + timedelta(days=day_offset, hours=dep_hour, minutes=dep_minute)
        sched_arr = sched_dep + timedelta(minutes=flight_time_min)

        # Weather-influenced delay probability (morning fog, afternoon thunderstorms)
        base_prob = 0.35
        if dep_hour in range(6, 10):   # morning fog window
            weather_prob_boost = 0.15
        elif dep_hour in range(14, 19):  # afternoon convective window
            weather_prob_boost = 0.20
        else:
            weather_prob_boost = 0.05
        delay_prob = base_prob + weather_prob_boost * rng.random()

        delay_min = _sample_delay(delay_prob, rng)
        delay_cause = _sample_delay_cause(rng) if delay_min > 0 else "none"

        actual_dep = sched_dep + timedelta(minutes=delay_min)
        actual_arr = sched_arr + timedelta(minutes=delay_min + rng.integers(0, 10))

        airline = rng.choice(AIRLINES)
        flight_number = f"{airline}{rng.integers(100, 9999)}"
        aircraft = rng.choice(AIRCRAFT_TYPES)

        records.append({
            "flight_id": f"FLT{i:04d}",
            "airline": airline,
            "flight_number": flight_number,
            "aircraft_type": aircraft,
            "origin": origin,
            "destination": dest,
            "distance_mi": round(distance_mi, 1),
            "scheduled_departure": sched_dep.isoformat(),
            "scheduled_arrival": sched_arr.isoformat(),
            "actual_departure": actual_dep.isoformat(),
            "actual_arrival": actual_arr.isoformat(),
            "delay_minutes": round(delay_min, 1),
            "delay_cause": delay_cause,
            "is_delayed": int(delay_min >= 15),
            "dep_hour": dep_hour,
            "day_of_week": sched_dep.weekday(),
            "month": sched_dep.month,
        })

    df = pd.DataFrame(records)
    return df


def save_flights(df: pd.DataFrame, path: str = "data/flights.csv") -> None:
    """Persist flights DataFrame to CSV."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[flight_data] Saved {len(df)} flights to {out}")


def load_flights(path: str = "data/flights.csv") -> pd.DataFrame:
    """Load flights from CSV, generating fresh data if file absent."""
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    df = generate_flights()
    save_flights(df, path)
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = generate_flights()
    save_flights(df)
    print(df.describe())
    print(f"\nDelay rate: {df['is_delayed'].mean():.1%}")
    print(f"Mean delay (when delayed): {df.loc[df['delay_minutes'] > 0, 'delay_minutes'].mean():.1f} min")
    print("\nDelay cause distribution:")
    print(df["delay_cause"].value_counts(normalize=True).round(3))

"""
weather_engine.py — Synthetic METAR-style weather data generator.
Produces realistic weather time series for 10 major US airports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WeatherReport:
    airport: str
    timestamp: datetime
    wind_speed_kts: float
    wind_dir_deg: float
    visibility_sm: float          # statute miles
    ceiling_ft: int               # feet AGL
    temperature_c: float
    dewpoint_c: float
    altimeter_inhg: float
    conditions: str               # VMC / MVMC / IMC
    weather_impact_score: float = field(init=False)

    def __post_init__(self) -> None:
        self.weather_impact_score = compute_impact_score(self)


# ---------------------------------------------------------------------------
# Flight condition classification
# ---------------------------------------------------------------------------

def classify_conditions(visibility_sm: float, ceiling_ft: int) -> str:
    """Classify VFR/IFR flight conditions per FAA standards."""
    if visibility_sm >= 3 and ceiling_ft >= 1000:
        return "VMC"
    if visibility_sm >= 1 and ceiling_ft >= 500:
        return "MVMC"
    return "IMC"


def compute_impact_score(report: "WeatherReport") -> float:
    """
    Compute weather impact score [0, 1].
    0 = perfect VMC, 1 = severe IFR / near-zero visibility.
    """
    # Visibility penalty: < 1 SM → max impact
    vis_score = max(0.0, 1.0 - report.visibility_sm / 10.0)
    vis_score = min(1.0, vis_score * 1.5)

    # Ceiling penalty: < 500 ft → max impact
    ceil_score = max(0.0, 1.0 - report.ceiling_ft / 5000.0)
    ceil_score = min(1.0, ceil_score * 2.0)

    # Wind penalty: gusts > 25 kts add significant impact
    wind_score = min(1.0, report.wind_speed_kts / 40.0)

    # Composite — weighted average
    score = 0.40 * vis_score + 0.40 * ceil_score + 0.20 * wind_score
    return round(float(np.clip(score, 0.0, 1.0)), 3)


# ---------------------------------------------------------------------------
# Airport climate profiles
# ---------------------------------------------------------------------------

# Each profile defines seasonal/diurnal behavior.
# Keys: base_visibility, base_ceiling, wind_mean, fog_morning_prob, storm_afternoon_prob
CLIMATE_PROFILES = {
    "JFK": {
        "base_visibility": 9.0,
        "base_ceiling": 4500,
        "wind_mean": 12,
        "fog_morning_prob": 0.15,
        "storm_afternoon_prob": 0.10,
        "base_temp_c": 12,
    },
    "LAX": {
        "base_visibility": 9.5,
        "base_ceiling": 5000,
        "wind_mean": 8,
        "fog_morning_prob": 0.25,   # marine layer
        "storm_afternoon_prob": 0.03,
        "base_temp_c": 18,
    },
    "ORD": {
        "base_visibility": 8.0,
        "base_ceiling": 3500,
        "wind_mean": 15,
        "fog_morning_prob": 0.12,
        "storm_afternoon_prob": 0.18,
        "base_temp_c": 8,
    },
    "DFW": {
        "base_visibility": 9.0,
        "base_ceiling": 4000,
        "wind_mean": 14,
        "fog_morning_prob": 0.08,
        "storm_afternoon_prob": 0.22,
        "base_temp_c": 20,
    },
    "ATL": {
        "base_visibility": 8.5,
        "base_ceiling": 3800,
        "wind_mean": 10,
        "fog_morning_prob": 0.10,
        "storm_afternoon_prob": 0.20,
        "base_temp_c": 16,
    },
    "SFO": {
        "base_visibility": 8.0,
        "base_ceiling": 3000,
        "wind_mean": 12,
        "fog_morning_prob": 0.30,   # famous bay area fog
        "storm_afternoon_prob": 0.05,
        "base_temp_c": 14,
    },
    "SEA": {
        "base_visibility": 7.5,
        "base_ceiling": 2500,
        "wind_mean": 10,
        "fog_morning_prob": 0.20,
        "storm_afternoon_prob": 0.08,
        "base_temp_c": 10,
    },
    "MIA": {
        "base_visibility": 9.5,
        "base_ceiling": 5000,
        "wind_mean": 12,
        "fog_morning_prob": 0.05,
        "storm_afternoon_prob": 0.35,   # tropical convection
        "base_temp_c": 26,
    },
    "BOS": {
        "base_visibility": 8.5,
        "base_ceiling": 3800,
        "wind_mean": 13,
        "fog_morning_prob": 0.18,
        "storm_afternoon_prob": 0.12,
        "base_temp_c": 11,
    },
    "DEN": {
        "base_visibility": 9.0,
        "base_ceiling": 6000,
        "wind_mean": 11,
        "fog_morning_prob": 0.05,
        "storm_afternoon_prob": 0.25,   # afternoon mountain thunderstorms
        "base_temp_c": 9,
    },
}


# ---------------------------------------------------------------------------
# Synthetic weather generation
# ---------------------------------------------------------------------------

class WeatherEngine:
    """Generate and cache synthetic METAR-style weather data."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self._cache: Dict[str, List[WeatherReport]] = {}

    def _generate_for_airport(
        self, airport: str, start: datetime, n_hours: int
    ) -> List[WeatherReport]:
        """Generate hourly weather reports for one airport over n_hours."""
        profile = CLIMATE_PROFILES.get(airport, CLIMATE_PROFILES["ORD"])
        reports: List[WeatherReport] = []

        vis = profile["base_visibility"]
        ceil = profile["base_ceiling"]
        wind = profile["wind_mean"]
        temp = profile["base_temp_c"]

        for h in range(n_hours):
            ts = start + timedelta(hours=h)
            hour = ts.hour
            day = h // 24

            # --- Diurnal patterns ---
            # Morning fog (hours 5-9): reduce visibility & ceiling
            morning_fog = False
            if 5 <= hour <= 9 and self.rng.random() < profile["fog_morning_prob"]:
                morning_fog = True

            # Afternoon convection (hours 14-19)
            afternoon_storm = False
            if 14 <= hour <= 19 and self.rng.random() < profile["storm_afternoon_prob"]:
                afternoon_storm = True

            # --- Visibility ---
            if morning_fog:
                vis = self.rng.uniform(0.5, 3.0)
            elif afternoon_storm:
                vis = self.rng.uniform(1.0, 5.0)
            else:
                vis = profile["base_visibility"] + self.rng.normal(0, 0.8)
                vis = float(np.clip(vis, 1.0, 10.0))

            # --- Ceiling ---
            if morning_fog:
                ceil = int(self.rng.integers(200, 1200))
            elif afternoon_storm:
                ceil = int(self.rng.integers(500, 3000))
            else:
                ceil = profile["base_ceiling"] + int(self.rng.normal(0, 500))
                ceil = int(np.clip(ceil, 300, 12000))

            # --- Wind ---
            wind_speed = profile["wind_mean"] + self.rng.normal(0, 4)
            if afternoon_storm:
                wind_speed += self.rng.uniform(5, 20)
            wind_speed = float(np.clip(wind_speed, 0, 50))
            wind_dir = float(self.rng.uniform(0, 360))

            # --- Temperature (diurnal: cooler at night) ---
            diurnal_offset = -3 * math.cos(2 * math.pi * hour / 24)
            temp_now = profile["base_temp_c"] + diurnal_offset + self.rng.normal(0, 1.5)
            dewpoint = temp_now - self.rng.uniform(2, 10)

            altimeter = 29.92 + self.rng.normal(0, 0.15)
            altimeter = float(np.clip(altimeter, 28.5, 31.0))

            conditions = classify_conditions(vis, ceil)

            report = WeatherReport(
                airport=airport,
                timestamp=ts,
                wind_speed_kts=round(wind_speed, 1),
                wind_dir_deg=round(wind_dir, 0),
                visibility_sm=round(vis, 1),
                ceiling_ft=ceil,
                temperature_c=round(temp_now, 1),
                dewpoint_c=round(dewpoint, 1),
                altimeter_inhg=round(altimeter, 2),
                conditions=conditions,
            )
            reports.append(report)

        return reports

    def preload(
        self,
        airports: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        n_hours: int = 720,  # 30 days
    ) -> None:
        """Pre-generate weather time series for all airports."""
        if airports is None:
            airports = list(CLIMATE_PROFILES.keys())
        if start is None:
            start = datetime(2024, 1, 1, 0, 0, 0)
        for apt in airports:
            self._cache[apt] = self._generate_for_airport(apt, start, n_hours)

    def get_weather(self, airport: str, timestamp: datetime) -> WeatherReport:
        """Return the closest weather report to the given timestamp."""
        if airport not in self._cache:
            self.preload(airports=[airport])
        reports = self._cache[airport]
        # Find closest report by timestamp
        base = reports[0].timestamp
        idx = int((timestamp - base).total_seconds() / 3600)
        idx = max(0, min(idx, len(reports) - 1))
        return reports[idx]

    def get_weather_impact_score(self, airport: str, timestamp: datetime) -> float:
        """Return impact score [0,1] for airport at timestamp."""
        return self.get_weather(airport, timestamp).weather_impact_score

    def to_dataframe(self, airport: str) -> pd.DataFrame:
        """Convert cached reports for an airport to a DataFrame."""
        if airport not in self._cache:
            self.preload(airports=[airport])
        rows = []
        for r in self._cache[airport]:
            rows.append({
                "airport": r.airport,
                "timestamp": r.timestamp,
                "hour": r.timestamp.hour,
                "wind_speed_kts": r.wind_speed_kts,
                "wind_dir_deg": r.wind_dir_deg,
                "visibility_sm": r.visibility_sm,
                "ceiling_ft": r.ceiling_ft,
                "temperature_c": r.temperature_c,
                "dewpoint_c": r.dewpoint_c,
                "altimeter_inhg": r.altimeter_inhg,
                "conditions": r.conditions,
                "weather_impact_score": r.weather_impact_score,
            })
        return pd.DataFrame(rows)

    def get_hourly_impact_matrix(self) -> pd.DataFrame:
        """
        Return a matrix: rows=airports, cols=hours 0-23,
        values=mean weather_impact_score (averaged over all days).
        """
        if not self._cache:
            self.preload()

        rows = {}
        for airport in self._cache:
            hourly = [0.0] * 24
            counts = [0] * 24
            for r in self._cache[airport]:
                h = r.timestamp.hour
                hourly[h] += r.weather_impact_score
                counts[h] += 1
            rows[airport] = [
                hourly[h] / counts[h] if counts[h] > 0 else 0.0
                for h in range(24)
            ]
        return pd.DataFrame(rows, index=list(range(24))).T


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_engine: Optional[WeatherEngine] = None


def get_engine() -> WeatherEngine:
    global _engine
    if _engine is None:
        _engine = WeatherEngine()
        _engine.preload()
    return _engine


def get_weather(airport: str, timestamp: datetime) -> WeatherReport:
    return get_engine().get_weather(airport, timestamp)


def get_weather_impact_score(airport: str, timestamp: datetime) -> float:
    return get_engine().get_weather_impact_score(airport, timestamp)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    engine = WeatherEngine()
    engine.preload()
    ts = datetime(2024, 1, 1, 7, 0)
    for apt in list(CLIMATE_PROFILES.keys())[:3]:
        rpt = engine.get_weather(apt, ts)
        print(
            f"{apt}: vis={rpt.visibility_sm}sm ceil={rpt.ceiling_ft}ft "
            f"wind={rpt.wind_speed_kts}kts cond={rpt.conditions} "
            f"impact={rpt.weather_impact_score}"
        )
    matrix = engine.get_hourly_impact_matrix()
    print("\nWeather impact matrix (airports × hours):")
    print(matrix.round(3))

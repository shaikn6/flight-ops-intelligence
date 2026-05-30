"""
atc_simulator.py — ATC sector load simulator.
Divides US airspace into 20 sectors and counts aircraft per sector per hour.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from intelligence.flight_data import AIRPORTS, generate_flights, load_flights
from intelligence.route_analyzer import great_circle_route

# ---------------------------------------------------------------------------
# Sector grid definition
# ---------------------------------------------------------------------------

# US airspace roughly bounded by:
#   lat: 25°N – 50°N
#   lon: -125°W – -65°W
# Divide into 5 cols × 4 rows = 20 sectors

LAT_MIN, LAT_MAX = 25.0, 50.0
LON_MIN, LON_MAX = -125.0, -65.0
N_ROWS, N_COLS = 4, 5
N_SECTORS = N_ROWS * N_COLS  # 20

LAT_STEP = (LAT_MAX - LAT_MIN) / N_ROWS
LON_STEP = (LON_MAX - LON_MIN) / N_COLS

SECTOR_OVERLOAD_THRESHOLD = 15  # aircraft/hour → flow control alert


@dataclass
class Sector:
    sector_id: int
    row: int
    col: int
    lat_lo: float
    lat_hi: float
    lon_lo: float
    lon_hi: float
    label: str

    def contains(self, lat: float, lon: float) -> bool:
        return (self.lat_lo <= lat < self.lat_hi) and (self.lon_lo <= lon < self.lon_hi)


def build_sectors() -> List[Sector]:
    sectors: List[Sector] = []
    sid = 0
    for row in range(N_ROWS):
        for col in range(N_COLS):
            lat_lo = LAT_MIN + row * LAT_STEP
            lat_hi = lat_lo + LAT_STEP
            lon_lo = LON_MIN + col * LON_STEP
            lon_hi = lon_lo + LON_STEP
            label = f"S{sid:02d}(R{row}C{col})"
            sectors.append(Sector(
                sector_id=sid,
                row=row, col=col,
                lat_lo=lat_lo, lat_hi=lat_hi,
                lon_lo=lon_lo, lon_hi=lon_hi,
                label=label,
            ))
            sid += 1
    return sectors


SECTORS = build_sectors()


def get_sector(lat: float, lon: float) -> Optional[int]:
    """Return sector_id for a (lat, lon) position, or None if outside US box."""
    for s in SECTORS:
        if s.contains(lat, lon):
            return s.sector_id
    return None


# ---------------------------------------------------------------------------
# Flight-path sector assignment
# ---------------------------------------------------------------------------

def _flight_waypoints_at_hour(
    origin: str,
    dest: str,
    dep_hour: int,
    flight_duration_min: int,
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Returns a mapping of hour → list of (lat, lon) positions where
    the aircraft is within each clock hour during its flight.
    """
    if origin not in AIRPORTS or dest not in AIRPORTS:
        return {}

    wps = great_circle_route(origin, dest, n_waypoints=30)
    if not wps:
        return {}

    # Linearly interpolate time along waypoints
    n = len(wps)
    hour_to_latlons: Dict[int, List[Tuple[float, float]]] = {}

    for i, (lat, lon) in enumerate(wps):
        # fractional position [0,1] along the flight
        frac = i / (n - 1) if n > 1 else 0.0
        elapsed_min = frac * flight_duration_min
        hour = dep_hour + int(elapsed_min // 60)
        hour = hour % 24  # wrap if flight crosses midnight

        hour_to_latlons.setdefault(hour, []).append((lat, lon))

    return hour_to_latlons


# ---------------------------------------------------------------------------
# Sector load computation
# ---------------------------------------------------------------------------

def compute_sector_load(
    df: Optional[pd.DataFrame] = None,
    day_filter: int = 0,
) -> pd.DataFrame:
    """
    Compute ATC sector load matrix: 20 sectors × 24 hours.
    Values = number of aircraft in sector during that hour.

    Parameters
    ----------
    df         : Flights DataFrame (generated if None)
    day_filter : Day offset from BASE_DATE to analyse (0 = Jan 1)

    Returns
    -------
    DataFrame with index=sector_id (0-19), columns=hour (0-23)
    """
    if df is None:
        df = load_flights()

    # Filter to the requested day
    base_date = datetime(2024, 1, 1) + timedelta(days=day_filter)
    base_str = base_date.date().isoformat()

    # Filter flights departing on this day
    mask = df["scheduled_departure"].str.startswith(base_str)
    day_df = df[mask]

    if day_df.empty:
        # Fall back to all flights if none match the day
        day_df = df

    # Initialize load matrix
    load = np.zeros((N_SECTORS, 24), dtype=int)

    for _, row in day_df.iterrows():
        origin = str(row["origin"])
        dest = str(row["destination"])
        dep_hour = int(row["dep_hour"])

        # Estimate flight duration from distance
        dist = float(row["distance_mi"])
        duration_min = int(dist / 520 * 60 + 40)

        hour_latlons = _flight_waypoints_at_hour(origin, dest, dep_hour, duration_min)

        for hour, positions in hour_latlons.items():
            h = int(hour) % 24
            for lat, lon in positions:
                sid = get_sector(lat, lon)
                if sid is not None:
                    load[sid, h] += 1

    sector_labels = [s.label for s in SECTORS]
    hours = list(range(24))
    result = pd.DataFrame(load, index=sector_labels, columns=hours)
    result.index.name = "sector"
    result.columns.name = "hour"
    return result


def identify_overloaded_sectors(load_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify sector-hour combinations exceeding the overload threshold.
    Returns DataFrame with columns: sector, hour, aircraft_count, alert.
    """
    records = []
    for sector in load_df.index:
        for hour in load_df.columns:
            count = int(load_df.loc[sector, hour])
            if count >= SECTOR_OVERLOAD_THRESHOLD:
                records.append({
                    "sector": sector,
                    "hour": hour,
                    "aircraft_count": count,
                    "alert": "FLOW CONTROL",
                })
    if not records:
        return pd.DataFrame(columns=["sector", "hour", "aircraft_count", "alert"])
    return pd.DataFrame(records).sort_values("aircraft_count", ascending=False)


def get_peak_sector_hours(load_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Return top-N (sector, hour) pairs by aircraft count."""
    flat = load_df.reset_index().melt(id_vars="sector", var_name="hour", value_name="count")
    return flat.nlargest(top_n, "count").reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = generate_flights()
    load = compute_sector_load(df)
    overloaded = identify_overloaded_sectors(load)

    print("Sector load matrix (20 sectors × 24 hours):")
    print(load.to_string())

    print(f"\nOverloaded sector-hours ({SECTOR_OVERLOAD_THRESHOLD}+ aircraft):")
    if overloaded.empty:
        print("  None")
    else:
        print(overloaded.to_string(index=False))

    print("\nPeak traffic periods (by sector):")
    print(get_peak_sector_hours(load).to_string(index=False))

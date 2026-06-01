"""
opensky_client.py — Live OpenSky Network ADS-B integration.

Fetches real aircraft states from https://opensky-network.org/api/states/all
No authentication required (public endpoint, ~10s refresh rate limit applies).

Caches results in SQLite (refreshes every 60s).
Falls back to cached data when API is unavailable.

Set MOCK_LIVE=true (default) to use synthetic aircraft positions instead.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENSKY_URL = "https://opensky-network.org/api/states/all"
CACHE_DB_PATH = Path("data/opensky_cache.db")
CACHE_TTL_SECONDS = 60

# US airspace bounding box
US_LAT_MIN = 24.0
US_LAT_MAX = 50.0
US_LON_MIN = -125.0
US_LON_MAX = -66.0

MOCK_LIVE = os.environ.get("MOCK_LIVE", "true").lower() in ("true", "1", "yes")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AircraftState:
    icao24: str
    callsign: str
    origin_country: str
    longitude: float
    latitude: float
    altitude_m: float          # geometric altitude in metres; 0 if unknown
    velocity_mps: float        # speed over ground m/s
    heading_deg: float         # true track / heading in degrees
    on_ground: bool

    @property
    def altitude_ft(self) -> float:
        return self.altitude_m * 3.28084

    def to_dict(self) -> dict:
        d = asdict(self)
        d["altitude_ft"] = self.altitude_ft
        return d


# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------

def _ensure_cache_db() -> sqlite3.Connection:
    CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CACHE_DB_PATH))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS aircraft_cache (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at INTEGER NOT NULL,
            payload   TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fetched_at ON aircraft_cache(fetched_at)"
    )
    conn.commit()
    return conn


def _write_cache(states: List[AircraftState]) -> None:
    conn = _ensure_cache_db()
    payload = json.dumps([s.to_dict() for s in states])
    now = int(time.time())
    conn.execute(
        "INSERT INTO aircraft_cache (fetched_at, payload) VALUES (?, ?)", (now, payload)
    )
    # Keep only last 10 snapshots to bound size
    conn.execute(
        "DELETE FROM aircraft_cache WHERE id NOT IN "
        "(SELECT id FROM aircraft_cache ORDER BY fetched_at DESC LIMIT 10)"
    )
    conn.commit()
    conn.close()


def _read_cache() -> Optional[List[AircraftState]]:
    if not CACHE_DB_PATH.exists():
        return None
    conn = _ensure_cache_db()
    row = conn.execute(
        "SELECT fetched_at, payload FROM aircraft_cache ORDER BY fetched_at DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if row is None:
        return None
    fetched_at, payload = row
    age = int(time.time()) - fetched_at
    if age > CACHE_TTL_SECONDS * 5:   # stale beyond 5 minutes — skip
        return None
    raw = json.loads(payload)
    states = []
    for item in raw:
        item.pop("altitude_ft", None)
        states.append(AircraftState(**item))
    return states


# ---------------------------------------------------------------------------
# Mock generator
# ---------------------------------------------------------------------------

_MOCK_CALLSIGNS = [
    "AAL", "UAL", "DAL", "SWA", "ASA", "JBU", "FFT", "NKS", "WJA", "SKW",
]
_MOCK_COUNTRIES = ["United States", "Canada", "Mexico"]


def _generate_mock_aircraft(n: int = 200, seed: int | None = None) -> List[AircraftState]:
    """Generate n synthetic aircraft positions scattered across US airspace."""
    rng = np.random.default_rng(seed if seed is not None else int(time.time()) % 10_000)
    aircraft = []
    for i in range(n):
        lat = float(rng.uniform(US_LAT_MIN + 1, US_LAT_MAX - 1))
        lon = float(rng.uniform(US_LON_MIN + 2, US_LON_MAX - 2))
        # Realistic altitude distribution: cruise 25k–40k ft, approach <5k ft
        alt_choice = rng.random()
        if alt_choice < 0.10:
            alt_m = float(rng.uniform(0, 1500))        # low / approach
        elif alt_choice < 0.20:
            alt_m = float(rng.uniform(1500, 7620))     # climb/descend
        else:
            alt_m = float(rng.uniform(7620, 12192))    # cruise (25k–40k ft)

        airline = _MOCK_CALLSIGNS[i % len(_MOCK_CALLSIGNS)]
        flight_num = int(rng.integers(100, 9999))
        icao = f"{i+1:06x}"
        country = _MOCK_COUNTRIES[i % len(_MOCK_COUNTRIES)]

        aircraft.append(AircraftState(
            icao24=icao,
            callsign=f"{airline}{flight_num}",
            origin_country=country,
            longitude=round(lon, 4),
            latitude=round(lat, 4),
            altitude_m=round(alt_m, 1),
            velocity_mps=round(float(rng.uniform(60, 270)), 1),
            heading_deg=round(float(rng.uniform(0, 360)), 1),
            on_ground=alt_m < 30,
        ))
    return aircraft


# ---------------------------------------------------------------------------
# OpenSky fetch
# ---------------------------------------------------------------------------

def _fetch_opensky() -> Optional[List[AircraftState]]:
    """Fetch from OpenSky REST API. Returns None on any error."""
    if not _REQUESTS_AVAILABLE:
        return None
    try:
        resp = requests.get(
            OPENSKY_URL,
            params={
                "lamin": US_LAT_MIN,
                "lamax": US_LAT_MAX,
                "lomin": US_LON_MIN,
                "lomax": US_LON_MAX,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        states_raw = data.get("states") or []
        aircraft = []
        for s in states_raw:
            # OpenSky vector format:
            # 0:icao24 1:callsign 2:origin_country 3:time_position 4:last_contact
            # 5:longitude 6:latitude 7:baro_altitude 8:on_ground 9:velocity
            # 10:true_track 11:vertical_rate 12:sensors 13:geo_altitude
            # 14:squawk 15:spi 16:position_source
            if s[5] is None or s[6] is None:
                continue
            aircraft.append(AircraftState(
                icao24=str(s[0] or ""),
                callsign=str(s[1] or "").strip(),
                origin_country=str(s[2] or ""),
                longitude=float(s[5]),
                latitude=float(s[6]),
                altitude_m=float(s[13] or s[7] or 0),
                velocity_mps=float(s[9] or 0),
                heading_deg=float(s[10] or 0),
                on_ground=bool(s[8]),
            ))
        return aircraft
    except Exception as exc:
        print(f"[opensky_client] API fetch failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def get_aircraft_states(force_refresh: bool = False) -> List[AircraftState]:
    """
    Return current aircraft states for US airspace.

    Resolution order:
    1. Mock mode (MOCK_LIVE=true, default): always return synthetic data.
    2. Live mode: try cache; if stale or missing, fetch from OpenSky.
    3. Fallback: return cached data even if stale, or mock on total failure.
    """
    if MOCK_LIVE:
        return _generate_mock_aircraft(n=200)

    # Attempt cache first
    if not force_refresh:
        cached = _read_cache()
        if cached is not None:
            age = _cache_age_seconds()
            if age <= CACHE_TTL_SECONDS:
                return cached

    # Fetch live
    live = _fetch_opensky()
    if live is not None:
        _write_cache(live)
        return live

    # Fallback: stale cache
    stale = _read_cache()
    if stale is not None:
        print("[opensky_client] Using stale cache due to API failure.")
        return stale

    # Last resort: mock
    print("[opensky_client] All sources unavailable — returning mock data.")
    return _generate_mock_aircraft(n=200)


def _cache_age_seconds() -> float:
    """Return age of most recent cache entry in seconds, or infinity if empty."""
    if not CACHE_DB_PATH.exists():
        return float("inf")
    conn = _ensure_cache_db()
    row = conn.execute(
        "SELECT fetched_at FROM aircraft_cache ORDER BY fetched_at DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if row is None:
        return float("inf")
    return time.time() - row[0]


def get_cache_age_seconds() -> float:
    """Public accessor for cache age."""
    return _cache_age_seconds()


def filter_us_airspace(states: List[AircraftState]) -> List[AircraftState]:
    """Filter aircraft states to US airspace bounding box."""
    return [
        s for s in states
        if US_LAT_MIN <= s.latitude <= US_LAT_MAX
        and US_LON_MIN <= s.longitude <= US_LON_MAX
    ]


def parse_opensky_response(raw_states: list) -> List[AircraftState]:
    """
    Parse raw OpenSky API states list into AircraftState objects.
    Exposed for testing without network calls.
    """
    result = []
    for s in raw_states:
        if s[5] is None or s[6] is None:
            continue
        result.append(AircraftState(
            icao24=str(s[0] or ""),
            callsign=str(s[1] or "").strip(),
            origin_country=str(s[2] or ""),
            longitude=float(s[5]),
            latitude=float(s[6]),
            altitude_m=float(s[13] or s[7] or 0),
            velocity_mps=float(s[9] or 0),
            heading_deg=float(s[10] or 0),
            on_ground=bool(s[8]),
        ))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    mock = "--live" not in sys.argv
    if mock:
        print("[opensky_client] Running in MOCK mode (pass --live to hit API)")
        states = _generate_mock_aircraft(n=50)
    else:
        states = get_aircraft_states(force_refresh=True)

    print(f"  Aircraft count: {len(states)}")
    for s in states[:5]:
        print(f"  {s.callsign:<10} {s.latitude:.2f}°N {abs(s.longitude):.2f}°W "
              f"alt={s.altitude_ft:.0f}ft  hdg={s.heading_deg:.0f}°")

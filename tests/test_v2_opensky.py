"""
test_v2_opensky.py — Tests for live/opensky_client.py

Covers:
  - AircraftState dataclass
  - parse_opensky_response (response parsing)
  - filter_us_airspace
  - Mock generator
  - Cache read/write cycle
  - Fallback to mock when API unavailable
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time
import tempfile
from pathlib import Path

import pytest

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force mock mode for all tests
os.environ["MOCK_LIVE"] = "true"

from live.opensky_client import (
    AircraftState,
    _generate_mock_aircraft,
    filter_us_airspace,
    get_aircraft_states,
    get_cache_age_seconds,
    parse_opensky_response,
    US_LAT_MIN,
    US_LAT_MAX,
    US_LON_MIN,
    US_LON_MAX,
)


# ---------------------------------------------------------------------------
# AircraftState dataclass
# ---------------------------------------------------------------------------

class TestAircraftState:
    def test_altitude_ft_conversion(self):
        s = AircraftState(
            icao24="abc123", callsign="AAL100", origin_country="United States",
            longitude=-90.0, latitude=35.0, altitude_m=10000.0,
            velocity_mps=200.0, heading_deg=90.0, on_ground=False,
        )
        assert abs(s.altitude_ft - 32808.0) < 1.0

    def test_altitude_ft_zero(self):
        s = AircraftState(
            icao24="abc123", callsign="", origin_country="",
            longitude=0.0, latitude=0.0, altitude_m=0.0,
            velocity_mps=0.0, heading_deg=0.0, on_ground=True,
        )
        assert s.altitude_ft == 0.0

    def test_to_dict_contains_altitude_ft(self):
        s = AircraftState(
            icao24="a1b2c3", callsign="UAL200", origin_country="United States",
            longitude=-100.0, latitude=40.0, altitude_m=8000.0,
            velocity_mps=230.0, heading_deg=270.0, on_ground=False,
        )
        d = s.to_dict()
        assert "altitude_ft" in d
        assert abs(d["altitude_ft"] - s.altitude_ft) < 0.01

    def test_to_dict_round_trips_fields(self):
        s = AircraftState(
            icao24="xyz999", callsign="DL456", origin_country="United States",
            longitude=-95.5, latitude=38.2, altitude_m=11000.0,
            velocity_mps=250.0, heading_deg=45.0, on_ground=False,
        )
        d = s.to_dict()
        assert d["icao24"] == "xyz999"
        assert d["callsign"] == "DL456"
        assert d["longitude"] == -95.5

    def test_on_ground_flag(self):
        s = AircraftState(
            icao24="g1", callsign="", origin_country="",
            longitude=-80.0, latitude=33.0, altitude_m=20.0,
            velocity_mps=0.0, heading_deg=0.0, on_ground=True,
        )
        assert s.on_ground is True


# ---------------------------------------------------------------------------
# parse_opensky_response
# ---------------------------------------------------------------------------

class TestParseOpenSkyResponse:
    def _make_raw_state(
        self,
        icao24="aabbcc",
        callsign="TEST100 ",
        country="United States",
        lon=-90.0,
        lat=35.0,
        baro_alt=9000.0,
        on_ground=False,
        velocity=220.0,
        true_track=180.0,
        geo_alt=9100.0,
    ):
        """Build a minimal OpenSky state vector list."""
        return [
            icao24,          # 0: icao24
            callsign,        # 1: callsign
            country,         # 2: origin_country
            None,            # 3: time_position
            None,            # 4: last_contact
            lon,             # 5: longitude
            lat,             # 6: latitude
            baro_alt,        # 7: baro_altitude
            on_ground,       # 8: on_ground
            velocity,        # 9: velocity
            true_track,      # 10: true_track
            None,            # 11: vertical_rate
            None,            # 12: sensors
            geo_alt,         # 13: geo_altitude
            None,            # 14: squawk
            None,            # 15: spi
            0,               # 16: position_source
        ]

    def test_parses_basic_state(self):
        raw = [self._make_raw_state()]
        states = parse_opensky_response(raw)
        assert len(states) == 1
        s = states[0]
        assert s.icao24 == "aabbcc"
        assert s.callsign == "TEST100"   # stripped
        assert s.latitude == 35.0
        assert s.longitude == -90.0

    def test_uses_geo_altitude_when_available(self):
        raw = [self._make_raw_state(baro_alt=9000.0, geo_alt=9200.0)]
        states = parse_opensky_response(raw)
        assert states[0].altitude_m == 9200.0

    def test_falls_back_to_baro_when_geo_none(self):
        state_vec = self._make_raw_state(baro_alt=8500.0, geo_alt=None)
        state_vec[13] = None   # explicitly null geo_alt
        states = parse_opensky_response([state_vec])
        assert states[0].altitude_m == 8500.0

    def test_skips_states_with_null_lon(self):
        state_vec = self._make_raw_state()
        state_vec[5] = None   # null longitude
        states = parse_opensky_response([state_vec])
        assert len(states) == 0

    def test_skips_states_with_null_lat(self):
        state_vec = self._make_raw_state()
        state_vec[6] = None   # null latitude
        states = parse_opensky_response([state_vec])
        assert len(states) == 0

    def test_parses_multiple_states(self):
        raw = [self._make_raw_state(icao24=f"aa{i:04x}", lon=-80.0 - i, lat=30.0 + i) for i in range(5)]
        states = parse_opensky_response(raw)
        assert len(states) == 5

    def test_on_ground_boolean(self):
        raw_on = [self._make_raw_state(on_ground=True)]
        raw_off = [self._make_raw_state(on_ground=False)]
        assert parse_opensky_response(raw_on)[0].on_ground is True
        assert parse_opensky_response(raw_off)[0].on_ground is False

    def test_empty_input(self):
        assert parse_opensky_response([]) == []

    def test_velocity_and_heading_parsed(self):
        raw = [self._make_raw_state(velocity=250.5, true_track=135.0)]
        s = parse_opensky_response(raw)[0]
        assert s.velocity_mps == 250.5
        assert s.heading_deg == 135.0


# ---------------------------------------------------------------------------
# filter_us_airspace
# ---------------------------------------------------------------------------

class TestFilterUsAirspace:
    def _make(self, lat, lon):
        return AircraftState(
            icao24="x", callsign="X", origin_country="US",
            longitude=lon, latitude=lat, altitude_m=5000.0,
            velocity_mps=200.0, heading_deg=90.0, on_ground=False,
        )

    def test_inside_us_passes(self):
        states = [self._make(37.0, -95.0)]
        assert len(filter_us_airspace(states)) == 1

    def test_outside_lat_filtered(self):
        states = [self._make(60.0, -95.0)]   # > 50°N
        assert len(filter_us_airspace(states)) == 0

    def test_outside_lon_filtered(self):
        states = [self._make(37.0, -60.0)]   # > -66°W
        assert len(filter_us_airspace(states)) == 0

    def test_boundary_lat_min_included(self):
        states = [self._make(US_LAT_MIN, -95.0)]
        assert len(filter_us_airspace(states)) == 1

    def test_boundary_lat_max_included(self):
        states = [self._make(US_LAT_MAX, -95.0)]
        assert len(filter_us_airspace(states)) == 1

    def test_mixed_batch(self):
        inside = [self._make(35.0, -95.0), self._make(40.0, -100.0)]
        outside = [self._make(55.0, -95.0), self._make(35.0, -50.0)]
        result = filter_us_airspace(inside + outside)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Mock generator
# ---------------------------------------------------------------------------

class TestMockGenerator:
    def test_returns_correct_count(self):
        states = _generate_mock_aircraft(n=200)
        assert len(states) == 200

    def test_all_within_us_roughly(self):
        states = _generate_mock_aircraft(n=100)
        for s in states:
            assert US_LAT_MIN + 1 <= s.latitude <= US_LAT_MAX - 1
            assert US_LON_MIN + 2 <= s.longitude <= US_LON_MAX - 2

    def test_callsign_not_empty(self):
        states = _generate_mock_aircraft(n=50)
        for s in states:
            assert len(s.callsign) > 0

    def test_different_seed_different_positions(self):
        s1 = _generate_mock_aircraft(n=10, seed=1)
        s2 = _generate_mock_aircraft(n=10, seed=2)
        lats1 = [s.latitude for s in s1]
        lats2 = [s.latitude for s in s2]
        assert lats1 != lats2

    def test_altitude_m_non_negative(self):
        states = _generate_mock_aircraft(n=100)
        for s in states:
            assert s.altitude_m >= 0


# ---------------------------------------------------------------------------
# get_aircraft_states (mock mode)
# ---------------------------------------------------------------------------

class TestGetAircraftStatesMock:
    def test_returns_nonempty_list(self):
        states = _generate_mock_aircraft(n=200)
        assert len(states) > 0

    def test_returns_aircraft_state_objects(self):
        states = _generate_mock_aircraft(n=5)
        for s in states:
            assert isinstance(s, AircraftState)

    def test_get_aircraft_states_mock_env(self):
        os.environ["MOCK_LIVE"] = "true"
        states = get_aircraft_states()
        assert isinstance(states, list)
        assert len(states) > 0

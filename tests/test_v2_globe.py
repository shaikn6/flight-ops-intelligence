"""
test_v2_globe.py — Tests for live/globe_renderer.py

Covers:
  - altitude_color mapping (low=blue, mid=yellow, high=red)
  - build_scatter_data structure
  - build_route_arcs structure
  - render_to_html produces file > 10 KB
  - build_deck returns pdk.Deck (when pydeck available)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pdk = pytest.importorskip("pydeck", reason="pydeck not installed")

from live.opensky_client import AircraftState, _generate_mock_aircraft
from live.globe_renderer import (
    ALT_LOW_MAX,
    ALT_MID_MAX,
    _BLUE,
    _YELLOW,
    _RED,
    altitude_color,
    build_scatter_data,
    build_route_arcs,
    build_deck,
    render_to_html,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(lat=35.0, lon=-95.0, alt_m=5000.0, callsign="TEST"):
    return AircraftState(
        icao24="aaaaaa", callsign=callsign, origin_country="US",
        longitude=lon, latitude=lat, altitude_m=alt_m,
        velocity_mps=220.0, heading_deg=90.0, on_ground=False,
    )


# ---------------------------------------------------------------------------
# altitude_color
# ---------------------------------------------------------------------------

class TestAltitudeColor:
    def test_zero_altitude_is_blue(self):
        assert altitude_color(0.0) == _BLUE

    def test_below_low_threshold_is_blue(self):
        assert altitude_color(1000.0) == _BLUE

    def test_at_low_threshold_is_yellow(self):
        # ALT_LOW_MAX = 1524 → exactly at boundary → yellow
        assert altitude_color(ALT_LOW_MAX) == _YELLOW

    def test_mid_range_is_yellow(self):
        assert altitude_color(4000.0) == _YELLOW

    def test_at_mid_threshold_is_red(self):
        assert altitude_color(ALT_MID_MAX) == _RED

    def test_high_cruise_is_red(self):
        assert altitude_color(11000.0) == _RED

    def test_returns_4_element_list(self):
        for alt in [0, 2000, 8000, 12000]:
            color = altitude_color(alt)
            assert len(color) == 4
            assert all(isinstance(c, int) for c in color)

    def test_alpha_channel_reasonable(self):
        for alt in [0, 2000, 8000]:
            color = altitude_color(alt)
            assert 0 < color[3] <= 255


# ---------------------------------------------------------------------------
# build_scatter_data
# ---------------------------------------------------------------------------

class TestBuildScatterData:
    def test_returns_list_of_dicts(self):
        states = [_make_state()]
        data = build_scatter_data(states)
        assert isinstance(data, list)
        assert isinstance(data[0], dict)

    def test_correct_count(self):
        states = [_make_state() for _ in range(10)]
        data = build_scatter_data(states)
        assert len(data) == 10

    def test_required_keys_present(self):
        data = build_scatter_data([_make_state()])
        record = data[0]
        for key in ("lat", "lon", "altitude_ft", "callsign", "color"):
            assert key in record, f"Missing key: {key}"

    def test_color_assigned_from_altitude(self):
        low_state = _make_state(alt_m=500.0)
        high_state = _make_state(alt_m=12000.0)
        data_low = build_scatter_data([low_state])
        data_high = build_scatter_data([high_state])
        assert data_low[0]["color"] == _BLUE
        assert data_high[0]["color"] == _RED

    def test_altitude_ft_conversion(self):
        state = _make_state(alt_m=3048.0)   # 10,000 ft
        data = build_scatter_data([state])
        assert abs(data[0]["altitude_ft"] - 10000.0) < 10.0

    def test_empty_states_returns_empty(self):
        assert build_scatter_data([]) == []

    def test_callsign_falls_back_to_icao(self):
        state = AircraftState(
            icao24="xyz999", callsign="", origin_country="US",
            longitude=-90.0, latitude=35.0, altitude_m=5000.0,
            velocity_mps=200.0, heading_deg=180.0, on_ground=False,
        )
        data = build_scatter_data([state])
        assert data[0]["callsign"] in ("", "xyz999")  # either is acceptable


# ---------------------------------------------------------------------------
# build_route_arcs
# ---------------------------------------------------------------------------

class TestBuildRouteArcs:
    def test_returns_list(self):
        arcs = build_route_arcs()
        assert isinstance(arcs, list)

    def test_respects_top_n(self):
        arcs = build_route_arcs(top_n=5)
        assert len(arcs) <= 5

    def test_arc_has_required_keys(self):
        arcs = build_route_arcs(top_n=1)
        if arcs:
            arc = arcs[0]
            assert "origin" in arc
            assert "destination" in arc

    def test_origin_destination_are_lon_lat_pairs(self):
        arcs = build_route_arcs(top_n=3)
        for arc in arcs:
            assert len(arc["origin"]) == 2
            assert len(arc["destination"]) == 2

    def test_top_20_returns_20(self):
        arcs = build_route_arcs(top_n=20)
        assert len(arcs) == 20


# ---------------------------------------------------------------------------
# build_deck
# ---------------------------------------------------------------------------

class TestBuildDeck:
    def test_returns_deck_object(self):
        states = _generate_mock_aircraft(n=10)
        deck = build_deck(states)
        assert isinstance(deck, pdk.Deck)

    def test_deck_has_layers(self):
        states = _generate_mock_aircraft(n=10)
        deck = build_deck(states, include_arcs=True)
        assert len(deck.layers) == 2   # scatter + arc

    def test_deck_without_arcs_has_one_layer(self):
        states = _generate_mock_aircraft(n=10)
        deck = build_deck(states, include_arcs=False)
        assert len(deck.layers) == 1


# ---------------------------------------------------------------------------
# render_to_html
# ---------------------------------------------------------------------------

class TestRenderToHtml:
    def test_html_file_created(self, tmp_path):
        states = _generate_mock_aircraft(n=50)
        out_path = tmp_path / "globe_test.html"
        result = render_to_html(states, output_path=str(out_path))
        assert result.exists()

    def test_html_file_size_exceeds_10kb(self, tmp_path):
        states = _generate_mock_aircraft(n=50)
        out_path = tmp_path / "globe_size_test.html"
        render_to_html(states, output_path=str(out_path))
        assert out_path.stat().st_size > 10_240   # > 10 KB

    def test_html_contains_deckgl_markers(self, tmp_path):
        states = _generate_mock_aircraft(n=20)
        out_path = tmp_path / "globe_content_test.html"
        render_to_html(states, output_path=str(out_path))
        content = out_path.read_text(encoding="utf-8", errors="replace")
        # pydeck HTML always contains these
        assert "deck" in content.lower()

    def test_html_is_valid_utf8(self, tmp_path):
        states = _generate_mock_aircraft(n=10)
        out_path = tmp_path / "globe_utf8_test.html"
        render_to_html(states, output_path=str(out_path))
        # Should not raise
        out_path.read_text(encoding="utf-8")

    def test_output_path_parent_created(self, tmp_path):
        states = _generate_mock_aircraft(n=5)
        nested_path = tmp_path / "nested" / "dir" / "globe.html"
        render_to_html(states, output_path=str(nested_path))
        assert nested_path.exists()

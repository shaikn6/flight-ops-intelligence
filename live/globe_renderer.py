"""
globe_renderer.py — PyDeck 3D globe / map renderer.

Renders:
  - ScatterplotLayer: aircraft positions coloured by altitude
    (low=blue, mid=yellow, high=red)
  - ArcLayer: top-20 busiest origin→destination routes

Exports a self-contained HTML file via pydeck.Deck.to_html().

Map style: dark (CartoDB dark-matter fallback, no Mapbox token required).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import pydeck as pdk
    _PYDECK_AVAILABLE = True
except ImportError:
    _PYDECK_AVAILABLE = False

from live.opensky_client import AircraftState

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

# Altitude thresholds (metres)
ALT_LOW_MAX = 1524       # < 5 000 ft  → blue
ALT_MID_MAX = 7620       # < 25 000 ft → yellow
# >= 25 000 ft            → red

_BLUE   = [30, 144, 255, 200]   # dodger-blue
_YELLOW = [255, 215, 0, 200]    # gold
_RED    = [255, 69, 0, 220]     # red-orange


def altitude_color(altitude_m: float) -> List[int]:
    """Return [R, G, B, A] colour based on altitude in metres."""
    if altitude_m < ALT_LOW_MAX:
        return _BLUE
    if altitude_m < ALT_MID_MAX:
        return _YELLOW
    return _RED


# ---------------------------------------------------------------------------
# Route helpers
# ---------------------------------------------------------------------------

# Representative hub coordinates used for synthetic arc endpoints
_HUB_COORDS = {
    "JFK": (40.6413, -73.7781),
    "LAX": (33.9425, -118.4081),
    "ORD": (41.9742, -87.9073),
    "DFW": (32.8998, -97.0403),
    "ATL": (33.6407, -84.4277),
    "SFO": (37.6213, -122.3790),
    "SEA": (47.4502, -122.3088),
    "MIA": (25.7959, -80.2870),
    "BOS": (42.3656, -71.0096),
    "DEN": (39.8561, -104.6737),
    "PHX": (33.4373, -112.0078),
    "MSP": (44.8848, -93.2223),
    "DTW": (42.2124, -83.3534),
    "LGA": (40.7769, -73.8740),
    "EWR": (40.6895, -74.1745),
    "CLT": (35.2144, -80.9473),
    "LAS": (36.0840, -115.1537),
    "PHL": (39.8729, -75.2437),
    "IAH": (29.9902, -95.3368),
    "BWI": (39.1754, -76.6683),
}

_ROUTE_PAIRS = [
    ("JFK", "LAX"), ("JFK", "SFO"), ("ORD", "ATL"), ("LAX", "SFO"),
    ("DFW", "ATL"), ("JFK", "MIA"), ("ORD", "DFW"), ("LAX", "DEN"),
    ("BOS", "JFK"), ("ATL", "MIA"), ("SEA", "SFO"), ("ORD", "BOS"),
    ("DEN", "ORD"), ("DFW", "LAX"), ("ATL", "DFW"), ("JFK", "ORD"),
    ("LAX", "PHX"), ("SFO", "SEA"), ("ATL", "CLT"), ("DFW", "IAH"),
]


def build_route_arcs(top_n: int = 20) -> List[dict]:
    """Return arc layer data for the top-N busiest synthetic routes."""
    arcs = []
    for origin, dest in _ROUTE_PAIRS[:top_n]:
        if origin not in _HUB_COORDS or dest not in _HUB_COORDS:
            continue
        o_lat, o_lon = _HUB_COORDS[origin]
        d_lat, d_lon = _HUB_COORDS[dest]
        arcs.append({
            "origin": [o_lon, o_lat],
            "destination": [d_lon, d_lat],
            "from_name": origin,
            "to_name": dest,
            "width": 3,
        })
    return arcs


# ---------------------------------------------------------------------------
# Layer builders
# ---------------------------------------------------------------------------

def build_scatter_data(states: List[AircraftState]) -> List[dict]:
    """Convert AircraftState list into pydeck ScatterplotLayer input records."""
    records = []
    for s in states:
        if s.longitude is None or s.latitude is None:
            continue
        records.append({
            "lon": s.longitude,
            "lat": s.latitude,
            "altitude_m": s.altitude_m,
            "altitude_ft": round(s.altitude_ft, 0),
            "callsign": s.callsign or s.icao24,
            "heading": s.heading_deg,
            "velocity_knots": round(s.velocity_mps * 1.944, 1),
            "color": altitude_color(s.altitude_m),
        })
    return records


def _make_scatter_layer(data: List[dict]) -> "pdk.Layer":
    return pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position=["lon", "lat"],
        get_color="color",
        get_radius=15000,
        pickable=True,
        opacity=0.85,
        stroked=False,
        filled=True,
        radius_scale=1,
        radius_min_pixels=3,
        radius_max_pixels=12,
    )


def _make_arc_layer(arc_data: List[dict]) -> "pdk.Layer":
    return pdk.Layer(
        "ArcLayer",
        data=arc_data,
        get_source_position="origin",
        get_target_position="destination",
        get_source_color=[0, 200, 255, 160],
        get_target_color=[255, 100, 0, 160],
        get_width="width",
        pickable=True,
        auto_highlight=True,
    )


# ---------------------------------------------------------------------------
# Deck builder
# ---------------------------------------------------------------------------

_DARK_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"


def build_deck(
    states: List[AircraftState],
    include_arcs: bool = True,
    top_routes: int = 20,
    map_style: str = _DARK_STYLE,
) -> "pdk.Deck":
    """
    Build a pydeck Deck with ScatterplotLayer + optional ArcLayer.

    Parameters
    ----------
    states : list of AircraftState
    include_arcs : bool  — include top-routes ArcLayer
    top_routes : int     — how many arc routes to render
    map_style : str      — map tile style URL

    Returns
    -------
    pdk.Deck ready for .to_html()
    """
    if not _PYDECK_AVAILABLE:
        raise ImportError(
            "pydeck is not installed. Run: pip install pydeck"
        )

    scatter_data = build_scatter_data(states)
    layers = [_make_scatter_layer(scatter_data)]

    if include_arcs:
        arc_data = build_route_arcs(top_n=top_routes)
        layers.append(_make_arc_layer(arc_data))

    view_state = pdk.ViewState(
        latitude=38.5,
        longitude=-96.0,
        zoom=3.5,
        pitch=30,
        bearing=0,
    )

    tooltip = {
        "html": (
            "<b>{callsign}</b><br/>"
            "Alt: {altitude_ft} ft<br/>"
            "Speed: {velocity_knots} kts<br/>"
            "Hdg: {heading}°"
        ),
        "style": {
            "background": "#0d1117",
            "color": "#e6edf3",
            "font-family": "monospace",
            "padding": "8px",
            "border-radius": "6px",
        },
    }

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=map_style,
        tooltip=tooltip,
        parameters={
            "clearColor": [0.05, 0.07, 0.09, 1.0],
        },
    )
    return deck


def render_to_html(
    states: List[AircraftState],
    output_path: str = "maps/globe_v2.html",
    include_arcs: bool = True,
    top_routes: int = 20,
) -> Path:
    """
    Render aircraft globe to self-contained HTML file.

    Returns the Path to the written file.
    """
    deck = build_deck(states, include_arcs=include_arcs, top_routes=top_routes)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    deck.to_html(str(out), open_browser=False)
    print(f"[globe_renderer] Globe HTML written → {out} ({out.stat().st_size // 1024} KB)")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from live.opensky_client import _generate_mock_aircraft

    print("[globe_renderer] Generating mock aircraft...")
    states = _generate_mock_aircraft(n=200)
    path = render_to_html(states, output_path="maps/globe_v2.html")
    print(f"[globe_renderer] Done. Open {path} in a browser.")

"""
route_map.py — V2 interactive Folium route risk map.

generate_route_risk_map(flights) → HTML string
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import folium

from src.weather_client import AIRPORT_COORDS


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FlightRoute:
    origin: str           # IATA code
    destination: str      # IATA code
    delay_probability: float   # 0.0 – 1.0
    weather_risk_score: float  # 0.0 – 1.0
    airline: str = ""
    flight_number: str = ""
    weather_summary: str = ""


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------


def _prob_to_color(probability: float) -> str:
    """Map delay probability to green / yellow / red."""
    if probability < 0.30:
        return "#2ea043"   # green — low risk
    if probability < 0.60:
        return "#d29922"   # yellow — moderate risk
    return "#f85149"       # red — high risk


def _prob_to_weight(probability: float) -> float:
    """Thicker lines for higher-risk routes."""
    return 2.0 + probability * 4.0


def _prob_to_opacity(probability: float) -> float:
    return 0.4 + probability * 0.5


# ---------------------------------------------------------------------------
# Map generator
# ---------------------------------------------------------------------------


def generate_route_risk_map(
    flights: List[FlightRoute],
    title: str = "Flight Ops Intelligence — Route Risk Map (V2)",
) -> str:
    """
    Generate an interactive Folium map showing route risk.

    Each route is drawn as a polyline arc colored by delay probability:
        green  → probability < 0.30
        yellow → probability < 0.60
        red    → probability >= 0.60

    Airport markers include weather tooltips.

    Parameters
    ----------
    flights : list[FlightRoute]
        Routes to render.
    title   : str
        Map title shown in header bar.

    Returns
    -------
    str
        Full Folium map rendered as an HTML string.
    """
    m = folium.Map(
        location=[39.5, -98.35],
        zoom_start=4,
        tiles=None,
    )

    # Dark base layer
    folium.TileLayer(
        tiles="https://cartodb-basemaps-{s}.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png",
        attr="© OpenStreetMap contributors, © CARTO",
        name="Dark",
        control=False,
    ).add_to(m)

    routes_group = folium.FeatureGroup(name="Routes", show=True)
    airports_group = folium.FeatureGroup(name="Airports", show=True)

    # Track airports used so we only draw each marker once
    seen_airports: dict[str, dict] = {}

    for flight in flights:
        orig = flight.origin.upper()
        dest = flight.destination.upper()

        orig_coords = AIRPORT_COORDS.get(orig)
        dest_coords = AIRPORT_COORDS.get(dest)

        if orig_coords is None or dest_coords is None:
            continue

        color = _prob_to_color(flight.delay_probability)
        weight = _prob_to_weight(flight.delay_probability)
        opacity = _prob_to_opacity(flight.delay_probability)

        latlons = [
            (orig_coords["lat"], orig_coords["lon"]),
            (dest_coords["lat"], dest_coords["lon"]),
        ]

        label = flight.flight_number or f"{orig}→{dest}"
        popup_html = (
            f"<b>{label}</b><br>"
            f"Route: {orig} → {dest}<br>"
            f"Airline: {flight.airline or 'N/A'}<br>"
            f"Delay prob: <b>{flight.delay_probability:.0%}</b><br>"
            f"Weather risk: {flight.weather_risk_score:.2f}<br>"
            f"<small>{flight.weather_summary}</small>"
        )

        folium.PolyLine(
            locations=latlons,
            color=color,
            weight=weight,
            opacity=opacity,
            tooltip=f"{orig}→{dest} | delay {flight.delay_probability:.0%}",
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(routes_group)

        # Collect unique airports
        for code, coords in [(orig, orig_coords), (dest, dest_coords)]:
            if code not in seen_airports:
                seen_airports[code] = coords

    # Draw airport markers
    for code, coords in seen_airports.items():
        folium.CircleMarker(
            location=[coords["lat"], coords["lon"]],
            radius=6,
            color="#58a6ff",
            fill=True,
            fill_color="#58a6ff",
            fill_opacity=0.9,
            tooltip=code,
            popup=folium.Popup(f"<b>{code}</b>", max_width=120),
        ).add_to(airports_group)

    routes_group.add_to(m)
    airports_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:#161b22;border:1px solid #30363d;border-radius:8px;
                padding:12px 16px;font-family:monospace;color:#e6edf3;
                font-size:12px;box-shadow:0 4px 16px #0005;">
        <b style="font-size:13px;">Route Risk</b><br><br>
        <span style="color:#2ea043">━━</span> Low (&lt; 30%)<br>
        <span style="color:#d29922">━━</span> Moderate (30–60%)<br>
        <span style="color:#f85149">━━</span> High (&ge; 60%)<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Title bar
    title_html = f"""
    <div style="position:fixed;top:12px;left:50%;transform:translateX(-50%);
                z-index:1000;background:#0d1117cc;border:1px solid #30363d;
                border-radius:8px;padding:8px 20px;font-family:monospace;
                color:#58a6ff;font-size:15px;font-weight:bold;
                letter-spacing:1px;backdrop-filter:blur(8px);">
        {title}
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    return m._repr_html_()

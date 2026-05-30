"""
map_generator.py — Folium interactive map generator.
Creates flight path maps with weather overlays and delay color-coding.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import folium
import numpy as np
import pandas as pd
from folium.plugins import HeatMap

from intelligence.flight_data import AIRPORTS, load_flights
from intelligence.route_analyzer import great_circle_route
from intelligence.weather_engine import WeatherEngine, get_engine

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _delay_color(delay_minutes: float) -> str:
    """Map delay minutes to a traffic-light color."""
    if delay_minutes < 15:
        return "#2ea043"   # green — on-time
    if delay_minutes < 60:
        return "#d29922"   # yellow — minor delay
    return "#f85149"       # red — major delay


def _opacity(delay_minutes: float) -> float:
    """Make major delays more visually prominent."""
    if delay_minutes < 15:
        return 0.25
    if delay_minutes < 60:
        return 0.55
    return 0.85


def _weather_circle_radius(impact_score: float) -> float:
    """Scale halo radius by weather severity (metres)."""
    return 50_000 + impact_score * 150_000


# ---------------------------------------------------------------------------
# Popup builders
# ---------------------------------------------------------------------------

def _airport_popup(code: str, weather_engine: WeatherEngine) -> str:
    """Build HTML popup for an airport marker."""
    info = AIRPORTS[code]
    ts = datetime(2024, 1, 1, 9, 0)
    rpt = weather_engine.get_weather(code, ts)
    return (
        f"<b>{code}</b> — {info['city']}<br>"
        f"<small>{info['name']}</small><br><hr>"
        f"Wind: {rpt.wind_speed_kts:.0f} kts @ {rpt.wind_dir_deg:.0f}°<br>"
        f"Visibility: {rpt.visibility_sm:.1f} sm<br>"
        f"Ceiling: {rpt.ceiling_ft:,} ft<br>"
        f"Conditions: <b>{rpt.conditions}</b><br>"
        f"Impact score: <b>{rpt.weather_impact_score:.2f}</b>"
    )


def _flight_popup(row: pd.Series) -> str:
    """Build HTML popup for a flight path."""
    delay = row.get("delay_minutes", 0)
    status = "On-time" if delay < 15 else f"+{delay:.0f} min"
    cause = row.get("delay_cause", "none")
    return (
        f"<b>{row.get('flight_number', row.get('flight_id', ''))}</b><br>"
        f"{row['origin']} → {row['destination']}<br>"
        f"Aircraft: {row.get('aircraft_type', 'N/A')}<br>"
        f"Delay: <b>{status}</b><br>"
        f"Cause: {cause}<br>"
        f"Distance: {row.get('distance_mi', 0):.0f} mi"
    )


# ---------------------------------------------------------------------------
# Map generator
# ---------------------------------------------------------------------------

def generate_flight_map(
    df: Optional[pd.DataFrame] = None,
    output_path: str = "maps/flight_map.html",
    weather_engine: Optional[WeatherEngine] = None,
    n_waypoints: int = 30,
) -> str:
    """
    Generate an interactive Folium map with:
      - Color-coded flight paths (green/yellow/red)
      - Airport markers with weather popups
      - Weather halo overlays (IMC airports)
      - Heatmap layer for delay density

    Parameters
    ----------
    df           : Flights DataFrame (loads from data/flights.csv if None)
    output_path  : Where to save the HTML file
    weather_engine : Uses singleton if None
    n_waypoints  : Waypoints per flight path

    Returns
    -------
    Absolute path to saved HTML file
    """
    if df is None:
        df = load_flights()
    if weather_engine is None:
        weather_engine = get_engine()

    # Ensure output directory exists
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Center map on continental US
    m = folium.Map(
        location=[39.5, -98.35],
        zoom_start=4,
        tiles=None,
    )

    # Dark tile layer
    folium.TileLayer(
        tiles="https://cartodb-basemaps-{s}.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png",
        attr="© OpenStreetMap contributors, © CARTO",
        name="Dark",
        control=False,
    ).add_to(m)

    # --- Layer groups ---
    on_time_group = folium.FeatureGroup(name="On-time Flights", show=True)
    delayed_group = folium.FeatureGroup(name="Delayed Flights", show=True)
    severe_group = folium.FeatureGroup(name="Severely Delayed", show=True)
    airport_group = folium.FeatureGroup(name="Airports", show=True)
    weather_group = folium.FeatureGroup(name="Weather Halos (IMC)", show=True)

    # --- Flight paths ---
    ts_eval = datetime(2024, 1, 1, 9, 0)

    for _, row in df.iterrows():
        origin = str(row["origin"])
        dest = str(row["destination"])

        if origin not in AIRPORTS or dest not in AIRPORTS:
            continue

        delay = float(row.get("delay_minutes", 0))
        color = _delay_color(delay)
        opacity = _opacity(delay)

        try:
            wps = great_circle_route(origin, dest, n_waypoints)
        except Exception:
            wps = [
                (AIRPORTS[origin]["lat"], AIRPORTS[origin]["lon"]),
                (AIRPORTS[dest]["lat"], AIRPORTS[dest]["lon"]),
            ]

        latlons = [(lat, lon) for lat, lon in wps]

        line = folium.PolyLine(
            locations=latlons,
            color=color,
            weight=1.5,
            opacity=opacity,
            tooltip=f"{origin}→{dest} | {row.get('flight_number', '')}",
            popup=folium.Popup(_flight_popup(row), max_width=280),
        )

        if delay < 15:
            line.add_to(on_time_group)
        elif delay < 60:
            line.add_to(delayed_group)
        else:
            line.add_to(severe_group)

    # --- Airport markers ---
    for code, info in AIRPORTS.items():
        rpt = weather_engine.get_weather(code, ts_eval)
        icon_color = "green" if rpt.conditions == "VMC" else ("orange" if rpt.conditions == "MVMC" else "red")

        folium.Marker(
            location=[info["lat"], info["lon"]],
            popup=folium.Popup(_airport_popup(code, weather_engine), max_width=300),
            tooltip=f"{code} — {rpt.conditions}",
            icon=folium.Icon(
                color=icon_color,
                icon="plane",
                prefix="fa",
            ),
        ).add_to(airport_group)

        # Weather halo for IMC / MVMC airports
        if rpt.conditions != "VMC":
            folium.Circle(
                location=[info["lat"], info["lon"]],
                radius=_weather_circle_radius(rpt.weather_impact_score),
                color="#f85149",
                fill=True,
                fill_color="#f85149",
                fill_opacity=0.08,
                weight=1,
                opacity=0.4,
                tooltip=f"{code} weather impact: {rpt.weather_impact_score:.2f}",
            ).add_to(weather_group)

    # --- Delay heatmap layer ---
    heat_data = []
    for _, row in df.iterrows():
        origin = str(row["origin"])
        if origin in AIRPORTS:
            lat = AIRPORTS[origin]["lat"] + np.random.uniform(-0.5, 0.5)
            lon = AIRPORTS[origin]["lon"] + np.random.uniform(-0.5, 0.5)
            weight = float(row.get("delay_minutes", 0)) / 180.0
            if weight > 0:
                heat_data.append([lat, lon, weight])

    heat_group = folium.FeatureGroup(name="Delay Heatmap", show=False)
    HeatMap(
        heat_data,
        min_opacity=0.2,
        max_zoom=8,
        radius=20,
        blur=15,
        gradient={0.2: "blue", 0.5: "yellow", 0.8: "red"},
    ).add_to(heat_group)

    # --- Add all groups to map ---
    on_time_group.add_to(m)
    delayed_group.add_to(m)
    severe_group.add_to(m)
    airport_group.add_to(m)
    weather_group.add_to(m)
    heat_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # --- Legend ---
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: #161b22; border: 1px solid #30363d;
                border-radius: 8px; padding: 12px 16px; font-family: monospace;
                color: #e6edf3; font-size: 12px; box-shadow: 0 4px 16px #0005;">
        <b style="font-size: 13px;">✈ Flight Status</b><br><br>
        <span style="color:#2ea043">━━</span> On-time (&lt; 15 min)<br>
        <span style="color:#d29922">━━</span> Minor delay (15–60 min)<br>
        <span style="color:#f85149">━━</span> Major delay (&gt; 60 min)<br><br>
        <b>Airport Conditions</b><br>
        🟢 VMC &nbsp; 🟠 MVMC &nbsp; 🔴 IMC<br><br>
        <small style="color:#8b949e">500 synthetic flights · Jan 2024</small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Title bar
    title_html = """
    <div style="position: fixed; top: 12px; left: 50%; transform: translateX(-50%);
                z-index: 1000; background: #0d1117cc; border: 1px solid #30363d;
                border-radius: 8px; padding: 8px 20px; font-family: monospace;
                color: #58a6ff; font-size: 15px; font-weight: bold;
                letter-spacing: 1px; backdrop-filter: blur(8px);">
        ✈ FLIGHT OPS INTELLIGENCE — US Flight Map (Jan 2024)
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    m.save(str(out))
    print(f"[map_generator] Saved flight map to {out.resolve()}")
    return str(out.resolve())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    path = generate_flight_map()
    print(f"Open in browser: file://{path}")

"""
app.py — Streamlit dashboard for Flight Ops Intelligence.
Run: streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from intelligence.flight_data import AIRPORTS, load_flights
from intelligence.weather_engine import WeatherEngine, CLIMATE_PROFILES
from intelligence.delay_predictor import predict_delay, get_feature_importances, load_models
from intelligence.route_analyzer import compute_route_risk_score
from intelligence.atc_simulator import compute_sector_load, identify_overloaded_sectors

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Flight Ops Intelligence",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme CSS
st.markdown(
    """
    <style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .stMetric { background: #161b22; border-radius: 8px; padding: 12px; }
    .stMetric label { color: #8b949e; }
    h1, h2, h3 { color: #58a6ff; }
    .stSelectbox label, .stSlider label { color: #8b949e; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("✈ Flight Ops Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard Overview", "Delay Predictor", "Route Risk Analyzer", "ATC Sector Load", "Weather Map"],
)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    return load_flights()


@st.cache_resource
def load_weather_engine():
    eng = WeatherEngine()
    eng.preload()
    return eng


@st.cache_data
def load_sector_matrix():
    df = load_flights()
    return compute_sector_load(df)


# ---------------------------------------------------------------------------
# Dashboard Overview
# ---------------------------------------------------------------------------

if page == "Dashboard Overview":
    st.title("✈ Flight Ops Intelligence — Dashboard")
    st.markdown("Real-time flight analytics powered by ML and weather data.")

    df = load_data()

    # --- KPI cards ---
    c1, c2, c3, c4 = st.columns(4)
    total = len(df)
    delayed = df["is_delayed"].sum()
    delay_rate = df["is_delayed"].mean()
    mean_delay = df.loc[df["delay_minutes"] > 0, "delay_minutes"].mean()

    c1.metric("Total Flights", f"{total:,}")
    c2.metric("Delayed Flights", f"{int(delayed):,}")
    c3.metric("Delay Rate", f"{delay_rate:.1%}")
    c4.metric("Avg Delay (when delayed)", f"{mean_delay:.0f} min")

    st.markdown("---")

    # --- Delay by cause ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Delay Causes")
        cause_counts = df[df["delay_minutes"] > 0]["delay_cause"].value_counts().reset_index()
        cause_counts.columns = ["Cause", "Count"]
        fig = px.pie(
            cause_counts, values="Count", names="Cause",
            color_discrete_sequence=px.colors.qualitative.Set3,
            template="plotly_dark",
        )
        fig.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Delay Distribution by Hour")
        hourly = df.groupby("dep_hour")["delay_minutes"].mean().reset_index()
        fig2 = px.bar(
            hourly, x="dep_hour", y="delay_minutes",
            color="delay_minutes",
            color_continuous_scale="RdYlGn_r",
            template="plotly_dark",
            labels={"dep_hour": "Departure Hour", "delay_minutes": "Avg Delay (min)"},
        )
        fig2.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22")
        st.plotly_chart(fig2, use_container_width=True)

    # --- Top delay routes ---
    st.subheader("Top 10 Routes by Average Delay")
    routes = (
        df.groupby(["origin", "destination"])["delay_minutes"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_delay", "count": "flights"})
        .assign(route=lambda x: x["origin"] + "→" + x["destination"])
        .nlargest(10, "avg_delay")
    )
    fig3 = px.bar(
        routes, x="route", y="avg_delay",
        color="avg_delay", color_continuous_scale="Reds",
        template="plotly_dark",
        labels={"route": "Route", "avg_delay": "Avg Delay (min)"},
    )
    fig3.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22")
    st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------------------------
# Delay Predictor
# ---------------------------------------------------------------------------

elif page == "Delay Predictor":
    st.title("🤖 ML Delay Predictor")
    st.markdown("Random Forest model predicting departure delay from weather + route features.")

    engine = load_weather_engine()

    col1, col2 = st.columns(2)
    with col1:
        origin = st.selectbox("Origin Airport", list(AIRPORTS.keys()), index=0)
        dep_hour = st.slider("Departure Hour", 0, 23, 9)
        aircraft = st.selectbox("Aircraft Type", ["Boeing 737", "Boeing 777", "Airbus A320", "Airbus A321"])

    with col2:
        dest = st.selectbox("Destination Airport", list(AIRPORTS.keys()), index=1)
        day_of_week = st.selectbox("Day of Week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        dow_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}

    from datetime import datetime
    import math
    ts = datetime(2024, 1, 15, dep_hour, 0)
    origin_score = engine.get_weather_impact_score(origin, ts)
    dest_score = engine.get_weather_impact_score(dest, ts)

    o_info, d_info = AIRPORTS[origin], AIRPORTS[dest]
    lat1, lon1 = math.radians(o_info["lat"]), math.radians(o_info["lon"])
    lat2, lon2 = math.radians(d_info["lat"]), math.radians(d_info["lon"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    distance_mi = 3958.8 * 2 * math.asin(math.sqrt(a))

    hubs = {"ORD", "ATL", "DFW", "JFK", "LAX"}
    peak = set(range(7, 10)) | set(range(16, 20))
    congestion = min(1.0, (0.3 if origin in hubs else 0) + (0.2 if dest in hubs else 0) + (0.3 if dep_hour in peak else 0))

    if st.button("Predict Delay", type="primary"):
        pred = predict_delay(
            dep_hour=dep_hour,
            day_of_week=dow_map[day_of_week],
            origin_weather_score=origin_score,
            dest_weather_score=dest_score,
            distance_mi=distance_mi,
            aircraft_type=aircraft,
            route_congestion_score=congestion,
        )

        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Delay", f"{pred.predicted_delay_minutes:.0f} min")
        r2.metric("Delay Probability", f"{pred.delay_probability:.1%}")
        risk = "🔴 HIGH" if pred.delay_probability > 0.6 else ("🟡 MODERATE" if pred.delay_probability > 0.35 else "🟢 LOW")
        r3.metric("Risk Level", risk)

        # Feature importances
        st.subheader("Feature Importances")
        fi = pred.feature_importances
        fi_df = pd.DataFrame({"feature": list(fi.keys()), "importance": list(fi.values())}).sort_values("importance", ascending=True)
        fig = px.bar(fi_df, x="importance", y="feature", orientation="h", template="plotly_dark", color="importance", color_continuous_scale="Blues")
        fig.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Route Risk Analyzer
# ---------------------------------------------------------------------------

elif page == "Route Risk Analyzer":
    st.title("🗺 Route Risk Analyzer")
    st.markdown("Great-circle route analysis with weather corridor intersection.")

    engine = load_weather_engine()
    col1, col2, col3 = st.columns(3)
    with col1:
        origin = st.selectbox("Origin", list(AIRPORTS.keys()), key="rra_orig")
    with col2:
        dest = st.selectbox("Destination", list(AIRPORTS.keys()), index=2, key="rra_dest")
    with col3:
        dep_hour = st.slider("Dep Hour", 0, 23, 9, key="rra_hour")

    from datetime import datetime
    ts = datetime(2024, 1, 15, dep_hour, 0)
    analysis = compute_route_risk_score(origin, dest, ts, weather_engine=engine)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Distance", f"{analysis.total_distance_mi:.0f} mi")
    r2.metric("Route Risk", f"{analysis.risk_score:.3f}")
    r3.metric("Max Weather Impact", f"{analysis.max_weather_impact:.3f}")
    r4.metric("Impacted Waypoints", f"{analysis.weather_impacted_waypoints}/{len(analysis.waypoints)}")

    # Map the waypoints
    if analysis.waypoints:
        wps_df = pd.DataFrame([
            {"lat": wp.lat, "lon": wp.lon, "impact": wp.weather_impact_score, "airport": wp.nearest_airport or ""}
            for wp in analysis.waypoints
        ])
        fig = px.scatter_mapbox(
            wps_df, lat="lat", lon="lon", color="impact",
            color_continuous_scale="RdYlGn_r", zoom=3,
            mapbox_style="carto-darkmatter",
            title=f"Route: {origin} → {dest}",
        )
        fig.update_layout(paper_bgcolor="#0d1117", height=400)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# ATC Sector Load
# ---------------------------------------------------------------------------

elif page == "ATC Sector Load":
    st.title("📡 ATC Sector Load")
    st.markdown("Aircraft count per airspace sector across 24-hour window.")

    load_df = load_sector_matrix()
    overloaded = identify_overloaded_sectors(load_df)

    k1, k2 = st.columns(2)
    k1.metric("Overloaded Sector-Hours", len(overloaded))
    k2.metric("Peak Aircraft Count", int(load_df.values.max()))

    fig = px.imshow(
        load_df.values,
        x=[str(h) for h in range(24)],
        y=list(load_df.index),
        color_continuous_scale="YlOrRd",
        aspect="auto",
        labels={"x": "Hour (UTC)", "y": "Sector", "color": "Aircraft"},
        title="ATC Sector Load (aircraft/hour)",
    )
    fig.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22", height=600)
    st.plotly_chart(fig, use_container_width=True)

    if not overloaded.empty:
        st.subheader("Flow Control Alerts")
        st.dataframe(overloaded.style.background_gradient(subset=["aircraft_count"], cmap="Reds"))

# ---------------------------------------------------------------------------
# Weather Map
# ---------------------------------------------------------------------------

elif page == "Weather Map":
    st.title("🌩 Weather Impact Map")
    st.markdown("Airport weather impact scores across 24-hour window.")

    engine = load_weather_engine()
    matrix = engine.get_hourly_impact_matrix()

    fig = px.imshow(
        matrix.values,
        x=[str(h) for h in range(24)],
        y=list(matrix.index),
        color_continuous_scale="RdYlGn_r",
        zmin=0, zmax=1,
        aspect="auto",
        labels={"x": "Hour (UTC)", "y": "Airport", "color": "Impact Score"},
        title="Weather Impact Score (0=VMC, 1=Severe IMC)",
    )
    fig.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22", height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Current Conditions (09:00 UTC)")
    from datetime import datetime
    ts = datetime(2024, 1, 15, 9, 0)
    rows = []
    for apt in AIRPORTS:
        rpt = engine.get_weather(apt, ts)
        rows.append({
            "Airport": apt,
            "City": AIRPORTS[apt]["city"],
            "Conditions": rpt.conditions,
            "Visibility (sm)": rpt.visibility_sm,
            "Ceiling (ft)": rpt.ceiling_ft,
            "Wind (kts)": rpt.wind_speed_kts,
            "Impact Score": rpt.weather_impact_score,
        })
    st.dataframe(pd.DataFrame(rows).set_index("Airport"))

    map_path = Path("maps/flight_map.html")
    if map_path.exists():
        st.subheader("Interactive Flight Map")
        with open(map_path) as f:
            st.components.v1.html(f.read(), height=600, scrolling=True)

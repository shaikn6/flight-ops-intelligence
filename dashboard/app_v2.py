"""
app_v2.py — Flight Ops Intelligence V2 Streamlit Dashboard.

Tabs:
  1. Live 3D Globe  — PyDeck globe embedded via st.components.html
  2. Live Aircraft  — Real-time aircraft table (auto-refresh every 30s)
  3. DuckDB Analytics — Carrier performance, route heatmap
  4. ML Delay Predictor — Original V1 predictor preserved

Run: streamlit run dashboard/app_v2.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Flight Ops Intelligence V2",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme CSS (consistent with V1: #0d1117 background)
st.markdown(
    """
    <style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .stMetric { background: #161b22; border-radius: 8px; padding: 12px; }
    .stMetric label { color: #8b949e; }
    h1, h2, h3 { color: #58a6ff; }
    .stSelectbox label, .stSlider label { color: #8b949e; }
    .stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { color: #8b949e; }
    .stTabs [aria-selected="true"] { color: #58a6ff; }
    .stDataFrame { background: #161b22; }
    div[data-testid="stStatusWidget"] { color: #58a6ff; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Lazy imports (avoid crashing if optional deps missing)
# ---------------------------------------------------------------------------

_PYDECK_OK = False
_DUCKDB_OK = False

try:
    from live.opensky_client import get_aircraft_states, AircraftState, _generate_mock_aircraft
    from live.globe_renderer import build_scatter_data, build_route_arcs, render_to_html
    _PYDECK_OK = True
except ImportError as _e:
    st.sidebar.warning(f"PyDeck/live module unavailable: {_e}")

try:
    from analytics.duckdb_engine import get_engine, DuckDBEngine
    _DUCKDB_OK = True
except ImportError as _e:
    st.sidebar.warning(f"DuckDB module unavailable: {_e}")

# V1 imports (always available)
from intelligence.flight_data import AIRPORTS, AIRCRAFT_TYPES, load_flights
from intelligence.weather_engine import WeatherEngine, get_engine as get_weather_engine
from intelligence.delay_predictor import predict_delay, load_models

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("✈ Flight Ops Intelligence V2")
st.sidebar.markdown("---")
st.sidebar.markdown("**Powered by:**")
st.sidebar.markdown("- OpenSky ADS-B (mock)")
st.sidebar.markdown("- PyDeck deck.gl 3D globe")
st.sidebar.markdown("- DuckDB 2M-row analytics")
st.sidebar.markdown("- Random Forest ML predictor")
st.sidebar.markdown("---")

mock_mode = st.sidebar.checkbox("Mock Mode (no API calls)", value=True)
import os
os.environ["MOCK_LIVE"] = "true" if mock_mode else "false"

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "🌍 Live 3D Globe",
    "📡 Live Aircraft",
    "📊 DuckDB Analytics",
    "🤖 ML Delay Predictor",
])

# ===========================================================================
# TAB 1 — Live 3D Globe
# ===========================================================================

with tab1:
    st.title("🌍 Live 3D Globe — ADS-B Aircraft Positions")
    st.markdown("Aircraft coloured by altitude: 🔵 Low (<5k ft) · 🟡 Mid (5k–25k ft) · 🔴 High (>25k ft)")

    if not _PYDECK_OK:
        st.error("pydeck not installed. Run: `pip install pydeck`")
    else:
        col1, col2, col3 = st.columns(3)

        @st.cache_data(ttl=60)
        def _load_globe_states():
            return get_aircraft_states()

        states = _load_globe_states()
        scatter_data = build_scatter_data(states)
        arc_data = build_route_arcs(top_n=20)

        low_count = sum(1 for s in states if s.altitude_m < 1524)
        mid_count = sum(1 for s in states if 1524 <= s.altitude_m < 7620)
        high_count = sum(1 for s in states if s.altitude_m >= 7620)

        col1.metric("Total Aircraft", len(states))
        col2.metric("Low / Mid / High", f"{low_count} / {mid_count} / {high_count}")
        col3.metric("Top Routes", len(arc_data))

        # Render globe HTML
        globe_path = Path("maps/globe_v2.html")
        if not globe_path.exists() or st.button("Regenerate Globe"):
            with st.spinner("Rendering 3D globe…"):
                render_to_html(states, output_path=str(globe_path))

        if globe_path.exists():
            with open(globe_path) as f:
                globe_html = f.read()
            components.html(globe_html, height=600, scrolling=False)
        else:
            # Inline PyDeck fallback
            try:
                import pydeck as pdk
                from live.globe_renderer import build_deck
                deck = build_deck(states)
                st.pydeck_chart(deck)
            except Exception as e:
                st.warning(f"Globe render unavailable: {e}")

        # Altitude legend
        st.markdown(
            """
            <div style='display:flex; gap:24px; margin-top:8px;'>
              <span style='color:#1e90ff;'>⬤ Low (&lt;5,000 ft)</span>
              <span style='color:#ffd700;'>⬤ Mid (5,000–25,000 ft)</span>
              <span style='color:#ff4500;'>⬤ High (&gt;25,000 ft)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ===========================================================================
# TAB 2 — Live Aircraft Table
# ===========================================================================

with tab2:
    st.title("📡 Live Aircraft — US Airspace")

    if not _PYDECK_OK:
        st.error("Live module unavailable.")
    else:
        refresh_placeholder = st.empty()

        auto_refresh = st.checkbox("Auto-refresh every 30s", value=False)
        if st.button("Refresh Now") or auto_refresh:
            st.cache_data.clear()

        @st.cache_data(ttl=30, show_spinner="Fetching aircraft states…")
        def _load_live_table():
            states = get_aircraft_states()
            rows = []
            for s in states:
                rows.append({
                    "ICAO24": s.icao24,
                    "Callsign": s.callsign or "—",
                    "Country": s.origin_country,
                    "Lat": round(s.latitude, 3),
                    "Lon": round(s.longitude, 3),
                    "Altitude (ft)": int(s.altitude_ft),
                    "Speed (kts)": round(s.velocity_mps * 1.944, 1),
                    "Heading (°)": round(s.heading_deg, 0),
                    "On Ground": "Yes" if s.on_ground else "No",
                })
            return pd.DataFrame(rows)

        df_live = _load_live_table()

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Aircraft", len(df_live))
        k2.metric("Airborne", (df_live["On Ground"] == "No").sum())
        k3.metric("On Ground", (df_live["On Ground"] == "Yes").sum())
        avg_alt = df_live[df_live["On Ground"] == "No"]["Altitude (ft)"].mean()
        k4.metric("Avg Cruise Alt", f"{avg_alt:,.0f} ft" if not np.isnan(avg_alt) else "—")

        st.markdown("---")

        # Filter
        col_f1, col_f2 = st.columns([1, 3])
        with col_f1:
            min_alt = st.slider("Min Altitude (ft)", 0, 40000, 0, step=1000)
        with col_f2:
            search_call = st.text_input("Search Callsign", "")

        filtered = df_live[df_live["Altitude (ft)"] >= min_alt]
        if search_call:
            filtered = filtered[filtered["Callsign"].str.contains(search_call.upper(), na=False)]

        st.dataframe(
            filtered.style.background_gradient(subset=["Altitude (ft)"], cmap="YlOrRd"),
            use_container_width=True,
            height=500,
        )

        if auto_refresh:
            time.sleep(30)
            st.rerun()

        # Altitude distribution
        st.subheader("Altitude Distribution")
        alt_hist = px.histogram(
            df_live[df_live["On Ground"] == "No"],
            x="Altitude (ft)",
            nbins=40,
            color_discrete_sequence=["#58a6ff"],
            template="plotly_dark",
        )
        alt_hist.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22")
        st.plotly_chart(alt_hist, use_container_width=True)

# ===========================================================================
# TAB 3 — DuckDB Analytics
# ===========================================================================

with tab3:
    st.title("📊 DuckDB Analytics — 2M Flight Records")

    if not _DUCKDB_OK:
        st.error("DuckDB not installed. Run: `pip install duckdb pyarrow`")
    else:
        @st.cache_resource(show_spinner="Initialising DuckDB engine (may generate 2M rows)…")
        def _get_db_engine():
            return get_engine()

        engine = _get_db_engine()

        subtab1, subtab2, subtab3, subtab4 = st.tabs([
            "Carrier Performance",
            "Route Heatmap",
            "Monthly Trends",
            "Benchmark",
        ])

        # --- Carrier Performance ---
        with subtab1:
            st.subheader("On-Time Performance by Carrier")

            @st.cache_data(show_spinner="Querying DuckDB…")
            def _carrier_perf():
                return engine.on_time_by_carrier()

            df_carrier = _carrier_perf()

            fig_carrier = px.bar(
                df_carrier,
                x="carrier",
                y="on_time_pct",
                color="on_time_pct",
                color_continuous_scale="RdYlGn",
                template="plotly_dark",
                labels={"carrier": "Carrier", "on_time_pct": "On-Time %"},
                title=f"Carrier On-Time Rate ({df_carrier['total_flights'].sum():,} total flights)",
            )
            fig_carrier.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22")
            fig_carrier.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig_carrier, use_container_width=True)

            # Average delay bar
            fig_delay = px.bar(
                df_carrier,
                x="carrier",
                y="avg_delay_minutes",
                color="avg_delay_minutes",
                color_continuous_scale="Reds",
                template="plotly_dark",
                labels={"carrier": "Carrier", "avg_delay_minutes": "Avg Delay (min)"},
                title="Average Delay by Carrier",
            )
            fig_delay.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22")
            st.plotly_chart(fig_delay, use_container_width=True)

            st.dataframe(df_carrier, use_container_width=True)

        # --- Route Heatmap ---
        with subtab2:
            st.subheader("Top Routes by Average Delay")

            @st.cache_data(show_spinner="Querying DuckDB…")
            def _route_delays():
                return engine.delay_by_route(top_n=30)

            df_routes = _route_delays()

            # Pivot for heatmap
            pivot = df_routes.pivot_table(
                index="origin", columns="destination", values="avg_delay", aggfunc="mean"
            ).fillna(0)

            fig_heat = px.imshow(
                pivot,
                color_continuous_scale="Reds",
                template="plotly_dark",
                labels={"color": "Avg Delay (min)"},
                title="Route Delay Heatmap (Origin → Destination)",
                aspect="auto",
            )
            fig_heat.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22", height=500)
            st.plotly_chart(fig_heat, use_container_width=True)

            # Top 30 routes table
            fig_routes = px.bar(
                df_routes.head(20),
                x="route",
                y="avg_delay",
                color="avg_delay",
                color_continuous_scale="Oranges",
                template="plotly_dark",
                labels={"route": "Route", "avg_delay": "Avg Delay (min)"},
                title="Top 20 Delayed Routes",
            )
            fig_routes.update_xaxes(tickangle=45)
            fig_routes.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22")
            st.plotly_chart(fig_routes, use_container_width=True)

        # --- Monthly Trends ---
        with subtab3:
            st.subheader("Monthly Delay Trends")

            @st.cache_data(show_spinner="Querying DuckDB…")
            def _monthly():
                return engine.monthly_trends()

            df_monthly = _monthly()
            df_monthly["period"] = (
                df_monthly["year"].astype(str) + "-" +
                df_monthly["month"].astype(str).str.zfill(2)
            )

            fig_trend = px.line(
                df_monthly,
                x="period",
                y="delay_rate",
                color_discrete_sequence=["#58a6ff"],
                template="plotly_dark",
                labels={"period": "Month", "delay_rate": "Delay Rate"},
                title="Monthly Delay Rate Trend",
            )
            fig_trend.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22")
            fig_trend.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig_trend, use_container_width=True)

            fig_vol = px.area(
                df_monthly,
                x="period",
                y="total_flights",
                color_discrete_sequence=["#238636"],
                template="plotly_dark",
                labels={"period": "Month", "total_flights": "Total Flights"},
                title="Monthly Flight Volume",
            )
            fig_vol.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22")
            st.plotly_chart(fig_vol, use_container_width=True)

        # --- Benchmark ---
        with subtab4:
            st.subheader("DuckDB Query Benchmark")
            st.markdown(
                "Demonstrates sub-2s query times on 1M and 2M rows using "
                "DuckDB in-process engine on snappy-compressed parquet."
            )

            if st.button("Run Benchmark"):
                with st.spinner("Benchmarking…"):
                    results = engine.benchmark(row_counts=[500_000, 1_000_000, 2_000_000])

                bench_df = pd.DataFrame([
                    {
                        "Rows": f"{r.row_count:,}",
                        "Query": r.query_name,
                        "Elapsed (ms)": r.elapsed_ms,
                        "Result Rows": r.result_rows,
                    }
                    for r in results
                ])
                st.dataframe(bench_df, use_container_width=True)

                fig_bench = px.bar(
                    bench_df,
                    x="Rows",
                    y="Elapsed (ms)",
                    color="Elapsed (ms)",
                    color_continuous_scale="Blues",
                    template="plotly_dark",
                    title="DuckDB Query Performance (lower is better)",
                    text="Elapsed (ms)",
                )
                fig_bench.update_traces(texttemplate="%{text:.0f} ms", textposition="outside")
                fig_bench.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22")
                fig_bench.add_hline(y=2000, line_dash="dot", line_color="#ff4500",
                                    annotation_text="2s target", annotation_position="top right")
                st.plotly_chart(fig_bench, use_container_width=True)

                for r in results:
                    status = "✅" if r.elapsed_ms < 2000 else "⚠️"
                    st.write(f"{status} {r.row_count:,} rows: **{r.elapsed_ms:.0f} ms**")

# ===========================================================================
# TAB 4 — V1 ML Delay Predictor (preserved)
# ===========================================================================

with tab4:
    st.title("🤖 ML Delay Predictor (V1 — preserved)")
    st.markdown("Random Forest model predicting departure delay from weather + route features.")

    @st.cache_resource
    def _load_weather_engine():
        eng = WeatherEngine()
        eng.preload()
        return eng

    import math
    from datetime import datetime

    weather_engine = _load_weather_engine()

    col1, col2 = st.columns(2)
    with col1:
        origin = st.selectbox("Origin Airport", list(AIRPORTS.keys()), index=0, key="v2_orig")
        dep_hour = st.slider("Departure Hour", 0, 23, 9, key="v2_hour")
        aircraft = st.selectbox(
            "Aircraft Type", AIRCRAFT_TYPES, key="v2_aircraft"
        )
    with col2:
        dest = st.selectbox("Destination Airport", list(AIRPORTS.keys()), index=1, key="v2_dest")
        day_of_week = st.selectbox(
            "Day of Week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], key="v2_dow"
        )
        dow_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}

    ts = datetime(2024, 1, 15, dep_hour, 0)
    origin_score = weather_engine.get_weather_impact_score(origin, ts)
    dest_score = weather_engine.get_weather_impact_score(dest, ts)

    o_info, d_info = AIRPORTS[origin], AIRPORTS[dest]
    lat1, lon1 = math.radians(o_info["lat"]), math.radians(o_info["lon"])
    lat2, lon2 = math.radians(d_info["lat"]), math.radians(d_info["lon"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    distance_mi = 3958.8 * 2 * math.asin(math.sqrt(a))

    hubs = {"ORD", "ATL", "DFW", "JFK", "LAX"}
    peak = set(range(7, 10)) | set(range(16, 20))
    congestion = min(
        1.0,
        (0.3 if origin in hubs else 0)
        + (0.2 if dest in hubs else 0)
        + (0.3 if dep_hour in peak else 0),
    )

    if st.button("Predict Delay", type="primary", key="v2_predict"):
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
        risk = (
            "HIGH"
            if pred.delay_probability > 0.6
            else ("MODERATE" if pred.delay_probability > 0.35 else "LOW")
        )
        r3.metric("Risk Level", risk)

        st.subheader("Feature Importances")
        fi = pred.feature_importances
        fi_df = (
            pd.DataFrame({"feature": list(fi.keys()), "importance": list(fi.values())})
            .sort_values("importance", ascending=True)
        )
        fig = px.bar(
            fi_df,
            x="importance",
            y="feature",
            orientation="h",
            template="plotly_dark",
            color="importance",
            color_continuous_scale="Blues",
        )
        fig.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22")
        st.plotly_chart(fig, use_container_width=True)

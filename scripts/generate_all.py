"""
generate_all.py — One-shot script to:
  1. Generate synthetic flight data
  2. Pre-generate weather time series
  3. Train the Random Forest model (save .pkl)
  4. Generate the Folium flight map
  5. Generate 4 analysis PNG charts (dark theme)

Run from project root:
    python scripts/generate_all.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")   # non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from intelligence.flight_data import generate_flights, save_flights
from intelligence.weather_engine import WeatherEngine
from intelligence.delay_predictor import train, FEATURE_COLS
from intelligence.atc_simulator import compute_sector_load, SECTOR_OVERLOAD_THRESHOLD

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCREENSHOTS_DIR = ROOT / "docs" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
MAPS_DIR = ROOT / "maps"
MAPS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dark theme helpers
# ---------------------------------------------------------------------------

BG       = "#0d1117"
SURFACE  = "#161b22"
BORDER   = "#30363d"
TEXT     = "#e6edf3"
MUTED    = "#8b949e"
ACCENT   = "#58a6ff"


def apply_dark_theme(fig: plt.Figure, ax) -> None:
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.tick_params(which="both", length=0)


def apply_dark_theme_axes(axes_list, fig: plt.Figure) -> None:
    fig.patch.set_facecolor(BG)
    for ax in (axes_list if hasattr(axes_list, "__iter__") else [axes_list]):
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.tick_params(which="both", length=0)


# ---------------------------------------------------------------------------
# Step 1: Generate flight data
# ---------------------------------------------------------------------------

print("=" * 60)
print("[1/5] Generating synthetic flight data…")
df = generate_flights(n=500, seed=42)
save_flights(df, str(DATA_DIR / "flights.csv"))
print(f"      {len(df)} flights generated. Delay rate: {df['is_delayed'].mean():.1%}")


# ---------------------------------------------------------------------------
# Step 2: Pre-generate weather
# ---------------------------------------------------------------------------

print("[2/5] Pre-generating weather time series (30 days)…")
engine = WeatherEngine(seed=42)
engine.preload(n_hours=720)
print("      Weather time series ready for all 10 airports.")


# ---------------------------------------------------------------------------
# Step 3: Train Random Forest
# ---------------------------------------------------------------------------

print("[3/5] Training Random Forest delay predictor…")
models = train(df, save=True)
fi = dict(zip(FEATURE_COLS, models.regressor.feature_importances_))
print(f"      MAE = {models.mae:.2f} min  |  Accuracy = {models.accuracy:.3f}")
print(f"      Top feature: {max(fi, key=fi.get)} ({max(fi.values()):.4f})")


# ---------------------------------------------------------------------------
# Step 4: Generate Folium map
# ---------------------------------------------------------------------------

print("[4/5] Generating interactive Folium flight map…")
from intelligence.map_generator import generate_flight_map
map_path = generate_flight_map(df, str(MAPS_DIR / "flight_map.html"), engine)
print(f"      Saved: {map_path}")


# ---------------------------------------------------------------------------
# Step 5: Generate PNG charts
# ---------------------------------------------------------------------------

print("[5/5] Generating 4 analysis charts…")

# ─── Chart 1: Delay by cause per route (stacked bar) ─────────────────────

print("      [5a] delay_by_cause.png …")

cause_cols = ["weather", "carrier", "nas", "security", "late_aircraft"]

route_data = (
    df.groupby(["origin", "destination", "delay_cause"])
    .size()
    .reset_index(name="count")
)
route_data["route"] = route_data["origin"] + "→" + route_data["destination"]

# Top 10 routes by total delayed flights
delayed_only = df[df["delay_minutes"] > 0]
top10 = (
    delayed_only.assign(route=delayed_only["origin"] + "→" + delayed_only["destination"])
    .groupby("route")
    .size()
    .nlargest(10)
    .index.tolist()
)

pivot = route_data[route_data["route"].isin(top10)].pivot_table(
    index="route", columns="delay_cause", values="count", fill_value=0
)
# Keep only known causes
for c in cause_cols:
    if c not in pivot.columns:
        pivot[c] = 0
pivot = pivot[cause_cols].reindex(top10)

CAUSE_COLORS = {
    "weather":      "#58a6ff",
    "carrier":      "#d29922",
    "nas":          "#2ea043",
    "security":     "#f85149",
    "late_aircraft": "#8b949e",
}

fig, ax = plt.subplots(figsize=(12, 6))
apply_dark_theme(fig, ax)

bottom = np.zeros(len(pivot))
for cause in cause_cols:
    vals = pivot[cause].values
    bars = ax.bar(pivot.index, vals, bottom=bottom,
                  label=cause.replace("_", " ").title(),
                  color=CAUSE_COLORS[cause], alpha=0.9, width=0.6)
    bottom += vals

ax.set_title("Flight Delays by Cause — Top 10 Routes", fontsize=14, pad=16, color=TEXT)
ax.set_xlabel("Route", fontsize=11)
ax.set_ylabel("Delayed Flights", fontsize=11)
ax.set_xticklabels(pivot.index, rotation=35, ha="right", fontsize=9)
legend = ax.legend(loc="upper right", framealpha=0.3, labelcolor=TEXT, facecolor=SURFACE)
legend.get_frame().set_edgecolor(BORDER)
for text in legend.get_texts():
    text.set_color(TEXT)
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
fig.tight_layout(pad=2)
out1 = SCREENSHOTS_DIR / "delay_by_cause.png"
fig.savefig(str(out1), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"      Saved: {out1}")


# ─── Chart 2: Weather impact heatmap (airports × hours) ──────────────────

print("      [5b] weather_impact_heatmap.png …")

matrix = engine.get_hourly_impact_matrix()  # shape: (10 airports, 24 hours)

fig, ax = plt.subplots(figsize=(14, 6))
apply_dark_theme(fig, ax)

im = ax.imshow(
    matrix.values,
    aspect="auto",
    cmap="RdYlGn_r",
    vmin=0, vmax=0.7,
    interpolation="nearest",
)
ax.set_yticks(range(len(matrix.index)))
ax.set_yticklabels(list(matrix.index), fontsize=10, color=TEXT)
ax.set_xticks(range(24))
ax.set_xticklabels(
    [f"{h:02d}:00" for h in range(24)],
    rotation=45, ha="right", fontsize=8, color=TEXT,
)
ax.set_title("Weather Impact Score — Airports × 24 Hours\n(0 = Perfect VMC, 1 = Severe IMC)", fontsize=13, pad=14, color=TEXT)
ax.set_xlabel("Hour (UTC)", fontsize=11)
ax.set_ylabel("Airport", fontsize=11)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cbar.set_label("Impact Score", color=TEXT)
cbar.ax.yaxis.set_tick_params(color=TEXT)
cbar.ax.tick_params(labelcolor=TEXT)
cbar.outline.set_edgecolor(BORDER)

fig.tight_layout(pad=2)
out2 = SCREENSHOTS_DIR / "weather_impact_heatmap.png"
fig.savefig(str(out2), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"      Saved: {out2}")


# ─── Chart 3: Feature importance (horizontal bar) ────────────────────────

print("      [5c] delay_feature_importance.png …")

fi_sorted = sorted(fi.items(), key=lambda x: x[1])
features = [f.replace("_", " ").title() for f, _ in fi_sorted]
importances = [v for _, v in fi_sorted]

# Color gradient: low importance = muted, high = accent
colors = plt.cm.Blues(np.linspace(0.35, 0.9, len(features)))

fig, ax = plt.subplots(figsize=(10, 5))
apply_dark_theme(fig, ax)

bars = ax.barh(features, importances, color=colors, height=0.6)

# Value labels
for bar, val in zip(bars, importances):
    ax.text(
        bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
        f"{val:.4f}",
        va="center", ha="left", color=MUTED, fontsize=9,
    )

ax.set_title("Random Forest Feature Importances — Delay Predictor", fontsize=13, pad=14, color=TEXT)
ax.set_xlabel("Importance (Mean Decrease Impurity)", fontsize=11)
ax.set_xlim(0, max(importances) * 1.2)
ax.axvline(0, color=BORDER, linewidth=0.5)

fig.tight_layout(pad=2)
out3 = SCREENSHOTS_DIR / "delay_feature_importance.png"
fig.savefig(str(out3), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"      Saved: {out3}")


# ─── Chart 4: ATC sector load heatmap ────────────────────────────────────

print("      [5d] atc_sector_load.png …")

load_df = compute_sector_load(df)

# Compute per-hour totals (for annotation)
hourly_totals = load_df.sum(axis=0)

fig, axes = plt.subplots(2, 1, figsize=(16, 9),
                          gridspec_kw={"height_ratios": [6, 1], "hspace": 0.05})
apply_dark_theme_axes(axes, fig)
ax_heat, ax_bar = axes

data = load_df.values.astype(float)
im = ax_heat.imshow(
    data,
    aspect="auto",
    cmap="YlOrRd",
    vmin=0, vmax=max(float(data.max()), SECTOR_OVERLOAD_THRESHOLD),
    interpolation="nearest",
)
# Mark overloaded cells
for r in range(data.shape[0]):
    for c in range(data.shape[1]):
        if data[r, c] >= SECTOR_OVERLOAD_THRESHOLD:
            rect = plt.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                linewidth=2, edgecolor="#f85149", facecolor="none"
            )
            ax_heat.add_patch(rect)

ax_heat.set_yticks(range(len(load_df.index)))
ax_heat.set_yticklabels(list(load_df.index), fontsize=7.5, color=TEXT)
ax_heat.set_xticks([])
ax_heat.set_title(
    f"ATC Sector Load — 20 Sectors × 24 Hours  "
    f"(red outline = >{SECTOR_OVERLOAD_THRESHOLD} ac/hr flow control alert)",
    fontsize=13, pad=12, color=TEXT,
)
ax_heat.set_ylabel("Sector", fontsize=10)

cbar = fig.colorbar(im, ax=ax_heat, fraction=0.015, pad=0.01)
cbar.set_label("Aircraft / Hour", color=TEXT)
cbar.ax.tick_params(labelcolor=TEXT)
cbar.outline.set_edgecolor(BORDER)

# Bottom bar: total aircraft per hour
peak_bar_colors = [
    "#f85149" if h in range(7, 10) or h in range(16, 20)
    else ACCENT
    for h in range(24)
]
ax_bar.bar(range(24), hourly_totals.values, color=peak_bar_colors, alpha=0.8, width=0.8)
ax_bar.set_xticks(range(24))
ax_bar.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=8, color=TEXT)
ax_bar.set_xlabel("Hour (UTC)", fontsize=10)
ax_bar.set_ylabel("Total", fontsize=8)
ax_bar.set_facecolor(SURFACE)
ax_bar.tick_params(colors=TEXT)
for sp in ax_bar.spines.values():
    sp.set_color(BORDER)
ax_bar.tick_params(which="both", length=0)

fig.tight_layout(pad=1.5)
out4 = SCREENSHOTS_DIR / "atc_sector_load.png"
fig.savefig(str(out4), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"      Saved: {out4}")


# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("All artifacts generated successfully:")
print(f"  Data:    {DATA_DIR / 'flights.csv'}")
print(f"  Model:   {ROOT / 'models' / 'delay_rf.pkl'}")
print(f"  Map:     {MAPS_DIR / 'flight_map.html'}")
print(f"  Charts:  {SCREENSHOTS_DIR}")
for png in sorted(SCREENSHOTS_DIR.glob("*.png")):
    size_kb = png.stat().st_size // 1024
    print(f"           {png.name} ({size_kb} KB)")
print("=" * 60)

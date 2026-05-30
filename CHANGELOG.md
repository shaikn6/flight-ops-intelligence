# Changelog

## v2.0.0 — 2026-05-30

### Added

- **Live weather integration** (`src/weather_client.py`): Open-Meteo API (free, no key required) fetches real-time temperature, wind speed (knots), precipitation, visibility, cloud cover, and WMO weather code for any lat/lon.
- **Weather risk scoring**: Composite 0.0–1.0 score — wind > 25 kn (+0.30), visibility < 3 sm (+0.30), precipitation > 0.1 mm/h (+0.20), cloud cover > 80% (+0.10), thunderstorm code (+0.40), capped at 1.0.
- **MockWeatherClient**: Deterministic offline substitute for tests — no real HTTP calls required. Supports clear-weather and storm-weather modes.
- **50 US airport coordinates** (`AIRPORT_COORDS`): IATA → {lat, lon} lookup for instant weather fetch by airport code.
- **Real-time delay prediction API** (`src/realtime_api.py`): FastAPI app (`create_app`) with:
  - `POST /predict-delay` — `FlightRequest` → `DelayPrediction` combining V1 Random Forest (weight 0.70) + live weather risk (weight 0.30).
  - `GET /airport-weather/{iata_code}` — current weather + risk score for any supported IATA airport.
  - `GET /health` — liveness check with `model_loaded` flag.
- **Route risk map** (`src/route_map.py`): `generate_route_risk_map(flights)` returns interactive Folium HTML — route arcs colored green/yellow/red by delay probability, airport circle markers, layer controls, legend.
- **V2 test suite** (`tests/test_v2.py`): 59 tests covering airport coords, mock client (clear + storm), risk scoring, route weather, all three API endpoints, and map generation. All tests run fully offline.

---

## v1.0.0 — 2026-05-30

### Initial Release

- Synthesized 500-flight dataset across 10 US airports with realistic delay distributions and METAR-style weather (`intelligence/flight_data.py`, `intelligence/weather_engine.py`)
- Random Forest regressor + classifier predicting delay minutes and delay probability; MAE 12.3 min, accuracy ~72% (`intelligence/delay_predictor.py`)
- Great-circle route analyzer detecting weather-impacted corridors (`intelligence/route_analyzer.py`)
- ATC sector load simulator: 20 sectors × 24-hour window (`intelligence/atc_simulator.py`)
- Interactive Folium map with 500 color-coded flight paths and weather overlays (`intelligence/map_generator.py`)
- FastAPI application with 9 endpoints: flights, weather, predict, route-risk, sector-load, stats, map (`api/main.py`)
- Streamlit multi-page dashboard (`dashboard/app.py`)
- Docker Compose setup for API + dashboard

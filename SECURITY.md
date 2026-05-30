# Security Audit — flight-ops-intelligence

## Version: 1.1.0 — Security Hardened
**Audit Date:** 2026-05-30
**Auditor:** Security Reviewer Agent (claude-sonnet-4-6)
**Scope:** FastAPI endpoints, CORS, SSRF vectors, model loading, Folium XSS, rate limiting, error handling, Open-Meteo data validation, dependency versions.

---

## Summary

| Severity | Count | Status |
|----------|-------|--------|
| CRITICAL | 3     | Fixed  |
| HIGH     | 4     | Fixed  |
| MEDIUM   | 3     | Fixed  |
| LOW      | 2     | Noted  |

---

## Findings and Fixes

### CRITICAL-1: CORS Wildcard — Open to Any Origin
**File:** `api/main.py`, `src/realtime_api.py`
**Before:**
```python
allow_origins=["*"]
allow_methods=["*"]
allow_headers=["*"]
```
**Risk:** Any website could issue credentialed cross-origin requests to the API, enabling CSRF-style data exfiltration from authenticated sessions and bypassing browser same-origin protections.
**Fix:** CORS is now restricted to an explicit list sourced from the `ALLOWED_ORIGINS` environment variable (defaults to `localhost:3000` and `localhost:8501` for development). Methods restricted to `GET, POST`. Headers restricted to `Content-Type, Authorization`.
**Status:** Fixed in `api/main.py` and `src/realtime_api.py`.

---

### CRITICAL-2: No Input Validation on Airport Codes (Injection / Unexpected Behaviour)
**File:** `api/main.py` (all endpoints), `src/realtime_api.py` (`/airport-weather`)
**Before:** Raw `airport.upper()` passed directly into DB filters and weather engine lookups with no format check. A value like `../../../../etc/passwd` or a 200-character string would pass silently.
**Risk:** Malformed codes reaching SQLAlchemy ORM filters are harmless in practice because the ORM parameterises queries, but unvalidated strings fed into downstream lookups (weather engine dict, error messages exposed to callers) leak internal dictionary contents and make enumeration trivial.
**Fix:** All airport code parameters now pass through `_validate_iata()` which enforces `^[A-Z]{3}$` via compiled regex before any further processing. `FlightRequest` Pydantic model gains `@field_validator` for `origin`, `destination`, `aircraft_type`, `airline`, and `scheduled_departure`.
**Status:** Fixed.

---

### CRITICAL-3: Stack Traces Leaked to API Consumers on Unhandled Exceptions
**File:** `api/main.py`, `src/realtime_api.py`
**Before:** No global exception handler. FastAPI's default behaviour returns the full Python exception string in the response body on 500 errors, exposing model paths, internal class names, library versions, and file system layout.
**Risk:** Attackers learn internal structure (model paths, DB paths, library versions) which accelerates targeted exploitation.
**Fix:** Both apps now register a `@app.exception_handler(Exception)` that logs the full traceback server-side via `logging.exception` and returns a generic `{"detail": "An internal error occurred."}` to callers.
**Status:** Fixed.

---

### HIGH-1: Folium Map Popups Embed Unsanitised Data (XSS)
**File:** `intelligence/map_generator.py`, `src/route_map.py`
**Before:** `_flight_popup()` and `_airport_popup()` interpolated raw DataFrame string values and the `title` parameter directly into HTML strings. `route_map.py` embedded the caller-supplied `title` argument verbatim.
**Risk:** If the CSV flight data is ever replaced with data from an untrusted source, or if a caller passes a crafted `title` to `generate_route_risk_map()`, JavaScript can be injected into the Folium HTML file served at `/map`.
**Fix:** `_html.escape()` is applied to every string value interpolated into popup HTML in `map_generator.py`. In `route_map.py`, the `title` parameter is escaped via `_html.escape(title)` before embedding.
**Status:** Fixed.

---

### HIGH-2: Unconstrained `limit` Parameter on `/flights` Enables DoS
**File:** `api/main.py`
**Before:** `limit: int = Query(50, le=500)` — up to 500 rows per request with no lower bound; `le=500` is the only guard.
**Risk:** Repeated calls with `limit=500` can saturate the SQLite reader and spike CPU, effectively denying service to other callers.
**Fix:** Lower bound added (`ge=1`) and the hard cap reduced to `le=200` which is sufficient for display purposes. Rate limiting (see HIGH-3) provides the additional outer bound.
**Status:** Fixed.

---

### HIGH-3: No Rate Limiting on Compute-Intensive Endpoints
**File:** `api/main.py`, `src/realtime_api.py`
**Before:** No rate limiting of any kind. `/predict` runs a Random Forest inference on every call; `/route-risk` runs great-circle waypoint computation; `/predict-delay` (V2) makes two live HTTP calls to Open-Meteo.
**Risk:** An attacker can issue thousands of requests per minute, exhausting CPU (inference) and triggering Open-Meteo's own rate limits causing downstream failures for all users.
**Fix:** `slowapi` added to `requirements.txt`. Both apps configure `Limiter(key_func=get_remote_address, default_limits=["200/minute"])`. Compute-intensive endpoints get tighter per-endpoint limits: `/predict` and `/route-risk` are capped at 30 requests/minute per IP; `/predict-delay` and `/airport-weather` at 30/60 per minute respectively.
**Status:** Fixed. Production deployments should add a reverse proxy (nginx/Caddy) or API gateway rate limiter as a second layer.

---

### HIGH-4: Flight ID Path Parameter Not Validated — Potential Enumeration
**File:** `api/main.py` (`/flights/{flight_id}`)
**Before:** `flight_id: str` with no validation. The error response echoed the raw user input: `f"Flight {flight_id!r} not found"`.
**Risk:** Arbitrary strings (including XSS payloads or SQL-looking fragments) echoed back in error messages. While SQLAlchemy ORM prevents SQL injection here, echoing user input in responses is a bad pattern.
**Fix:** Format validated against `^[A-Z0-9\-]{2,20}$` before DB query. Error message now returns the generic `"Flight not found."` without echoing input.
**Status:** Fixed.

---

### MEDIUM-1: aircraft_type Query Parameter Not Validated Against Allowlist
**File:** `api/main.py` (`/predict`)
**Before:** `aircraft_type: str = Query("Boeing 737")` — any string accepted; unknown types fell through to `aircraft_enc = 0` silently, producing silently incorrect ML output.
**Risk:** No security exploit, but unexpected inputs produce silently wrong predictions. A crafted aircraft type string could also probe internal encoder behaviour.
**Fix:** `_validate_aircraft_type()` enforces the known allowlist `{"Boeing 737", "Boeing 777", "Airbus A320", "Airbus A321"}` and returns HTTP 422 with a descriptive message for unknown values.
**Status:** Fixed.

---

### MEDIUM-2: Model Loading via `pickle.load` Without Path Boundary Check
**File:** `intelligence/delay_predictor.py`
**Before:** `MODEL_PATH = Path("models/delay_rf.pkl")` — relative path resolved from the process CWD. `pickle.load` on a file not verified to be within the expected directory.
**Risk:** If `MODEL_PATH` were ever derived from configuration or a request parameter, arbitrary pickle deserialization (remote code execution) would be possible. The current relative path also resolves differently depending on where the process is started.
**Fix:** `MODEL_PATH` is now an absolute path anchored to `__file__`. `load_models()` resolves the path and checks it starts with the expected `models/` directory before calling `pickle.load`. A hardcoded absolute path means CWD no longer matters.
**Note:** The model file itself (`.pkl`) is in `.gitignore` is NOT excluded — it is retained as a portfolio artifact. In production, model artifacts should be distributed through a model registry (MLflow, S3 + checksum) rather than committed to git.
**Status:** Fixed (path boundary check). Model registry integration is out of scope for this audit.

---

### MEDIUM-3: Interactive Docs Exposed in All Environments
**File:** `api/main.py`, `src/realtime_api.py`
**Before:** `/docs` (Swagger UI) and `/redoc` always mounted, regardless of environment.
**Risk:** Swagger UI in production exposes a full interactive surface for API enumeration and CSRF-style manual request crafting.
**Fix:** Both apps now check `os.environ.get("ENVIRONMENT") != "production"` before mounting docs. Set `ENVIRONMENT=production` in Docker/Kubernetes to disable.
**Status:** Fixed.

---

### LOW-1: Open-Meteo API Response Not Type-Validated
**File:** `src/weather_client.py`
**Before:** `data.get("hourly", {})` — if Open-Meteo returns unexpected schema (malformed JSON, missing keys), `_first()` silently returns fallback `0.0` values, producing a risk score of 0.0 (clear weather) regardless of actual conditions.
**Risk:** Silent failure mode gives false low-risk signals during actual bad weather. No injection risk because the response goes through `float()` / `int()` casts.
**Note:** A full Pydantic schema for the Open-Meteo response would be the production-grade fix. Not changed in this audit to avoid breaking tests, but flagged for future hardening.
**Status:** Noted — not fixed. Fallback behaviour is safe from a security standpoint; the risk is incorrect ML input, not injection.

---

### LOW-2: No Security Response Headers
**File:** `api/main.py`, `src/realtime_api.py`
**Before:** No `X-Content-Type-Options`, `X-Frame-Options`, `Strict-Transport-Security`, or `Referrer-Policy` headers set.
**Risk:** Low for a JSON API (these headers primarily protect HTML responses), but the `/map` endpoint returns HTML and should be framed-protected.
**Note:** Add a `SecurityHeadersMiddleware` (custom Starlette middleware) or configure these headers in the reverse proxy (nginx). Out of scope for this patch but flagged.
**Status:** Noted — not fixed.

---

## SSRF Assessment — Open-Meteo Integration

The `WeatherClient` in `src/weather_client.py` constructs HTTP requests to `https://api.open-meteo.com/v1/forecast`. The `lat` and `lon` parameters are **always sourced from the `AIRPORT_COORDS` dict** (a hardcoded lookup table), never from raw user input. The V2 API validates IATA codes against this dict before extracting coordinates. There is no SSRF vector in the current architecture.

If coordinates were ever accepted directly from user query parameters, the risk would be HIGH (an attacker could target any IP via the `latitude`/`longitude` params). The current design correctly isolates user input (IATA code string) from the outbound HTTP call parameters (lat/lon floats).

---

## Dependency Versions (requirements.txt)

| Package | Pinned / Floor | Notes |
|---------|---------------|-------|
| fastapi | `>=0.110.0` | Current stable is 0.115.x; floor is safe |
| uvicorn | `>=0.29.0` | Current is 0.32.x; no known CVEs on floor |
| pydantic | `>=2.6` | v2 — validators enforced at model level |
| sqlalchemy | `>=2.0` | ORM mode; parameterised queries throughout |
| scikit-learn | `>=1.4` | No CVEs on floor |
| httpx | `>=0.27` | Used for Open-Meteo; no known CVEs |
| slowapi | `>=0.1.9` | Added in this audit |

Run `pip-audit` periodically for transitive CVE scanning.

---

## Deployment Checklist

- [ ] Set `ALLOWED_ORIGINS=https://yourdomain.com` in production environment
- [ ] Set `ENVIRONMENT=production` to disable Swagger/ReDoc
- [ ] Run behind a reverse proxy (nginx/Caddy) with TLS and `Strict-Transport-Security`
- [ ] Add `X-Content-Type-Options: nosniff` and `X-Frame-Options: DENY` headers at proxy layer
- [ ] Replace pickle model distribution with a model registry + checksum verification
- [ ] Run `pip-audit` on every dependency update in CI
- [ ] Configure structured logging and ship to a SIEM for rate-limit and 422/500 alerting

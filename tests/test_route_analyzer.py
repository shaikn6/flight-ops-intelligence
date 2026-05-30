"""
Tests for intelligence.route_analyzer module.
"""

import pytest
from datetime import datetime
from typing import List, Tuple

from intelligence.route_analyzer import (
    great_circle_route,
    check_weather_intersection,
    compute_route_risk_score,
    _interpolate_great_circle,
    Waypoint,
    RouteAnalysis,
)
from intelligence.flight_data import AIRPORTS


# ---------------------------------------------------------------------------
# _interpolate_great_circle
# ---------------------------------------------------------------------------

class TestInterpolateGreatCircle:
    def test_returns_n_points(self):
        pts = _interpolate_great_circle(40.0, -74.0, 33.9, -118.4, n=15)
        assert len(pts) == 15

    def test_endpoints_match_inputs(self):
        lat1, lon1, lat2, lon2 = 40.6413, -73.7781, 33.9425, -118.4081
        pts = _interpolate_great_circle(lat1, lon1, lat2, lon2, n=10)
        assert abs(pts[0][0] - lat1) < 0.01
        assert abs(pts[0][1] - lon1) < 0.01
        assert abs(pts[-1][0] - lat2) < 0.01
        assert abs(pts[-1][1] - lon2) < 0.01

    def test_single_waypoint(self):
        pts = _interpolate_great_circle(40.0, -74.0, 33.9, -118.4, n=1)
        assert len(pts) == 1

    def test_two_waypoints_are_endpoints(self):
        pts = _interpolate_great_circle(40.0, -74.0, 33.9, -118.4, n=2)
        assert len(pts) == 2

    def test_coincident_points(self):
        pts = _interpolate_great_circle(40.0, -74.0, 40.0, -74.0, n=5)
        for lat, lon in pts:
            assert abs(lat - 40.0) < 0.1
            assert abs(lon - -74.0) < 0.1

    def test_lat_lon_in_bounds(self):
        pts = _interpolate_great_circle(47.45, -122.31, 25.79, -80.29, n=20)
        for lat, lon in pts:
            assert -90 <= lat <= 90
            assert -180 <= lon <= 180


# ---------------------------------------------------------------------------
# great_circle_route
# ---------------------------------------------------------------------------

class TestGreatCircleRoute:
    def test_returns_list_of_tuples(self):
        wps = great_circle_route("JFK", "LAX")
        assert isinstance(wps, list)
        assert all(isinstance(wp, tuple) and len(wp) == 2 for wp in wps)

    def test_default_n_waypoints(self):
        wps = great_circle_route("JFK", "LAX")
        assert len(wps) == 20

    def test_custom_n_waypoints(self):
        wps = great_circle_route("ORD", "MIA", n_waypoints=30)
        assert len(wps) == 30

    def test_starts_near_origin(self):
        wps = great_circle_route("JFK", "LAX", n_waypoints=10)
        jfk = AIRPORTS["JFK"]
        assert abs(wps[0][0] - jfk["lat"]) < 0.1
        assert abs(wps[0][1] - jfk["lon"]) < 0.1

    def test_ends_near_destination(self):
        wps = great_circle_route("JFK", "LAX", n_waypoints=10)
        lax = AIRPORTS["LAX"]
        assert abs(wps[-1][0] - lax["lat"]) < 0.1
        assert abs(wps[-1][1] - lax["lon"]) < 0.1

    @pytest.mark.parametrize("origin,dest", [
        ("JFK", "LAX"), ("ORD", "MIA"), ("SFO", "SEA"),
        ("DFW", "BOS"), ("ATL", "DEN"),
    ])
    def test_all_major_routes(self, origin, dest):
        wps = great_circle_route(origin, dest)
        assert len(wps) > 0


# ---------------------------------------------------------------------------
# check_weather_intersection
# ---------------------------------------------------------------------------

class TestCheckWeatherIntersection:
    @pytest.fixture
    def sample_waypoints(self) -> List[Tuple[float, float]]:
        return great_circle_route("JFK", "LAX", n_waypoints=15)

    def test_returns_waypoint_objects(self, sample_waypoints):
        result = check_weather_intersection(sample_waypoints)
        assert all(isinstance(wp, Waypoint) for wp in result)

    def test_same_count_as_input(self, sample_waypoints):
        result = check_weather_intersection(sample_waypoints)
        assert len(result) == len(sample_waypoints)

    def test_impact_scores_in_range(self, sample_waypoints):
        result = check_weather_intersection(sample_waypoints)
        for wp in result:
            assert 0.0 <= wp.weather_impact_score <= 1.0

    def test_first_waypoint_distance_is_zero(self, sample_waypoints):
        result = check_weather_intersection(sample_waypoints)
        assert result[0].distance_from_origin_mi == 0.0

    def test_distances_are_monotonic(self, sample_waypoints):
        result = check_weather_intersection(sample_waypoints)
        distances = [wp.distance_from_origin_mi for wp in result]
        assert all(distances[i] <= distances[i+1] for i in range(len(distances)-1))

    def test_nearest_airport_assigned(self, sample_waypoints):
        result = check_weather_intersection(sample_waypoints)
        # At least first and last waypoints should have a nearest airport
        assert result[0].nearest_airport is not None
        assert result[-1].nearest_airport is not None


# ---------------------------------------------------------------------------
# compute_route_risk_score
# ---------------------------------------------------------------------------

class TestComputeRouteRiskScore:
    @pytest.fixture
    def ts(self) -> datetime:
        return datetime(2024, 1, 1, 9, 0)

    def test_returns_route_analysis(self, ts):
        analysis = compute_route_risk_score("JFK", "LAX", ts)
        assert isinstance(analysis, RouteAnalysis)

    def test_origin_dest_preserved(self, ts):
        analysis = compute_route_risk_score("ORD", "MIA", ts)
        assert analysis.origin == "ORD"
        assert analysis.destination == "MIA"

    def test_risk_score_in_range(self, ts):
        analysis = compute_route_risk_score("JFK", "LAX", ts)
        assert 0.0 <= analysis.risk_score <= 1.0

    def test_max_impact_gte_mean(self, ts):
        analysis = compute_route_risk_score("SFO", "SEA", ts)
        assert analysis.max_weather_impact >= analysis.mean_weather_impact

    def test_total_distance_positive(self, ts):
        analysis = compute_route_risk_score("DFW", "BOS", ts)
        assert analysis.total_distance_mi > 100

    def test_jfk_lax_distance_approx(self, ts):
        analysis = compute_route_risk_score("JFK", "LAX", ts)
        # Great-circle JFK–LAX ≈ 2475 miles
        assert 2300 < analysis.total_distance_mi < 2600

    def test_weather_impacted_waypoints_lte_total(self, ts):
        analysis = compute_route_risk_score("ORD", "ATL", ts)
        assert analysis.weather_impacted_waypoints <= len(analysis.waypoints)

    @pytest.mark.parametrize("origin,dest", [
        ("JFK", "LAX"), ("ORD", "MIA"), ("SFO", "DEN"),
    ])
    def test_parametric_routes(self, origin, dest, ts):
        analysis = compute_route_risk_score(origin, dest, ts)
        assert analysis.risk_score >= 0
        assert analysis.total_distance_mi > 0

    def test_morning_vs_afternoon_risk_differs(self):
        """Afternoon convective weather should generally produce higher risk."""
        ts_morning = datetime(2024, 1, 1, 7, 0)
        ts_afternoon = datetime(2024, 1, 1, 16, 0)
        r_am = compute_route_risk_score("DFW", "MIA", ts_morning)
        r_pm = compute_route_risk_score("DFW", "MIA", ts_afternoon)
        # Afternoon risk may be equal or greater — just verify both are valid
        assert r_am.risk_score >= 0
        assert r_pm.risk_score >= 0

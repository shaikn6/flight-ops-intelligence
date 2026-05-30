"""
Tests for intelligence.flight_data module.
"""

import pytest
import pandas as pd
from datetime import datetime

from intelligence.flight_data import (
    generate_flights,
    AIRPORTS,
    AIRCRAFT_TYPES,
    DELAY_CAUSES,
    _haversine_mi,
    _expected_flight_time,
)


# ---------------------------------------------------------------------------
# generate_flights
# ---------------------------------------------------------------------------

class TestGenerateFlights:
    def test_returns_dataframe(self):
        df = generate_flights(n=20, seed=0)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self):
        df = generate_flights(n=100, seed=1)
        assert len(df) == 100

    def test_required_columns_present(self):
        df = generate_flights(n=20, seed=2)
        required = [
            "flight_id", "airline", "flight_number", "aircraft_type",
            "origin", "destination", "distance_mi",
            "scheduled_departure", "scheduled_arrival",
            "actual_departure", "actual_arrival",
            "delay_minutes", "delay_cause", "is_delayed",
            "dep_hour", "day_of_week", "month",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_all_airports_are_valid(self):
        df = generate_flights(n=50, seed=3)
        valid_codes = set(AIRPORTS.keys())
        assert df["origin"].isin(valid_codes).all()
        assert df["destination"].isin(valid_codes).all()

    def test_no_self_loops(self):
        df = generate_flights(n=50, seed=4)
        assert (df["origin"] != df["destination"]).all()

    def test_aircraft_types_are_valid(self):
        df = generate_flights(n=50, seed=5)
        assert df["aircraft_type"].isin(AIRCRAFT_TYPES).all()

    def test_delay_minutes_non_negative(self):
        df = generate_flights(n=100, seed=6)
        assert (df["delay_minutes"] >= 0).all()

    def test_delay_minutes_bounded(self):
        df = generate_flights(n=200, seed=7)
        assert (df["delay_minutes"] <= 180).all()

    def test_is_delayed_matches_threshold(self):
        df = generate_flights(n=100, seed=8)
        expected = (df["delay_minutes"] >= 15).astype(int)
        assert (df["is_delayed"] == expected).all()

    def test_dep_hour_range(self):
        df = generate_flights(n=100, seed=9)
        assert (df["dep_hour"] >= 0).all()
        assert (df["dep_hour"] <= 23).all()

    def test_day_of_week_range(self):
        df = generate_flights(n=50, seed=10)
        assert (df["day_of_week"] >= 0).all()
        assert (df["day_of_week"] <= 6).all()

    def test_distance_mi_positive(self):
        df = generate_flights(n=50, seed=11)
        assert (df["distance_mi"] > 0).all()

    def test_delay_causes_are_valid(self):
        df = generate_flights(n=100, seed=12)
        valid_causes = set(DELAY_CAUSES.keys()) | {"none"}
        assert df["delay_cause"].isin(valid_causes).all()

    def test_no_delay_cause_for_on_time_flights(self):
        df = generate_flights(n=100, seed=13)
        on_time = df[df["delay_minutes"] == 0]
        assert (on_time["delay_cause"] == "none").all()

    def test_reproducible_with_same_seed(self):
        df1 = generate_flights(n=50, seed=42)
        df2 = generate_flights(n=50, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        df1 = generate_flights(n=50, seed=1)
        df2 = generate_flights(n=50, seed=2)
        assert not df1["delay_minutes"].equals(df2["delay_minutes"])

    def test_scheduled_before_actual_departure_when_delayed(self):
        df = generate_flights(n=100, seed=14)
        delayed = df[df["delay_minutes"] > 0]
        for _, row in delayed.iterrows():
            sched = datetime.fromisoformat(row["scheduled_departure"])
            actual = datetime.fromisoformat(row["actual_departure"])
            assert actual >= sched, f"actual < scheduled for flight {row['flight_id']}"

    def test_full_500_flights(self):
        df = generate_flights(n=500, seed=42)
        assert len(df) == 500
        # Basic sanity: at least some delayed flights
        assert df["is_delayed"].sum() > 0


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestHaversine:
    def test_jfk_lax_approx(self):
        # JFK (40.64, -73.78) → LAX (33.94, -118.41) ≈ 2475 miles
        dist = _haversine_mi(40.6413, -73.7781, 33.9425, -118.4081)
        assert 2400 < dist < 2550

    def test_same_point_is_zero(self):
        dist = _haversine_mi(40.0, -74.0, 40.0, -74.0)
        assert dist < 0.01

    def test_symmetric(self):
        d1 = _haversine_mi(40.0, -74.0, 33.9, -118.4)
        d2 = _haversine_mi(33.9, -118.4, 40.0, -74.0)
        assert abs(d1 - d2) < 0.01


class TestFlightTime:
    def test_short_hop(self):
        # ~300 miles → should be roughly 75+ min
        t = _expected_flight_time(300)
        assert t > 60

    def test_cross_country(self):
        # ~2500 miles → should be 300+ min
        t = _expected_flight_time(2500)
        assert t > 280

    def test_longer_is_more(self):
        assert _expected_flight_time(2000) > _expected_flight_time(500)

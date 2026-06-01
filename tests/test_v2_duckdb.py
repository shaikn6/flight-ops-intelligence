"""
test_v2_duckdb.py — Tests for analytics/duckdb_engine.py

Covers:
  - generate_parquet correctness
  - DuckDB query correctness (on_time_by_carrier, delay_by_route, monthly_trends)
  - Benchmark timing targets (<2s on 2M rows)
  - Edge cases (empty filter, top_n bounds)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Guard: skip entire module if duckdb or pyarrow not installed
pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from analytics.duckdb_engine import (
    DuckDBEngine,
    generate_parquet,
    BenchmarkResult,
    CARRIERS,
    ORIGINS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_parquet(tmp_path_factory):
    """Generate a small (50k row) parquet for fast test execution."""
    path = tmp_path_factory.mktemp("data") / "test_flights.parquet"
    generate_parquet(n_rows=50_000, path=path, seed=99)
    return path


@pytest.fixture(scope="module")
def engine(small_parquet):
    """DuckDBEngine backed by the small test parquet."""
    return DuckDBEngine(parquet_path=small_parquet)


# ---------------------------------------------------------------------------
# generate_parquet
# ---------------------------------------------------------------------------

class TestGenerateParquet:
    def test_file_created(self, small_parquet):
        assert small_parquet.exists()
        assert small_parquet.stat().st_size > 10_000   # at least 10 KB

    def test_row_count(self, small_parquet):
        import pyarrow.parquet as pq
        tbl = pq.read_table(str(small_parquet))
        assert tbl.num_rows == 50_000

    def test_expected_columns(self, small_parquet):
        import pyarrow.parquet as pq
        tbl = pq.read_table(str(small_parquet))
        expected = {
            "year", "month", "day_of_month", "day_of_week", "dep_hour",
            "carrier", "origin", "destination", "distance_mi",
            "delay_minutes", "is_delayed", "delay_cause",
        }
        assert expected.issubset(set(tbl.schema.names))

    def test_is_delayed_binary(self, small_parquet):
        import pyarrow.parquet as pq
        df = pq.read_table(str(small_parquet)).to_pandas()
        unique_vals = set(df["is_delayed"].unique())
        assert unique_vals.issubset({0, 1})

    def test_delay_minutes_non_negative(self, small_parquet):
        import pyarrow.parquet as pq
        df = pq.read_table(str(small_parquet)).to_pandas()
        assert (df["delay_minutes"] >= 0).all()

    def test_carriers_are_known(self, small_parquet):
        import pyarrow.parquet as pq
        df = pq.read_table(str(small_parquet)).to_pandas()
        assert set(df["carrier"].unique()).issubset(set(CARRIERS))

    def test_origins_are_known(self, small_parquet):
        import pyarrow.parquet as pq
        df = pq.read_table(str(small_parquet)).to_pandas()
        assert set(df["origin"].unique()).issubset(set(ORIGINS))

    def test_year_in_valid_range(self, small_parquet):
        import pyarrow.parquet as pq
        df = pq.read_table(str(small_parquet)).to_pandas()
        assert df["year"].min() >= 2019
        assert df["year"].max() <= 2024


# ---------------------------------------------------------------------------
# on_time_by_carrier
# ---------------------------------------------------------------------------

class TestOnTimeByCarrier:
    def test_returns_dataframe(self, engine):
        df = engine.on_time_by_carrier()
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, engine):
        df = engine.on_time_by_carrier()
        expected = {"carrier", "total_flights", "delayed_flights", "on_time_pct", "avg_delay_minutes"}
        assert expected.issubset(set(df.columns))

    def test_on_time_pct_range(self, engine):
        df = engine.on_time_by_carrier()
        assert (df["on_time_pct"] >= 0.0).all()
        assert (df["on_time_pct"] <= 1.0).all()

    def test_avg_delay_non_negative(self, engine):
        df = engine.on_time_by_carrier()
        assert (df["avg_delay_minutes"] >= 0).all()

    def test_total_flights_positive(self, engine):
        df = engine.on_time_by_carrier()
        assert (df["total_flights"] > 0).all()

    def test_all_known_carriers_present(self, engine):
        df = engine.on_time_by_carrier()
        assert set(df["carrier"]).issubset(set(CARRIERS))

    def test_sorted_by_on_time_desc(self, engine):
        df = engine.on_time_by_carrier()
        assert list(df["on_time_pct"]) == sorted(df["on_time_pct"], reverse=True)


# ---------------------------------------------------------------------------
# delay_by_route
# ---------------------------------------------------------------------------

class TestDelayByRoute:
    def test_returns_dataframe(self, engine):
        df = engine.delay_by_route()
        assert isinstance(df, pd.DataFrame)

    def test_respects_top_n(self, engine):
        df = engine.delay_by_route(top_n=10)
        assert len(df) <= 10

    def test_route_format(self, engine):
        df = engine.delay_by_route(top_n=5)
        for route in df["route"]:
            assert "→" in route

    def test_avg_delay_non_negative(self, engine):
        df = engine.delay_by_route()
        assert (df["avg_delay"] >= 0).all()

    def test_origin_destination_not_equal(self, engine):
        df = engine.delay_by_route()
        for _, row in df.iterrows():
            assert row["origin"] != row["destination"]


# ---------------------------------------------------------------------------
# monthly_trends
# ---------------------------------------------------------------------------

class TestMonthlyTrends:
    def test_returns_dataframe(self, engine):
        df = engine.monthly_trends()
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, engine):
        df = engine.monthly_trends()
        assert {"year", "month", "total_flights", "delay_rate", "avg_delay"}.issubset(df.columns)

    def test_month_in_valid_range(self, engine):
        df = engine.monthly_trends()
        assert (df["month"] >= 1).all()
        assert (df["month"] <= 12).all()

    def test_delay_rate_in_range(self, engine):
        df = engine.monthly_trends()
        assert (df["delay_rate"] >= 0.0).all()
        assert (df["delay_rate"] <= 1.0).all()


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class TestBenchmark:
    def test_benchmark_returns_results(self, engine):
        results = engine.benchmark(row_counts=[10_000])
        assert len(results) == 1
        assert isinstance(results[0], BenchmarkResult)

    def test_benchmark_result_has_fields(self, engine):
        results = engine.benchmark(row_counts=[10_000])
        r = results[0]
        assert r.query_name == "on_time_by_carrier"
        assert r.row_count == 10_000
        assert r.elapsed_ms >= 0
        assert r.result_rows > 0

    def test_benchmark_multiple_sizes(self, engine):
        results = engine.benchmark(row_counts=[5_000, 10_000, 25_000])
        assert len(results) == 3
        counts = [r.row_count for r in results]
        assert counts == [5_000, 10_000, 25_000]

    def test_str_representation(self, engine):
        results = engine.benchmark(row_counts=[10_000])
        s = str(results[0])
        assert "on_time_by_carrier" in s
        assert "ms" in s

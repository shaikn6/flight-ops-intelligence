"""
duckdb_engine.py — DuckDB in-process analytics layer for flight data.

Loads BTS-style parquet files (generates 2M synthetic records if none exist).
Runs sub-2-second queries on 2M rows.

Public API
----------
get_engine()                  → DuckDBEngine singleton
engine.on_time_by_carrier()   → pd.DataFrame
engine.delay_by_route()       → pd.DataFrame
engine.monthly_trends()       → pd.DataFrame
engine.benchmark(rows)        → BenchmarkResult
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import duckdb
    _DUCKDB_AVAILABLE = True
except ImportError:
    _DUCKDB_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _ARROW_AVAILABLE = True
except ImportError:
    _ARROW_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
PARQUET_PATH = DATA_DIR / "flights_2m.parquet"

# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

CARRIERS = ["AA", "UA", "DL", "WN", "AS", "B6", "NK", "F9", "HA", "G4"]
ORIGINS = [
    "JFK", "LAX", "ORD", "DFW", "ATL", "SFO", "SEA", "MIA", "BOS", "DEN",
    "PHX", "MSP", "DTW", "CLT", "LGA", "EWR", "LAS", "PHL", "IAH", "BWI",
]
DELAY_CAUSES = ["weather", "carrier", "nas", "security", "late_aircraft", "none"]
# Probabilities for delayed-only causes (excluding "none"); must sum to 1.0
DELAY_CAUSE_PROBS = [0.30, 0.30, 0.20, 0.07, 0.13]   # len == 5 (no "none")


def generate_parquet(n_rows: int = 2_000_000, path: Path = PARQUET_PATH, seed: int = 42) -> Path:
    """
    Generate n_rows synthetic BTS-style flight records and save as parquet.
    Optimised for speed: uses vectorised NumPy operations.
    """
    if not _ARROW_AVAILABLE:
        raise ImportError("pyarrow is required to generate parquet files. pip install pyarrow")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    print(f"[duckdb_engine] Generating {n_rows:,} synthetic flight records…")
    t0 = time.time()

    # Core numeric columns
    year = rng.integers(2019, 2025, size=n_rows)
    month = rng.integers(1, 13, size=n_rows)
    day_of_month = rng.integers(1, 29, size=n_rows)
    day_of_week = rng.integers(0, 7, size=n_rows)
    dep_hour = rng.integers(0, 24, size=n_rows)

    carrier_idx = rng.integers(0, len(CARRIERS), size=n_rows)
    origin_idx = rng.integers(0, len(ORIGINS), size=n_rows)
    # Ensure dest != origin
    dest_idx = (origin_idx + rng.integers(1, len(ORIGINS), size=n_rows)) % len(ORIGINS)

    # Distance: roughly correlated with airport pairs
    distance_mi = np.abs(rng.normal(loc=1200, scale=700, size=n_rows)).clip(100, 5500).round(0)

    # Delay logic: weather/peak hours increase probability
    base_delay_prob = 0.35
    peak_boost = np.where((dep_hour >= 7) & (dep_hour <= 9), 0.15, 0.0)
    peak_boost += np.where((dep_hour >= 16) & (dep_hour <= 19), 0.20, 0.0)
    delay_probs = np.clip(base_delay_prob + peak_boost * rng.uniform(0, 1, size=n_rows), 0, 0.9)
    is_delayed_mask = rng.random(size=n_rows) < delay_probs

    delay_minutes = np.where(
        is_delayed_mask,
        np.clip(rng.lognormal(mean=3.0, sigma=0.8, size=n_rows), 1, 300),
        0.0,
    ).round(1)

    is_delayed = (delay_minutes >= 15).astype(np.int8)

    delay_cause_arr = np.where(
        delay_minutes > 0,
        rng.choice(DELAY_CAUSES[:-1], size=n_rows, p=DELAY_CAUSE_PROBS),
        "none",
    )

    # Build arrow table for efficient parquet write
    table = pa.table({
        "year": pa.array(year, type=pa.int16()),
        "month": pa.array(month, type=pa.int8()),
        "day_of_month": pa.array(day_of_month, type=pa.int8()),
        "day_of_week": pa.array(day_of_week, type=pa.int8()),
        "dep_hour": pa.array(dep_hour, type=pa.int8()),
        "carrier": pa.array([CARRIERS[i] for i in carrier_idx]),
        "origin": pa.array([ORIGINS[i] for i in origin_idx]),
        "destination": pa.array([ORIGINS[i] for i in dest_idx]),
        "distance_mi": pa.array(distance_mi, type=pa.float32()),
        "delay_minutes": pa.array(delay_minutes, type=pa.float32()),
        "is_delayed": pa.array(is_delayed, type=pa.int8()),
        "delay_cause": pa.array(delay_cause_arr),
    })

    pq.write_table(
        table,
        str(path),
        compression="snappy",
        row_group_size=200_000,
    )

    elapsed = time.time() - t0
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"[duckdb_engine] Wrote {n_rows:,} rows → {path} ({size_mb:.1f} MB) in {elapsed:.1f}s")
    return path


# ---------------------------------------------------------------------------
# Benchmark result
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    query_name: str
    row_count: int
    elapsed_ms: float
    result_rows: int

    def __str__(self) -> str:
        return (
            f"{self.query_name}: {self.row_count:,} rows → "
            f"{self.elapsed_ms:.0f} ms ({self.result_rows} result rows)"
        )


# ---------------------------------------------------------------------------
# DuckDB engine
# ---------------------------------------------------------------------------

class DuckDBEngine:
    """
    In-process DuckDB analytics engine over parquet flight data.

    Usage
    -----
    engine = DuckDBEngine()
    df = engine.on_time_by_carrier()
    """

    def __init__(self, parquet_path: Path = PARQUET_PATH) -> None:
        if not _DUCKDB_AVAILABLE:
            raise ImportError("duckdb is not installed. Run: pip install duckdb")
        self._parquet_path = parquet_path
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._ensure_data()
        self._connect()

    def _ensure_data(self) -> None:
        """Generate parquet data if it doesn't exist."""
        if not self._parquet_path.exists():
            generate_parquet(n_rows=2_000_000, path=self._parquet_path)

    def _connect(self) -> None:
        self._conn = duckdb.connect(database=":memory:")
        # Register parquet as a view for zero-copy scans
        self._conn.execute(
            f"CREATE VIEW flights AS SELECT * FROM read_parquet('{self._parquet_path}')"
        )

    def _query(self, sql: str) -> pd.DataFrame:
        return self._conn.execute(sql).df()

    # ------------------------------------------------------------------
    # Analytics queries
    # ------------------------------------------------------------------

    def on_time_by_carrier(self) -> pd.DataFrame:
        """
        On-time performance by carrier.

        Returns columns: carrier, total_flights, delayed_flights, on_time_pct,
                          avg_delay_minutes, median_delay
        """
        return self._query(
            """
            SELECT
                carrier,
                COUNT(*)                                    AS total_flights,
                SUM(is_delayed)                             AS delayed_flights,
                ROUND(1.0 - AVG(is_delayed::FLOAT), 4)     AS on_time_pct,
                ROUND(AVG(delay_minutes), 2)                AS avg_delay_minutes,
                ROUND(MEDIAN(delay_minutes), 2)             AS median_delay
            FROM flights
            GROUP BY carrier
            ORDER BY on_time_pct DESC
            """
        )

    def delay_by_route(self, top_n: int = 30) -> pd.DataFrame:
        """
        Average delay by origin→destination route.

        Returns columns: route, origin, destination, avg_delay, total_flights
        """
        return self._query(
            f"""
            SELECT
                origin || '→' || destination                AS route,
                origin,
                destination,
                ROUND(AVG(delay_minutes), 2)                AS avg_delay,
                COUNT(*)                                    AS total_flights
            FROM flights
            GROUP BY origin, destination
            ORDER BY avg_delay DESC
            LIMIT {top_n}
            """
        )

    def monthly_trends(self) -> pd.DataFrame:
        """
        Monthly delay trend across all years.

        Returns columns: year, month, total_flights, delay_rate, avg_delay
        """
        return self._query(
            """
            SELECT
                year,
                month,
                COUNT(*)                                    AS total_flights,
                ROUND(AVG(is_delayed::FLOAT), 4)            AS delay_rate,
                ROUND(AVG(delay_minutes), 2)                AS avg_delay
            FROM flights
            GROUP BY year, month
            ORDER BY year, month
            """
        )

    def delay_cause_distribution(self) -> pd.DataFrame:
        """Breakdown of delay causes."""
        return self._query(
            """
            SELECT
                delay_cause,
                COUNT(*)            AS count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct
            FROM flights
            WHERE delay_minutes > 0
            GROUP BY delay_cause
            ORDER BY count DESC
            """
        )

    def hourly_delay_pattern(self) -> pd.DataFrame:
        """Average delay by departure hour."""
        return self._query(
            """
            SELECT
                dep_hour,
                ROUND(AVG(delay_minutes), 2)    AS avg_delay,
                ROUND(AVG(is_delayed::FLOAT), 4) AS delay_rate,
                COUNT(*)                         AS flight_count
            FROM flights
            GROUP BY dep_hour
            ORDER BY dep_hour
            """
        )

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def benchmark(self, row_counts: Optional[List[int]] = None) -> List[BenchmarkResult]:
        """
        Run timing benchmarks on the on_time_by_carrier query at multiple scales.

        If row_counts is None, uses [1_000_000, 2_000_000].
        Demonstrates sub-2s query times on 2M rows.
        """
        if row_counts is None:
            row_counts = [1_000_000, 2_000_000]

        results = []
        for n in row_counts:
            # Use LIMIT to simulate different dataset sizes
            sql = f"""
            SELECT
                carrier,
                COUNT(*) AS total_flights,
                ROUND(1.0 - AVG(is_delayed::FLOAT), 4) AS on_time_pct,
                ROUND(AVG(delay_minutes), 2) AS avg_delay
            FROM (SELECT * FROM flights LIMIT {n})
            GROUP BY carrier
            ORDER BY on_time_pct DESC
            """
            t0 = time.perf_counter()
            df = self._query(sql)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            results.append(BenchmarkResult(
                query_name="on_time_by_carrier",
                row_count=n,
                elapsed_ms=round(elapsed_ms, 1),
                result_rows=len(df),
            ))

        return results

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_engine_instance: Optional[DuckDBEngine] = None


def get_engine() -> DuckDBEngine:
    """Return (or create) the module-level DuckDB engine singleton."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = DuckDBEngine()
    return _engine_instance


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[duckdb_engine] Initialising engine…")
    engine = get_engine()

    print("\n--- On-time by carrier ---")
    print(engine.on_time_by_carrier().to_string(index=False))

    print("\n--- Top 10 delay routes ---")
    print(engine.delay_by_route(top_n=10).to_string(index=False))

    print("\n--- Benchmark ---")
    for result in engine.benchmark():
        print(f"  {result}")

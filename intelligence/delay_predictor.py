"""
delay_predictor.py — Random Forest delay predictor.
Trains on synthetic flight data. Predicts delay minutes and delay probability.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from intelligence.flight_data import AIRPORTS, AIRCRAFT_TYPES, generate_flights, load_flights
from intelligence.weather_engine import WeatherEngine, get_engine

# Model path is intentionally hardcoded — it is never derived from user input.
# The resolved-path guard in load_models() below prevents any future
# path-traversal if this value were ever made configurable.
MODEL_PATH = (Path(__file__).parent.parent / "models" / "delay_rf.pkl").resolve()

FEATURE_COLS = [
    "dep_hour",
    "day_of_week",
    "origin_weather_score",
    "dest_weather_score",
    "distance_mi",
    "aircraft_type_encoded",
    "route_congestion_score",
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DelayPrediction:
    predicted_delay_minutes: float
    delay_probability: float          # P(delay >= 15 min)
    feature_importances: Dict[str, float]
    confidence_interval: Tuple[float, float]  # rough 80% CI


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

class FeatureBuilder:
    """Build ML feature matrix from raw flight DataFrame + weather engine."""

    def __init__(self, weather_engine: Optional[WeatherEngine] = None) -> None:
        self.weather_engine = weather_engine or get_engine()
        self.aircraft_encoder = LabelEncoder()
        self.aircraft_encoder.fit(AIRCRAFT_TYPES)
        self._fitted = False

    def _get_impact(self, airport: str, hour: int) -> float:
        from datetime import datetime
        ts = datetime(2024, 1, 15, hour, 0)
        return self.weather_engine.get_weather_impact_score(airport, ts)

    def _congestion_score(self, origin: str, dest: str, dep_hour: int) -> float:
        """Heuristic: hub routes during peak hours are congested."""
        hubs = {"ORD", "ATL", "DFW", "JFK", "LAX"}
        peak_hours = set(range(7, 10)) | set(range(16, 20))
        score = 0.0
        if origin in hubs:
            score += 0.3
        if dest in hubs:
            score += 0.2
        if dep_hour in peak_hours:
            score += 0.3
        # Small deterministic jitter — clipped to [0, 1]
        jitter = np.random.RandomState(hash((origin, dest, dep_hour)) % (2**31)).uniform(0.0, 0.05)
        return float(np.clip(score + jitter, 0.0, 1.0))

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add feature columns to a flights DataFrame.
        Returns new DataFrame with FEATURE_COLS present.
        """
        out = df.copy()

        # Aircraft type encoding
        try:
            out["aircraft_type_encoded"] = self.aircraft_encoder.transform(
                out["aircraft_type"]
            )
        except ValueError:
            out["aircraft_type_encoded"] = 0

        # Weather scores (vectorised via apply — fast enough for 500 rows)
        out["origin_weather_score"] = out.apply(
            lambda r: self._get_impact(r["origin"], int(r["dep_hour"])), axis=1
        )
        out["dest_weather_score"] = out.apply(
            lambda r: self._get_impact(r["destination"], int(r["dep_hour"])), axis=1
        )

        # Route congestion score
        out["route_congestion_score"] = out.apply(
            lambda r: self._congestion_score(r["origin"], r["destination"], int(r["dep_hour"])),
            axis=1,
        )

        return out


# ---------------------------------------------------------------------------
# Model training + persistence
# ---------------------------------------------------------------------------

@dataclass
class TrainedModels:
    regressor: RandomForestRegressor
    classifier: RandomForestClassifier
    feature_cols: List[str]
    mae: float
    accuracy: float


def train(df: Optional[pd.DataFrame] = None, save: bool = True) -> TrainedModels:
    """
    Train Random Forest regressor + classifier on the flight dataset.
    Saves models to MODEL_PATH.
    """
    if df is None:
        df = load_flights()

    builder = FeatureBuilder()
    df_feat = builder.build(df)

    X = df_feat[FEATURE_COLS].values.astype(float)
    y_reg = df_feat["delay_minutes"].values.astype(float)
    y_clf = df_feat["is_delayed"].values.astype(int)

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )

    # Regressor
    regressor = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    regressor.fit(X_train, y_reg_train)
    y_pred_reg = regressor.predict(X_test)
    mae = mean_absolute_error(y_reg_test, y_pred_reg)

    # Classifier (binary: delayed >= 15 min)
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    classifier.fit(X_train, y_clf_train)
    y_pred_clf = classifier.predict(X_test)
    accuracy = (y_pred_clf == y_clf_test).mean()

    print(f"[delay_predictor] Regressor MAE: {mae:.2f} min")
    print(f"[delay_predictor] Classifier accuracy: {accuracy:.3f}")
    print("[delay_predictor] Classification report:")
    print(classification_report(y_clf_test, y_pred_clf, target_names=["on-time", "delayed"]))

    models = TrainedModels(
        regressor=regressor,
        classifier=classifier,
        feature_cols=FEATURE_COLS,
        mae=mae,
        accuracy=accuracy,
    )

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(models, f)
        print(f"[delay_predictor] Models saved to {MODEL_PATH}")

    return models


def load_models() -> TrainedModels:
    """Load trained models from disk, training fresh if absent.

    The resolved path is verified to be inside the expected models/ directory
    so that this function is safe even if MODEL_PATH were ever derived from
    external configuration.
    """
    expected_dir = (Path(__file__).parent.parent / "models").resolve()
    resolved = MODEL_PATH.resolve()
    if not str(resolved).startswith(str(expected_dir)):
        raise ValueError(
            f"Model path '{resolved}' is outside the expected models/ directory. "
            "Refusing to load."
        )
    if not MODEL_PATH.exists():
        print("[delay_predictor] No saved model found — training now…")
        return train()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Inference API
# ---------------------------------------------------------------------------

_models: Optional[TrainedModels] = None


def _get_models() -> TrainedModels:
    global _models
    if _models is None:
        _models = load_models()
    return _models


def predict_delay(
    dep_hour: int,
    day_of_week: int,
    origin_weather_score: float,
    dest_weather_score: float,
    distance_mi: float,
    aircraft_type: str,
    route_congestion_score: float,
) -> DelayPrediction:
    """
    Predict delay for a single flight.

    Returns DelayPrediction with:
        - predicted_delay_minutes
        - delay_probability (P >= 15 min delay)
        - feature_importances dict
        - rough 80% confidence interval
    """
    models = _get_models()

    # Encode aircraft type
    encoder = LabelEncoder()
    encoder.fit(AIRCRAFT_TYPES)
    try:
        aircraft_enc = int(encoder.transform([aircraft_type])[0])
    except ValueError:
        aircraft_enc = 0

    x = np.array([[
        dep_hour,
        day_of_week,
        origin_weather_score,
        dest_weather_score,
        distance_mi,
        aircraft_enc,
        route_congestion_score,
    ]], dtype=float)

    predicted_delay = float(max(0.0, models.regressor.predict(x)[0]))
    delay_prob = float(models.classifier.predict_proba(x)[0][1])

    # Per-tree predictions for rough confidence interval
    tree_preds = np.array([tree.predict(x)[0] for tree in models.regressor.estimators_])
    ci_low = float(max(0.0, np.percentile(tree_preds, 10)))
    ci_high = float(np.percentile(tree_preds, 90))

    importances = dict(zip(FEATURE_COLS, models.regressor.feature_importances_.tolist()))

    return DelayPrediction(
        predicted_delay_minutes=round(predicted_delay, 1),
        delay_probability=round(delay_prob, 3),
        feature_importances=importances,
        confidence_interval=(round(ci_low, 1), round(ci_high, 1)),
    )


def get_feature_importances() -> Dict[str, float]:
    """Return feature importances from the trained regressor."""
    models = _get_models()
    return dict(zip(FEATURE_COLS, models.regressor.feature_importances_.tolist()))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    models = train()
    print(f"\nFeature importances:")
    fi = dict(zip(FEATURE_COLS, models.regressor.feature_importances_))
    for feat, imp in sorted(fi.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"  {feat:<30} {imp:.4f}  {bar}")

    pred = predict_delay(
        dep_hour=8,
        day_of_week=0,
        origin_weather_score=0.7,
        dest_weather_score=0.3,
        distance_mi=2400.0,
        aircraft_type="Boeing 737",
        route_congestion_score=0.6,
    )
    print(f"\nSample prediction: {pred}")

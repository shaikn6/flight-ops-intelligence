"""
models.py — SQLAlchemy ORM models for Flight Ops Intelligence.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, Text, ForeignKey
)
from sqlalchemy.orm import relationship

from api.database import Base


class Flight(Base):
    __tablename__ = "flights"

    id = Column(Integer, primary_key=True, index=True)
    flight_id = Column(String(16), unique=True, index=True, nullable=False)
    airline = Column(String(4), nullable=False)
    flight_number = Column(String(10), nullable=False)
    aircraft_type = Column(String(32), nullable=False)
    origin = Column(String(4), nullable=False, index=True)
    destination = Column(String(4), nullable=False, index=True)
    distance_mi = Column(Float, nullable=False)
    scheduled_departure = Column(DateTime, nullable=False)
    scheduled_arrival = Column(DateTime, nullable=False)
    actual_departure = Column(DateTime, nullable=True)
    actual_arrival = Column(DateTime, nullable=True)
    delay_minutes = Column(Float, default=0.0)
    delay_cause = Column(String(32), default="none")
    is_delayed = Column(Boolean, default=False)
    dep_hour = Column(Integer)
    day_of_week = Column(Integer)
    month = Column(Integer)

    predictions = relationship("DelayPrediction", back_populates="flight", cascade="all, delete-orphan")


class WeatherReport(Base):
    __tablename__ = "weather_reports"

    id = Column(Integer, primary_key=True, index=True)
    airport = Column(String(4), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    wind_speed_kts = Column(Float)
    wind_dir_deg = Column(Float)
    visibility_sm = Column(Float)
    ceiling_ft = Column(Integer)
    temperature_c = Column(Float)
    dewpoint_c = Column(Float)
    altimeter_inhg = Column(Float)
    conditions = Column(String(8))
    weather_impact_score = Column(Float)


class DelayPrediction(Base):
    __tablename__ = "delay_predictions"

    id = Column(Integer, primary_key=True, index=True)
    flight_id = Column(String(16), ForeignKey("flights.flight_id"), index=True)
    predicted_delay_minutes = Column(Float)
    delay_probability = Column(Float)
    ci_low = Column(Float)
    ci_high = Column(Float)
    feature_importances_json = Column(Text)     # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)

    flight = relationship("Flight", back_populates="predictions")

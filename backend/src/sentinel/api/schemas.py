"""Pydantic response schemas for the API."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel


class HealthOut(BaseModel):
    status: str
    version: str


class RegionOut(BaseModel):
    id: int
    name: str


class RiskScoreOut(BaseModel):
    region_id: int
    region_name: str
    date: date
    ensemble_score: float
    cnn_score: float | None
    lstm_score: float | None
    model_version: str | None
    computed_at: datetime | None

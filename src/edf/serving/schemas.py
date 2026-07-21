from __future__ import annotations

from pydantic import BaseModel


class ForecastPoint(BaseModel):
    timestamp: str
    p10: float
    p50: float
    p90: float


class ForecastResponse(BaseModel):
    model: str
    generated_at: str
    horizon_steps: int
    points: list[ForecastPoint]


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    trained_at: str | None = None
    data_source: str | None = None


class ModelSummary(BaseModel):
    model: str
    mae: float
    rmse: float
    mape: float
    pinball_10: float
    pinball_90: float
    coverage_80: float

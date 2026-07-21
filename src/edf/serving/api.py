"""FastAPI service exposing the trained forecasters.

Run with: uvicorn edf.serving.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from edf.config import MODELS_DIR, REPORTS_DIR
from edf.serving.schemas import ForecastPoint, ForecastResponse, HealthResponse, ModelSummary

app = FastAPI(
    title="UK Energy Demand Forecasting API",
    description="Serves seasonal-naive, LightGBM, and LSTM half-hourly GB demand forecasts with prediction intervals.",
    version="1.0.0",
)

KNOWN_MODELS = ["seasonal_naive", "lightgbm", "lstm"]


@lru_cache(maxsize=1)
def _load_context() -> pd.DataFrame:
    path = MODELS_DIR / "latest_context.parquet"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Model artifacts not found; run the training pipeline first.")
    return pd.read_parquet(path)


@lru_cache(maxsize=8)
def _load_model(name: str):
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Model {name!r} artifact not found")
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_metadata() -> dict:
    path = MODELS_DIR / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    metadata = _load_metadata()
    loaded = [name for name in KNOWN_MODELS if (MODELS_DIR / f"{name}.pkl").exists()]
    return HealthResponse(
        status="ok" if loaded else "no_models",
        models_loaded=loaded,
        trained_at=metadata.get("trained_at"),
        data_source=metadata.get("data_source"),
    )


@app.get("/forecast", response_model=ForecastResponse)
def forecast(model: str = "lightgbm", horizon: int = 48) -> ForecastResponse:
    if model not in KNOWN_MODELS:
        raise HTTPException(status_code=400, detail=f"model must be one of {KNOWN_MODELS}")

    forecaster = _load_model(model)
    context = _load_context()
    preds = forecaster.predict(context, horizon)

    points = [
        ForecastPoint(timestamp=ts.isoformat(), p10=float(row.p10), p50=float(row.p50), p90=float(row.p90))
        for ts, row in preds.iterrows()
    ]
    metadata = _load_metadata()
    return ForecastResponse(
        model=model,
        generated_at=metadata.get("trained_at", ""),
        horizon_steps=len(points),
        points=points,
    )


@app.get("/backtest/metrics", response_model=list[ModelSummary])
def backtest_metrics() -> list[ModelSummary]:
    path = REPORTS_DIR / "summary_metrics.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail="No backtest report found; run the training pipeline first.")
    return json.loads(path.read_text())


@app.get("/backtest/latest")
def backtest_latest() -> dict:
    path = REPORTS_DIR / "latest_forecast.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail="No backtest report found; run the training pipeline first.")
    return json.loads(path.read_text())

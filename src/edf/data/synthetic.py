"""Synthetic fallback dataset generator.

Live network access to NESO / Open-Meteo is not always available (e.g. in a
sandboxed dev environment with restricted egress). This module generates a
schema-compatible synthetic demand + weather series so the rest of the
pipeline (features, models, backtesting, API, dashboard) can be developed and
unit-tested offline. It is NOT a substitute for real data in production: the
scheduled GitHub Actions retrain job always fetches real NESO + Open-Meteo
data since GitHub-hosted runners have unrestricted internet access.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from edf.config import settings


def _uk_temperature(hours: np.ndarray, day_of_year: np.ndarray) -> np.ndarray:
    annual = 9.0 - 7.0 * np.cos(2 * np.pi * (day_of_year - 30) / 365.25)
    diurnal = 3.0 * np.sin(2 * np.pi * (hours - 6) / 24)
    noise = np.random.default_rng(7).normal(0, 1.5, size=hours.shape)
    return annual + diurnal + noise


def generate_synthetic_dataset(start: str | None = None, end: str | None = None, seed: int = 42) -> pd.DataFrame:
    """Return a half-hourly DataFrame with columns [demand_mw, temp_c, humidity_pct,
    wind_speed_ms, cloud_cover_pct, precip_mm] resembling GB national demand."""
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start or settings.history_start)
    end_ts = pd.Timestamp(end or settings.history_end)
    idx = pd.date_range(start_ts, end_ts, freq="30min", inclusive="left")

    hours = idx.hour + idx.minute / 60.0
    dow = idx.dayofweek.values
    day_of_year = idx.dayofyear.values

    temp_c = _uk_temperature(hours.values, day_of_year)

    base_load = 32000.0
    weekday_factor = np.where(dow < 5, 1.0, 0.86)
    daily_shape = (
        0.55
        + 0.30 * np.exp(-((hours.values - 8.5) ** 2) / (2 * 2.0 ** 2))
        + 0.42 * np.exp(-((hours.values - 18.0) ** 2) / (2 * 2.2 ** 2))
        - 0.18 * np.exp(-((hours.values - 3.5) ** 2) / (2 * 2.5 ** 2))
    )
    seasonal_factor = 1.0 + 0.18 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25)
    heating_response = np.clip(15.0 - temp_c, 0, None) * 220.0
    cooling_response = np.clip(temp_c - 22.0, 0, None) * 140.0

    noise = rng.normal(0, 450, size=len(idx))
    demand_mw = (
        base_load * daily_shape * weekday_factor * seasonal_factor
        + heating_response
        + cooling_response
        + noise
    )
    demand_mw = np.clip(demand_mw, 15000, None)

    humidity_pct = np.clip(75 - 1.2 * temp_c + rng.normal(0, 6, size=len(idx)), 30, 100)
    wind_speed_ms = np.clip(rng.gamma(shape=2.2, scale=2.3, size=len(idx)), 0, None)
    cloud_cover_pct = np.clip(rng.beta(2, 1.5, size=len(idx)) * 100, 0, 100)
    precip_mm = rng.exponential(0.15, size=len(idx)) * (rng.random(len(idx)) < 0.25)

    df = pd.DataFrame(
        {
            "demand_mw": demand_mw,
            "temp_c": temp_c,
            "humidity_pct": humidity_pct,
            "wind_speed_ms": wind_speed_ms,
            "cloud_cover_pct": cloud_cover_pct,
            "precip_mm": precip_mm,
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df

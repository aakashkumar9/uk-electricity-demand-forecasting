"""Central configuration for the UK energy demand forecasting pipeline."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EDF_", env_file=".env", extra="ignore")

    # NESO (National Energy System Operator) open data portal, CKAN API.
    neso_api_base: str = "https://api.neso.energy/api/3/action"
    neso_package_query: str = "historic demand data"

    # Open-Meteo historical weather archive, no API key required.
    weather_api_base: str = "https://archive-api.open-meteo.com/v1/archive"
    # London (Bishop's Stortford / national grid demand centre proxy) lat/lon.
    weather_latitude: float = 51.5074
    weather_longitude: float = -0.1278
    weather_hourly_vars: str = (
        "temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover,precipitation"
    )

    # Data window for training/backtesting.
    history_start: str = "2022-01-01"
    history_end: str = "2024-12-31"

    # Resample resolution: "30min" (native GB settlement period) or "1h".
    resample_freq: str = "30min"

    # Forecast horizon in number of resample_freq steps (24h at half-hourly = 48).
    horizon_steps: int = 48
    # Lookback window fed to the LSTM, in steps.
    lookback_steps: int = 336  # 7 days at half-hourly resolution

    # Rolling-origin backtesting.
    n_backtest_folds: int = 6
    backtest_step_days: int = 14

    request_timeout_s: int = 30


settings = Settings()

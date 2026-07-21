"""Combine calendar, lag, and weather features into a single model-ready frame."""
from __future__ import annotations

import pandas as pd

from edf.features.calendar import add_calendar_features
from edf.features.lags import add_lag_features

WEATHER_COLS = ["temp_c", "humidity_pct", "wind_speed_ms", "cloud_cover_pct", "precip_mm"]

TARGET_COL = "demand_mw"


def build_feature_frame(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Return a feature matrix (including the target column, unshifted) with all
    engineered columns. Rows with insufficient history for the longest lag are
    dropped (NaNs from warmup)."""
    out = add_calendar_features(df)
    out = add_lag_features(out, target_col=target_col)
    out = out.dropna()
    return out


def feature_columns(df: pd.DataFrame, target_col: str = TARGET_COL) -> list[str]:
    exclude = {target_col, "source"}
    return [c for c in df.columns if c not in exclude]

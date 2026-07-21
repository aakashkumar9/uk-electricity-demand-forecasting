"""Lag and rolling-window features for the demand target."""
from __future__ import annotations

import pandas as pd

# In half-hourly (30min) steps.
DEFAULT_LAG_STEPS = (1, 2, 4, 48, 96, 336)  # 30m, 1h, 2h, 1d, 2d, 1 week
DEFAULT_ROLLING_WINDOWS = (48, 336)  # 1 day, 1 week


def add_lag_features(df: pd.DataFrame, target_col: str = "demand_mw") -> pd.DataFrame:
    out = df.copy()
    for lag in DEFAULT_LAG_STEPS:
        out[f"lag_{lag}"] = out[target_col].shift(lag)
    for window in DEFAULT_ROLLING_WINDOWS:
        shifted = out[target_col].shift(1)
        out[f"roll_mean_{window}"] = shifted.rolling(window).mean()
        out[f"roll_std_{window}"] = shifted.rolling(window).std()
    return out

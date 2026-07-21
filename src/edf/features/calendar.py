"""Calendar / seasonal features: time-of-day, day-of-week, and UK bank holidays."""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import holidays as pyholidays
except ImportError:  # pragma: no cover
    pyholidays = None

# Columns added by add_calendar_features(), i.e. those knowable in advance for
# any future timestamp. Models that need to decode a specific future step
# (e.g. the LSTM decoder) condition on exactly this set.
CALENDAR_COLS = [
    "settlement_period",
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "tod_sin",
    "tod_cos",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "is_bank_holiday",
    "is_working_day",
]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index

    minutes_since_midnight = idx.hour * 60 + idx.minute
    steps_per_day = 24 * 60 / 30  # native half-hourly settlement periods

    out["settlement_period"] = (minutes_since_midnight // 30 + 1).astype(int)
    out["hour"] = idx.hour
    out["day_of_week"] = idx.dayofweek
    out["day_of_month"] = idx.day
    out["month"] = idx.month
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    out["tod_sin"] = np.sin(2 * np.pi * minutes_since_midnight / (steps_per_day * 30))
    out["tod_cos"] = np.cos(2 * np.pi * minutes_since_midnight / (steps_per_day * 30))
    out["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    out["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)

    if pyholidays is not None:
        years = sorted(set(idx.year))
        uk_holidays = pyholidays.country_holidays("GB", subdiv="ENG", years=years)
        out["is_bank_holiday"] = idx.normalize().isin(pd.to_datetime(list(uk_holidays.keys()))).astype(int)
    else:  # pragma: no cover
        out["is_bank_holiday"] = 0

    out["is_working_day"] = ((out["is_weekend"] == 0) & (out["is_bank_holiday"] == 0)).astype(int)
    return out

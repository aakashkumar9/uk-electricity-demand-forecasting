"""Seasonal-naive baseline: forecast(t) = actual(t - 1 week), i.e. the same
settlement period on the same weekday last week. This is the honest floor
that the LightGBM and LSTM models must beat."""
from __future__ import annotations

import numpy as np
import pandas as pd

from edf.models.base import Forecaster, make_quantile_frame

WEEK_STEPS = 336  # 7 days * 48 half-hourly settlement periods


class SeasonalNaiveForecaster(Forecaster):
    name = "seasonal_naive"

    def __init__(self, season_steps: int = WEEK_STEPS):
        self.season_steps = season_steps
        self._history: pd.Series | None = None
        self._residual_std: float = 0.0

    def fit(self, train_df: pd.DataFrame, target_col: str) -> "SeasonalNaiveForecaster":
        self._history = train_df[target_col].copy()
        shifted = self._history.shift(self.season_steps)
        resid = (self._history - shifted).dropna()
        self._residual_std = float(resid.std()) if len(resid) else 0.0
        return self

    def predict(self, context_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        target_col = "demand_mw"
        history = pd.concat([self._history, context_df[target_col]]) if self._history is not None else context_df[target_col]
        history = history[~history.index.duplicated(keep="last")].sort_index()

        future_index = self._future_index(context_df, horizon)
        preds = []
        for ts in future_index:
            source_ts = ts - pd.Timedelta(minutes=30 * self.season_steps)
            if source_ts in history.index:
                preds.append(history.loc[source_ts])
            else:
                preds.append(history.iloc[-1])
        p50 = np.array(preds, dtype=float)
        spread = np.full(horizon, self._residual_std)
        return make_quantile_frame(future_index, p50, spread)

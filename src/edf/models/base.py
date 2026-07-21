"""Common interface all forecasters implement, so the backtester can treat
the seasonal-naive baseline, LightGBM, and LSTM models uniformly."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Forecaster(ABC):
    """A model that, given a feature frame up to time t, predicts the next
    `horizon` steps of the target column, with P10/P50/P90 quantiles."""

    name: str = "forecaster"

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, target_col: str) -> "Forecaster":
        ...

    @abstractmethod
    def predict(self, context_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Return a DataFrame of length `horizon` indexed by future timestamps
        with columns ['p10', 'p50', 'p90']."""
        ...

    def _future_index(self, context_df: pd.DataFrame, horizon: int) -> pd.DatetimeIndex:
        freq = context_df.index.freq or pd.infer_freq(context_df.index[-10:])
        last_ts = context_df.index[-1]
        return pd.date_range(last_ts, periods=horizon + 1, freq=freq)[1:]


def make_quantile_frame(index: pd.DatetimeIndex, p50: np.ndarray, spread: np.ndarray) -> pd.DataFrame:
    """Helper: build a symmetric-ish quantile frame from a point forecast and a
    per-step spread (e.g. residual std), using a Gaussian approximation for the
    10th/90th percentiles (z=1.2816)."""
    z = 1.2816
    return pd.DataFrame(
        {
            "p10": p50 - z * spread,
            "p50": p50,
            "p90": p50 + z * spread,
        },
        index=index,
    )

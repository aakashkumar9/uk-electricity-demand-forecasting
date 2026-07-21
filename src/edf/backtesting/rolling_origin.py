"""Rolling-origin (walk-forward) cross-validation for time series.

Unlike a single train/test split, this repeatedly moves the "origin" forward
in time, retrains (or reuses a model fit on the data available up to that
origin), forecasts the next `horizon` steps, and scores against the actuals
that follow. This is the honest way to backtest a forecaster: at no point
does the model see data from after the origin it is forecasting from.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Fold:
    fold_id: int
    origin: pd.Timestamp
    train_df: pd.DataFrame
    actual: pd.Series


def rolling_origin_splits(
    df: pd.DataFrame,
    horizon: int,
    n_folds: int,
    step: int,
    min_train_steps: int,
    target_col: str = "demand_mw",
) -> list[Fold]:
    """Build `n_folds` folds, each `step` rows apart, ending at the latest
    point that still leaves a full `horizon`-length actuals window."""
    n = len(df)
    last_origin_idx = n - horizon
    if last_origin_idx <= min_train_steps:
        raise ValueError("Not enough data for even one backtest fold with this configuration")

    first_origin_idx = max(min_train_steps, last_origin_idx - (n_folds - 1) * step)
    origin_indices = sorted(set(range(first_origin_idx, last_origin_idx + 1, step)))
    origin_indices = origin_indices[-n_folds:]

    folds = []
    for i, origin_idx in enumerate(origin_indices):
        train_df = df.iloc[:origin_idx]
        actual = df[target_col].iloc[origin_idx : origin_idx + horizon]
        folds.append(Fold(fold_id=i, origin=df.index[origin_idx], train_df=train_df, actual=actual))
    return folds

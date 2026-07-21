"""LightGBM forecaster: three quantile regressors (P10/P50/P90) trained on
one-step-ahead lag+calendar+weather features, applied recursively to produce
a multi-step horizon forecast."""
from __future__ import annotations

import pandas as pd
from lightgbm import LGBMRegressor

from edf.features.calendar import add_calendar_features
from edf.features.lags import DEFAULT_LAG_STEPS, DEFAULT_ROLLING_WINDOWS
from edf.models.base import Forecaster

WEATHER_COLS = ["temp_c", "humidity_pct", "wind_speed_ms", "cloud_cover_pct", "precip_mm"]


class LightGBMForecaster(Forecaster):
    name = "lightgbm"

    def __init__(self, **lgbm_kwargs):
        base_params = dict(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=42,
            verbosity=-1,
        )
        base_params.update(lgbm_kwargs)
        self.model_p50 = LGBMRegressor(objective="regression_l1", **base_params)
        self.model_p10 = LGBMRegressor(objective="quantile", alpha=0.1, **base_params)
        self.model_p90 = LGBMRegressor(objective="quantile", alpha=0.9, **base_params)
        self.feature_cols: list[str] = []
        self.target_col = "demand_mw"

    def fit(self, train_df: pd.DataFrame, target_col: str = "demand_mw") -> "LightGBMForecaster":
        self.target_col = target_col
        self.feature_cols = [c for c in train_df.columns if c not in (target_col, "source")]
        X = train_df[self.feature_cols]
        y = train_df[target_col]
        self.model_p50.fit(X, y)
        self.model_p10.fit(X, y)
        self.model_p90.fit(X, y)
        return self

    def predict(self, context_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        future_index = self._future_index(context_df, horizon)

        history = context_df[self.target_col].copy()
        has_weather = all(c in context_df.columns for c in WEATHER_COLS)
        last_weather = context_df.iloc[-1][WEATHER_COLS] if has_weather else None

        p10s, p50s, p90s = [], [], []
        for ts in future_index:
            row = {}
            for lag in DEFAULT_LAG_STEPS:
                src_ts = ts - pd.Timedelta(minutes=30 * lag)
                row[f"lag_{lag}"] = history.get(src_ts, history.iloc[-1])
            for window in DEFAULT_ROLLING_WINDOWS:
                tail = history.iloc[-window:]
                row[f"roll_mean_{window}"] = tail.mean()
                row[f"roll_std_{window}"] = tail.std()
            cal = add_calendar_features(pd.DataFrame(index=[ts]))
            for col in cal.columns:
                row[col] = cal.iloc[0][col]
            if last_weather is not None:
                for col in WEATHER_COLS:
                    row[col] = last_weather[col]

            X = pd.DataFrame([row])[self.feature_cols]
            p10 = float(self.model_p10.predict(X)[0])
            p50 = float(self.model_p50.predict(X)[0])
            p90 = float(self.model_p90.predict(X)[0])
            lo, hi = min(p10, p90), max(p10, p90)
            p50 = min(max(p50, lo), hi)

            p10s.append(lo)
            p50s.append(p50)
            p90s.append(hi)
            history.loc[ts] = p50  # feed forecast back for recursive multi-step lags

        return pd.DataFrame({"p10": p10s, "p50": p50s, "p90": p90s}, index=future_index)

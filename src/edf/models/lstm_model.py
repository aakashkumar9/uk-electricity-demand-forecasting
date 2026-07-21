"""PyTorch LSTM forecaster: an encoder LSTM compresses the lookback window,
then a decoder steps through the forecast horizon one settlement period at a
time, conditioned on that step's *future calendar features* (hour, day of
week, bank holiday, etc. — all knowable in advance). Without that
conditioning the decoder has no way to know which future timestep it is
predicting and performs far worse than the baselines; this is the standard
"known future covariates" pattern for demand forecasting.

Prediction intervals come from MC-dropout (multiple stochastic forward
passes at inference, empirical P10/P50/P90 across samples).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from edf.features.calendar import CALENDAR_COLS, add_calendar_features
from edf.models.base import Forecaster


class _Seq2SeqLSTM(nn.Module):
    def __init__(self, n_features: int, n_calendar: int, hidden_size: int, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_size, num_layers=2, batch_first=True, dropout=dropout)
        self.decoder_cell = nn.LSTMCell(n_calendar, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, future_calendar: torch.Tensor) -> torch.Tensor:
        _, (h_n, c_n) = self.encoder(x)
        h, c = h_n[-1], c_n[-1]
        outputs = []
        for t in range(future_calendar.shape[1]):
            h, c = self.decoder_cell(future_calendar[:, t, :], (h, c))
            outputs.append(self.output_head(self.dropout(h)))
        return torch.cat(outputs, dim=1)


class LSTMForecaster(Forecaster):
    name = "lstm"

    def __init__(
        self,
        lookback: int = 336,
        horizon: int = 48,
        hidden_size: int = 64,
        epochs: int = 8,
        batch_size: int = 64,
        lr: float = 1e-3,
        mc_samples: int = 30,
        stride: int = 1,
        device: str | None = None,
    ):
        self.lookback = lookback
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.mc_samples = mc_samples
        self.stride = max(1, stride)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_cols: list[str] = []
        self.calendar_cols: list[str] = CALENDAR_COLS
        self.target_col = "demand_mw"
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._cal_mean: np.ndarray | None = None
        self._cal_std: np.ndarray | None = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self.net: _Seq2SeqLSTM | None = None

    def _make_sequences(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = df[self.feature_cols].to_numpy(dtype="float32")
        C = df[self.calendar_cols].to_numpy(dtype="float32")
        y = df[self.target_col].to_numpy(dtype="float32")
        n = len(df)
        xs, cs, ys = [], [], []
        last_start = n - self.lookback - self.horizon + 1
        for start in range(0, max(last_start, 0), self.stride):
            xs.append(X[start : start + self.lookback])
            cs.append(C[start + self.lookback : start + self.lookback + self.horizon])
            ys.append(y[start + self.lookback : start + self.lookback + self.horizon])
        if not xs:
            return (
                np.empty((0, self.lookback, X.shape[1]), dtype="float32"),
                np.empty((0, self.horizon, C.shape[1]), dtype="float32"),
                np.empty((0, self.horizon), dtype="float32"),
            )
        return np.stack(xs), np.stack(cs), np.stack(ys)

    def fit(self, train_df: pd.DataFrame, target_col: str = "demand_mw") -> "LSTMForecaster":
        self.target_col = target_col
        self.feature_cols = [c for c in train_df.columns if c not in (target_col, "source")]

        X_raw, C_raw, y_raw = self._make_sequences(train_df)
        if len(X_raw) < 4:
            raise ValueError("Not enough training rows for the configured lookback+horizon window")

        self._x_mean = X_raw.reshape(-1, X_raw.shape[-1]).mean(axis=0)
        self._x_std = X_raw.reshape(-1, X_raw.shape[-1]).std(axis=0) + 1e-6
        self._cal_mean = C_raw.reshape(-1, C_raw.shape[-1]).mean(axis=0)
        self._cal_std = C_raw.reshape(-1, C_raw.shape[-1]).std(axis=0) + 1e-6
        self._y_mean = float(y_raw.mean())
        self._y_std = float(y_raw.std() + 1e-6)

        Xn = (X_raw - self._x_mean) / self._x_std
        Cn = (C_raw - self._cal_mean) / self._cal_std
        yn = (y_raw - self._y_mean) / self._y_std

        dataset = TensorDataset(
            torch.tensor(Xn, dtype=torch.float32),
            torch.tensor(Cn, dtype=torch.float32),
            torch.tensor(yn, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.net = _Seq2SeqLSTM(len(self.feature_cols), len(self.calendar_cols), self.hidden_size).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.net.train()
        for _ in range(self.epochs):
            for xb, cb, yb in loader:
                xb, cb, yb = xb.to(self.device), cb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.net(xb, cb), yb)
                loss.backward()
                opt.step()
        return self

    def predict(self, context_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        if self.net is None:
            raise RuntimeError("Model must be fit before predict")
        horizon = min(horizon, self.horizon)
        future_index = self._future_index(context_df, horizon)

        window = context_df[self.feature_cols].iloc[-self.lookback :].to_numpy(dtype="float32")
        if len(window) < self.lookback:
            pad = np.repeat(window[:1], self.lookback - len(window), axis=0)
            window = np.concatenate([pad, window], axis=0)

        future_calendar = add_calendar_features(pd.DataFrame(index=future_index))[self.calendar_cols].to_numpy(
            dtype="float32"
        )

        Xn = (window - self._x_mean) / self._x_std
        Cn = (future_calendar - self._cal_mean) / self._cal_std
        x_t = torch.tensor(Xn[None, ...], dtype=torch.float32).to(self.device)
        c_t = torch.tensor(Cn[None, ...], dtype=torch.float32).to(self.device)

        self.net.train()  # keep dropout active for MC-dropout uncertainty estimation
        samples = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                pred = self.net(x_t, c_t).cpu().numpy()[0]
                samples.append(pred * self._y_std + self._y_mean)
        self.net.eval()

        samples = np.stack(samples)[:, :horizon]
        p10 = np.percentile(samples, 10, axis=0)
        p50 = np.percentile(samples, 50, axis=0)
        p90 = np.percentile(samples, 90, axis=0)
        return pd.DataFrame({"p10": p10, "p50": p50, "p90": p90}, index=future_index)

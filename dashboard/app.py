"""Streamlit dashboard: forecast vs actual with prediction intervals, plus a
model comparison table from the rolling-origin backtest.

Tries the FastAPI service first (set EDF_API_URL, defaults to
http://localhost:8000); falls back to reading the training pipeline's local
artifacts directly (and running inference in-process) so the dashboard also
works standalone with no separate API process — e.g. on Streamlit Community
Cloud, which only runs this one script and has nowhere to run a second
`uvicorn` process alongside it.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edf.config import MODELS_DIR, REPORTS_DIR  # noqa: E402

API_URL = os.environ.get("EDF_API_URL", "http://localhost:8000")


@st.cache_resource
def _load_local_model(model_name: str):
    path = MODELS_DIR / f"{model_name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def _load_local_context() -> pd.DataFrame | None:
    path = MODELS_DIR / "latest_context.parquet"
    return pd.read_parquet(path) if path.exists() else None


def get_forecast(model_name: str, horizon: int) -> pd.DataFrame:
    """Try the FastAPI service; fall back to running the pickled model
    in-process against the locally-committed context if the API isn't
    reachable (no separate server needed)."""
    try:
        resp = requests.get(f"{API_URL}/forecast", params={"model": model_name, "horizon": horizon}, timeout=3)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json()["points"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception:
        model = _load_local_model(model_name)
        context = _load_local_context()
        if model is None or context is None:
            raise RuntimeError(
                f"No API reachable at {API_URL} and no local model artifact for {model_name!r} "
                "in artifacts/models/ — run the training pipeline first."
            )
        preds = model.predict(context, horizon)
        df = preds.reset_index().rename(columns={"index": "timestamp"})
        return df

st.set_page_config(page_title="UK Energy Demand Forecast", layout="wide")
st.title("UK Energy Demand Forecasting")
st.caption("Half-hourly GB national demand — seasonal-naive vs LightGBM vs LSTM, backtested with rolling-origin CV")


@st.cache_data(ttl=60)
def get_summary_metrics() -> pd.DataFrame:
    try:
        resp = requests.get(f"{API_URL}/backtest/metrics", timeout=3)
        resp.raise_for_status()
        return pd.DataFrame(resp.json())
    except Exception:
        path = REPORTS_DIR / "summary_metrics.json"
        if path.exists():
            return pd.DataFrame(json.loads(path.read_text()))
        return pd.DataFrame()


@st.cache_data(ttl=60)
def get_latest_backtest() -> dict:
    try:
        resp = requests.get(f"{API_URL}/backtest/latest", timeout=3)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        path = REPORTS_DIR / "latest_forecast.json"
        if path.exists():
            return json.loads(path.read_text())
        return {}


@st.cache_data(ttl=60)
def get_metadata() -> dict:
    path = MODELS_DIR / "metadata.json"
    return json.loads(path.read_text()) if path.exists() else {}


metadata = get_metadata()
if metadata:
    cols = st.columns(4)
    cols[0].metric("Data source", metadata.get("data_source", "?"))
    cols[1].metric("Rows trained on", f"{metadata.get('n_rows', '?'):,}" if isinstance(metadata.get("n_rows"), int) else "?")
    cols[2].metric("Backtest folds", metadata.get("n_backtest_folds", "?"))
    cols[3].metric("Best model (MAE)", metadata.get("best_model_by_mae", "?"))

summary = get_summary_metrics()
st.subheader("Model comparison — rolling-origin backtest (mean over folds)")
if summary.empty:
    st.warning("No backtest report found yet. Run `python -m edf.training.train` first.")
else:
    st.dataframe(summary.set_index("model").style.highlight_min(axis=0, color="#1f6f43"), use_container_width=True)

st.subheader("Forecast vs actual — most recent backtest fold")
latest = get_latest_backtest()
if not latest:
    st.info("No forecast artifact found yet.")
else:
    model_choice = st.selectbox("Model", list(latest.keys()), index=list(latest.keys()).index("lightgbm") if "lightgbm" in latest else 0)
    rows = latest[model_choice]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["p90"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["p10"], fill="tonexty", fillcolor="rgba(31,111,67,0.2)",
            line=dict(width=0), name="P10-P90 interval",
        )
    )
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["p50"], name=f"{model_choice} forecast (P50)", line=dict(color="#1f6f43")))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["actual"], name="actual", line=dict(color="black", dash="dash")))
    fig.update_layout(height=500, yaxis_title="Demand (MW)", legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Live forecast")
st.caption(f"Tries the API at {API_URL} first, falls back to running the trained model directly if it's unreachable.")
horizon = st.slider("Horizon (half-hour steps)", min_value=6, max_value=96, value=48, step=6)
model_for_api = st.selectbox("Model to query", ["lightgbm", "lstm", "seasonal_naive"], key="api_model")
if st.button("Request forecast"):
    try:
        df_live = get_forecast(model_for_api, horizon)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_live["timestamp"], y=df_live["p90"], line=dict(width=0), showlegend=False))
        fig2.add_trace(go.Scatter(x=df_live["timestamp"], y=df_live["p10"], fill="tonexty", line=dict(width=0), name="P10-P90"))
        fig2.add_trace(go.Scatter(x=df_live["timestamp"], y=df_live["p50"], name="forecast (P50)"))
        fig2.update_layout(height=400, yaxis_title="Demand (MW)")
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))

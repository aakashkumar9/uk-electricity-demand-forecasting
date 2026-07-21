"""Runs rolling-origin backtests across a set of forecasters and produces a
comparison report (JSON + Markdown + PNG + a "latest forecast" artifact for
the dashboard/API)."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from edf.backtesting.metrics import interval_coverage, mae, mape, pinball_loss, rmse
from edf.backtesting.rolling_origin import Fold, rolling_origin_splits
from edf.models.base import Forecaster


def run_backtest(
    df: pd.DataFrame,
    model_factories: dict[str, Callable[[], Forecaster]],
    horizon: int,
    n_folds: int,
    step: int,
    min_train_steps: int,
    target_col: str = "demand_mw",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[Fold]]:
    """Returns (fold_metrics_df, {model_name: last_fold_forecast_df}, folds)."""
    folds = rolling_origin_splits(df, horizon, n_folds, step, min_train_steps, target_col)

    rows = []
    last_forecasts: dict[str, pd.DataFrame] = {}

    for model_name, factory in model_factories.items():
        for fold in folds:
            model = factory()
            model.fit(fold.train_df, target_col=target_col)
            forecast = model.predict(fold.train_df, horizon)

            actual = fold.actual.reindex(forecast.index)
            rows.append(
                {
                    "model": model_name,
                    "fold_id": fold.fold_id,
                    "origin": fold.origin,
                    "mae": mae(actual.values, forecast["p50"].values),
                    "rmse": rmse(actual.values, forecast["p50"].values),
                    "mape": mape(actual.values, forecast["p50"].values),
                    "pinball_10": pinball_loss(actual.values, forecast["p10"].values, 0.1),
                    "pinball_90": pinball_loss(actual.values, forecast["p90"].values, 0.9),
                    "coverage_80": interval_coverage(actual.values, forecast["p10"].values, forecast["p90"].values),
                }
            )
            if fold.fold_id == folds[-1].fold_id:
                out = forecast.copy()
                out["actual"] = actual.values
                last_forecasts[model_name] = out

    return pd.DataFrame(rows), last_forecasts, folds


def summarize(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    agg = fold_metrics.groupby("model")[["mae", "rmse", "mape", "pinball_10", "pinball_90", "coverage_80"]].mean()
    return agg.sort_values("mae").round(3)


def write_report(
    fold_metrics: pd.DataFrame,
    last_forecasts: dict[str, pd.DataFrame],
    reports_dir: Path,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(fold_metrics)

    fold_metrics.to_json(reports_dir / "fold_metrics.json", orient="records", date_format="iso", indent=2)
    summary.reset_index().to_json(reports_dir / "summary_metrics.json", orient="records", indent=2)

    md_lines = ["# Backtest results (rolling-origin CV)\n", summary.reset_index().to_markdown(index=False), "\n"]
    (reports_dir / "comparison.md").write_text("\n".join(md_lines))

    fig, ax = plt.subplots(figsize=(11, 5))
    for model_name, forecast in last_forecasts.items():
        ax.plot(forecast.index, forecast["p50"], label=f"{model_name} forecast")
        ax.fill_between(forecast.index, forecast["p10"], forecast["p90"], alpha=0.15)
    any_forecast = next(iter(last_forecasts.values()))
    ax.plot(any_forecast.index, any_forecast["actual"], label="actual", color="black", linewidth=2, linestyle="--")
    ax.set_title("Forecast vs actual — last backtest fold (shaded: P10-P90)")
    ax.set_ylabel("Demand (MW)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports_dir / "forecast_vs_actual.png", dpi=120)
    plt.close(fig)

    latest_payload = {
        model_name: json.loads(forecast.reset_index().rename(columns={"index": "timestamp"}).to_json(orient="records", date_format="iso"))
        for model_name, forecast in last_forecasts.items()
    }
    (reports_dir / "latest_forecast.json").write_text(json.dumps(latest_payload, indent=2))

    return reports_dir / "comparison.md"

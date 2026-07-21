"""End-to-end training CLI:

1. Build the dataset (real NESO + Open-Meteo, falling back to synthetic).
2. Engineer calendar/lag/weather features.
3. Rolling-origin backtest: seasonal-naive baseline vs LightGBM vs LSTM.
4. Persist a comparison report + refit each model on the full dataset and
   save the artifacts the FastAPI service and Streamlit dashboard read.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

from edf.backtesting.report import run_backtest, summarize, write_report
from edf.config import MODELS_DIR, REPORTS_DIR, settings
from edf.data.build_dataset import build_dataset
from edf.features.build_features import build_feature_frame
from edf.models.lightgbm_model import LightGBMForecaster
from edf.models.lstm_model import LSTMForecaster
from edf.models.seasonal_naive import SeasonalNaiveForecaster

logger = logging.getLogger(__name__)


def get_model_factories(horizon: int, lookback: int, lstm_epochs: int, lstm_stride: int):
    return {
        "seasonal_naive": lambda: SeasonalNaiveForecaster(),
        "lightgbm": lambda: LightGBMForecaster(),
        "lstm": lambda: LSTMForecaster(lookback=lookback, horizon=horizon, epochs=lstm_epochs, stride=lstm_stride),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--horizon", type=int, default=settings.horizon_steps)
    parser.add_argument("--lookback", type=int, default=settings.lookback_steps)
    parser.add_argument("--n-folds", type=int, default=settings.n_backtest_folds)
    parser.add_argument("--step-days", type=int, default=settings.backtest_step_days)
    parser.add_argument("--lstm-epochs", type=int, default=8)
    parser.add_argument("--lstm-stride", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    t0 = time.time()
    raw = build_dataset(args.start, args.end, force_synthetic=args.synthetic)
    data_source = raw["source"].iloc[0] if "source" in raw.columns else "unknown"
    logger.info("Dataset ready: %d rows (%s)", len(raw), data_source)

    features = build_feature_frame(raw)
    logger.info("Feature frame ready: %d rows, %d columns", *features.shape)

    steps_per_day = 48  # native half-hourly settlement periods
    step = args.step_days * steps_per_day
    min_train_steps = args.lookback + args.horizon + steps_per_day * 30  # >= ~1 month warmup

    factories = get_model_factories(args.horizon, args.lookback, args.lstm_epochs, args.lstm_stride)
    fold_metrics, last_forecasts, folds = run_backtest(
        features,
        factories,
        horizon=args.horizon,
        n_folds=args.n_folds,
        step=step,
        min_train_steps=min_train_steps,
    )

    summary = summarize(fold_metrics)
    logger.info("Backtest summary (mean over %d folds):\n%s", args.n_folds, summary)

    report_path = write_report(fold_metrics, last_forecasts, REPORTS_DIR)
    logger.info("Report written to %s", report_path)

    logger.info("Refitting each model on the full dataset for production artifacts...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, factory in factories.items():
        model = factory()
        model.fit(features, target_col="demand_mw")
        with open(MODELS_DIR / f"{name}.pkl", "wb") as f:
            pickle.dump(model, f)
        logger.info("Saved %s -> %s", name, MODELS_DIR / f"{name}.pkl")

    # Save the tail of the feature frame so the API can serve forecasts without
    # needing to refetch data at request time.
    features.tail(args.lookback + steps_per_day * 3).to_parquet(MODELS_DIR / "latest_context.parquet")

    metadata = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_source": raw["source"].iloc[0] if "source" in raw else "unknown",
        "n_rows": len(raw),
        "horizon_steps": args.horizon,
        "lookback_steps": args.lookback,
        "n_backtest_folds": args.n_folds,
        "best_model_by_mae": summary.index[0],
        "training_seconds": round(time.time() - t0, 1),
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info("Training pipeline complete in %.1fs. Metadata: %s", metadata["training_seconds"], metadata)


if __name__ == "__main__":
    main()

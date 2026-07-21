# UK Energy Demand Forecasting

A time-series forecasting system for short-term GB electricity demand, built
to mirror how a team at Octopus Energy / Kaluza / NESO / Ofgem would actually
approach it: a real baseline, an honest backtest, a couple of models that
have to beat that baseline, and a served, dashboarded, containerised,
CI-tested end product.

## What it does

1. **Ingests** half-hourly GB national demand from [NESO's open data
   portal](https://www.neso.energy/data-portal) (CKAN API) and hourly
   weather from the [Open-Meteo historical archive](https://open-meteo.com/)
   (no API key needed), and merges them onto a half-hourly (settlement
   period) grid.
2. **Engineers features**: calendar/cyclical encodings, UK bank holidays,
   lag features (30min/1h/2h/1d/2d/1wk), and rolling stats.
3. **Forecasts** with three models of increasing sophistication:
   - **Seasonal-naive baseline** — `demand(t) = demand(t - 1 week)`. This is
     the floor. If a fancier model can't beat this, it isn't earning its
     complexity.
   - **LightGBM** — three quantile regressors (P10/P50/P90) on lag +
     calendar + weather features, applied recursively over the forecast
     horizon.
   - **LSTM** (PyTorch) — encodes a multi-day lookback window and decodes
     the full horizon directly, with MC-dropout for prediction intervals.
4. **Backtests honestly** with rolling-origin (walk-forward) cross-validation
   — the model is retrained/re-evaluated from multiple points in time moving
   forward, never a single train/test split — and reports MAE, RMSE, MAPE,
   pinball loss, and 80% interval coverage per fold.
5. **Serves** the trained models behind a FastAPI app (`/forecast`,
   `/backtest/metrics`, `/health`).
6. **Visualises** forecast vs. actual with prediction intervals, and the
   model comparison table, in a Streamlit dashboard.
7. **Ships** in Docker (API + dashboard as separate services via
   `docker-compose`) with a GitHub Actions pipeline that runs tests on every
   push/PR and retrains on a weekly schedule (or on demand).

## A note on data access

NESO and Open-Meteo are fetched live wherever the pipeline actually runs
with real internet access (your laptop, a GitHub Actions runner). Some
sandboxed / egress-restricted environments cannot reach those hosts; in that
case `build_dataset()` falls back to a schema-identical **synthetic**
generator (`src/edf/data/synthetic.py`) so the rest of the pipeline can still
be developed and tested offline. The fallback is logged loudly and stamped
into `artifacts/models/metadata.json` as `"data_source"` — check that field
before trusting any numbers. The scheduled GitHub Actions retrain job always
has real internet access, so its results are real-data results.

## Architecture

```
NESO CKAN API ─┐
                ├─▶ build_dataset ─▶ build_feature_frame ─▶ rolling-origin backtest ─▶ artifacts/
Open-Meteo API ─┘                                                 │                        │
                                                                   ▼                        ▼
                                          seasonal-naive / LightGBM / LSTM        FastAPI  ◀── Streamlit
```

## Project layout

```
src/edf/
  config.py              settings (data window, paths, model hyperparameters)
  data/                   NESO client, Open-Meteo client, synthetic fallback, dataset builder
  features/               calendar features, lag/rolling features
  models/                 seasonal-naive, LightGBM, LSTM (common Forecaster interface)
  backtesting/            rolling-origin CV splitter, metrics, report writer
  training/train.py       end-to-end CLI: fetch -> features -> backtest -> save artifacts
  serving/api.py          FastAPI app
dashboard/app.py          Streamlit dashboard
tests/                    pytest suite (synthetic-data fixtures, no network required)
docker/                   Dockerfile.api, Dockerfile.dashboard
docker-compose.yml
```

## Running it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Train + backtest + save artifacts. Add --synthetic to force the offline fallback.
python -m edf.training.train --n-folds 6 --lstm-epochs 8

# Serve the API
uvicorn edf.serving.api:app --reload

# In another terminal: the dashboard
streamlit run dashboard/app.py
```

Or with Docker:

```bash
docker compose up --build
# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

## Testing

```bash
pytest -v
```

Tests run entirely against the synthetic fixture generator — no network
access required, so they're deterministic in CI.

## Backtest results

Rolling-origin CV results are written to `artifacts/reports/` on every
training run: `comparison.md` (summary table), `fold_metrics.json` (per-fold
detail), `forecast_vs_actual.png`, and `latest_forecast.json` (consumed by
the dashboard/API). Re-run `python -m edf.training.train` to regenerate them
against the current data window — the numbers in `artifacts/` reflect
whatever was most recently trained (see `metadata.json` for the data source
and timestamp).

## Design choices worth calling out

- **Half-hourly resolution**, not hourly: this matches NESO's native GB
  settlement period, which is what balancing/demand teams actually operate
  on (48 periods/day, occasionally 46 or 50 on clock-change days).
- **Recursive LightGBM, direct-decode LSTM**: two different multi-step
  strategies deliberately, so the comparison isn't just "which library" but
  "which forecasting strategy" — recursive error accumulation vs. a model
  that has to learn the whole horizon shape at once.
- **Rolling-origin CV over a single split**: a single train/test split can
  make a model look great or terrible depending on which season it happens
  to land on. Multiple folds moving forward in time is the only way to get
  an honest, low-variance read on relative model quality.
- **Prediction intervals from two different mechanisms**: LightGBM quantile
  regression (pinball loss objective) vs. LSTM MC-dropout — again, on
  purpose, to show both approaches.

"""Builds the merged demand + weather dataset used for feature engineering.

Tries real NESO + Open-Meteo data first; falls back to the synthetic
generator (with a loud warning) if network access is unavailable. This means
the exact same CLI works both in a restricted local sandbox and in the
GitHub Actions retrain job, which has real internet access.
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from edf.config import PROCESSED_DIR, RAW_DIR, settings
from edf.data.neso_client import NesoFetchError, fetch_historic_demand
from edf.data.synthetic import generate_synthetic_dataset
from edf.data.weather_client import WeatherFetchError, fetch_historic_weather

logger = logging.getLogger(__name__)

DATASET_PATH = PROCESSED_DIR / "dataset.parquet"


def _fetch_real(start: str | None, end: str | None) -> pd.DataFrame:
    demand = fetch_historic_demand(start, end)
    demand.to_parquet(RAW_DIR / "neso_demand.parquet")

    weather = fetch_historic_weather(start, end)
    weather.to_parquet(RAW_DIR / "open_meteo_weather.parquet")

    freq = settings.resample_freq
    demand_r = demand.resample(freq).mean().interpolate(limit=4)
    weather_r = weather.resample(freq).interpolate(limit=8)

    merged = demand_r.join(weather_r, how="inner").dropna(subset=["demand_mw"])
    merged["source"] = "neso+open-meteo"
    return merged


def build_dataset(
    start: str | None = None,
    end: str | None = None,
    force_synthetic: bool = False,
) -> pd.DataFrame:
    if not force_synthetic:
        try:
            merged = _fetch_real(start, end)
            merged.to_parquet(DATASET_PATH)
            logger.info("Built dataset from real NESO + Open-Meteo data: %d rows", len(merged))
            return merged
        except (NesoFetchError, WeatherFetchError, Exception) as exc:  # noqa: BLE001
            logger.warning(
                "Falling back to synthetic dataset because real data fetch failed: %s", exc
            )

    synth = generate_synthetic_dataset(start, end)
    synth["source"] = "synthetic"
    synth.to_parquet(DATASET_PATH)
    logger.info("Built synthetic dataset: %d rows", len(synth))
    return synth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data, skip network calls")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    df = build_dataset(args.start, args.end, force_synthetic=args.synthetic)
    print(df.describe())
    print(f"Saved to {DATASET_PATH}")


if __name__ == "__main__":
    main()

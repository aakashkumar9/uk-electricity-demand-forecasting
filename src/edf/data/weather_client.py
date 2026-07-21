"""Client for historical GB weather from the Open-Meteo archive API (no API key)."""
from __future__ import annotations

import logging

import pandas as pd
import requests

from edf.config import settings

logger = logging.getLogger(__name__)


class WeatherFetchError(RuntimeError):
    pass


def fetch_historic_weather(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Fetch hourly historical weather for the configured lat/lon and return a
    DataFrame indexed by tz-naive local timestamp."""
    start_date = str(pd.Timestamp(start or settings.history_start).date())
    end_date = str(pd.Timestamp(end or settings.history_end).date())

    params = {
        "latitude": settings.weather_latitude,
        "longitude": settings.weather_longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": settings.weather_hourly_vars,
        "timezone": "Europe/London",
    }
    resp = requests.get(settings.weather_api_base, params=params, timeout=settings.request_timeout_s)
    resp.raise_for_status()
    payload = resp.json()
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise WeatherFetchError(f"Unexpected Open-Meteo response: {payload}")

    df = pd.DataFrame(hourly)
    df["timestamp"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"]).set_index("timestamp").sort_index()
    df = df.rename(columns={
        "temperature_2m": "temp_c",
        "relative_humidity_2m": "humidity_pct",
        "wind_speed_10m": "wind_speed_ms",
        "cloud_cover": "cloud_cover_pct",
        "precipitation": "precip_mm",
    })
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = fetch_historic_weather()
    print(out.head())
    print(f"Fetched {len(out)} hourly weather rows")

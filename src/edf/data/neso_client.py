"""Client for NESO (National Energy System Operator) historic GB demand data.

NESO publishes half-hourly national demand (ND) and transmission system demand
(TSD) figures on its open CKAN data portal (https://www.neso.energy/data-portal).
Resource identifiers are not hardcoded here: we discover the relevant datasets
and CSV resource URLs at run time via the CKAN `package_search` / `package_show`
actions, then download the CSVs directly. This keeps the client working even
as NESO rolls datasets over from year to year.
"""
from __future__ import annotations

import logging
from datetime import datetime
from io import StringIO

import pandas as pd
import requests

from edf.config import settings

logger = logging.getLogger(__name__)

_SETTLEMENT_PERIOD_MINUTES = 30


class NesoFetchError(RuntimeError):
    """Raised when NESO data cannot be discovered or downloaded."""


def _ckan_get(action: str, **params) -> dict:
    url = f"{settings.neso_api_base}/{action}"
    resp = requests.get(url, params=params, timeout=settings.request_timeout_s)
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("success", False):
        raise NesoFetchError(f"CKAN action {action!r} did not succeed: {payload}")
    return payload["result"]


def discover_csv_resources(query: str | None = None) -> list[dict]:
    """Search the NESO CKAN portal for historic-demand-data packages and return
    the list of CSV resource dicts (with 'url', 'name', 'format')."""
    query = query or settings.neso_package_query
    result = _ckan_get("package_search", q=query, rows=50)
    resources: list[dict] = []
    for pkg in result.get("results", []):
        for res in pkg.get("resources", []):
            if str(res.get("format", "")).upper() == "CSV":
                resources.append(res)
    if not resources:
        raise NesoFetchError(f"No CSV resources found for query {query!r}")
    return resources


def _download_csv(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=settings.request_timeout_s)
    resp.raise_for_status()
    return pd.read_csv(StringIO(resp.text))


def _settlement_to_timestamp(date_col: pd.Series, period_col: pd.Series) -> pd.Series:
    base = pd.to_datetime(date_col, dayfirst=True, errors="coerce")
    minutes = (period_col.astype(float) - 1) * _SETTLEMENT_PERIOD_MINUTES
    return base + pd.to_timedelta(minutes, unit="m")


def fetch_historic_demand(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Fetch and concatenate historic half-hourly GB national demand (ND, MW).

    Returns a DataFrame indexed by tz-naive local (Europe/London clock time)
    timestamp with a single column ``demand_mw``.
    """
    start_ts = pd.Timestamp(start or settings.history_start)
    end_ts = pd.Timestamp(end or settings.history_end)

    resources = discover_csv_resources()
    frames = []
    for res in resources:
        try:
            df = _download_csv(res["url"])
        except Exception as exc:  # noqa: BLE001 - keep going with other resources
            logger.warning("Skipping NESO resource %s: %s", res.get("name"), exc)
            continue
        cols = {c.upper(): c for c in df.columns}
        if "SETTLEMENT_DATE" not in cols or "SETTLEMENT_PERIOD" not in cols:
            continue
        demand_col = None
        for candidate in ("ND", "TSD", "ENGLAND_WALES_DEMAND"):
            if candidate in cols:
                demand_col = cols[candidate]
                break
        if demand_col is None:
            continue
        ts = _settlement_to_timestamp(df[cols["SETTLEMENT_DATE"]], df[cols["SETTLEMENT_PERIOD"]])
        frame = pd.DataFrame({"timestamp": ts, "demand_mw": pd.to_numeric(df[demand_col], errors="coerce")})
        frames.append(frame)

    if not frames:
        raise NesoFetchError("Downloaded NESO resources did not contain usable demand columns")

    full = pd.concat(frames, ignore_index=True).dropna(subset=["timestamp", "demand_mw"])
    full = full.drop_duplicates(subset="timestamp").sort_values("timestamp")
    full = full[(full["timestamp"] >= start_ts) & (full["timestamp"] <= end_ts)]
    full = full.set_index("timestamp")

    if full.empty:
        raise NesoFetchError(
            f"No NESO demand rows found in range {start_ts.date()}..{end_ts.date()}"
        )
    return full


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = fetch_historic_demand()
    print(out.head())
    print(f"Fetched {len(out)} half-hourly demand rows spanning {out.index.min()} to {out.index.max()}")

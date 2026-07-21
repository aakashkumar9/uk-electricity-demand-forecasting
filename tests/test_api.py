import pickle
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture()
def client(tmp_path, monkeypatch, feature_frame):
    import edf.config as config_module

    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    models_dir.mkdir()
    reports_dir.mkdir()

    monkeypatch.setattr(config_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(config_module, "REPORTS_DIR", reports_dir)

    import edf.serving.api as api_module

    monkeypatch.setattr(api_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(api_module, "REPORTS_DIR", reports_dir)
    api_module._load_context.cache_clear()
    api_module._load_model.cache_clear()

    from edf.models.seasonal_naive import SeasonalNaiveForecaster

    horizon = 48
    train = feature_frame.iloc[:-horizon]
    model = SeasonalNaiveForecaster().fit(train, target_col="demand_mw")
    with open(models_dir / "seasonal_naive.pkl", "wb") as f:
        pickle.dump(model, f)

    train.tail(500).to_parquet(models_dir / "latest_context.parquet")
    (models_dir / "metadata.json").write_text('{"trained_at": "2024-01-01T00:00:00Z", "data_source": "synthetic"}')
    (reports_dir / "summary_metrics.json").write_text(
        '[{"model": "seasonal_naive", "mae": 100.0, "rmse": 120.0, "mape": 1.0, "pinball_10": 10.0, "pinball_90": 10.0, "coverage_80": 0.8}]'
    )

    return TestClient(api_module.app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "seasonal_naive" in body["models_loaded"]


def test_forecast_endpoint(client):
    resp = client.get("/forecast", params={"model": "seasonal_naive", "horizon": 12})
    assert resp.status_code == 200
    body = resp.json()
    assert body["horizon_steps"] == 12
    assert len(body["points"]) == 12
    assert body["points"][0]["p10"] <= body["points"][0]["p50"] <= body["points"][0]["p90"]


def test_forecast_unknown_model_400(client):
    resp = client.get("/forecast", params={"model": "not_a_model"})
    assert resp.status_code == 400


def test_backtest_metrics(client):
    resp = client.get("/backtest/metrics")
    assert resp.status_code == 200
    assert resp.json()[0]["model"] == "seasonal_naive"

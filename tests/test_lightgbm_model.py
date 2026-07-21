from edf.models.lightgbm_model import LightGBMForecaster


def test_lightgbm_forecast_shape_and_monotonic_quantiles(feature_frame):
    horizon = 24
    train = feature_frame.iloc[:-horizon]
    model = LightGBMForecaster(n_estimators=30).fit(train, target_col="demand_mw")
    forecast = model.predict(train, horizon)

    assert len(forecast) == horizon
    assert (forecast["p10"] <= forecast["p50"]).all()
    assert (forecast["p50"] <= forecast["p90"]).all()
    assert forecast["p50"].notna().all()

from edf.models.seasonal_naive import WEEK_STEPS, SeasonalNaiveForecaster


def test_seasonal_naive_matches_last_week(feature_frame):
    horizon = 48
    train = feature_frame.iloc[:-horizon]
    model = SeasonalNaiveForecaster().fit(train, target_col="demand_mw")
    forecast = model.predict(train, horizon)

    assert list(forecast.columns) == ["p10", "p50", "p90"]
    assert len(forecast) == horizon
    # p10 <= p50 <= p90 always
    assert (forecast["p10"] <= forecast["p50"]).all()
    assert (forecast["p50"] <= forecast["p90"]).all()


def test_seasonal_naive_beats_naive_last_value_on_seasonal_data(feature_frame):
    """The seasonal-naive baseline (t - 1 week) should track a strongly
    seasonal series far better than always predicting the last observed value."""
    horizon = 48
    train = feature_frame.iloc[:-horizon]
    actual = feature_frame["demand_mw"].iloc[-horizon:]

    model = SeasonalNaiveForecaster(season_steps=WEEK_STEPS).fit(train, target_col="demand_mw")
    seasonal_forecast = model.predict(train, horizon)["p50"].values

    last_value_forecast = train["demand_mw"].iloc[-1]
    seasonal_mae = abs(seasonal_forecast - actual.values).mean()
    naive_mae = abs(last_value_forecast - actual.values).mean()

    assert seasonal_mae < naive_mae

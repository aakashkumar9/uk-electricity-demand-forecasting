from edf.models.lstm_model import LSTMForecaster


def test_lstm_forecast_shape(feature_frame):
    horizon = 12
    lookback = 96
    train = feature_frame.iloc[:-horizon]
    model = LSTMForecaster(lookback=lookback, horizon=horizon, hidden_size=8, epochs=1, mc_samples=3).fit(
        train, target_col="demand_mw"
    )
    forecast = model.predict(train, horizon)

    assert len(forecast) == horizon
    assert (forecast["p10"] <= forecast["p90"]).all()

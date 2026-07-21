from edf.features.lags import DEFAULT_LAG_STEPS, add_lag_features


def test_lag_features_alignment(raw_dataset):
    out = add_lag_features(raw_dataset)
    for lag in DEFAULT_LAG_STEPS:
        col = f"lag_{lag}"
        assert col in out.columns
        # lag_N at row i should equal demand_mw at row i-N
        valid = out.iloc[lag:]
        assert (valid[col].values == raw_dataset["demand_mw"].iloc[:-lag].values).all()


def test_lag_features_leading_nans(raw_dataset):
    out = add_lag_features(raw_dataset)
    max_lag = max(DEFAULT_LAG_STEPS)
    assert out[f"lag_{max_lag}"].iloc[:max_lag].isna().all()

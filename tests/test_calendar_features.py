import pandas as pd

from edf.features.calendar import add_calendar_features


def test_calendar_features_basic_shape(raw_dataset):
    out = add_calendar_features(raw_dataset)
    for col in ["settlement_period", "hour", "day_of_week", "is_weekend", "tod_sin", "tod_cos", "is_bank_holiday"]:
        assert col in out.columns
    assert len(out) == len(raw_dataset)


def test_settlement_period_range(raw_dataset):
    out = add_calendar_features(raw_dataset)
    assert out["settlement_period"].min() >= 1
    assert out["settlement_period"].max() <= 48


def test_christmas_day_is_bank_holiday():
    idx = pd.date_range("2023-12-24", "2023-12-26", freq="30min", inclusive="left")
    df = pd.DataFrame({"demand_mw": 1.0}, index=idx)
    out = add_calendar_features(df)
    christmas_rows = out[out.index.date == pd.Timestamp("2023-12-25").date()]
    assert (christmas_rows["is_bank_holiday"] == 1).all()

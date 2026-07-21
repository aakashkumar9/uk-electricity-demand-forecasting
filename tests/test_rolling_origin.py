import pytest

from edf.backtesting.rolling_origin import rolling_origin_splits


def test_rolling_origin_fold_count_and_no_lookahead(feature_frame):
    horizon = 48
    n_folds = 3
    step = 96
    folds = rolling_origin_splits(feature_frame, horizon=horizon, n_folds=n_folds, step=step, min_train_steps=500)

    assert len(folds) == n_folds
    for fold in folds:
        assert len(fold.actual) == horizon
        # No overlap between train data and the actuals being forecast.
        assert fold.train_df.index.max() < fold.actual.index.min()

    # Origins should be strictly increasing across folds.
    origins = [f.origin for f in folds]
    assert origins == sorted(origins)


def test_rolling_origin_raises_when_insufficient_data(feature_frame):
    with pytest.raises(ValueError):
        rolling_origin_splits(feature_frame, horizon=48, n_folds=3, step=96, min_train_steps=10**9)

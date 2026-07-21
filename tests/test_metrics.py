import numpy as np

from edf.backtesting.metrics import interval_coverage, mae, mape, pinball_loss, rmse


def test_mae_zero_for_perfect_forecast():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert mape(y, y) == 0.0


def test_mae_rmse_known_values():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    assert mae(y_true, y_pred) == np.mean([2, 2, 3])
    assert rmse(y_true, y_pred) == np.sqrt(np.mean([4, 4, 9]))


def test_pinball_loss_zero_at_perfect_prediction():
    y = np.array([5.0, 6.0])
    assert pinball_loss(y, y, 0.1) == 0.0
    assert pinball_loss(y, y, 0.9) == 0.0


def test_interval_coverage():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    lower = np.array([0.0, 0.0, 0.0, 10.0])
    upper = np.array([5.0, 5.0, 5.0, 20.0])
    assert interval_coverage(y_true, lower, upper) == 0.75

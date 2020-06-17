from other_costs import mse_, rmse_, r2score_, mae_
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import numpy as np

x = np.array([0, 15, -9, 7, 12, 3, -21])
y = np.array([2, 14, -13, 5, 12, 4, -19])


def test_mse():
    assert mse_(x, y) == mean_squared_error(x, y)


def test_rmse():
    assert rmse_(x, y) == sqrt(mean_squared_error(x, y))


def test_r2score():
    assert r2score_(x, y) == r2_score(x, y)


def test_mae():
    assert mae_(x, y) == mean_absolute_error(x, y)

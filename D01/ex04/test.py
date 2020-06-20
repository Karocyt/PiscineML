from gradient import gradient
from other_costs import mse_
from prediction import predict_

import numpy as np

x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733])
y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554])

def test1():
    theta1 = np.array([2, 0.7])
    res = gradient(x, y, theta1)
    print(res)
    assert (res == np.array([21.0342574, 587.36875564])).all()

def test2():
    theta2 = np.array([1, -0.4])
    res = gradient(x, y, theta2)
    loss = mse_(x, predict_(x, theta2))
    print(res, loss)
    theta2 -= 0.001 * res
    res = gradient(x, y, theta2)
    loss = mse_(x, predict_(x, theta2))
    print(res, loss)
    theta2 -= 0.001 * res
    res = gradient(x, y, theta2)
    loss = mse_(x, predict_(x, theta2))
    print(res, loss)
    theta2 -= 0.0001 * res
    res = gradient(x, y, theta2)
    loss = mse_(x, predict_(x, theta2))
    print(res, loss)
    theta2 -= 0.0001 * res
    res = gradient(x, y, theta2)
    loss = mse_(x, predict_(x, theta2))
    print(res, loss)

    assert (res == np.array([58.86823748, 2229.72297889])).all()
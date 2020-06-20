import numpy as np
from tools import add_intercept


def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop. The
    ,→ three arrays must have compatible dimensions.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    11
    theta: has to be an numpy.ndarray, a 2 * 1 vector.
    Returns:
    The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
    None if x, y, or theta are empty numpy.ndarray.
    None if x, y and theta do not have compatible dimensions.
    Raises:
    This function should not raise any Exception.
    """
    # try:
    x_bias = add_intercept(x)
    y_pred = np.dot(x_bias, theta)
    y_pred = y_pred.reshape(len(y),1)
    error = y_pred - y
    # print(y_pred)
    # print(y)
    # print(error)
    error_columns = error.reshape(len(y), 1)
    error_dot_x = np.dot(error_columns.T, x_bias)
    grad = 1/len(x) * error_dot_x
    return grad.reshape(len(theta),)
    # except:
    # return None

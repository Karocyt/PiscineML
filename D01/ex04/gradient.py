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
    try:
        x_bias = add_intercept(x)
        y_pred = np.dot(x_bias, theta)
        return (1/len(x)* np.dot((y_pred - y).reshape(len(y), 1).T, x_bias)).reshape(len(theta),)
    except:
        return None
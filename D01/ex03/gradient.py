import numpy as np

from prediction import predict_

def simple_gradient(x, y, theta):
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
    return np.array([1.0 / len(x) * np.sum(predict_(x, theta) - y), 1.0 / len(x) * np.sum((predict_(x, theta) - y) * x)])
    # except:
    #     return None
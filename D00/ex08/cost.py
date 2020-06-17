import numpy as np


def cost_(y, y_hat):
    """
    Description:
    Calculates the value of cost function.
    Args:
    y: has to be an numpy.ndarray, a vector.
    y_hat: has to be an numpy.ndarray, a vector.
    Returns:
    26
    J_value : has to be a float.
    None if there is a dimension matching problem between X, Y or theta.
    Raises:
    This function should not raise any Exception.
    """
    if len(y) > 1 and len(y.shape) == 1:
        print("convert")
        y_hat = y_hat.flatten()
    try:
        return np.sum(1.0 / (2*len(y)) * (y_hat - y)**2)
    except:
        return None

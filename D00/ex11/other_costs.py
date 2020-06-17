import numpy as np


def mse_elem_(y, y_hat):
    """
    Description:
    Calculate the MSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.ndarray, a vector of dimension m * 1.
    y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
    Returns:
    mse: has to be a float.
    35
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    if len(y) > 1 and len(y.shape) == 1:
        print("convert")
        y_hat = y_hat.flatten()
    try:
        return (y_hat - y)**2
    except:
        return None


def mse_(y, y_hat):
    """
    Description:
    Calculate the MSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.ndarray, a vector of dimension m * 1.
    y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
    Returns:
    mse: has to be a float.
    35
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    if len(y) > 1 and len(y.shape) == 1:
        print("convert")
        y_hat = y_hat.flatten()
    try:
        return np.sum(mse_elem_(y, y_hat) / len(y))
    except:
        return None


def rmse_(y, y_hat):
    """
    Description:
    Calculate the RMSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.ndarray, a vector of dimension m * 1.
    y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
    Returns:
    rmse: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    try:
        return np.sqrt(mse_(y, y_hat))
    except:
        return None


def mae_elem_(y, y_hat):
    return np.abs(y - y_hat)


def mae_(y, y_hat):
    """
    Description:
    Calculate the MAE between the predicted output and the real output.
    Args:
    y: has to be a numpy.ndarray, a vector of dimension m * 1.
    y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
    Returns:
    mae: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    try:
        return mae_elem_(y, y_hat).mean()
    except:
        return None


def r2score_(y, y_hat):
    """
    Description:
    Calculate the R2score between the predicted output and the output.
    Args:
    y: has to be a numpy.ndarray, a vector of dimension m * 1.
    y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
    Returns:
    r2score: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    try:
        return 1.0 - np.sum(mse_elem_(y, y_hat)) / np.sum((y_hat - y.mean())**2)
    except:
        return None

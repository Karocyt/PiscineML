#! /usr/bin/env python3

import numpy as np


def add_polynomial_features(x, power):
    """Add polynomial features to vector x by raising its values up to the power given in argument.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
    power: has to be an int, the power up to which the components of vector x are going to be raised.
    Returns:
    The matrix of polynomial features as a numpy.ndarray, of dimension m * n, containg the polynomial feature values for all training examples.
    None if x is an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    if type(power) is int and type(x) is np.ndarray:
        return np.concatenate([x**i for i in range(1, power+1)], axis=1)
    return None


if __name__ == "__main__":
    x = np.arange(1,6).reshape(-1, 1)
    # Example 1:
    print(x)
    print(add_polynomial_features(x, 3))
    
    # Example 2:
    print(add_polynomial_features(x, 6))
    # Output:
    # array([[ 1, 1, 1, 1, 1, 1],
    # [ 2, 4, 8, 16, 32, 64],
    # [ 3, 9, 27, 81, 243, 729],
    # [ 4, 16, 64, 256, 1024, 4096],
    # [ 5, 25, 125, 625, 3125, 15625]])
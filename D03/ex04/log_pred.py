#! /usr/bin/env python3

import numpy as np
from sigmoid import sigmoid_

def logistic_predict(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * n.
    theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
    Returns:
    y_hat as a numpy.ndarray, a vector of dimension m * 1.
    None if x or theta are empty numpy.ndarray.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exception.
    """
    if x.ndim == 1:
        x = x.reshape(len(x), 1)
        print(x.shape, theta.shape)
    intercept = np.ones((x.shape[0], 1))
    x = np.append(intercept, x, axis=1)
    return sigmoid_(x.dot(theta))

if __name__ == "__main__":
    # Example 1
    x = np.array([4])
    theta = np.array([[2], [0.5]])
    print(logistic_predict(x, theta))
    # Output:
    print(np.array([[0.98201379]]))
    print()
    # Example 1
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    print(logistic_predict(x2, theta2))
    # Output:
    print(np.array([[0.98201379],
    [0.99624161],
    [0.97340301],
    [0.99875204],
    [0.90720705]]))
    print()
    # Example 3
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print(logistic_predict(x3, theta3))
    # Output:
    print(np.array([[0.03916572],
    [0.00045262],
    [0.2890505 ]]))
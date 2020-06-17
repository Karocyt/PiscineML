#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt

from prediction import predict_


def plot(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exceptions.
    """
    plt.plot(x, y, 'ro')

    linex = np.linspace(np.min(x), np.max(x), 2)
    plt.plot(linex, predict_(linex, theta), color="blue", linestyle="dotted")

    plt.show()


if __name__ == "__main__":
    print("--- Plot tests ---")

    x = np.arange(1, 6)
    y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])
    # Example 1:
    theta1 = np.array([4.5, -0.2])
    plot(x, y, theta1)
    # Output:

    # Example 2:
    theta2 = np.array([-1.5, 2])
    plot(x, y, theta2)
    # Output:

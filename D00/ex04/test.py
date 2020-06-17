from tools import add_intercept
import numpy as np


def test():
    # Example 1:
    x = np.arange(1, 6)
    assert (add_intercept(x) == np.array([[1., 1.],
                                          [1., 2.],
                                          [1., 3.],
                                          [1., 4.],
                                          [1., 5.]])).all()

    # Example 2:
    y = np.arange(1, 10).reshape((3, 3))
    assert (add_intercept(y) == np.array([[1., 1., 2., 3.],
                                          [1., 4., 5., 6.],
                                          [1., 7., 8., 9.]])).all()

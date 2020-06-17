from prediction import predict_

import numpy as np


def test_correc():
    x = np.arange(1, 6)
    # Example 1:
    theta1 = np.array([5, 0])
    assert np.equal(predict_(x, theta1),
                    np.array([5., 5., 5., 5., 5.])).all()
    # Do you understand why y_hat contains only 5's here?

    # Example 2:
    theta2 = np.array([0, 1])
    assert (predict_(x, theta2) == np.array([1., 2., 3., 4., 5.])).all()
    # Do you understand why y_hat == x here?

    # Example 3:
    theta3 = np.array([5, 3])
    assert (predict_(x, theta3) == np.array(
        [8., 11., 14., 17., 20.])).all()

    # Example 4:
    theta4 = np.array([-3, 1])
    assert (predict_(x, theta4) ==
            np.array([-2., -1., 0., 1., 2.])).all()


def test_better():
    x = np.arange(1, 6)
    # Example 1:
    theta1 = np.array([5, 1])
    assert np.equal(predict_(x, theta1),
                    np.array([6., 7., 8., 9., 10.])).all()


def test_error():
    x = 56
    # Example 1:
    theta1 = np.array([5, 1])
    assert np.isnan(predict_(x, theta1))

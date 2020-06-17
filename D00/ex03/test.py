from prediction import simple_predict

import numpy as np

def test_correc():
    x = np.arange(1,6)
    #Example 1:
    theta1 = np.array([5, 0])
    assert np.equal(simple_predict(x, theta1), np.array([5., 5., 5., 5., 5.])).all()
    # Do you understand why y_hat contains only 5's here?

def test_better():
    x = np.arange(1,6)
    #Example 1:
    theta1 = np.array([5, 1])
    assert np.equal(simple_predict(x, theta1), np.array([6., 7., 8., 9., 10.])).all()
from prediction import predict_
from cost import cost_, cost_elem_
import numpy as np


x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
theta1 = np.array([[2.], [4.]])
y_hat1 = predict_(x1, theta1)
y1 = np.array([[2.], [7.], [12.], [17.], [22.]])

x2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.],
               [0.6, 6., 60.], [0.8, 8., 80.]])
theta2 = np.array([[0.05], [1.], [1.], [1.]])
y_hat2 = predict_(x2, theta2)
y2 = np.array([[19.], [42.], [67.], [93.]])

x3 = np.array([0, 15, -9, 7, 12, 3, -21])
theta3 = np.array([[0.], [1.]])
y_hat3 = predict_(x3, theta3)
y3 = np.array([2, 14, -13, 5, 12, 4, -19])


def test1():
    # Example 1:
    assert (cost_elem_(y1, y_hat1) == np.array(
        [[0.], [0.1], [0.4], [0.9], [1.6]])).all()


def test2():
    # Example 2:
    assert cost_(y1, y_hat1) == 3.0


def test3():
    # Example 3:
    print(cost_elem_(y2, y_hat2))
    print(np.array(
        [[1.3203125], [0.7503125], [0.0153125], [2.1528125]]))
    assert (cost_elem_(y2, y_hat2) == np.array(
        [[1.3203125], [0.7503125], [0.0153125], [2.1528125]])).all()


def test4():
    # Example 4:
    assert cost_(y2, y_hat2) == 4.238750000000004


def test5():
    # Example 5:
    print(y3)
    print(y_hat3)
    assert cost_(y3, y_hat3) == 4.285714285714286/2


def test6():
    # Example 6:
    assert cost_(y3, y3) == 0.0

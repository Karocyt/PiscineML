#! /usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True


class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, n_cycle=1000):
        self.alpha = alpha
        self.n_cycle = n_cycle
        self.thetas = thetas

    def fit_(self, x, y):
        """
        Description:
        Fits the model to the training dataset contained in x and y.
        Args:
        x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
        Returns:
        new_theta: numpy.ndarray, a vector of dimension 2 * 1.
        None if there is a matching dimension problem.
        Raises:
        This function should not raise any Exception.
        """
        for i in range(self.n_cycle):
            grad = MyLinearRegression.gradient(x, y, self.thetas)
            self.thetas -= grad*self.alpha
        return self.thetas

    @staticmethod
    def gradient(x, y, theta):
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
        #None if x, y and theta do not have compatible dimensions.
        Raises:
        This function should not raise any Exception.
        """
        if (type(x) is not np.ndarray or type(y) is not np.ndarray or
                type(theta) is not np.ndarray or len(x) == 0 or len(theta) == 0):
            return None
        x_bias = MyLinearRegression.add_intercept(x)
        y_pred = np.dot(x_bias, theta)
        y_pred = y_pred.reshape(len(y), 1)
        error = y_pred - y
        error_columns = error.reshape(len(y), 1)
        error_dot_x = np.dot(error_columns.T, x_bias)
        grad = 1/len(x) * error_dot_x
        return grad.reshape(len(theta),)

    @staticmethod
    def add_intercept(x):
        """Adds a column of 1's to the non-empty numpy.ndarray x.
        Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        Returns:
        X as a numpy.ndarray, a vector of dimension m * 2.
        None if x is not a numpy.ndarray.
        None if x is a empty numpy.ndarray.
        Raises:
        This function should not raise any Exception.
        """
        if type(x) is not np.ndarray or len(x) == 0:
            return None
        if len(x.shape) is 1:
            x = x.reshape((x.shape[0], 1))
        intercept = np.ones((x.shape[0], 1))
        return np.append(intercept, x, axis=1)

    def predict_(self, x, thetas=None):
        if thetas is None:
            thetas = self.thetas
        y_pred = np.dot(MyLinearRegression.add_intercept(x), thetas)
        return y_pred.reshape(len(y_pred), 1)

    @staticmethod
    def cost_elem_(y_hat, y):
        """
        Description:
        Calculates all the elements (1/2*M)*(y_pred - y)^2 of the cost function.
        Args:
        y: has to be an numpy.ndarray, a vector.
        y_hat: has to be an numpy.ndarray, a vector.
        Returns:
        J_elem: numpy.ndarray, a vector of dimension (number of the training examples,1).
        None if there is a dimension matching problem between X, Y or theta.
        Raises:
        This function should not raise any Exception.
        """
        if len(y) > 1 and len(y.shape) == 1:
            y_hat = y_hat.flatten()
        if y_hat.shape != y.shape:
            return None
        return (y_hat - y)**2

    @staticmethod
    def cost_(y_hat, y):
        """
        Description:
        Calculates the value of cost function.
        Args:
        y: has to be an numpy.ndarray, a vector.
        y_hat: has to be an numpy.ndarray, a vector.
        Returns:
        J_value : has to be a float.
        None if there is a dimension matching problem between X, Y or theta.
        Raises:
        This function should not raise any Exception.
        """
        if len(y) > 1 and len(y.shape) == 1:
            y_hat = y_hat.flatten()
        if y_hat.shape != y.shape:
            return None
        return np.mean((y_hat - y)**2)

    @staticmethod
    def plot(x, y, y_pred):
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
        y = y.flatten()
        x = x.flatten()
        y_pred = y_pred.flatten()

        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ['Tahoma',
                                           'Lucida Grande', 'Verdana', 'DejaVu Sans']
        plt.plot(x, y_pred, "g--", label="Regression line")

        plt.plot([x, x], [y, y_pred], "r--")
        plt.plot([], [], "r--", label="Error")
        plt.plot(x, y_pred, "gx", label="$S_{predict}(pills)$")

        strue = plt.plot(x, y, 'co', label="$S_{true}(pills)$")

        plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2),
                   loc="lower left", mode="expand", ncol=4, frameon=False)
        plt.grid(True)
        plt.xlabel("Quantity of blue pill (in microcrams)")
        plt.ylabel("Space driving score")
        plt.show()

    def plotcost(self, x, y, bias=np.linspace(-70, 70, num=6)):
        """Plot the cost as function of the weights (theta1)
        for different bias values (theta0)
        """

        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ['Tahoma',
                                           'Lucida Grande', 'Verdana', 'DejaVu Sans']

        t1list = np.linspace(-40, 40, num=100)
        greys = np.linspace(0.3, 0.9, num=len(y))

        colors = [tuple((g, g, g)) for g in greys]
        coloridx = -1
        for i, b in enumerate(bias):
            t0 = self.thetas[0] + b
            coloridx += 1 
            yline = []
            for t1 in t1list:
                yline += [self.cost_(self.predict_(x, np.array([t0, t1])), y)]
            plt.plot(t1list, yline, color=colors[coloridx], label=f"J($\\theta_0$=$c_{i}$,$\\theta_1$)")

        plt.legend(loc="lower right")
        plt.grid(True)
        plt.xlabel("$\\theta_1$")
        plt.ylabel("cost function J($\\theta_0$,$\\theta_1$)")
        plt.show()


if __name__ == "__main__":

    x = np.array([[12.4956442], [21.5007972], [
                 31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [
                 45.7655287], [46.6793434], [59.5585554]])
    lr1 = MyLinearRegression([2, 0.7])
    print(lr1.cost_(lr1.predict_(x), y))
    # Example 0.0:
    20
    print(lr1.predict_(x))

    # Example 0.1:
    print(lr1.cost_elem_(lr1.predict_(x), y))

    # Example 0.2:
    print(lr1.cost_(lr1.predict_(x), y))

    print()

    # Example 1.0:
    lr2 = MyLinearRegression([0, 0])
    print(lr2.cost_(lr2.predict_(x), y))
    lr2.fit_(x, y)
    print(lr2.thetas)
    # Example 1.1:
    print(lr2.predict_(x))

    # Example 1.2:
    print(lr2.cost_elem_(lr2.predict_(x), y))

    # Example 1.3:
    print(lr2.cost_(lr2.predict_(x), y))

    print()

    y = 1.234 + 4.321*x
    # Example 2.0:
    lr3 = MyLinearRegression([0, 0], alpha=0.001, n_cycle=110000)
    print(lr3.cost_(lr3.predict_(x), y))
    lr3.fit_(x, y)
    print(lr3.thetas)
    # Example 2.1:
    print(lr3.predict_(x))

    # Example 2.2:
    print(lr3.cost_elem_(lr3.predict_(x), y))

    # Example 2.3:
    print(lr3.cost_(lr3.predict_(x), y))

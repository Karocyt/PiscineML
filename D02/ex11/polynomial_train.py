#! /usr/bin/env python3
import pandas as pd
import numpy as np
from my_linear_regression import MyLinearRegression as MyLR
from polynomial_model import add_polynomial_features
from minmax import minmax
from matplotlib import pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv("../../ressources/are_blue_pills_magics.csv")
    print(df)
    x, y = df.iloc[:, 1:2].to_numpy(), df.iloc[:, 2:3].to_numpy()

    y = minmax(y)
    res = []
    alpha = [0.1,
             0.1,
             0.1,
             0.1,
             0.1,
             0.1,
             0.1,
             0.1,
             0.1]
    cycles = [10000,
              10000,
              10000,
              100000,
              100000,
              100000,
              100000,
              100000,
              100000]
    
    model = MyLR(alpha=1, n_cycle=1000000) #alpha=alpha[i], n_cycle=cycles[i])
    model.thetas = np.zeros(1)
    for i in range(0, 9):
        model.alpha, model.n_cycle = alpha[i], cycles[i]
        tmp = add_polynomial_features(x, i + 1)
        for j in range(len(tmp[0])):
            tmp[:, j] = minmax(tmp[:, j])
        #thetas = np.zeros(len(tmp[0]) + 1) #np.random.rand(len(tmp[0]) + 1)
        model.thetas = np.zeros(len(tmp[0]) + 1)#np.concatenate([model.thetas, np.zeros(1)])
        model.fit_(tmp, y)
        y_pred = model.predict_(tmp)
        cost = MyLR.cost_(y_pred, y)
        res += [cost]
        print(model.thetas)
        print(cost)
        model.plot(x, y -i*0.5, y_pred - i*0.5)
        lin = np.linspace(x.min(), x.max(), num=100).reshape((100, 1))
        linpol = add_polynomial_features(lin, i + 1)
        for j in range(len(linpol[0])):
            linpol[:, j] = minmax(linpol[:, j])
        plt.plot(lin, model.predict_(linpol)*(y.max() - y.min()) + y.min() -i*0.5, "b--", label="Regression line")

    # tmp = np.concatenate([minmax(x**7), minmax(x**9)], axis=1)
    # model = MyLR(alpha=1, n_cycle=1000000) #alpha=alpha[i], n_cycle=cycles[i])
    # model.fit_(tmp, y)
    # y_pred = model.predict_(tmp)
    # res += [MyLR.cost_(y_pred, y)]
    print(res)
    # model.plot(x, y -10*0.5, y_pred - 10*0.5)

    plt.show()

    for i in range(len(res)):

        plt.plot(res)

    plt.show()

    print(y)
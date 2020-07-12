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
    alpha = [1.,
             0.0001,
             0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    cycles = [50,
              1000000,
              500000, 100, 100, 100, 100, 100, 100]
    for i in range(2):
        tmp = minmax(add_polynomial_features(x, i + 1))
        model = MyLR(np.random.rand(len(tmp[0]) + 1), alpha=alpha[i], n_cycle=cycles[i])
        model.fit_(tmp, y)
        res += [MyLR.cost_(model.predict_(tmp), y)]
        print(model.thetas)
        print(res)

    for i in range(len(res)):

        plt.plot(res)

    plt.show()
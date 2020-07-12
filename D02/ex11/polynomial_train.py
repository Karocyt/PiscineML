#! /usr/bin/env python3
import pandas as pd
from my_linear_regression import MyLinearRegression as MyLR
from polynomial_model import add_polynomial_features
from matplotlib import pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv("../../ressources/are_blue_pills_magics.csv")
    print(df)
    x, y = df.iloc[:, 1:2].to_numpy(), df.iloc[:, 2:3].to_numpy()

    res = []
    alpha = [0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    cycles = [100, 100, 100, 100, 100, 100, 100, 100, 100]
    for i in range(2):
        tmp = add_polynomial_features(x, i + 1)
        print(tmp, y)
        model = MyLR(alpha=alpha[i], n_cycle=cycles[i])
        model.fit_(tmp, y)
        res += [MyLR.cost_(model.predict_(tmp), y)]

    print(res)
    for i in range(len(res)):

        plt.plot(res)

    plt.show()
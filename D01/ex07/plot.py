#! /usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from mylinearregression import MyLinearRegression as MyLR


# import  matplotlib.font_manager
# flist = matplotlib.font_manager.get_fontconfig_fonts()
# names = [print(matplotlib.font_manager.FontProperties(fname=fname).get_name()) for fname in flist]
# print("fonts:", "\n".join(flist))

data = pd.read_csv("../../ressources/are_blue_pills_magics.csv")
Xpill = np.array(data["Micrograms"]).reshape(-1, 1)
Yscore = np.array(data["Score"]).reshape(-1, 1)
linear_model1 = MyLR(np.array([[89.0], [-8]]))
linear_model2 = MyLR(np.array([[89.0], [-6]]))
Y_model1 = linear_model1.predict_(Xpill)
Y_model2 = linear_model2.predict_(Xpill)
print(Xpill)
print(linear_model1.cost_(Y_model1, Yscore))
# 57.60304285714282
print(mean_squared_error(Yscore, Y_model1))
# 57.603042857142825
MyLR.plot(Xpill, Yscore, Y_model1)
linear_model1.plotcost(Xpill, Yscore)

print(linear_model2.cost_(Y_model2, Yscore))
# 232.16344285714285
print(mean_squared_error(Yscore, Y_model2))
# 232.16344285714285
MyLR.plot(Xpill, Yscore, Y_model2)

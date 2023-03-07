import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

# parameter_space = {
    #     "min_samples_leaf": range(6, 9, 1),
    #     "min_samples_split": range(2, 7, 2),
    #     "max_depth": range(6, 16, 4),
    #     "max_leaf_nodes": [None] + list(range(20, 70, 20)),
    #     "n_estimators": range(10, 131, 60),
    #     "max_features": ['sqrt', 'log2'] + list(range(1, 4, 1)),
    #     "max_samples": [None, 0.55, 0.6, 0.65]
    # }

    # 数据集处理
data = pd.read_excel("郑州处理后.xlsx")
rng = np.random.RandomState(1)
x = data.iloc[0:400, 0:5]
y = data.iloc[0:400, 5]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=22)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

RF_0 = RandomForestRegressor(random_state=22)
# grid_RF_0 = GridSearchCV(RF_0, parameter_space, n_jobs=15)
RF_0.fit(x_train, y_train)
# print(RF_0.best_score_)
# print(RF_0.score(x_train, y_train), grid_RF_0.score(x_test, y_test))

y_pr = RF_0.predict(x_test)

plt.plot(range(1, 41), y_pr, "r-", label='predict')
plt.legend()
plt.plot(range(1, 41), y_test, "black", label='true')
plt.legend()
# plt.ylim((0, 24))
plt.savefig(r'RF2.jpg')
plt.show()

# print(grid_RF_0.best_params_)

R2 = RF_0.score(x_test, y_test)
MSE = mean_squared_error(y_test, y_pr)
RMSE = np.sqrt(mean_squared_error(y_test, y_pr))
MAE = mean_absolute_error(y_test, y_pr)

print(R2, MSE, RMSE, MAE)



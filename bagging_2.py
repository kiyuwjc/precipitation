import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

parameter_space = {
    "min_samples_leaf": range(6, 9, 1),
    "min_samples_split": range(2, 7, 2),
    "max_depth": range(6, 16, 4),
    "max_leaf_nodes": [None] + list(range(20, 70, 20)),
    "n_estimators": range(10, 131, 60),
    "max_features": ['sqrt', 'log2'] + list(range(1, 4, 1)),
    "max_samples": [None, 0.55, 0.6, 0.65]
}
parameter = {
    "min_samples_leaf": 7,
    "min_samples_split": 2,
    "max_depth": 6,
    "max_leaf_nodes": None,
    "n_estimators": 10,
    "max_features": 'sqrt',
    "max_samples": None
}

# 数据集处理
city = '西安'
data = pd.read_excel("../处理后文件/{}处理后.xlsx".format(city))
rng = np.random.RandomState(1)
x = data.iloc[0:400, 0:5]
y = data.iloc[0:400, 5]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)

RF_0 = RandomForestRegressor(min_samples_leaf=7,
                             min_samples_split=2,
                             max_depth=6,
                             max_leaf_nodes=None,
                             n_estimators=10,
                             max_features='sqrt',
                             max_samples=None)
# grid_RF_0 = GridSearchCV(RF_0, parameter_space, n_jobs=15)

RF_0.fit(x_train, y_train)

# print(RF_0.best_score_)
print(RF_0.score(x_train, y_train), RF_0.score(x_test, y_test))

y_pr = RF_0.predict(x_test)

plt.plot(range(1, 41), y_pr, "red", label='predict',linestyle='dashed')
plt.xlabel('Sample Number')
plt.ylabel('Precipitation')
plt.legend()
plt.plot(range(1, 41), y_test, "black", label='true')
plt.legend()
plt.savefig(r'../图片/{}/RF.jpg'.format(city))
plt.show()

# print(RF_0.best_params_)

R2 = RF_0.score(x_test, y_test)
MSE = mean_squared_error(y_test, y_pr)
RMSE = np.sqrt(mean_squared_error(y_test, y_pr))
SD = np.std(y_test)
RPD = SD / RMSE
MAE = mean_absolute_error(y_test, y_pr)
print(R2, RPD, RMSE, MAE)

with open('../图片/{}/{}.csv'.format(city, city), mode='x') as f:
    writer = csv.writer(f)
    writer.writerow(['RF', RPD])




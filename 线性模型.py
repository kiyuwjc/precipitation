import csv

from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

city = '西安'
data = pd.read_excel("../处理后文件/{}处理后.xlsx".format(city))
rng = np.random.RandomState(1)
x = data.iloc[0:400, 0:5]
y = data.iloc[0:400, 5]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

li = SGDRegressor()
li.fit(x_train, y_train)
y_pr = li.predict(x_test)
#print(li.score(x_train, y_train), li.score(x_test, y_test))

plt.plot(range(1, 41), y_pr, "red", label='predict',linestyle='dashed')
plt.xlabel('Sample Number')
plt.ylabel('Precipitation')
plt.legend()
plt.plot(range(1, 41), y_test, "black", label='true')
plt.legend()
plt.savefig(r'../图片/{}/LI.jpg'.format(city))
plt.show()

R2 = li.score(x_test, y_test)
MSE = mean_squared_error(y_test, y_pr)
RMSE = np.sqrt(mean_squared_error(y_test, y_pr))
SD = np.std(y_test)
RPD = SD / RMSE
MAE = mean_absolute_error(y_test, y_pr)
print(R2, RPD, RMSE, MAE)

with open('../图片/{}/{}.csv'.format(city, city), mode='a') as f:
    writer = csv.writer(f)
    writer.writerow(['LI', RPD])
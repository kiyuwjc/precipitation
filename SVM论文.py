import csv

import numpy as np
import pylab as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
city = '西安'
data = pd.read_excel("../处理后文件/{}处理后.xlsx".format(city))

x = data.iloc[0:400, 0:5]
# print(x)
y = data.iloc[0:400, 5]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
np.random.seed(123)

# model = SVR(degree=3, kernel='poly', gamma=1)
model = LinearSVR(epsilon=1.5, random_state=22)

# model = SVR(kernel='linear', gamma=1)
# print(model)
model.fit(x_train, y_train)


y_pr = model.predict(x_test)
print(model.score(x_train, y_train), model.score(x_test, y_test))

R2 = model.score(x_test, y_test)
MSE = mean_squared_error(y_test, y_pr)
RMSE = np.sqrt(mean_squared_error(y_test, y_pr))
SD = np.std(y_test)
RPD = SD / RMSE
MAE = mean_absolute_error(y_test, y_pr)

with open('../图片/{}/{}.csv'.format(city, city), mode='a') as f:
    writer = csv.writer(f)
    writer.writerow(['SVM', RPD])

print(R2, RPD, RMSE, MAE)
plt.plot(range(1, 41), y_pr, "red", label='predict',linestyle='dashed')
plt.xlabel('Sample Number')
plt.ylabel('Precipitation')
plt.legend()
plt.plot(range(1, 41), y_test, "black", label='true')
plt.legend()
plt.savefig(r'../图片/{}/SVM.jpg'.format(city))
plt.show()



# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:04:53 2020

@author: bala.vivek
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


dat = pd.read_csv('C:/Users/bala.vivek/Desktop/BostonHousing.csv')

matrix= dat.corr().round(2)


sns.heatmap(data=matrix, annot = True)

X = pd.DataFrame(np.c_[dat['lstat'], dat['rm'],dat['ptratio']], columns =['lstat','rm','ptratio'])
Y = dat['medv']
 

X_train, X_test,Y_train, Y_test = train_test_split(X , Y, test_size = 0.2, random_state = 0)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
#X_test= X_test.astype('float64')
#Y_test= Y_test.astype('float64')

"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
Y_train = sc_x.fit_transform(Y_train)
"""
dat.isnull().sum()
#X_train.reshape(1, -1)
#X_test.reshape(1, -1)
#X_train= X_train.astype('float64')
#Y_train= Y_train.astype('float64')

clas = LinearRegression()

clas.fit(X_train, Y_train)


#model evaluation

y_pre = clas.predict(X_train)
rmse = np.sqrt(mean_squared_error(Y_train, y_pre))
r2 = r2_score (Y_train, y_pre)
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")


y_pre_test = clas.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(Y_test, y_pre_test))
r2_test = r2_score (Y_test, y_pre_test)

print('RMSE_test is {}'.format(rmse_test))
print('R2_test score is {}'.format(r2_test))
print("\n")











# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 22:20:17 2019

@author: bala.vivek 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data  = pd.read_csv('C:/Users/bala.vivek/Downloads/Machine Learning A-Z New/Part 2 - Regression/Section 8 - Decision Tree Regression/Position_Salaries.csv')

X=data.iloc[:, 1:2].values
Y=data.iloc[:, 2].values


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,Y)
y_pred = regressor.predict([[6.5]])


X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,Y,color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('sal vs exp')
plt.xlabel('exp')
plt.ylabel('sal')
plt.show()



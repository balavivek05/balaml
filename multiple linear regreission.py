# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 21:37:42 2019

@author: bala.vivek
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('C:/Users/bala.vivek/Desktop/50_Startups.csv')
#print(dataset)
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, 1].values

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3]) 
onehotencoder =OneHotEncoder(columntransformer =[3])
X = onehotencoder.fit_transform(X).toarray()

X = X[: , 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
pred = regressor.predict(X_test)

plt.scatter(X_train, Y_train, color = 'red' )

plt.plot(X_train, regressor.predict(X_train), color = 'blue')



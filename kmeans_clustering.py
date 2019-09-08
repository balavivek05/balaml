# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 17:08:59 2019

@author: bala.vivek
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('C:/Users/bala.vivek/Downloads/Machine Learning A-Z New/Part 4 - Clustering/Section 24 - K-Means Clustering/Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values


## in the elbow method calculate the sum of squares and it says done by getting the inertia of the kmeans algorithem
from sklearn.cluster import KMeans
wscc =[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wscc.append(kmeans.inertia_)
plt.title('elbow method')
plt.plot(range(1,11), wscc)
plt.xlabel('number of cluster')
plt.ylabel('wcss')
plt.show()

kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter=500, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

#c = ['red', 'green', 'cyan', 'magenta']
#for i in range (0, 4):
   # plt.scatter(X[y_kmeans == (i 0)], X[y_kmeans = (i, 1)], s = 200, c = [i])
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 20, c = 'green', label = 'cluster1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 20, c = 'red', label = 'cluster2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 20, c = 'blue', label = 'cluster3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 20, c = 'cyan', label = 'cluster4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 20, c = 'magenta', label = 'cluster5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s= 400, c = 'yellow', label = 'centroid')
     
plt.show()
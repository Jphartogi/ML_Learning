# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 21:39:37 2019

@author: Joshua Phartogi
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing

df = pd.read_csv('teleCust1000t.csv')
df.head()
a = df['custcat'].value_counts()


X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
print(X[0:5])


#Data Standardization give data zero mean and unit variance, it is good practice, especially for algorithms such as KNN which is based on distance of cases:
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print('after normalizing',X[0:5])

y = df['custcat'].values


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

from sklearn.neighbors import KNeighborsClassifier
k = 5
neighbour = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
a = neighbour.predict(X_test)
print('real value y:',y_test[0:5])
print('predicted value y:',a[0:5])
print(a.shape)
xs = np.linspace(0, 200, num=200)
print(xs.shape)
plt.scatter(xs,a)
plt.scatter(xs,y_test)



from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neighbour.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, a))


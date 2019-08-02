# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 22:50:38 2019

@author: User
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("drug200.csv",delimiter=",")

X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

y = data["Drug"]


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.5,random_state=3)

drugTree = DecisionTreeClassifier(criterion="entropy",max_depth= 4)
drugTree.fit(x_train,y_train)

prediction = drugTree.predict(x_test)

print(prediction)
print(y_test)

from sklearn import metrics
import matplotlib.pyplot as plt

print("Decision Tree Accuracy: ",metrics.accuracy_score(y_test,prediction))





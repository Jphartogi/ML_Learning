# -*- coding: utf-8 -*-
"""
Created on Fri May 24 22:51:59 2019

@author: Joshua Phartogi
"""
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
    

""" type of non linear model """
# quadratic
x = np.arange(-5.0, 5.0, 0.1)

##You can adjust the slope and intercept to verify the changes in the graph

y = np.power(x,2)
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

# exponential

X = np.arange(-5.0, 5.0, 0.1)

##You can adjust the slope and intercept to verify the changes in the graph

Y= np.exp(X)

plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

# logaritmic

X = np.arange(-5.0, 5.0, 0.1)

Y = np.log(X)

plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

# sigmoidal 

X = np.arange(-5.0, 5.0, 0.1)


Y = 1-4/(1+np.power(3, X-2))

plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

""" this is the part for non-linear regression with dataset """

def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y



data_set = pd.read_csv("china_gdp.csv")

plt.figure(figsize=(8,5))
x_data, y_data = (data_set["Year"].values, data_set["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

#guessing the beta
beta_1 = 0.10
beta_2 = 1990.0

#logistic function
Y_pred = sigmoid(x_data, beta_1 , beta_2)


# Lets normalize our data
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)


# try to find the best parameter
from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)
#print the final parameters
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

Y_optimized = sigmoid(xdata,popt[0],popt[1])

#plot initial prediction against datapoints

plt.plot(xdata, ydata, 'ro')
plt.plot(xdata, Y_optimized)




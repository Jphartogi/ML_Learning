# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:50:32 2019

@author: User
"""
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

celsius_q= np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a= np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

a = tf.keras.layers.Dense(units=1,input_shape=[1])
model = tf.keras.Sequential([a])

model.compile(optimizer=tf.keras.optimizers.Adam(0.1),loss='mean_squared_error')

history = model.fit(celsius_q,fahrenheit_a,epochs=1000,verbose=False)
print('Finished training model')


plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])

print(model.predict([100]))
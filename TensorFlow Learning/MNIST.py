# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 08:37:42 2019

@author: User
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import math
import tqdm
import tqdm.auto
import matplotlib.pyplot as plt



#fashion MNIST dataset

tqdm.tqdm = tqdm.auto.tqdm
#tf.enable_eager_execution()

dataset,metadata = tfds.load('fashion_mnist',as_supervised=True,with_info=True)
train_dataset,test_dataset= dataset['train'],dataset['test']
print('finished taking dataset')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']

num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))

#create a normalize function change from 0-255 to 0-1
def normalize(images,labels):
    images = tf.cast(images,tf.float32)
    images /= 255
    return images,labels

train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)


model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=[28,28,1]),
        tf.keras.layers.Dense(128,activation=tf.nn.relu),
        tf.keras.layers.Dense(10,activation=tf.nn.softmax)])
    
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# USE FOR TRAINING ANY MODEL
BATCH_SIZE=32
train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model.fit(train_dataset,epochs=5,steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))

test_loss,test_accuracy = model.evaluate(test_dataset,steps=math.ceil(num_test_examples/BATCH_SIZE))
print('accuracynya:',test_accuracy)


for image,label in test_dataset.take(1):
    image = image.numpy()
    label = label.numpy()
    

prediction = model.predict(image)
print('the prediciton = ',np.argmax(prediction[4]))
print('the real data = ',label[4])

    
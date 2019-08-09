# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:21:21 2019

@author: JosephMcParland
"""

import tensorflow as tf
import datetime as dt
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard

#%%
LOG_DIR = r'C:/Users/JosephMcParland/Documents/mnest/run_'

def get_log_dir():
    now = dt.datetime.now().strftime('%Y%m%d_%H%M')
    my_dir = LOG_DIR +now
    return my_dir

#%%
    
mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train =X_train/255
X_test =X_test/255
    

#%%

fig, ax = plt.subplots(4,4)
for i in range(16):
    curr_ax = ax.ravel()[i]
    curr_ax.imshow(X_train[i], cmap=plt.cm.gray)
    curr_ax.set_title(y_train[i])
    
#%%

tf.keras.backend.clear_session()

model = keras.Sequential([
        keras.layers.Flatten(input_shape = (28, 28)),
        keras.layers.Dense(128, activation= tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
    
model.compile(optimizer ='sgd',
              loss ='sparse_categorical_crossentropy',
              metrics = ['accuracy'])

tensorboard = TensorBoard(get_log_dir())

#%%

model.fit(X_train, y_train, epochs =10,validation_data=[X_test, y_test], callbacks = [tensorboard])
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 14:57:14 2019

@author: JosephMcParland
"""

#%%
import tensorflow as tf
import datetime as dt


#%%
# Logging
LOG_DIR =  r'C:\Users\JosephMcParland\Documents\junkyard/run_'
LOG_DIR += dt.datetime.now().strftime('%Y%m%d_%H%M')

#%%
tf.reset_default_graph()

a = tf.Variable(1, name = 'a')
b = tf.Variable(1, name = 'b')
c = tf.Variable(1, name = 'c')
d = tf.Variable(1, name = 'd')
j = 3*(a*b +c)

# Initialiation

file_writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())
init = tf.global_variables_initializer()

#%%

with tf.Session() as sess:
    init.run()
    print(j.eval())
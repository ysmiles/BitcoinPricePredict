# Import Libraries
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

# TensorFlow
import tensorflow as tf
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn

# Transmitter
from transmitter import getBCdata
import psutil

# MongoDB
from pymongo import MongoClient


startdate = '2017-01-01'
enddate = '2017-11-18'

rng = pd.date_range(start=startdate, end=enddate, freq='D')
ts = getBCdata(startdate, enddate)
# ts = pd.Series([v for (k, v) in ts.items()])
TS = np.array([v for (k, v) in ts.items()])

num_periods = 20
f_horizon = 1  # forecast horizon, one period into the future

x_data = TS[:(len(TS) - (len(TS) % num_periods))]
x_batches = x_data.reshape(-1, 20, 1)

y_data = TS[1:(len(TS) - (len(TS) % num_periods)) + f_horizon]
y_batches = y_data.reshape(-1, 20, 1)
# print(len(x_batches))
print('x batches shape', x_batches.shape)
# print(x_batches[0:2])
# print(y_batches[0:1])
print('y batches shape', y_batches.shape)


def test_data(series, forecast, num_periods):
    test_x_setup = TS[-(num_periods + forecast):]
    testX = test_x_setup[:num_periods].reshape(-1, 20, 1)
    testY = TS[-(num_periods):].reshape(-1, 20, 1)
    return testX, testY


X_test, Y_test = test_data(TS, f_horizon, num_periods)

print('Test shape', X_test.shape)
# print(X_test)

# We didn't have any previous graph objects running, but this would reset the graphs
tf.reset_default_graph()

# setted before
# num_periods = 20 # number of periods per vector we are using to predict one period ahead
inputs = 1  # number of vectors submitted
hidden = 500  # number of neurons we will recursively work through, can be changed to improve accuracy
output = 1  # number of output vectors

# create variable objects
X = tf.placeholder(tf.float32, [None, num_periods, inputs])
y = tf.placeholder(tf.float32, [None, num_periods, output])

basic_cell = tf.contrib.rnn.BasicRNNCell(
    num_units=hidden, activation=tf.nn.relu)  # create our RNN object
rnn_output, states = tf.nn.dynamic_rnn(
    basic_cell, X, dtype=tf.float32)  # choose dynamic over static

learning_rate = 0.001  # small learning rate so we don't overshoot the minimum

# change the form into a tensor
stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])
# specify the type of layer (dense)
stacked_outputs = tf.layers.dense(stacked_rnn_output, output)
# shape of results
outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])

# define the cost function which evaluates the quality of our model
loss = tf.reduce_sum(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(
    learning_rate=learning_rate)  # gradient descent method
# train the result of the application of the cost_function
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()  # initialize all the variables

epochs = 1000  # number of iterations or training cycles, includes both the FeedFoward and Backpropogation

# sample usage of cpu and memory
# psutil.cpu_percent()
# psutil.virtual_memory().used

cpus = np.array([])
mems = np.array([])
mses = np.array([])

cpuho = tf.placeholder(tf.float32)
memho = tf.placeholder(tf.float32)
lossho = tf.placeholder(tf.float32)

tf.summary.scalar('loss', lossho)
tf.summary.scalar('cpu-usage', cpuho)
tf.summary.scalar('memory-usage', memho)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('./', sess.graph)
    init.run()
    for ep in range(epochs):
        sess.run(training_op, feed_dict={X: x_batches, y: y_batches})

        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().used
        mse = loss.eval(feed_dict={X: x_batches, y: y_batches})

        cpus = np.append(cpus, cpu)
        mems = np.append(mems, mem)
        mses = np.append(mses, mse)
        sumary = sess.run(merged, feed_dict={
                          lossho: mse, cpuho: cpu, memho: mem})
        train_writer.add_summary(sumary, ep)

        if ep % 100 == 0:
            # mean square error
            print(ep + 100, "\tMSE:", mse)

    y_pred = sess.run(outputs, feed_dict={X: X_test})
    # print(y_pred)

epochs = np.arange(epochs)

# Prepare to save data
print("prepare to save data")

# database client
client = MongoClient()

# will saved processed data into this database collection
collection = client.bitcoin_predict.parameters

# remove old parameters
collection.remove()

collection.insert_many([
    {
        'name': 'Y_test',
        'val': Y_test.tolist()
    },
    {
        'name': 'y_pred',
        'val': y_pred.tolist()
    },
    {
        'name': 'epochs',
        'val': epochs.tolist()
    },
    {
        'name': 'mses',
        'val': mses.tolist()
    },
    {
        'name': 'cpus',
        'val': cpus.tolist()
    },
    {
        'name': 'mems',
        'val': mems.tolist()
    },
])

print("data saved")
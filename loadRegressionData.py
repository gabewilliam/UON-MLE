import numpy as np
import tensorflow as tf
import math
import logging
logging.basicConfig(level=logging.DEBUG)
import matplotlib.pyplot as plt
from pandas import DataFrame
import sklearn.model_selection as skm

import pandas as pd

x = pd.read_csv("predx_for_regression.csv", header=0)
x_array = np.asarray(x, dtype = "float")
x_arrayt=x_array

y = pd.read_csv("predy_for_regression.csv", header=0)
y_array = np.asarray(y, dtype = "float")
y_arrayt=y_array

whole_data=np.concatenate((x_arrayt, y_arrayt),axis=1)


angle = pd.read_csv("angle.csv", header=0)
angle_array = np.asarray(angle, dtype = "float")

#Network parameters
n_hidden1 = 72
n_hidden2 = 48
n_hidden3 = 24
n_input = 98
n_output = 3

#Learning parameters
learning_constant = 0.006
number_epochs = 10000

#Defining the input and the output
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])

#DEFINING WEIGHTS AND BIASES

#First hidden layer
b1 = tf.Variable(tf.random_normal([n_hidden1]))
w1 = tf.Variable(tf.random_normal([n_input, n_hidden1]))
#Second hidden layer
b2 = tf.Variable(tf.random_normal([n_hidden2]))
w2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]))
#Third hidden layer
b3 = tf.Variable(tf.random_normal([n_hidden3]))
w3 = tf.Variable(tf.random_normal([n_hidden2, n_hidden3]))
#Output layer
b4 = tf.Variable(tf.random_normal([n_output]))
w4 = tf.Variable(tf.random_normal([n_hidden3, n_output]))

#The incoming data given to the
#network is input_d
def multilayer_perceptron(input_d):
    
    #Task of neurons of first hidden layer
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_d, w1), b1))
    #Task of neurons of second hidden layer
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w2), b2))
    #Task of neurons of output layer
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, w3), b3))
    #Task of neurons of output layer
    out_layer = tf.add(tf.matmul(layer_3, w4),b4)
    
    return out_layer

#Create model
neural_network = multilayer_perceptron(X)

#Define loss and optimizer
loss_op = tf.reduce_mean(tf.math.squared_difference(neural_network,Y))

#loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neural_network,labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_constant).minimize(loss_op)

#Initializing the variables
init = tf.global_variables_initializer()
#Create a session


label=angle_array#+1e-50-1e-50

batch_x=(whole_data-200)/2000 #Normalisation

batch_y=angle_array

# Training split
batch_x_train=batch_x[0:275712, :]
batch_y_train=batch_y[0:275712, :]

label_train=label

# Testing split
batch_x_test=batch_x[275713:,:]
batch_y_test=batch_y[275713:,:]

label_test=label

with tf.Session() as sess:
    sess.run(init)
    #Training epoch
    for epoch in range(number_epochs):

        #Run the optimizer feeding the network with the batch
        sess.run(optimizer, feed_dict={X: batch_x_train, Y: batch_y_train})
        
        #Display the epoch
        if epoch % 100 == 0 and epoch>10:
            print("Epoch:", '%d' % (epoch))
            print("Loss:", loss_op.eval({X: batch_x_train, Y: batch_y_train}) )


    # Test model
    pred = (neural_network)
    accuracy=tf.keras.losses.MSE(pred,Y)

    print("Accuracy:", np.square(accuracy.eval({X: batch_x_test, Y: batch_y_test})).mean() )
    print("Prediction:", pred.eval({X: batch_x_test}))

    

    output=neural_network.eval({X: batch_x_train})
    
    df = DataFrame(output)

    export_csv = df.to_csv ('output.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

    print (df)
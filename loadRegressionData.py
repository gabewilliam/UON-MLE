import numpy as np
import tensorflow as tf
import math
import logging
logging.basicConfig(level=logging.DEBUG)
import matplotlib.pyplot as plt
from pandas import DataFrame

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
angle_arrayt=angle_array.transpose()


#Network parameters
n_hidden1 = 147
n_hidden2 = 74
n_hidden3 = 50
n_input = 98
n_output = 1
#Learning parameters
learning_constant = 0.002
number_epochs = 10000
batch_size = 10000

#Defining the input and the output
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])

#DEFINING WEIGHTS AND BIASES

#Biases first hidden layer
b1 = tf.Variable(tf.random_normal([n_hidden1]))
#Biases second hidden layer
b2 = tf.Variable(tf.random_normal([n_hidden2]))
#Biases output layer
b3 = tf.Variable(tf.random_normal([n_hidden3]))

b4 = tf.Variable(tf.random_normal([n_output]))


#Weights connecting input layer with first hidden layer
w1 = tf.Variable(tf.random_normal([n_input, n_hidden1]))
#Weights connecting first hidden layer with second hidden layer
w2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]))
#Weights connecting second hidden layer with output layer
w3 = tf.Variable(tf.random_normal([n_hidden2, n_hidden3]))

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

batch_x=(whole_data-200)/2000
temp=np.array([angle_array[:,0]])
batch_y=temp.transpose()

batch_x_train=batch_x
batch_y_train=batch_y

batch_x_test=batch_x
batch_y_test=batch_y

label_train=label

label_test=label


with tf.Session() as sess:
    sess.run(init)
    #Training epoch
    for epoch in range(number_epochs):
        #Get one batch of images
        #batch_x, batch_y = mnist.train.next_batch(batch_size)
        
        #print (batch_x)
        #print ((batch_x.shape))
        #Run the optimizer feeding the network with the batch
        sess.run(optimizer, feed_dict={X: batch_x_train, Y: batch_y_train})
        #Display the epoch
        if epoch % 100 == 0 and epoch>10:
            print("Epoch:", '%d' % (epoch))
            print("Accuracy:", loss_op.eval({X: batch_x_train, Y: batch_y_train}) )


    # Test model
    pred = (neural_network)  # Apply softmax to logits
    accuracy=tf.keras.losses.MSE(pred,Y)
    print("Accuracy:", np.square(accuracy.eval({X: batch_x_train, Y: batch_y_train})).mean() )
    #tf.keras.evaluate(pred,batch_x)

    print("Prediction:", pred.eval({X: batch_x_train}))
    print(batch_y)

    output=neural_network.eval({X: batch_x_train})
    plt.plot(batch_y_train, 'r', output, 'b')
    plt.ylabel('some numbers')
    plt.show()
   

    plt.plot(batch_y_train[30000:300020], 'r', output[30000:300020], 'b')
    plt.ylabel('some numbers')
    plt.show()
    
    print(batch_y_train[30000:300020])
    print(output[30000:300020])
    
    df = DataFrame(output)

    export_csv = df.to_csv ('output.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

    print (df)

    
    #correct_prediction = tf.math.subtract((pred), (Y))
    # Calculate accuracy
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #print("Accuracy:", accuracy.eval({X: batch_x, Y: batch_y}))
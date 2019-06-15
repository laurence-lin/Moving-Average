import tensorflow as tf
import numpy as np

'''
Define simple neural network arthitecture
'''

# Normalization: (x - mean) / var => [mean = 0]
def Normalization(train_set, test_set):
    '''
    Define normalization to training set and test set
    train_set: 1D array of [batchsize, features]
    test_set: 1D array of [batchsize, features]
    return: same size 1D array of train set & test set after normalization
    '''
    mean = np.mean(train_set[:])
    var = np.var(train_set[:])
    train_set[:] = (train_set[:] - mean) / var
    
    mean_test = np.mean(test_set[:])
    var_test = np.var(test_set[:])
    test_set[:] = (test_set[:] - mean_test) / var_test
        
    return train_set, test_set
       
def batch_norm(layer_input):
    '''
    Define batch normalization btw each hidden layers
    layer_input: 2D tensor [batchsize, features]
    return: same size as input after batch normalization
    '''
    epsilon = 0.001 # to avoid BN divided by zero
    mean, var = tf.nn.moments(layer_input, axes = 0) # compute along batch dimension
    beta = tf.get_variable('beta', shape = [1], initializer = tf.constant_initializer(0.0))
    gamma = tf.get_variable('gamma', shape = [1], initializer = tf.constant_initializer(1.0))
    bn = tf.nn.batch_normalization(layer_input, mean, var, beta, gamma, epsilon)
    return bn
    
def hidden_bn_relu(layer_input, out_feature):
    '''
    Computation of hidden layer => batch normalization => activation function
    '''
    features = layer_input.get_shape().as_list()[-1]
    initializer = tf.truncated_normal_initializer(stddev = 0.01)
    w1 = tf.get_variable('weight', shape = [features, out_feature], initializer = initializer)
    b1 = tf.get_variable('bias', shape = [out_feature], initializer = tf.constant_initializer(0.0))
    layer_out = batch_norm(tf.matmul(layer_input, w1) + b1)
    layer_out = tf.nn.relu(layer_out)
    
    return layer_out
    
    
# Prediction network
n1_hidden = 10
n2_hidden = 5
n3_hidden = 3
n_out = 1
def inference(input_x, reuse):
    
    layer = []
    with tf.variable_scope('hidden_1', reuse = reuse):
        h1_out = hidden_bn_relu(input_x, n1_hidden)
        layer.append(h1_out)
        
    with tf.variable_scope('hidden_2', reuse = reuse):
        h2_out = hidden_bn_relu(h1_out, n2_hidden)
        layer.append(h2_out)
        
    with tf.variable_scope('hidden_3', reuse = reuse):
        h3_out = hidden_bn_relu(h2_out, n3_hidden)
        layer.append(h3_out)
        
    with tf.variable_scope('output_layer', reuse = reuse): # no need BN in output layer
        initializer = tf.truncated_normal_initializer(stddev = 0.01)
        w_out = tf.get_variable('weight', shape = [n3_hidden, n_out], initializer = initializer)
        b_out = tf.get_variable('bias', shape = [n_out], initializer = tf.constant_initializer(0.0))
        output = tf.matmul(h3_out, w_out) + b_out
        layer.append(output)
    
    return layer[-1]



    
    
    

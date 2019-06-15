import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import network as net

'''
This code is to test if moving average could help remove noise in data, and improve model predict performance.
Use stock prices prediction for testing, test if moving average smoothing effect helps.
'''

'''
IBM stock price data, contains 360 data samples
'''
ibm = pd.read_csv('ibm_stock.csv')
ibm = np.array(ibm.iloc[:, 1])

'''
Data 2: S & P 500 stock price data, contains 610000 data samples
'''
sp500 = pd.read_csv('all_stocks_5yr.csv')
stock_price = np.array(sp500.iloc[:, 1:5])

total_size = 610000
croped_size = 150000  # don't train all dataset at first
#close_price = stock_price[0:croped_size, 3]
data = stock_price[0:croped_size, :]

# Data preprocessing
# 80% as training set, 20% as test set
train_size = int(np.floor(croped_size * 0.8))
test_size = int(np.floor(croped_size * 0.2))
train_end = train_size
test_start = train_end

train_set = data[0:train_end, :]
test_set = data[test_start:, :]

def fillnan(matrix):
    '''
    Fill in NaN missing values
    '''
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isnan(matrix[i, j]):
                matrix[i, j] = np.mean([matrix[i-1, j], matrix[i+1, j]])
                
    return matrix

def Denormalize(y):
    '''
    Denormalize the output of network for prediction comparison
    '''
    mean = np.mean(y)
    var = np.var(y)
    denorm = (y * var) + mean
    
    return denorm

train_set = fillnan(train_set)
test_set = fillnan(test_set)

#train_set, test_set = net.Normalization(train_set, test_set)
print(train_set[0, :])
x_train = train_set[:-1, :] # high price, low price, open prices of today close price
y_train = train_set[1:, 0] # predict open price of tomorrow
x_test = test_set[:-1, :]
y_test = test_set[1:, 0]

features = train_set.shape[1]
num_of_days = 1 # number of past days to predict future stock price
x = tf.placeholder(tf.float32, [None, features * num_of_days])
y = tf.placeholder(tf.float32, [None])

# Create train & test data pipeline
total_train_data = x_train.shape[0]
batch_size = 256
epochs = 10
learn_rate = 0.05    
total_batch = int(np.floor(total_train_data/batch_size)) + 1

predict_y = net.inference(x, True)
predict_y = tf.squeeze(predict_y) #remove size 1 dimension
mse = tf.losses.mean_squared_error(labels = y, predictions = predict_y)
with tf.variable_scope('opt', reuse = True):
     train_op = tf.train.AdamOptimizer(learn_rate).minimize(mse)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    print(tf.global_variables())
    print('Train data size: ', x_train.shape[0])
    print('Total batch:', total_batch)
    print('Start training...')
    
    loss_curve = []
    for iterate in range(epochs):
        
        mean_err = 0
        for batch in range(total_batch):
            offset = batch * batch_size
            if batch == (total_batch - 1):
                x_batch = x_train[offset:, :]
                y_batch = y_train[offset:]
            else:
                x_batch = x_train[offset:offset + batch_size, :]
                y_batch = y_train[offset:offset + batch_size]
            
            _, err = sess.run([train_op, mse], {x:x_batch, y:y_batch})
            
            mean_err += err
                
        mean_err /= total_batch
        print('Epoch: ', iterate + 1, 'MSE = ', mean_err)
        loss_curve.append(mean_err)
        
        validate_y = sess.run(predict_y, {x:x_test})
        plt.figure(iterate)
        plt.title('Predict output')
        plt.plot(validate_y, label = 'predict output')
        plt.legend(loc = 'upper right')
        
        '''weight = [v for v in tf.global_variables() if v.name == 'hidden_1/weight:0']    
        weight = np.array(sess.run(weight))
        weight1 = weight
        
        print('weight:', weight1)'''
        
            
        '''if iterate % (epochs/2) == 0:
            validate_y = sess.run(predict_y, {x:x_test})
            plt.figure(iterate)
            plt.title('Predict output')
            plt.plot(validate_y)
    
            plt.figure(1000)
            plt.title('Test output')
            plt.plot(y_test)'''

    plt.figure(iterate)
    plt.title('Test output')
    plt.plot(y_test, label = 'test output')
    plt.legend(loc = 'upper right')
            
    plt.figure(100000)
    plt.title('Loss curve')
    plt.plot(loss_curve)
       
    plt.show()
























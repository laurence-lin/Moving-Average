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
data = stock_price[0:croped_size, 0]

# Data preprocessing
# 80% as training set, 20% as test set
train_size = int(np.floor(croped_size * 0.8))
test_size = int(np.floor(croped_size * 0.2))
train_end = train_size
test_start = train_end
train_set = data[0:train_end]
test_set = data[test_start:]

def create_pipeline(x, predict_slide = 10):
    '''
    x: 1D time series data sequence
    predict_slide: how many days to predict future day
    Create dataset pipeline: 1~n days data => predict n + 1 day data
    return: 
    '''
    data = x[:-1]
    x_data = []
    y_data = []
    
    for i in range(0, len(data) - predict_slide):
        x_data.append(data[i:i + predict_slide])
        y_data.append(data[i + predict_slide])

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    return x_data, y_data

def fillnan(matrix):
    '''
    Fill in NaN missing values
    '''
    for i in range(len(matrix)):
        if np.isnan(matrix[i]):
                matrix[i] = np.mean([matrix[i-1], matrix[i+1]])
                
    return matrix

def Denormalize(y):
    '''
    Denormalize the output of network for prediction comparison
    '''
    mean = np.mean(y)
    var = np.var(y)
    denorm = (y * var) + mean
    
    return denorm

def MovingAverage(x):
    '''
    Preprocessing: compute moving average for input to smooth the data
    '''
    decay = 0.9
    global_step = tf.Variable(0, dtype = tf.float32, trainabale = False)
    x_t = x[0] # x_t: new current input x 
    ema = tf.train.ExponentialMovingAverage(decay, global_step)
    ema_op = ema.apply([x_t])
    ema_list = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in range(1, len(x)):
            sess.run(tf.assign(x_t, x[i]))
            sess.run(ema_op)
            ema_list.append(sess.run(ema.average(x_t))) # average = s_t 
    
    ema_list = tf.conver_to_tensor(ema_list)
    return ema_list

# 1. Fill in missing value
train_set = fillnan(train_set)
test_set = fillnan(test_set)

# 2. Normalization
predict_slide = 10
train_set, test_set = net.Normalization(train_set, test_set)
x_train, y_train = create_pipeline(train_set, predict_slide)
x_test, y_test = create_pipeline(test_set, predict_slide)
print('Train shape: ', x_train.shape)
print('Test shape: ', x_test.shape)

print(x_test[0:3, :])

x = tf.placeholder(tf.float32, [None, predict_slide])
y = tf.placeholder(tf.float32, [None])

# Create train & test data pipeline
total_train_data = x_train.shape[0]
batch_size = 256
epochs = 10
learn_rate = 0.05    
total_batch = int(np.floor(total_train_data/batch_size)) + 1

predict_y = net.inference(x, reuse = tf.AUTO_REUSE)
predict_y = tf.squeeze(predict_y) #remove size 1 dimension
mse = tf.losses.mean_squared_error(labels = y, predictions = predict_y)
#mse = tf.reduce_mean(tf.squared_difference(predict_y, y))
with tf.variable_scope('opt', reuse = tf.AUTO_REUSE):
     train_op = tf.train.AdamOptimizer(learn_rate).minimize(mse)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    #print(tf.global_variables())
    print('Train data size: ', x_train.shape[0])
    print('Total batch:', total_batch)
    print('Start training...')
    print(x_test[0:3, :])
    loss_curve = []
    
    for i in range(epochs):
        _, err = sess.run([train_op, mse], {x:x_train, y:y_train})
        print('MSE:', err)
        
    test_output = sess.run(predict_y, {x:x_test})
    plt.figure(1)
    plt.title('Predict output')
    plt.plot(test_output, label = 'predict output')
    plt.legend(loc = 'upper right')
    plt.plot(y_test, label = 'test output')
    plt.legend(loc = 'upper right')
    
    
    
    plt.show()
    
    
    
    '''for iterate in range(epochs):
        
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
        
    
    validate_y = sess.run(predict_y, {x:x_test})
    plt.figure(20)
    plt.title('Predict output & Test output')
    plt.plot(validate_y, label = 'predict output')
    plt.legend(loc = 'upper right')
    plt.plot(y_test, label = 'test output')
    plt.legend(loc = 'upper right')
            
    plt.figure(21)
    plt.title('Loss curve')
    plt.plot(loss_curve)
       
    plt.show()'''
























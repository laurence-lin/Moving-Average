import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

stock = pd.read_csv('ibm_stock.csv')
stock = np.array(stock.iloc[:, 1])
print(stock.shape)

decay = 0.9
s_t_list = []
s_t = stock[0]
s_t_list.append(s_t)
for i in range(1, len(stock)):
    s_t = s_t * decay + (1 - decay) * stock[i]
    s_t_list.append(s_t)


with tf.Session() as sess:
    stock_t = tf.convert_to_tensor(stock, dtype = tf.float32)
    s_t = tf.Variable(stock[0], dtype = tf.float32)
    global_step = tf.Variable(0, dtype = tf.float32, trainable = False)
    ema = tf.train.ExponentialMovingAverage(decay, global_step)
    ema_op = ema.apply([s_t])
    sess.run(tf.global_variables_initializer())
    ema_list = []
    
    for i in range(1, len(stock)):
        sess.run(tf.assign(s_t, stock_t[i]))   
        sess.run(ema_op)
        ema_list.append(sess.run(ema.average(s_t)))
    
plt.figure(0)
plt.title('Moving average result: Fixed weight decay = %s'%str(decay))
plt.plot(stock, label = 'Original')
plt.legend(loc = 'upper right')
plt.plot(s_t_list[:], label = 'Moving average')
plt.legend(loc = 'upper right')

plt.figure(1)
plt.title('Exponential Moving Average')
plt.plot(ema_list[:], label = 'EMA')
plt.legend(loc = 'upper right')
plt.plot(stock, label = 'Original')
plt.legend(loc = 'upper right')

plt.show()
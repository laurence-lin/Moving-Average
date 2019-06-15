# Moving-Average
Experiment the implementation of moving average, and test if its smoothing effect helps training performance.


Dataset: IBM stock price dataset, 362 samples
Set decay = 0.9

Result:
1. Calculate the moving average by hand directly:
![image](https://github.com/laurence-lin/Moving-Average/blob/master/moving%20average%20result.png)

2. Calculate the exponential moving average by tensorflow EMA function:
The decay rate is set by min(decay, (1 + num_update)/(10 + num_update)), which means actual decay rate increase from 1/10 to decay rate 0.9.
![image](https://github.com/laurence-lin/Moving-Average/blob/master/EMA.png)

#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#这里是假设A=1，H=1, B=0的情况
# 故动态模型 X(k) = X(k-1) + 噪声
#            Z(K) = X(k)
# 动态模型是一个常量

# intial parameters
n_iter = 50
sz = (n_iter,) # size of array
n_iter2 = 25
sz2 = (n_iter2,)
x = 2000 # truth value (typo in example at top of p. 13 calls this z)
x1=1000
z1 = np.random.normal(x, 200,size=sz2) # observations (normal about x, sigma=0.1)
z2 = np.random.normal(1000, 100, size=sz2)
z = np.append(z1,z2)
Q = 5 # process variance

# allocate space for arrays
xhat=np.zeros(sz)      # a posteri estimate of x
P=np.zeros(sz)         # a posteri error estimate
xhatminus=np.zeros(sz) # a priori estimate of x
Pminus=np.zeros(sz)    # a priori error estimate
K=np.zeros(sz)         # gain or blending factor

R = 0.1**2 # estimate of measurement variance, change to see effect

# intial guesses
xhat[0] = 2000.0
P[0] = 50

for k in range(1,n_iter):
    # time update
    xhatminus[k] = xhat[k-1]  #X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0
    Pminus[k] = P[k-1]+Q      #P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1
    print("xxxx:",xhatminus[k], z[k])
    temp = 10 * abs(z[k]-xhatminus[k]) / xhatminus[k]
    R = temp*temp
    # measurement update
    K[k] = Pminus[k]/( Pminus[k]+R ) #Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
    print("####", R, Pminus[k], K[k])
    xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k]) #X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
    P[k] = (1-K[k])*Pminus[k] #P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1

plt.figure()
plt.plot(z, 'k+', label='noisy measurements')  # 测量值
plt.plot(xhatminus, 'r+', label='a pminus estimate')  # 过滤后的值
plt.plot(xhat, 'b-', label='a posteri estimate')  # 过滤后的值
plt.axhline(x, color='g', label='truth value')  # 系统值
plt.axhline(x1, color='g', label='truth value')  # 系统值
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Voltage')
plt.show()

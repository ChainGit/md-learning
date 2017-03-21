#!/usr/bin/python
# -*- coding:UTF-8 -*-

# from 51CTO Yudi Tang 
# 三层的简单神经网络

import numpy as np

print "init data and label ..."

def sigmoid(x,deriv = False):
	if(deriv == True):
		return x*(1-x);
	return 1/(1+np.exp(-x))
	
# data
x = np.array([
			[0,0,1],
			[0,1,1],
			[1,0,1],
			[1,1,1],
			[0,0,1],
			])
			
print "data array rows and columns: " + str(x.shape)
print "data: "
print x

# label
y = np.array([
			[0],
			[1],
			[1],
			[0],
			[0]
			])

print "label array rows and columns: " + str(y.shape)
print "label: "
print y

# set random
np.random.seed(1)

# 3层神经网络(Neural Network)：输入层，隐藏层和输出层
# w
w0 = 2 * np.random.random((3,4)) - 1
w1 = 2 * np.random.random((4,1)) - 1

print "w0: "
print w0
print "w1: "
print w1

for j in xrange(60000):
	# 输入层
	l0 = x
	# 第一层到第二层的前向传播
	l1 = sigmoid(np.dot(l0,w0))
	# 第二层到第三层的前向传播
	l2 = sigmoid(np.dot(l1,w1))
	# 误差
	l2_error = y - l2;
	# 反向传播
	l2_delta = l2_error * sigmoid(l2,deriv = True)
	if j % 10000 == 0:
		print "l2_error.shape: " + str(l2_error.shape)
		print "l2_error: " + str(np.mean(np.abs(l2_error)))
		print "l2_delta.shape: " + str(l2_delta.shape)
		print "l2_delta: " + str(np.mean(np.abs(l2_delta)))
	# 误差
	l1_error = l2_delta.dot(w1.T)	
	# 反向传播
	l1_delta = l1_error * sigmoid(l1,deriv = True)
	if j % 10000 == 0:
		print "l1_error.shape: " + str(l1_error.shape)
		print "l1_error: " + str(np.mean(np.abs(l1_error)))
		print "l1_delta.shape: " + str(l1_delta.shape)
		print "l1_delta: " + str(np.mean(np.abs(l1_delta)))
	# 修改（更新）权值
	w1 += l1.T.dot(l2_delta)
	w0 += l0.T.dot(l1_delta)


# 输出优化后的权重
print "w0: "
print w0
print "w1: "
print w1

# 再次计算
out_0 = sigmoid(np.dot(x,w0))
out_1 = sigmoid(np.dot(out_0,w1))
print(y)
print(out_1)
	
	














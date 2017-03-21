#!/usr/bin/python
# -*- coding: utf-8 -*-

# 卷积神经网络(CNN)
# 基于谷歌官方示范

'''
Tensorflow依赖于一个高效的C++后端来进行计算。
与后端的这个连接叫做session。
一般而言，使用TensorFlow程序的流程是先创建一个图，然后在session中启动它。
这里，我们使用更加方便的InteractiveSession类。
通过它，你可以更加灵活地构建你的代码。
它能让你在运行图的时候，插入一些计算图，这些计算图是由某些操作(operations)构成的。
这对于工作在交互式环境中的人们来说非常便利，比如使用IPython。

训练20000次后，再进行测试，测试精度可以达到99%。
'''

import numpy as np
import tensorflow as tf 
import tensorflow.examples.tutorials.mnist.input_data as input_data

## train or test,True is train
FLAG = False

## 下载并加载mnist数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 输入的数据占位符		 
x = tf.placeholder(tf.float32, [None, 784])		
# 输入的标签占位符										
y_actual = tf.placeholder(tf.float32, shape=[None, 10])						

## 定义四个函数，分别用于初始化权值W，初始化偏置项b, 构建卷积层和构建池化层
#定义一个函数，用于初始化所有的权值 W
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

# 定义一个函数，用于初始化所有的偏置项 b
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)
	
#定义一个函数，用于构建卷积层
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#定义一个函数，用于构建池化层
def max_pool(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

## 构建网络
## 整个网络由两个卷积层（包含激活层和池化层），一个全连接层，一个dropout层和一个softmax层组成
# 转换输入数据shape,以便于用于网络中
x_image = tf.reshape(x, [-1,28,28,1])	
		 
W_conv1 = weight_variable([5, 5, 1, 32])			
b_conv1 = bias_variable([32])			 
# 第一个卷积层（ReLU）
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)		 
# 第一个池化层
h_pool1 = max_pool(h_conv1)																	

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
# 第二个卷积层（ReLU）
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# 第二个池化层	
h_pool2 = max_pool(h_conv2)																	 

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# reshape成向量
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])			
# 第一个全连接层				
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)		

keep_prob = tf.placeholder("float") 
# dropout层
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)									

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
# softmax层
y_predict=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)	 

# 交叉熵
cross_entropy = -tf.reduce_sum(y_actual*tf.log(y_predict))		 
# 梯度下降法(参数是学习指数)	
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)	

## 用于测试
correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))	
# 精确度计算（均方误差）	
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))						

save_path = str("save_model_cnn/mnist_softmax_cnn")
save = tf.train.Saver()

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

if FLAG:
	## 训练
	for i in range(20000):
		batch_xs, batch_ys = mnist.train.next_batch(50)
		train_step.run(feed_dict={x: batch_xs, y_actual: batch_ys, keep_prob: 0.5})
		# 训练100次，验证一次
		if i%100 == 0:									
			train_acc = accuracy.eval(feed_dict={x:batch_xs, y_actual: batch_ys, keep_prob: 1.0})
			print('step',i,'training accuracy',train_acc)

	save.save(sess,save_path)
else:
	## 测试
	save.restore(sess,save_path)
	
	# 进行单独测试
	nsample = 20
	randidx = np.random.randint(mnist.test.images.shape[0],size=nsample)
		
	for i in randidx:
		curr_image = np.reshape(mnist.test.images[i,:],(1,784))
		curr_label = np.reshape(mnist.test.labels[i,:],(1,10))
		test_acc=accuracy.eval(feed_dict={x: curr_image, y_actual: curr_label, keep_prob: 1.0})
		print("test accuracy",i,test_acc)
	
	# 测试test集
	# test_acc=accuracy.eval(feed_dict={x: mnist.test.images, y_actual: mnist.test.labels, keep_prob: 1.0})
	# print("test accuracy",test_acc)


#!/usr/bin/python
# -*- coding: utf-8 -*-

# ���������(CNN)
# ���ڹȸ�ٷ�ʾ��

'''
Tensorflow������һ����Ч��C++��������м��㡣
���˵�������ӽ���session��
һ����ԣ�ʹ��TensorFlow������������ȴ���һ��ͼ��Ȼ����session����������
�������ʹ�ø��ӷ����InteractiveSession�ࡣ
ͨ����������Ը������ع�����Ĵ��롣
��������������ͼ��ʱ�򣬲���һЩ����ͼ����Щ����ͼ����ĳЩ����(operations)���ɵġ�
����ڹ����ڽ���ʽ�����е�������˵�ǳ�����������ʹ��IPython��

ѵ��20000�κ��ٽ��в��ԣ����Ծ��ȿ��Դﵽ99%��
'''

import numpy as np
import tensorflow as tf 
import tensorflow.examples.tutorials.mnist.input_data as input_data

## train or test,True is train
FLAG = False

## ���ز�����mnist����
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# ���������ռλ��		 
x = tf.placeholder(tf.float32, [None, 784])		
# ����ı�ǩռλ��										
y_actual = tf.placeholder(tf.float32, shape=[None, 10])						

## �����ĸ��������ֱ����ڳ�ʼ��ȨֵW����ʼ��ƫ����b, ���������͹����ػ���
#����һ�����������ڳ�ʼ�����е�Ȩֵ W
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

# ����һ�����������ڳ�ʼ�����е�ƫ���� b
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)
	
#����һ�����������ڹ��������
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#����һ�����������ڹ����ػ���
def max_pool(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

## ��������
## ������������������㣨���������ͳػ��㣩��һ��ȫ���Ӳ㣬һ��dropout���һ��softmax�����
# ת����������shape,�Ա�������������
x_image = tf.reshape(x, [-1,28,28,1])	
		 
W_conv1 = weight_variable([5, 5, 1, 32])			
b_conv1 = bias_variable([32])			 
# ��һ������㣨ReLU��
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)		 
# ��һ���ػ���
h_pool1 = max_pool(h_conv1)																	

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
# �ڶ�������㣨ReLU��
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# �ڶ����ػ���	
h_pool2 = max_pool(h_conv2)																	 

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# reshape������
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])			
# ��һ��ȫ���Ӳ�				
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)		

keep_prob = tf.placeholder("float") 
# dropout��
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)									

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
# softmax��
y_predict=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)	 

# ������
cross_entropy = -tf.reduce_sum(y_actual*tf.log(y_predict))		 
# �ݶ��½���(������ѧϰָ��)	
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)	

## ���ڲ���
correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))	
# ��ȷ�ȼ��㣨������	
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))						

save_path = str("save_model_cnn/mnist_softmax_cnn")
save = tf.train.Saver()

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

if FLAG:
	## ѵ��
	for i in range(20000):
		batch_xs, batch_ys = mnist.train.next_batch(50)
		train_step.run(feed_dict={x: batch_xs, y_actual: batch_ys, keep_prob: 0.5})
		# ѵ��100�Σ���֤һ��
		if i%100 == 0:									
			train_acc = accuracy.eval(feed_dict={x:batch_xs, y_actual: batch_ys, keep_prob: 1.0})
			print('step',i,'training accuracy',train_acc)

	save.save(sess,save_path)
else:
	## ����
	save.restore(sess,save_path)
	
	# ���е�������
	nsample = 20
	randidx = np.random.randint(mnist.test.images.shape[0],size=nsample)
		
	for i in randidx:
		curr_image = np.reshape(mnist.test.images[i,:],(1,784))
		curr_label = np.reshape(mnist.test.labels[i,:],(1,10))
		test_acc=accuracy.eval(feed_dict={x: curr_image, y_actual: curr_label, keep_prob: 1.0})
		print("test accuracy",i,test_acc)
	
	# ����test��
	# test_acc=accuracy.eval(feed_dict={x: mnist.test.images, y_actual: mnist.test.labels, keep_prob: 1.0})
	# print("test accuracy",test_acc)


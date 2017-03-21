#!/usr/bin/python
# -*- coding:UTF-8 -*-

# 简单神经网络
# 基于谷歌官方示范

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import matplotlib
matplotlib.use('Agg')

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

FLAGS = None

# 是否随机提取数据集的数据并保存为图片
SAVE = False

# 训练还是测试
TEST = True


def main(_):
	# Import data
	'''
	数据集被分三个子集：
	5.5W行的训练数据集（mnist.train），
	5千行的验证数据集（mnist.validation),
	1W行的测试数据集（mnist.test）
	'''
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	if SAVE:
		print("type of mnist is %s" % (type(mnist)))
		print("type of mnist_train_images is %s" % (type(mnist.train.images)))
		print("type of mnist_train_labels is %s" % (type(mnist.train.labels)))
		print("type of mnist_test_images is %s" % (type(mnist.test.images)))
		print("type of mnist_test_labels is %s" % (type(mnist.test.labels)))
		# 28x28=784，灰度图
		print("shape of mnist_train_images is %s" % (mnist.train.images.shape,))
		print("shape of mnist_train_labels is %s" % (mnist.train.labels.shape,))
		print("shape of mnist_test_images is %s" % (mnist.test.images.shape,))
		print("shape of mnist_test_labels is %s" % (mnist.test.labels.shape,))
		# MNIST数据集的数据个数
		print("num of train data is %d" % (mnist.train.num_examples))
		print("num of test data is %d" % (mnist.test.num_examples))

		# 随机提取数据集并保存为图片
		nsample = 5
		randidx = np.random.randint(mnist.train.images.shape[0],size=nsample)
		
		for i in randidx:
			curr_image = np.reshape(mnist.train.images[i,:],(28,28))
			curr_label = np.argmax(mnist.train.labels[i,:])
			plt.matshow(curr_image,cmap=plt.get_cmap('gray'))
			plt.title("The "+str(i)+"th Training data,and label is "+str(curr_label))
			plt.savefig("pics_train/"+str(i)+"th.jpg")
			
		nsample = 5
		randidx = np.random.randint(mnist.test.images.shape[0],size=nsample)
		
		for i in randidx:
			curr_image = np.reshape(mnist.test.images[i,:],(28,28))
			curr_label = np.argmax(mnist.test.labels[i,:])
			plt.matshow(curr_image,cmap=plt.get_cmap('gray'))
			plt.title("The "+str(i)+"th Testing data,and label is "+str(curr_label))
			plt.savefig("pics_test/"+str(i)+"th.jpg")
	
	# Create the model
	# 占位符placeholder
	x = tf.placeholder(tf.float32, [None, 784])
	# 初始化权值W
	# 每张图片为28x28的黑白图片，所以每行为784维的向量
	# 0~9,故为10行768列的权值
	W = tf.Variable(tf.zeros([784, 10]))
	# 初始化偏置项b
	b = tf.Variable(tf.zeros([10]))
	# 加权变换并进行softmax分类器回归，得到预测概率
	# 线性得分函数
	# y_predict
	y = tf.matmul(x, W) + b

	# Define loss and optimizer
	# y_actual
	y_ = tf.placeholder(tf.float32, [None, 10])

	# The raw formulation of cross-entropy,
	#
	#	 tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
	#																 reduction_indices=[1]))
	#
	# can be numerically unstable.
	#
	# So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
	# outputs of 'y', and then average across the batch.
	# 求交叉熵
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	# 用梯度下降法使得残差最小
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
			
	# Test trained model
	# 在测试阶段，测试准确度计算
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	# 多个批次的准确度均值
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	save_path = str("save_model_nn/mnist_softmax_nn")
	save = tf.train.Saver()
	
	if not TEST:
		# Train
		# 训练阶段，迭代1000次
		for _ in range(1000):
			# 按批次训练，每批100行数据
			batch_xs, batch_ys = mnist.train.next_batch(100)
			# 执行训练
			sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
			# 每训练100次，测试一次
			if _ % 100 == 0:
				print(sess.run(accuracy, feed_dict={x: mnist.test.images,y_: mnist.test.labels}))
		
		save.save(sess,save_path)
	else:
		save.restore(sess,save_path)
		
		# 集体测试取平均
		# print("last: ")
		# print(sess.run(accuracy, feed_dict={x: mnist.test.images,y_: mnist.test.labels}))
		
		# 单独测试
		nsample = 20
		randidx = np.random.randint(mnist.test.images.shape[0],size=nsample)
			
		for i in randidx:
			curr_image = np.reshape(mnist.test.images[i,:],(1,784))
			curr_label = np.reshape(mnist.test.labels[i,:],(1,10))
			tmp = tf.matmul(curr_image, W) + b
			print("actual: ",str(np.argmax(curr_label))," predict: ",str(np.argmax(sess.run(tmp))),
				" correct: ",str(sess.run(correct_prediction,feed_dict={x: curr_image,y_: curr_label})))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',help='Directory for storing input data')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

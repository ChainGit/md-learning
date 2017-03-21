#!/usr/bin/python
# -*- coding:UTF-8 -*-

# �߼��ع飨����ࣩ
# ʹ��MNIST
# From 51CTO Yudi Tang

import numpy as np
import tensorflow as tf
import input_data

# �������ݼ�
mnist = input_data.read_data_sets('MNIST_data/',one_hot = True)
train_images = mnist.train.images
train_labels = mnist.train.labels
test_images = mnist.test.images
test_labels = mnist.test.images

# ���ò���
x = tf.placeholder('float',[None,784])
y_actual = tf.placeholder('float',[None,10])
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# �߼��ع�ģ��
activate = tf.nn.softmax(tf.matmul(x,w)+b)

# ��ʧ
loss = tf.reduce_mean(-tf.reduce_sum(y_actual*tf.log(activate),reduction_indices=1))

# �Ż�����(�ݶ�)
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Ԥ��ֵ
y_predict = tf.equal(tf.argmax(activate,1),tf.argmax(y_actual,1))

# ׼ȷ��
accuray = tf.reduce_mean(tf.cast(y_predict,'float'))

training_epochs = 50
batch_size = 100
display_step = 5

# ��ʼ��Session
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(training_epochs):
		avg_loss = 0
		num_batch = int(mnist.train.num_examples/batch_size)
		for i in range(num_batch):
			batch_xs,batch_ys = mnist.train.next_batch(batch_size)
			feeds = {x:batch_xs,y_actual:batch_ys}
			sess.run(optimizer,feed_dict=feeds)
			avg_loss += sess.run(loss,feed_dict=feeds)/num_batch
		if epoch % display_step == 0 :
			train_acc = sess.run(accuray,feed_dict=feeds)
			test_acc = sess.run(accuray,feed_dict={x:mnist.test.images,y_actual:mnist.test.labels})
			print(epoch,training_epochs,avg_loss,train_acc,test_acc)
	
	print(sess.run(w))
	print(sess.run(b))





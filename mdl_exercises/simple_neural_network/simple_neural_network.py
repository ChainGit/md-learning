#!/usr/bin/python
# -*- coding:UTF-8 -*-

# �������磨���㣩
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

# ����Ĳ���(��Ԫ����)
n_hidden_1 = 256
n_hidden_2 = 128
n_input = 784
n_classes = 10

# ��������
x = tf.placeholder('float',[None,n_input])
y = tf.placeholder('float',[None,n_classes])

# �������
stddev = 0.1
# Ȩֵ
weights = {
	'w1':tf.Variable(tf.random_normal([n_input,n_hidden_1],stddev=stddev)),
	'w2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2],stddev=stddev)),
	'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes],stddev=stddev))
}
# ƫ��
biases = {
	'b1':tf.Variable(tf.random_normal([n_hidden_1])),
	'b2':tf.Variable(tf.random_normal([n_hidden_2])),
	'out':tf.Variable(tf.random_normal([n_classes]))
}

# ����
def multilayer_perceptron(_X,_weights,_biases):
	# ��һ��
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X,_weights['w1']),_biases['b1']))
	# �ڶ���
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,_weights['w2']),_biases['b2']))
	# ���
	return (tf.matmul(layer_2,_weights['out'])+_biases['out'])
	
# Ԥ��
y_predict = multilayer_perceptron(x,weights,biases)

# ��ʧ(������)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_predict))
# ����(�ݶ��½�)
optimizer = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)

# ������ȷ��
correct = tf.equal(tf.argmax(y_predict,1),tf.argmax(y,1))
accuray = tf.reduce_mean(tf.cast(correct,'float'))

training_epochs = 20
batch_size = 100
display_step = 4

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(training_epochs):
		avg_loss = 0
		num_batch = int(mnist.train.num_examples/batch_size)
		for i in range(num_batch):
			batch_xs,batch_ys = mnist.train.next_batch(batch_size)
			feeds = {x:batch_xs,y:batch_ys}
			sess.run(optimizer,feed_dict=feeds)
			avg_loss += sess.run(loss,feed_dict=feeds)/num_batch
		if epoch % display_step == 0 :
			train_acc = sess.run(accuray,feed_dict=feeds)
			test_acc = sess.run(accuray,feed_dict={x:mnist.test.images,y:mnist.test.labels})
			print(epoch,training_epochs,avg_loss,train_acc,test_acc)



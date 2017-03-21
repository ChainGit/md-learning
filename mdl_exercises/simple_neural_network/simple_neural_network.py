#!/usr/bin/python
# -*- coding:UTF-8 -*-

# 简单神经网络（两层）
# 使用MNIST
# From 51CTO Yudi Tang

import numpy as np
import tensorflow as tf
import input_data

# 读入数据集
mnist = input_data.read_data_sets('MNIST_data/',one_hot = True)
train_images = mnist.train.images
train_labels = mnist.train.labels
test_images = mnist.test.images
test_labels = mnist.test.images

# 网络的参数(神经元个数)
n_hidden_1 = 256
n_hidden_2 = 128
n_input = 784
n_classes = 10

# 输入和输出
x = tf.placeholder('float',[None,n_input])
y = tf.placeholder('float',[None,n_classes])

# 网络参数
stddev = 0.1
# 权值
weights = {
	'w1':tf.Variable(tf.random_normal([n_input,n_hidden_1],stddev=stddev)),
	'w2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2],stddev=stddev)),
	'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes],stddev=stddev))
}
# 偏置
biases = {
	'b1':tf.Variable(tf.random_normal([n_hidden_1])),
	'b2':tf.Variable(tf.random_normal([n_hidden_2])),
	'out':tf.Variable(tf.random_normal([n_classes]))
}

# 两层
def multilayer_perceptron(_X,_weights,_biases):
	# 第一层
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X,_weights['w1']),_biases['b1']))
	# 第二层
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,_weights['w2']),_biases['b2']))
	# 输出
	return (tf.matmul(layer_2,_weights['out'])+_biases['out'])
	
# 预测
y_predict = multilayer_perceptron(x,weights,biases)

# 损失(交叉熵)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_predict))
# 调整(梯度下降)
optimizer = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)

# 计算正确度
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



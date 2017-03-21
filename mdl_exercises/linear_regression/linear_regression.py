#!/usr/bin/python
# -*- coding:UTF-8 -*-

# ���Իع飨�����ࣩ
# From 51CTO Yudi Tang

import numpy as np
import tensorflow as tf
# ���ڻ���ͼ��(���������в���ʾ���Ǳ���ͼƬ)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# �������1000����
# Χ����y=0.1x+0.3
num_of_points = 1000
# ����������
vectors_set = []
for i in range(num_of_points):
	xt = np.random.normal(0.0,0.55)
	yt = xt * 0.1 + 0.3 + np.random.normal(0.0,0.03)
	vectors_set.append([xt,yt])

# �����ѡȡԤ��ֵ
data_x = [v[0] for v in vectors_set]
data_y = [v[1] for v in vectors_set]

# ���ͼƬ
plt.scatter(data_x,data_y,c='r')
# plt.show()
plt.savefig('dots.jpg')
	
# ����1ά��w����ȡֵ��[-1,1]֮��������
w = tf.Variable(tf.random_uniform([1],-1.0,1.0,name='w'))
# ����1ά��b����ȡֵ��ʼֵΪ0
b = tf.Variable(tf.zeros([1]),name='b')
# ��������õ���Ԥ��ֵ
y_predict = w * data_x + b

# �Թ���ֵy_predict��ʵ��ֵdata_y֮��ľ��������Ϊ��ʧ
loss = tf.reduce_mean(tf.square(y_predict - data_y),name='loss')
# �����ݶ��½������Ż�����
optimizer = tf.train.GradientDescentOptimizer(0.5)
# ѵ���Ĺ��̾�����С��������ֵloss
train = optimizer.minimize(loss,name='train')

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	# ��ʼʱ��w��b��ֵ
	print('w=',str(sess.run(w)),'b=',str(sess.run(b)),'loss=',str(sess.run(loss)))
	# 20�εĵ���ѵ��
	for i in range(20):
		sess.run(train)
		print('w=',str(sess.run(w)),'b=',str(sess.run(b)),'loss=',str(sess.run(loss)))

	print('y=0.1x+0.3')

	# ���ͼƬ
	plt.scatter(data_x,data_y,c='r')
	plt.plot(data_x,(sess.run(w) * data_x + sess.run(b)),c='b')
	# plt.show()
	plt.savefig('dots_res.jpg')



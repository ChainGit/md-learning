#!/usr/bin/python
# -*- coding:UTF-8 -*-

# 线性回归（单分类）
# From 51CTO Yudi Tang

import numpy as np
import tensorflow as tf
# 用于绘制图表(这里命令行不显示而是保存图片)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 随机生成1000个点
# 围绕在y=0.1x+0.3
num_of_points = 1000
# 随机点的坐标
vectors_set = []
for i in range(num_of_points):
	xt = np.random.normal(0.0,0.55)
	yt = xt * 0.1 + 0.3 + np.random.normal(0.0,0.03)
	vectors_set.append([xt,yt])

# 随机再选取预估值
data_x = [v[0] for v in vectors_set]
data_y = [v[1] for v in vectors_set]

# 输出图片
plt.scatter(data_x,data_y,c='r')
# plt.show()
plt.savefig('dots.jpg')
	
# 生成1维的w矩阵，取值是[-1,1]之间的随机数
w = tf.Variable(tf.random_uniform([1],-1.0,1.0,name='w'))
# 生成1维的b矩阵，取值初始值为0
b = tf.Variable(tf.zeros([1]),name='b')
# 进过计算得到的预估值
y_predict = w * data_x + b

# 以估计值y_predict和实际值data_y之间的均方误差作为损失
loss = tf.reduce_mean(tf.square(y_predict - data_y),name='loss')
# 采用梯度下降法来优化参数
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 训练的过程就是最小化这个误差值loss
train = optimizer.minimize(loss,name='train')

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	# 初始时的w和b的值
	print('w=',str(sess.run(w)),'b=',str(sess.run(b)),'loss=',str(sess.run(loss)))
	# 20次的迭代训练
	for i in range(20):
		sess.run(train)
		print('w=',str(sess.run(w)),'b=',str(sess.run(b)),'loss=',str(sess.run(loss)))

	print('y=0.1x+0.3')

	# 输出图片
	plt.scatter(data_x,data_y,c='r')
	plt.plot(data_x,(sess.run(w) * data_x + sess.run(b)),c='b')
	# plt.show()
	plt.savefig('dots_res.jpg')



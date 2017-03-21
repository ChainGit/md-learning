#!/usr/bin/python
# -*- coding:UTF-8 -*-

# TensorFlow 基本使用 4
# http://edu.51cto.com/course/course_id-6511.html?edu_recommend_adid=87

import tensorflow as tf

# 先执行learn3_basic.py
v1 = tf.Variable([[0.0,0.0]], name='v1')
v2 = tf.Variable([[0.0,0.0],[0.0,0.0]], name='v2')

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver.restore(sess,"save_model/test_save_model")
	# 观察打印出的值
	print(sess.run(v1))
	print(sess.run(v2))
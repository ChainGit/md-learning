#!/usr/bin/python
# -*- coding:UTF-8 -*-

# TensorFlow ����ʹ�� 2
# http://edu.51cto.com/course/course_id-6511.html?edu_recommend_adid=87

import tensorflow as tf

start = tf.Variable(0)
delta = tf.constant(1)
newval = tf.add(start,delta)
# assign��newval���䣨��ֵ����start
update = tf.assign(start,newval)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(start))
for _ in range(3):
	sess.run(update)
	print(sess.run(start))

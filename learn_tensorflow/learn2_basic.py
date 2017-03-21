#!/usr/bin/python
# -*- coding:UTF-8 -*-

# TensorFlow 基本使用 2
# http://edu.51cto.com/course/course_id-6511.html?edu_recommend_adid=87

import tensorflow as tf

start = tf.Variable(0)
delta = tf.constant(1)
newval = tf.add(start,delta)
# assign将newval分配（赋值）给start
update = tf.assign(start,newval)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(start))
for _ in range(3):
	sess.run(update)
	print(sess.run(start))

#!/usr/bin/python
# -*- coding:UTF-8 -*-

# TensorFlow 基本使用 3
# http://edu.51cto.com/course/course_id-6511.html?edu_recommend_adid=87

import tensorflow as tf

v1 = tf.Variable([[0.5,1]], name='v1')
v2 = tf.Variable([[2.0,1.0],[1.0,0.5]], name='v2')

# Pass the variables as a dict:
saver = tf.train.Saver({'v1': v1, 'v2': v2})
# Or pass them as a list.
# saver = tf.train.Saver([v1, v2])
# Or Passing a list is equivalent to passing a dict with the variable op names
# as keys:
# saver = tf.train.Saver({v.op.name: v for v in [v1, v2]})

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver.save(sess,"save_model/test_save_model")
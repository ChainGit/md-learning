#!/usr/bin/python
# -*- coding:UTF-8 -*-

# TensorFlow 基本使用 5
# http://edu.51cto.com/course/course_id-6511.html?edu_recommend_adid=87

import tensorflow as tf
import numpy as np

np_a = np.zeros((3,3),np.float32)
print(np_a.shape)
tf_a = tf.convert_to_tensor(np_a)
print(tf_a.shape)

with tf.Session() as sess:
	print(sess.run(tf_a))
#!/usr/bin/python
# -*- coding:UTF-8 -*-

# TensorFlow ����ʹ�� 1
# http://edu.51cto.com/course/course_id-6511.html?edu_recommend_adid=87

import tensorflow as tf

# python ���������ķ���
# a = 3
# tensorflow ���������ķ���
w = tf.Variable([[0.5,1.0]])
x = tf.Variable([[2.0],[1.0]])
# �������
y = tf.matmul(w,x)

# �鿴�����Ľṹ
print(w)
print(x)
print(y)

tmp1 = tf.zeros([3,4],tf.float32)
print(tmp1.shape)
tmp2 = tf.zeros_like(tmp1)
print(tmp2.shape)
tmp3 = tf.ones([2,3],tf.float32)
print(tmp3.shape)
tmp4 = tf.ones_like(tmp3)
print(tmp4.shape)
tmp5 = tf.constant([1,2,3,4,5,6,7])
print(tmp5.shape)
tmp6 = tf.constant(-1.0,shape=[2,3])
print(tmp6.shape)
tmp7 = tf.linspace(10.0,12.0,3,name='linspace')
print(tmp7.shape)
# start limit delta
tmp8 = tf.range(3,18,3)
print(tmp8.shape)
# ��˹���
tmp9 = tf.random_normal([2,3],mean=-1,stddev=4)
print(tmp9.shape)
tmp10 = tf.constant([[1,2],[3,4],[5,6]])
print(tmp10)
# ���ϴ��
tmp11 = tf.random_shuffle(tmp10)
print(tmp11)
# ռλ��
tmp12 = tf.placeholder(tf.float32)
tmp13 = tf.placeholder(tf.float32)
tmp14 = tf.add(tmp12,tmp13)

# ��ʼ������
init_tf = tf.global_variables_initializer()

# ��ʼ��Session������ͼ�ļ���ģʽ��
with tf.Session() as sess:
	sess.run(init_tf)
	# ���㣨���ۣ�
	print(y.eval())
	#��ӡ
	print(sess.run(tmp1))
	print(sess.run(tmp2))
	print(sess.run(tmp3))
	print(sess.run(tmp4))
	print(sess.run(tmp5))
	print(sess.run(tmp6))
	print(sess.run(tmp7))
	print(sess.run(tmp8))
	print(sess.run(tmp9))
	print(sess.run(tmp10))
	print(sess.run(tmp11))
	print(sess.run([tmp14],feed_dict={tmp12:[7.0],tmp13:[2.]}))
	

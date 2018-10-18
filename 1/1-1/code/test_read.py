# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 18:52:37 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

plotdata = {"batchsize": [], "loss": []}


def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx - w):idx]) / w for idx, val in enumerate(a)]


# 生成模拟数据


# 创建模型
# 占位符
X = tf.placeholder("float")
Y = tf.placeholder("float")
# 模型参数
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")

# 前向结构
z = tf.multiply(X, W) + b

# 反向优化
cost = tf.reduce_mean(tf.square(Y - z))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)  # Gradient descent

# 初始化变量
init = tf.global_variables_initializer()
# 训练参数
training_epochs = 10000
display_step = 2

saver = tf.train.Saver(max_to_keep=1)
# 启动session
with tf.Session() as sess:
    sess.run(init)
    new_saver = tf.train.import_meta_graph('model/my-model-19.meta')
    new_saver.restore(sess, 'model/my-model-19')
    sess.run(z, feed_dict={X: 0.2})
    print("x=0.2，z=", sess.run(z, feed_dict={X: 0.2}))
    print("x=0.4，z=", sess.run(z, feed_dict={X: 0.4}))
    print("x=0.5，z=", sess.run(z, feed_dict={X: 0.5}))
    print("x=0.6，z=", sess.run(z, feed_dict={X: 0.6}))
    print("x=10，z=", sess.run(z, feed_dict={X: 10.5}))
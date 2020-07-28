# 例子一
# 利用TensorFlow模拟简单的一次函数
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 建立训练所需的数据，数据的type统一为float32
x_data = np.random.rand(100).astype(np.float32)
y_data = 3 * x_data + 1.5
# 建立训练的结构，随机初始化权重和偏置
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
Bias = tf.Variable(tf.zeros([1]))
y = Weights * x_data + Bias
# 定义loss以及优化目标
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
# 初始化所有变量
init_variables = tf.initialize_all_variables()
# 初始化任务
sess = tf.Session()
sess.run(init_variables)
for step in range(301):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(Bias))
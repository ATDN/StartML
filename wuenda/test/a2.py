# 例子二
# 利用TensorFlow训练神经网络模拟二/三次函数（线性回归）
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# 定义单个全连接层
def add_layer(inputs, in_size, out_size, activitation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    bias = tf.Variable(tf.zeros([1, out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + bias
    if activitation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activitation_function(Wx_plus_b)
    return outputs
# 生成带噪声的训练数据
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) * x_data - 0.5 + noise
# 预置input和label的位置
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
# 搭建神经网络（两层，第一层激活函数为tanh）
l1 = add_layer(xs, 1, 10, activitation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activitation_function=None)
# 定义loss和优化器
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 初始化所有变量
init = tf.initialize_all_variables()
# 初始化任务
with tf.Session() as sess:
    sess.run(init)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    plt.ion()
    for i in range(10000):
        # 训练网络
        sess.run(train, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            print(sess.run(loss,feed_dict={xs: x_data, ys: y_data}))
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            plt.pause(0.1)
    plt.show()
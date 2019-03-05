import tensorflow as tf
import numpy
rdn = numpy.random

# 0. 训练数据
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]


# 1. 构建计算图
# - 输入、权重变量
# - 模型
# - 损失函数
# - 优化器
# - 训练（参数初始化、）


# 输入、权重变量
X = tf.placeholder(name='X')
y = tf.placeholder(name='y')

W = tf.Variable(initial_value=rdn.rand(), name='weight')
b = tf.Variable(initial_value=rdn.rand(), name='bias')

# 模型
linear_model = tf.add(tf.multiply(X,W),b)

# 损失函数
loss = tf.reduce_mean(tf.pow(linear_model-y),2) / (2 * n_samples)

# 优化器
opt = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss=loss)

# 训练
with tf.get_default_session() as sess:
    tf.variables_initializer()
    sess.run()




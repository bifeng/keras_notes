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
# - 优化器（参数-learning rate/）
# - 参数初始化
# - 训练（参数-epoch/）（初始化/优化器）
#   把数据喂进去


# 输入、权重变量
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

W = tf.Variable(initial_value=rdn.rand(), name='weight')
b = tf.Variable(initial_value=rdn.rand(), name='bias')

# 模型
linear_model = tf.add(tf.multiply(X,W),b)

# 损失函数
loss = tf.reduce_mean(tf.pow(linear_model-Y,2))

# 优化器
learning_rate = 0.01
opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=loss)

# 参数初始化
init = tf.global_variables_initializer()

# 训练
epochs = 1000
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        for (x,y) in zip(train_X, train_Y):
            sess.run(opt, feed_dict={X:x, Y:y})

        # 查看loss的变化
        if (epoch + 1) % 50 == 0:
            c = sess.run(loss, feed_dict={X:train_X, Y:train_Y})
            print('current epoch {} loss is {}'.format(epoch+1, c), "W=", sess.run(W), "b=", sess.run(b))

    print("optimize done")
    loss = sess.run(loss, feed_dict={X:train_X, Y:train_Y})
    print('training loss is {}'.format(loss), "W=", sess.run(W), "b=", sess.run(b))


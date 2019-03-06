import tensorflow as tf
import numpy

rdn = numpy.random

# 0. 训练数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# 1. 构建计算图
# - 输入、权重变量
# - 模型
# - 损失函数
# - 优化器（参数-learning rate/）
# - 参数初始化
# - 训练（参数-epoch/batch size）（初始化/优化器）
#   把数据喂进去


# 输入、权重变量
X = tf.placeholder(tf.float32, shape=[None, 784], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 10], name='Y')

W = tf.Variable(initial_value=tf.zeros([784, 10]), name='weight')
b = tf.Variable(initial_value=tf.zeros([10]), name='bias')

# 模型
logistic_model = tf.nn.softmax(tf.matmul(X,W)+b)

# 损失函数
loss = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(logistic_model), reduction_indices=1))

# 优化器
learning_rate = 0.01
opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=loss)

# 参数初始化
init = tf.global_variables_initializer()

# 训练
epochs = 25
batch_size = 100
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        ave_loss = 0.
        total_batch = int(mnist.train.num_examples/100)
        for batch in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            _, loss_value = sess.run([opt,loss], feed_dict={X:batch_xs,Y:batch_ys})
            ave_loss += loss_value/total_batch

        # 查看loss的变化
        if (epoch + 1) % 1 == 0:
            print('current epoch {} loss is {}'.format(epoch+1, ave_loss))

    print("optimize done")

    # Test model
    correct_prediction = tf.equal(tf.argmax(logistic_model, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))


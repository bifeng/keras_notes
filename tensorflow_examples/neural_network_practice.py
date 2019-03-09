# 1. 构建计算图
# - 输入、权重变量
# - 模型
# - 损失函数
# - 优化器（参数-learning rate/）
# - 参数初始化
# - 训练（参数-epoch/）（初始化/优化器）
#   把数据喂进去


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

input_dimension = 784
out_dimension = 10
num_hiddens_1 = 256
num_hiddens_2 = 256

X = tf.placeholder(tf.float32, shape=[None,input_dimension], name='X')
Y = tf.placeholder(tf.float32,shape=[None,out_dimension], name='Y')

weight = {
    'w1': tf.Variable(tf.random_normal([input_dimension, num_hiddens_1])),
    'w2': tf.Variable(tf.random_normal([num_hiddens_1, num_hiddens_2])),
    'out': tf.Variable(tf.random_normal([num_hiddens_2,out_dimension]))
}


bias = {
    'b1': tf.Variable(tf.random_normal([num_hiddens_1])),
    'b2': tf.Variable(tf.random_normal([num_hiddens_2])),
    'out': tf.Variable(tf.random_normal([out_dimension]))
}


def neural_net(x):
    # layer 1
    layer1 = tf.add(tf.matmul(x,weight['w1']), bias['b1'])
    # layer 2
    layer2 = tf.add(tf.matmul(layer1,weight['w2']), bias['b2'])
    # out_layer
    out_layer = tf.add(tf.matmul(layer2,weight['out']), bias['out'])
    return out_layer


# model
logits = neural_net(X)

# prediction
pred = tf.nn.softmax(logits)

# loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))

# optimizer
learning_rate = 0.1
# opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# evaluate
correct = tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# init
init = tf.global_variables_initializer()


# train
num_steps = 500
batch_size = 128
with tf.Session() as sess:

    sess.run(init)

    for step in range(1, num_steps+1):
        batchx, batchy = mnist.train.next_batch(batch_size)
        _, loss_, acc_ = sess.run([opt, loss, accuracy], feed_dict={X:batchx, Y:batchy})

        if step % 100 == 0 or step == 1:
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss_) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc_))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels}))


# todo visualize gradient of GradientDescentOptimizer/AdamOptimizer
'''question
为什么这里必须使用AdamOptimizer，而不是GradientDescentOptimizer？
'''


# Compare result:
# GradientDescentOptimizer vs. AdamOptimizer
'''
GradientDescentOptimizer
Step 1, Minibatch Loss= 3536.2437, Training Accuracy= 0.102
Step 100, Minibatch Loss= nan, Training Accuracy= 0.086
Step 200, Minibatch Loss= nan, Training Accuracy= 0.156
Step 300, Minibatch Loss= nan, Training Accuracy= 0.086
Step 400, Minibatch Loss= nan, Training Accuracy= 0.094
Step 500, Minibatch Loss= nan, Training Accuracy= 0.055
Optimization Finished!
Testing Accuracy: 0.098


AdamOptimizer
Step 1, Minibatch Loss= 3377.3652, Training Accuracy= 0.141
Step 100, Minibatch Loss= 208.3779, Training Accuracy= 0.883
Step 200, Minibatch Loss= 71.6571, Training Accuracy= 0.914
Step 300, Minibatch Loss= 66.3778, Training Accuracy= 0.820
Step 400, Minibatch Loss= 114.3219, Training Accuracy= 0.805
Step 500, Minibatch Loss= 110.8882, Training Accuracy= 0.797
Optimization Finished!
Testing Accuracy: 0.8603
'''


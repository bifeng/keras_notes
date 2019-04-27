'''
cs231n - Lecture 8 - Deep Learning Software
Train a two-layer ReLU network on random data with L2 loss
'''
import numpy as np

np.random.seed(0)
import tensorflow as tf

N, D, H = 64, 1000, 100

x = tf.placeholder(tf.float32, shape=(N, D))
y = tf.placeholder(tf.float32, shape=(N, D))

ini = tf.contrib.layers.xavier_initializer()
h = tf.layers.dense(inputs=x, units=H, activation=tf.nn.relu, kernel_initializer=ini)
pred = tf.layers.dense(inputs=h, units=D, kernel_initializer=ini)

loss = tf.losses.mean_squared_error(y, pred)

learning_rate = 0.01
opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    values = {
        x: np.random.randn(N, D),
        y: np.random.randn(N, D)}
    _, loss_ = sess.run([opt, loss], feed_dict=values)

'''

'''
import numpy as np

np.random.seed(0)
import tensorflow as tf

N, D, H = 64, 1000, 100

x = tf.placeholder(tf.float32, shape=(N, D))
y = tf.placeholder(tf.float32, shape=(N, D))

w1 = tf.Variable(initial_value=tf.truncated_normal((D, H)))
w2 = tf.Variable(initial_value=tf.truncated_normal((H, D)))

h = tf.maximum(tf.matmul(x, w1), 0)

pred = tf.matmul(h, w2)

loss = tf.losses.mean_squared_error(y, pred)

learning_rate = 0.01
opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    values = {
        x: np.random.randn(N, D),
        y: np.random.randn(N, D)}
    _, loss_ = sess.run([opt, loss], feed_dict=values)

'''

'''
import numpy as np

np.random.seed(0)
import tensorflow as tf

N, D, H = 64, 1000, 100

x = tf.placeholder(tf.float32, shape=(N, D))
y = tf.placeholder(tf.float32, shape=(N, D))

w1 = tf.Variable(initial_value=tf.truncated_normal((D, H)))
w2 = tf.Variable(initial_value=tf.truncated_normal((H, D)))

h = tf.maximum(tf.matmul(x, w1), 0)

pred = tf.matmul(h, w2)

loss = tf.reduce_mean(tf.reduce_sum((pred - y) ** 2, axis=1))

grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

learning_rate = 0.01
new_w1 = w1.assign(w1 - learning_rate * grad_w1)
new_w2 = w2.assign(w2 - learning_rate * grad_w2)
updates = tf.group(new_w1, new_w2)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    values = {
        x: np.random.randn(N, D),
        y: np.random.randn(N, D)}
    _, loss_ = sess.run([updates, loss], feed_dict=values)

'''

Train the network: Run the graph over and over, use gradient to update weights
Problem: copying weights between CPU/GPU each step
'''
import numpy as np

np.random.seed(0)
import tensorflow as tf

N, D, H = 64, 1000, 100

x = tf.placeholder(tf.float32, shape=(N, D))
y = tf.placeholder(tf.float32, shape=(N, D))

w1 = tf.placeholder(tf.float32, shape=(D, H))
w2 = tf.placeholder(tf.float32, shape=(H, D))

h = tf.maximum(tf.matmul(x, w1), 0)

pred = tf.matmul(h, w2)

loss = tf.reduce_mean(tf.reduce_sum((pred - y) ** 2, axis=1))

grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

with tf.Session() as sess:
    values = {
        x: np.random.randn(N, D),
        y: np.random.randn(N, D),
        w1: np.random.randn(D, N),
        w2: np.random.randn(N, D)}
    learning_rate = 1 * 10 ** (-5)
    for i in range(50):
        grad_w1_v, grad_w2_v, loss_ = sess.run([grad_w1, grad_w2, loss], feed_dict=values)
        values[w1] -= learning_rate * grad_w1_v
        values[w2] -= learning_rate * grad_w2_v

import tensorflow as tf


a = tf.constant([[4.0, 4.0, 4.0], [3.0, 3.0, 3.0], [1.0, 1.0, 1.0]])
b = tf.constant([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])


mse1 = tf.reduce_mean(tf.pow((a-b),2))
mse2 = tf.reduce_mean(tf.square(a-b))
mse3 = tf.losses.mean_squared_error(a,b)

with tf.Session() as sess:
    print(sess.run(mse1))
    print(sess.run(mse2))
    print(sess.run(mse3))


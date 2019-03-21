import tensorflow as tf

c = tf.random_uniform([], -10, 10, seed=2)
d = tf.random_uniform([], -10, 10, seed=2)
e = tf.random_uniform([], -10, 10, seed=2)

with tf.Session() as sess:
    print(sess.run(c))  # >> 3.57493
    print(sess.run(d))  # >> 3.57493
    print(sess.run(e))  # >> 3.57493


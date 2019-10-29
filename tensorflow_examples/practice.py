
# Practice 1
# import tensorflow as tf
# input = tf.constant([[0, 2],
#                      [1, 2],
#                      [0, 1]])
# # output:
# [[0, 0], [0, 2], [1, 1], [1, 2], [2, 0], [2, 1]]

import tensorflow as tf

input = tf.constant([[0, 2], [1, 2], [0, 1]])
B, S = input.shape
input2 = tf.reshape(input, [-1, 1])
index = tf.reshape(tf.constant(sorted((list(range(B))) * S)), [-1, 1])
# or
index = tf.reshape(tf.tile(tf.reshape(tf.range(B),(B,1)),[1,S]),[-1,1])
input = tf.concat([index, input2], axis=1)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
with sess.as_default():
    print(input2.eval())
    print(input.eval())

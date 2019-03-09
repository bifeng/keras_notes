import numpy as np
import tensorflow as tf


y = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,0]])
x = np.array([[12,3,2],[3,10,1],[1,2,5],[4,6.5,1.2],[3,6,1]])

pred1 = tf.nn.softmax(x)

cost1 = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred1), reduction_indices=1))
softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=x)
cost2 = tf.reduce_mean(softmax_cross_entropy)

with tf.Session() as sess:
    print(sess.run(cost1))
    print(sess.run(cost2))


tf.nn.softmax_cross_entropy_with_logits
tf.nn.softmax_cross_entropy_with_logits_v2
tf.nn.sigmoid_cross_entropy_with_logits
tf.nn.sparse_softmax_cross_entropy_with_logits
tf.nn.weighted_cross_entropy_with_logits


'''

https://stackoverflow.com/questions/37312421/whats-the-difference-between-sparse-softmax-cross-entropy-with-logits-and-softm

'''
import tensorflow as tf
from random import randint

dims = 8
pos  = randint(0, dims - 1)

logits = tf.random_uniform([dims], maxval=3, dtype=tf.float32)
labels = tf.one_hot(pos, dims)

res1 = tf.nn.softmax_cross_entropy_with_logits(       logits=logits, labels=labels)
res2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.constant(pos))

with tf.Session() as sess:
    a, b = sess.run([res1, res2])
    print(a, b)
    print(a == b)

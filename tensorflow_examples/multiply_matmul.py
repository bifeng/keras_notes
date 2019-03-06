import tensorflow as tf
from pprint import pprint as print
a = tf.reshape(tf.constant([1,2,3,4,5,6]), [2,3])
b = tf.reshape(tf.constant([1,2,3,4,5,6]), [3,2])
c = tf.reshape(tf.constant([1,2,3,4,5,6]), [2,3])

x = tf.matmul(a, b)
y = a * c
z = tf.multiply(a,c)


with tf.Session() as sess:
    print(sess.run([a,b,c, x, y,z]))

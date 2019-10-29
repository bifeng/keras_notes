# case 1
import tensorflow as tf

sess = tf.InteractiveSession()

values = tf.constant([[1,3],
                      [0,2],
                      [1,3]])

T = tf.constant([[0, 1, 2 ,  3],
                 [4, 5, 6 ,  7],
                 [8, 9, 10, 11]])


result = tf.gather_nd(T, values)
print(result.eval())
# [7 2 7]

result2 = tf.reshape(values, (2,3))
print(result2.eval())

refer: CS20si



#### feeding values to TF ops

**Extremely helpful for testing Feed in dummy values to test parts of a large graph**




```python
import tensorflow as tf

# create operations, tensors, etc (using the default graph)
a = tf.add(2, 5)
b = tf.multiply(a, 3)

with tf.Session() as sess:
    # compute the value of b given a is 15
    print(sess.run(b))
    print(sess.run(b, feed_dict={a: 15}))
```
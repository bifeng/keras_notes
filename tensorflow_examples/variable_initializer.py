import tensorflow as tf

x = tf.Variable(initial_value=tf.constant(0.0), name='x')
y = tf.constant(0.5)
z = tf.Variable(initial_value=tf.constant(1.0), name='z', collections=[tf.GraphKeys.LOCAL_VARIABLES])

init = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    sess.run(init_local)
    print(sess.run([x,y]))
    print(sess.run(z))


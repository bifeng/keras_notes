'''
https://stackoverflow.com/questions/34775522/tensorflow-multiple-sessions-with-multiple-gpus

'''

'''
multiple session - you need separate running the session in each script
'''
import tensorflow as tf

W = tf.Variable(10)

sess1 = tf.Session()
sess2 = tf.Session()

sess1.run(W.initializer)
sess2.run(W.initializer)

print(sess1.run(W.assign_add(10)))  # >> 20
print(sess2.run(W.assign_sub(2)))  # >> 8

print(sess1.run(W.assign_add(100)))  # >> 120
print(sess2.run(W.assign_sub(50)))  # >> -42

sess1.close()
sess2.close()


'''
multiple graph
'''
g1 = tf.get_default_graph()
g2 = tf.Graph()
# add ops to the default graph
with g1.as_default():
    a = tf.constant(3)
# add ops to the user created graph
with g2.as_default():
    b = tf.constant(5)

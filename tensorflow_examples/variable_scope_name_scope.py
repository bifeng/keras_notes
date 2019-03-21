'''
Variable is not share
'''
import tensorflow as tf

x1 = tf.truncated_normal([200, 100], name='x1')
x2 = tf.truncated_normal([200, 100], name='x2')


def two_hidden_layer(x):
    w1 = tf.Variable(initial_value=tf.random_normal([100, 50]), name='w1_weight')
    b1 = tf.Variable(tf.zeros([50]), name='b1_bias')
    h1 = tf.matmul(x, w1) + b1

    w2 = tf.Variable(initial_value=tf.random_normal([50, 10]), name='w2_weight')
    b2 = tf.Variable(tf.zeros([10]), name='b2_bias')
    h2 = tf.matmul(h1, w2) + b2
    return h2


logit1 = two_hidden_layer(x1)
logit2 = two_hidden_layer(x2)

init = tf.global_variables_initializer()

with tf.Session() as sess1:
    sess1.run(init)
    writer = tf.summary.FileWriter('./graphs/scope1', sess1.graph)
    sess1.run([logit1, logit2])
    writer.close()

'''
Variable is sharing
'''
import tensorflow as tf

x1 = tf.truncated_normal([200, 100], name='x1')
x2 = tf.truncated_normal([200, 100], name='x2')


def two_hidden_layer(x):
    w1 = tf.get_variable(initializer=tf.random_normal([100, 50]), name='w1_weight')
    b1 = tf.get_variable(initializer=tf.zeros([50]), name='b1_bias')
    h1 = tf.matmul(x, w1) + b1

    w2 = tf.get_variable(initializer=tf.random_normal([50, 10]), name='w2_weight')
    b2 = tf.get_variable(initializer=tf.zeros([10]), name='b2_bias')
    h2 = tf.matmul(h1, w2) + b2
    return h2


with tf.variable_scope('two_layers') as scope:
    logit1 = two_hidden_layer(x1)
    scope.reuse_variables()  # reuse variables
    logit2 = two_hidden_layer(x2)

init = tf.global_variables_initializer()

with tf.Session() as sess2:
    sess2.run(init)
    writer = tf.summary.FileWriter('./graphs/scope2', sess2.graph)
    sess2.run([logit1, logit2])
    writer.close()

'''
Variable is sharing - simplified version
'''
import tensorflow as tf

x1 = tf.truncated_normal([200, 100], name='x1')
x2 = tf.truncated_normal([200, 100], name='x2')


def fully_connect_layer(x, out_dim, scope):
    with tf.variable_scope(scope) as scope:
        w = tf.get_variable("weights", [x.shape[1], out_dim], initializer=tf.random_normal_initializer())
        b = tf.get_variable("biases", [out_dim], initializer=tf.constant_initializer(0.0))
        # w = tf.get_variable(initializer=tf.random_normal([x.shape[1], out_dim]), name='weights')
        # b = tf.get_variable(initializer=tf.zeros([out_dim]), name='biases')
        # Wrong! todo why?
        '''
        TypeError: Failed to convert object of type <class 'list'> to Tensor. Contents: [Dimension(100), 50]. 
        Consider casting elements to a supported type.
        '''
        h = tf.matmul(x, w) + b
    return h


def two_hidden_layer(x):
    h1 = fully_connect_layer(x, 50, 'h1')
    h2 = fully_connect_layer(h1, 10, 'h2')
    return h2


with tf.variable_scope('two_layers') as scope:
    logit1 = two_hidden_layer(x1)
    scope.reuse_variables()  # reuse variables
    logit2 = two_hidden_layer(x2)

init = tf.global_variables_initializer()

with tf.Session() as sess3:
    sess3.run(init)
    writer = tf.summary.FileWriter('./graphs/scope3', sess3.graph)
    sess3.run([logit1, logit2])
    writer.close()

'''
Variable is sharing - simplified version2
'''
import tensorflow as tf

x1 = tf.truncated_normal([200, 100], name='x1')
x2 = tf.truncated_normal([200, 100], name='x2')


def fully_connect_layer(x, out_dim, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:  # reuse variables
        w = tf.get_variable("weights", [x.shape[1], out_dim], initializer=tf.random_normal_initializer())
        b = tf.get_variable("biases", [out_dim], initializer=tf.constant_initializer(0.0))
        h = tf.matmul(x, w) + b
    return h


def two_hidden_layer(x):
    h1 = fully_connect_layer(x, 50, 'h1')
    h2 = fully_connect_layer(h1, 10, 'h2')
    return h2


with tf.variable_scope('two_layers') as scope:
    logit1 = two_hidden_layer(x1)
    logit2 = two_hidden_layer(x2)

init = tf.global_variables_initializer()

with tf.Session() as sess4:
    sess4.run(init)
    writer = tf.summary.FileWriter('./graphs/scope4', sess4.graph)
    sess4.run([logit1, logit2])
    writer.close()

'''
tensorboard --logdir="./graphs/scope1"
'''
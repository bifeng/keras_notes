"""
why the dynamic_rnn have two outputs: output and state?

The cell output equals to the hidden state.
But for tf.contrib.nn.dynamic_rnn, the returned state may be different when the sequence is shorter (sequence_length argument).

The state is a convenient tensor that holds the last actual RNN state, ignoring the zeros.
The output tensor holds the outputs of all cells, so it doesn't ignore the zeros.
That's the reason for returning both of them.

https://stats.stackexchange.com/questions/330176/what-is-the-output-of-a-tf-nn-dynamic-rnn
https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html
"""

"""
dynamic rnn
"""
import tensorflow as tf
import numpy as np

n_steps = 2
n_inputs = 3
n_neurons = 5

X = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_inputs])
seq_length = tf.placeholder(tf.int32, [None])

basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, sequence_length=seq_length, dtype=tf.float32)

X_batch = np.array([
  # t = 0      t = 1
  [[0, 1, 2], [9, 8, 7]], # instance 0
  [[3, 4, 5], [0, 0, 0]], # instance 1
  [[6, 7, 8], [6, 5, 4]], # instance 2
  [[9, 0, 1], [3, 2, 1]], # instance 3
])
seq_length_batch = np.array([2, 1, 2, 2])

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  outputs_val, states_val = sess.run([outputs, states],
                                     feed_dict={X: X_batch, seq_length: seq_length_batch})

  print(outputs_val)
  print('-------------')
  print(states_val)

"""
[[[-0.31307542  0.7681713  -0.06069343 -0.13103582  0.38810092]
  [ 0.90247655  0.99998987  0.9663651  -0.9997993   0.99775183]]
 [[-0.06315047  0.9962653   0.55165243 -0.95261353  0.9524636 ]
  [ 0.          0.          0.          0.          0.        ]]
 [[ 0.19495435  0.9999468   0.8623077  -0.99846804  0.99731416]
  [ 0.6090871   0.9999036   0.91982526 -0.9891903   0.98514694]]
 [[ 0.9792592   0.99914706 -0.40777573 -0.9994195   0.994314  ]
  [ 0.51258177  0.9659592   0.5879667  -0.91654384  0.63013095]]]
 -------------
[[ 0.90247655  0.99998987  0.9663651  -0.9997993   0.99775183]
 [-0.06315047  0.9962653   0.55165243 -0.95261353  0.9524636 ]
 [ 0.6090871   0.9999036   0.91982526 -0.9891903   0.98514694]
 [ 0.51258177  0.9659592   0.5879667  -0.91654384  0.63013095]]
"""

#################################################
"""
dynamic rnn
"""
#################################################
"""
Placeholders
"""

x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')
init_state = tf.zeros([batch_size, state_size])

"""
Inputs
"""

rnn_inputs = tf.one_hot(x, num_classes)

"""
RNN
"""

cell = tf.contrib.rnn.BasicRNNCell(state_size)
rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

"""
Predictions, loss, training step
"""

with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
logits = tf.reshape(
            tf.matmul(tf.reshape(rnn_outputs, [-1, state_size]), W) + b,
            [batch_size, num_steps, num_classes])
predictions = tf.nn.softmax(logits)

losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)


#################################################
""" 
static rnn
"""
#################################################

"""
Placeholders
"""

x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')
init_state = tf.zeros([batch_size, state_size])

"""
Inputs
"""

x_one_hot = tf.one_hot(x, num_classes)
rnn_inputs = tf.unstack(x_one_hot, axis=1)

"""
RNN
"""

cell = tf.contrib.rnn.BasicRNNCell(state_size)
rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state=init_state)

"""
Predictions, loss, training step
"""

with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
predictions = [tf.nn.softmax(logit) for logit in logits]

y_as_list = tf.unstack(y, num=num_steps, axis=1)

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit) for \
          logit, label in zip(logits, y_as_list)]
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)
import numpy as np
import tensorflow as tf

batch = 2
dim = 3
hidden = 4

lengths = tf.placeholder(dtype=tf.int32, shape=[batch])
inputs = tf.placeholder(dtype=tf.float32, shape=[batch, None, dim])
cell = tf.nn.rnn_cell.GRUCell(hidden)
cell_state = cell.zero_state(batch, tf.float32)
output, state_ = tf.nn.dynamic_rnn(cell, inputs, lengths, initial_state=cell_state)

inputs_ = np.asarray([[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
                    [[6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]]],
                    dtype=np.int32)  # (2,4,3) (batch_size,sequence_length,dim)
lengths_ = np.asarray([3, 1], dtype=np.int32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output_, state_ = sess.run([output,state_], {inputs: inputs_, lengths: lengths_})
    print(output_)  # (2,4,4) (batch_size,sequence_length,hidden_size)
    print('--------')
    print(state_)  # (2,4) (batch_size, hidden_size)


"""
[[[ 0.          0.          0.          0.        ]
  [-0.25622493  0.08511242 -0.08345975  0.17080714]
  [-0.6236931   0.15141508 -0.22935116  0.41420087]
  [ 0.          0.          0.          0.        ]]

 [[-0.85967606  0.01181258 -0.7444219   0.6083782 ]
  [ 0.          0.          0.          0.        ]
  [ 0.          0.          0.          0.        ]
  [ 0.          0.          0.          0.        ]]]
--------
[[-0.6236931   0.15141508 -0.22935116  0.41420087]
 [-0.85967606  0.01181258 -0.7444219   0.6083782 ]]
 """


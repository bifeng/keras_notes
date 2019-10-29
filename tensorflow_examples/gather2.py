# Select rows
import tensorflow as tf

values2 = tf.constant([[0, 2],
                       [1, 2],
                       [0, 1]])  # batch_size=3 entity_pos=2

T = tf.constant([[[0, 1, 2, 3],
                  [4, 5, 6, 7],
                  [8, 9, 10, 11]],
                 [[1, 0, 3, 1],
                  [6, 4, 2, 1],
                  [0, 3, 5, 8]],
                 [[3, 3, 8, 2],
                  [4, 9, 0, 3],
                  [7, 4, 6, 2]]])  # batch_size=3 sequence_length=3 hidden_size=4


# # method 1
# B, S = values2.shape
# input2 = tf.reshape(values2, [-1, 1])
# index = tf.reshape(tf.constant(sorted((list(range(B))) * S)), [-1, 1])
# input = tf.concat([index, input2], axis=1)
# # [[0, 0], [0, 2], [1, 1], [1, 2], [2, 0], [2, 1]]
# gather_value = tf.gather_nd(T, input)
# print(gather_value.shape)

# method 2
sequence_shape = T.shape.as_list()
batch_size = sequence_shape[0]
seq_length = sequence_shape[1]
width = sequence_shape[2]

flat_offsets = tf.reshape(
    tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
flat_positions = tf.reshape(values2 + flat_offsets, [-1])
flat_sequence_tensor = tf.reshape(T,
                                  [batch_size * seq_length, width])
output_tensor = tf.gather(flat_sequence_tensor, flat_positions)


# method 3
# values = tf.constant([[1, 0, 1],
#                       [0, 1, 1],
#                       [1, 0, 1]])  # batch_size=3 sequence_length=3
# mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
# entity_pos_tensor = mul_mask(T, values)
# intermediate_tensor = tf.reduce_sum(tf.abs(entity_pos_tensor), 1)
# zero_vector = tf.zeros(shape=(1,1))
# bool_mask = tf.not_equal(tf.cast(intermediate_tensor, tf.float32), zero_vector)
# omit_zeros = tf.boolean_mask(entity_pos_tensor, bool_mask)


with tf.Session() as sess:
    # print(sess.run(input))
    print(sess.run(flat_offsets))
    print(sess.run(flat_positions))
    print(sess.run(flat_sequence_tensor))
    print(sess.run(output_tensor))

"""
[[0 0]
 [0 2]
 [1 1]
 [1 2]
 [2 0]
 [2 1]]
[[ 0  1  2  3]
 [ 8  9 10 11]
 [ 6  4  2  1]
 [ 0  3  5  8]
 [ 3  3  8  2]
 [ 4  9  0  3]]
"""
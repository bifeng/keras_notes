import tensorflow as tf

A = tf.constant([[1, 2], [3, 4]])
indices = tf.constant([1, 0])

# prepare row indices
row_indices = tf.range(tf.shape(indices)[0])

# zip row indices with column indices
full_indices = tf.stack([row_indices, indices], axis=1)

# retrieve values by indices
S = tf.gather_nd(A, full_indices)

session = tf.InteractiveSession()
print(full_indices.eval())
print(session.run(S))

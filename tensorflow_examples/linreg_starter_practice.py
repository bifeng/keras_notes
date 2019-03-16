import numpy as np
def read_birth_life_data(filename):
    """
    Read in birth_life_2010.txt and return:
    data in the form of NumPy array
    n_samples: number of samples
    """
    text = open(filename, 'r').readlines()[1:]
    data = [line[:-1].split('\t') for line in text]
    births = [float(line[1]) for line in data]
    lifes = [float(line[2]) for line in data]
    data = list(zip(births, lifes))
    n_samples = len(data)
    data = np.asarray(data, dtype=np.float32)
    return data, n_samples

###########################


import tensorflow as tf

path = 'C:\\Users\\kiss\\deep_coding_notes\\tensorflow_examples\\stanford-tensorflow-tutorials\\examples\\'
DATA_FILE = path + 'data\\birth_life_2010.txt'
data, n_samples = read_birth_life_data(DATA_FILE)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.get_variable(initializer=tf.constant_initializer)
b = tf.get_variable(initializer=tf.constant_initializer)

pred = tf.multiply(X,W) + b
loss = tf.square(pred-Y)

learning_rate = 0.01
opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimize = opt.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run([opt], feed_dict=[data,])




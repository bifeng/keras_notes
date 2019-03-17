import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a, b, name='add')
writer = tf.summary.FileWriter('./graph', tf.get_default_graph())
with tf.Session() as sess:
	print(sess.run(x))
writer.close()

''' Check in tensorboard:
tensorboard --logdir="./graph" --port 6006      # 6006 or any port you want

注意 - "./graph" 路径两边是双引号
'''


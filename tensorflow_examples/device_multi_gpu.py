'''
tf.device
tf.Graph.device

https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py
'''
import tensorflow as tf

with tf.device('/gpu:0'):  # ...


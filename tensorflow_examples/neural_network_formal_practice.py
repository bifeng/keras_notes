# 1. 构建计算图
# - 输入、权重变量
# - 模型
# - 损失函数
# - 优化器（参数-learning rate/）
# - 参数初始化
# - 训练（参数-epoch/）（初始化/优化器）
#   把数据喂进去

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)


out_dimension = 10
num_hiddens_1 = 256
num_hiddens_2 = 256


def neural_net(x_dict):
    x = x_dict['images']
    layer1 = tf.layers.dense(x, num_hiddens_1)
    layer2 = tf.layers.dense(layer1, num_hiddens_2)
    out_layer = tf.layers.dense(layer2, out_dimension)
    return out_layer


learning_rate = 0.1


def model_fn(features, labels, mode):
    # model
    logits = neural_net(features)

    # Predictions
    pred = tf.nn.softmax(logits)
    pred_classes = tf.argmax(pred, axis=1)

    # # or
    # pred_classes = tf.argmax(logits, axis=1)
    # pred_probas = tf.nn.softmax(logits)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # loss
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(labels, dtype=tf.int32),logits=logits))

    # opt
    global_step = tf.train.get_global_step()
    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)

    # evaluate
    acc = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss,
        train_op=opt,
        eval_metric_ops={'accuracy':acc}
    )
    return estim_specs


# Build the Estimator
model = tf.estimator.Estimator(model_fn)

batch_size = 128
# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images}, y=mnist.train.labels,
    batch_size=batch_size, num_epochs=None, shuffle=True)

# Train the Model
num_steps = 1000
model.train(input_fn, steps=num_steps)

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images}, y=mnist.test.labels,
    batch_size=batch_size, shuffle=False)

# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])


# todo visualize gradient of GradientDescentOptimizer/AdamOptimizer too
'''question
为什么这里可以使用GradientDescentOptimizer - 居然可以迭代？
（nerual_network_practice.py必须用AdamOptimizer才可以.）
'''


''' Notes:
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)  # change one_hot from False to True

raise error:
softmax_cross_entropy_with_logits
ValueError: Can not squeeze dim[1], expected a dimension of 1, got 10 for 'remove_squeezable_dimensions/Squeeze' (op: 'Squeeze') with input shapes: [128,10].

sparse_softmax_cross_entropy_with_logits
ValueError: Rank mismatch: Rank of labels (received 2) should equal rank of logits minus 1 (received 2).
'''

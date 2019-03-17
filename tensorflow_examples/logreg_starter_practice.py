import tensorflow as tf
from tensorflow_examples.stanford_tensorflow_tutorials.examples import utils

path = 'C:/Users/kiss/deep_coding_notes/tensorflow_examples/'
mnist_folder = path + 'tmp/mnist'
# utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)

# todo struct.error: unpack requires a buffer of 8 bytes

batchsize = 128
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000) # optional
train_data = train_data.batch(batch_size=batchsize)

n_test = 10000
test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.batch(batch_size=batchsize)

# Creating an iterator with different initializers!
iterator = tf.data.Iterator.from_structure(train_data.output_shapes, train_data.output_types)
img, label = iterator.get_next()
train_init = iterator.make_initializer(train_data)
test_init = iterator.make_initializer(test_data)

W = tf.get_variable(name='weights', shape=(784, 10), initializer=tf.random_normal_initializer(0, 0.01))
b = tf.get_variable(name='bias', shape=(1, 10), initializer=tf.zeros_initializer())

logits = tf.matmul(img, W) + b
pred = tf.nn.softmax(logits)

entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label,logits=logits)
loss = tf.reduce_mean(entropy)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(label,1))
accuracy = tf.reduce_sum(tf.cast(correct_pred,tf.float32))

learning_rate = 0.01
opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer = opt.minimize(loss)


init = tf.global_variables_initializer()

writer = tf.summary.FileWriter('./graph/logreg', tf.get_default_graph())

'''
We have to catch the OutOfRangeError because miraculously, TensorFlow doesn’t automatically catch it for us.
'''
epoch = 30
with tf.Session() as sess:

    sess.run(init)

    for i in range(epoch):
        sess.run(train_init)  # use train_init during training loop
        total_loss = 0
        batches = 0
        try:
            while True:
                _, loss_ = sess.run([optimizer, loss])
                total_loss += loss_
                batches += 1
        except tf.errors.OutOfRangeError:
            pass

        print('Average loss epoch {0}:{1} '.format(i, total_loss/batches))

    sess.run(test_init)
    total_acc = 0
    try:
        while True:
            acc_ = sess.run(accuracy)
            total_acc += acc_
    except tf.errors.OutOfRangeError:
        pass

    print('Accuracy {0}'.format(total_acc / n_test))

writer.close()


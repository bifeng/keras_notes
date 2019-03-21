'''
question
1. what's the difference between input weight and output weight?
2. log linear activation function?
3. training/inference - softmax used in inference?
4. how to create dataset and training for multiple context words?
5. how many parameters?
6. how to coding the NCE?


1. yield
2.


'''
import tensorflow as tf
from tensorflow_examples.stanford_tensorflow_tutorials.examples import utils
from tensorflow_examples.stanford_tensorflow_tutorials.examples import word2vec_utils

# Parameters for downloading data
DOWNLOAD_URL = 'http://mattmahoney.net/dc/text8.zip'
EXPECTED_BYTES = 31344016
VISUAL_FLD = 'visualization'

VOCAB_SIZE = 50000
SKIP_WINDOW = 1
BATCH_SIZE = 128
EMBED_SIZE = 128
NUM_SAMPLED = 64

NUM_TRAIN_STEPS = 100000
LEARNING_RATE = 1.0


def gen():
    yield from word2vec_utils.batch_gen(DOWNLOAD_URL, EXPECTED_BYTES, VOCAB_SIZE,
                                        BATCH_SIZE, SKIP_WINDOW, VISUAL_FLD)


dataset = tf.data.Dataset.from_generator(gen,
                                         (tf.int32, tf.int32),
                                         (tf.TensorShape([BATCH_SIZE]), tf.TensorShape([BATCH_SIZE, 1])))
iterator = dataset.make_initializable_iterator()
center_words, target_words = iterator.get_next()

embed_matrix = tf.get_variable('embed_matrix',
                               shape=[VOCAB_SIZE, EMBED_SIZE],
                               initializer=tf.random_uniform_initializer())

embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')

nce_weight = tf.get_variable('nce_weight',
                             shape=[VOCAB_SIZE, EMBED_SIZE],
                             initializer=tf.truncated_normal_initializer(stddev=1.0 / (EMBED_SIZE ** 0.5)))
nce_bias = tf.get_variable('nce_bias', initializer=tf.zeros([VOCAB_SIZE]))

loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                     biases=nce_bias,
                                     labels=target_words,
                                     inputs=embed,
                                     num_sampled=NUM_SAMPLED,
                                     num_classes=VOCAB_SIZE))

optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)


init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run([init, iterator.initializer])  # initializer for first epoch

    writer = tf.summary.FileWriter('graphs/word2vec_simple', sess.graph)

    for index in range(NUM_TRAIN_STEPS):
        try:
            _, loss_ = sess.run([optimizer,loss])
        except tf.errors.OutOfRangeError:
            sess.run(iterator.initializer)  # initializer for next epoch

    writer.close()

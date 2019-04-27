# site-packages\bert_serving\server\graph.py
# change three place:

# one - add the 'CLASSIFICATION' choice for PoolingStrategy
class PoolingStrategy(Enum):
    NONE = 0
    REDUCE_MAX = 1
    REDUCE_MEAN = 2
    REDUCE_MEAN_MAX = 3
    FIRST_TOKEN = 4  # corresponds to [CLS] for single sequences
    LAST_TOKEN = 5  # corresponds to [SEP] for single sequences
    CLS_TOKEN = 4  # corresponds to the first token for single seq.
    SEP_TOKEN = 5  # corresponds to the last token for single seq.
    CLASSIFICATION = 4

# two - add this before 'tvars = tf.trainable_variables()'
if args.pooling_strategy == PoolingStrategy.CLASSIFICATION:
    hidden_size = 768
    output_weights = tf.get_variable(
        "output_weights", [args.num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [args.num_labels], initializer=tf.zeros_initializer())

tvars = tf.trainable_variables()

# three - add 'PoolingStrategy.CLASSIFICATION' before 'PoolingStrategy.FIRST_TOKEN or PoolingStrategy.CLS_TOKEN'
elif args.pooling_strategy == PoolingStrategy.CLASSIFICATION:
pooled = tf.squeeze(encoder_layer[:, 0:1, :], axis=1)
logits = tf.matmul(pooled, output_weights, transpose_b=True)
logits = tf.nn.bias_add(logits, output_bias)
pooled = tf.nn.softmax(logits, axis=-1)
elif args.pooling_strategy == PoolingStrategy.FIRST_TOKEN or \
     args.pooling_strategy == PoolingStrategy.CLS_TOKEN:
pooled = tf.squeeze(encoder_layer[:, 0:1, :], axis=1)
elif args.pooling_strategy == PoolingStrategy.LAST_TOKEN or \
     args.pooling_strategy == PoolingStrategy.SEP_TOKEN:
seq_len = tf.cast(tf.reduce_sum(input_mask, axis=1), tf.int32)
rng = tf.range(0, tf.shape(seq_len)[0])
indexes = tf.stack([rng, seq_len - 1], 1)
pooled = tf.gather_nd(encoder_layer, indexes)

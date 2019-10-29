https://github.com/taki0112/Tensorflow-Cookbook

https://github.com/nfmcclure/tensorflow_cookbook

https://github.com/vahidk/EffectiveTensorflow

refer:<br>https://hanxiao.github.io/2017/11/08/Optimizing-Contrastive-Rank-Triplet-Loss-in-Tensorflow-for-Neural/



#### embedding



##### multiple ways of using pre-trained embedding

1. tf.Variable(trainable=True) - Create `W` as a `tf.Variable` and initialize it from the NumPy array via a [`tf.placeholder()`](https://www.tensorflow.org/api_guides/python/io_ops#placeholder):

```python
W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                trainable=False, name="W")  
# trainable=False for static/trainable=True for fine-tuning

embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
embedding_init = W.assign(embedding_placeholder)

# ...
sess = tf.Session()

sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})
```



```python
W_text = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.25, 0.25), name="W_text")
embedded_chars = tf.nn.embedding_lookup(W_text, input_text)

# ...
sess = tf.Session()

sess.run(W_text.assign(embedding))
```



2. tf.get_variable(trainable=None) - 

   ```python
       with tf.variable_scope(var_scope or 'word_embedding', reuse=tf.AUTO_REUSE):
           word_embedding = tf.get_variable('word_embedding', initializer=word_vec_mat, dtype=tf.float32)
           if add_unk_and_blank:
               word_embedding = tf.concat([word_embedding,
                                           tf.get_variable("unk_word_embedding", [1, word_embedding_dim], dtype=tf.float32,
                                               initializer=tf.contrib.layers.xavier_initializer()),
                                           tf.constant(np.zeros((1, word_embedding_dim), dtype=np.float32))], 0)
           x = tf.nn.embedding_lookup(word_embedding, word)
   ```

   

3. one hot embedding

```python
  use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
    embeddings or tf.embedding_lookup() for the word embeddings.
    如果True，使用矩阵乘法实现提取词的Embedding；否则用tf.embedding_lookup()
    对于TPU，使用前者更快，对于GPU和CPU，后者更快。
    
  ...
  embedding_table = tf.get_variable(
      name=word_embedding_name,
      shape=[vocab_size, embedding_size],
      initializer=create_initializer(initializer_range))

  flat_input_ids = tf.reshape(input_ids, [-1])
  if use_one_hot_embeddings:
    one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
    output = tf.matmul(one_hot_input_ids, embedding_table)
  else:
    output = tf.gather(embedding_table, flat_input_ids)  # embedding_lookup
```



4. If the embedding was trained as part of another TensorFlow model, you can use a [`tf.train.Saver`](https://www.tensorflow.org/api_guides/python/state_ops#Saver) to load the value from the other model's checkpoint file. This means that the embedding matrix can bypass Python altogether. 

```python
W = tf.Variable(...)

embedding_saver = tf.train.Saver({"name_of_variable_in_other_model": W})

# ...
sess = tf.Session()
embedding_saver.restore(sess, "checkpoint_filename.ckpt")
```





refer:

https://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow?rq=1



##### unk

1. Ignore the UNK word
2. Use a common random vector for all UNK word
3. Use a unique random vector for each UNK word
4. Use fasttext if there are lots of UNK word

##### only training parts of embedding

- Two embedding layers - one trainable and one not
- The non-trainable one has all the Glove embeddings for in-vocab words and zero vectors for others
- The trainable one only maps the OOV words and special symbols
- The output of these two layers is added (I was thinking of this like ResNet)
- The Conv/LSTM/etc below the embedding is unchanged

```python
    # Normal embs - '+2' for empty token and OOV token
    embedding_matrix = np.zeros((vocab_len + 2, emb_dim))
    # Special embs
    special_embedding_matrix = np.zeros((special_tokens_len + 2, emb_dim))

    # Here we may apply pre-trained embeddings to embedding_matrix

    embedding_layer = Embedding(vocab_len + 2,
                        emb_dim,
                        mask_zero = True,
                        weights = [embedding_matrix],
                        input_length = MAX_SENT_LEN,
                        trainable = False)

    special_embedding_layer = Embedding(special_tokens_len + 2,
                            emb_dim,
                            mask_zero = True,
                            weights = [special_embedding_matrix],
                            input_length = MAX_SENT_LEN,
                            trainable = True)

    valid_words = vocab_len - special_tokens_len

    sentence_input = Input(shape=(MAX_SENT_LEN,), dtype='int32')

    # Create a vector of special tokens, e.g: [0,0,1,0,3,0,0]
    special_tokens_input = Lambda(lambda x: x - valid_words)(sentence_input)
    special_tokens_input = Activation('relu')(special_tokens_input)

    # Apply both 'normal' embeddings and special token embeddings
    embedded_sequences = embedding_layer(sentence_input)
    embedded_special = special_embedding_layer(special_tokens_input)

    # Add the matrices
    embedded_sequences = Add()([embedded_sequences, embedded_special])
```

```python
import tensorflow as tf
import numpy as np

EMB_DIM = 300
def load_pretrained_glove():
    return ["a", "cat", "sat", "on", "the", "mat"], np.random.rand(6, EMB_DIM)

def get_train_vocab():
    return ["a", "dog", "sat", "on", "the", "mat"]

def embed_tensor(string_tensor, trainable=True):
  """
  Convert List of strings into list of indices then into 300d vectors
  """
  # ordered lists of vocab and corresponding (by index) 300d vector
  pretrained_vocab, pretrained_embs = load_pretrained_glove()
  train_vocab = get_train_vocab()
  only_in_train = list(set(train_vocab) - set(pretrained_vocab))
  vocab = pretrained_vocab + only_in_train

  # Set up tensorflow look up from string word to unique integer
  vocab_lookup = tf.contrib.lookup.index_table_from_tensor(
    mapping=tf.constant(vocab),
    default_value=len(vocab))
  string_tensor = vocab_lookup.lookup(string_tensor)

  # define the word embedding
  pretrained_embs = tf.get_variable(
      name="embs_pretrained",
      initializer=tf.constant_initializer(np.asarray(pretrained_embs), dtype=tf.float32),
      shape=pretrained_embs.shape,
      trainable=trainable)
  train_embeddings = tf.get_variable(
      name="embs_only_in_train",
      shape=[len(only_in_train), EMB_DIM],
      initializer=tf.random_uniform_initializer(-0.04, 0.04),
      trainable=trainable)
  unk_embedding = tf.get_variable(
      name="unk_embedding",
      shape=[1, EMB_DIM],
      initializer=tf.random_uniform_initializer(-0.04, 0.04),
      trainable=False)

  embeddings = tf.concat([pretrained_embs, train_embeddings, unk_embedding], axis=0)

  return tf.nn.embedding_lookup(embeddings, string_tensor)
```

https://stackoverflow.com/questions/49009386/train-only-some-word-embeddings-keras?rq=1

https://stackoverflow.com/questions/45113130/how-to-add-new-embeddings-for-unknown-words-in-tensorflow-training-pre-set-fo

more refer:

https://stackoverflow.com/questions/35803425/update-only-part-of-the-word-embedding-matrix-in-tensorflow

#### loss

##### rank loss

define the objective function:

1. $\min \ g(q,d^-) - g(q,d^+)$ 

   backwards: it's unbounded hence won't converge

2. $\min \  \max(0, \epsilon + g(q,d^-) - g(q,d^+))$ 

   The max function can ignore large negative values, and $\epsilon$ is to avoid $ g(q,d^-) = g(q,d^+)$.

   It is basically a hinge loss. In fact, we could use any loss function besides the hinge loss, e.g. logistic loss, exponential loss. As for the metric, we also have plenty of options, e.g. cosine, $\ell_1/\ell_2$-norm. We could even parametrize the metric function with a multi-layer perceptron and learn it from the data.

   Given a training set of $N$ triplets $\{ q_i, \{d^{+}_{i,j},\ldots\}, \{d^{-}_{i,k},\ldots\} \}$, the final loss function of the model has the following general form:
   $$
   min \sum_{i=1}^{N} \sum_{j=1}^{∣d_{i}^{+}∣} \sum_{k=1}^{∣d_{i}^{−}∣} w_{i,j}ℓ(g(q_i,d_{i,j}^{+}),g(q_i,d_{i,k}^{−}))
   $$
   where $w_{i,j}$ is the weight of the positive query-document pair. In practice, this could be the click-through rate, or log number of clicks mined from the query-log. 

   (You can use the weight to balance those few-shot/long-tail queries. You may also use something like `log1p(num_clicks_queryi_on_productj)` to value more on those popular (q,d) pairs.) 

   

   For functions $\ell$ and $g$, the options are:

   - **Loss function $\ell$**: logistic, exponential, hinge loss, etc.
   - **Metric function $g$**: cosine similarity, euclidean distance (i.e. $\ell_2$-norm), MLP, etc.

   

   ## Metric Layer Implementation

   ```python
   with tf.variable_scope('Metric_Layer'):
       q_norm = tf.nn.l2_normalize(tf.expand_dims(query, 1), 2)
       d_pos_norm = tf.nn.l2_normalize(doc_pos, 2)
       d_neg_norm = tf.nn.l2_normalize(doc_neg, 2)
       if metric == 'cosine':
           metric_p = tf.reduce_sum(q_norm * d_pos_norm, axis=2, name='cos_sim_pos')
           metric_n = tf.reduce_sum(q_norm * d_neg_norm, axis=2, name='cos_sim_neg')
       elif metric == 'l2':
           metric_p = - tf.norm(q_norm - d_pos_norm, axis=2)
           metric_n = - tf.norm(q_norm - d_neg_norm, axis=2)
       elif metric == 'mlp':
           q_dp = tf.concat([tf.tile(q_norm, [1, NUM_POS, 1]), d_pos_norm], axis=2)
           q_dn = tf.concat([tf.tile(q_norm, [1, NUM_NEG, 1]), d_neg_norm], axis=2)
           metric_p = q_dp
           metric_n = q_dn
           for l_size in ([64, 32, 16][:mlp_metric_layer] + [1]):
               metric_p = tf.layers.dense(inputs=metric_p, units=l_size,
                    name='output_layer_%d' % l_size,
                    kernel_initializer=glorot_uniform_initializer(),
                    bias_initializer=constant_initializer(0.0),
                    activation=tf.nn.softplus)
               metric_n = tf.layers.dense(inputs=metric_n, units=l_size,
                    name='output_layer_%d' % l_size,
                    kernel_initializer=glorot_uniform_initializer(),
                    bias_initializer=constant_initializer(0.0),
                    activation=tf.nn.softplus,
                    reuse=True)
       else:
           raise NotImplementedError
   ```

   

   ## Loss Layer Implementation

   ```python
   with tf.variable_scope('Loss_layer'):
       metric_p = tf.tile(tf.expand_dims(metric_p, axis=2), [1, 1, NUM_NEG])
       metric_n = tf.tile(tf.expand_dims(metric_n, axis=1), [1, NUM_POS, 1])
       delta = metric_n - metric_p
       
       # loss per query-pos doc pair
       if tr_conf.loss == 'logistic':    
           loss_q_pos = tf.log1p(tf.reduce_sum(tf.exp(delta), axis=2))
       elif tr_conf.loss == 'hinge':
           loss_q_pos = tf.reduce_sum(tf.nn.relu(margin + delta), axis=2)
       elif tr_conf.loss == 'exp':
           loss_q_pos = tf.reduce_sum(tf.exp(delta), axis=2)
       else:
           raise NotImplementedError
       
       model_loss = tf.reduce_sum(weight * loss_q_pos)
   ```

   First we compute the difference of each triplet (query, positive document, negative document). Then, we feed `delta` to the loss function and aggregate over all negative documents, via `tf.reduce_sum(..., axis=2)`. Finally we rescale the loss of each query-(postive) document pair by `weight` and reduce them into a scalar.

3. ...



#### early stopping

<https://tensorflow.google.cn/versions/r1.12/api_docs/python/tf/contrib/estimator/stop_if_no_decrease_hook>

<https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/estimator/python/estimator/early_stopping.py>

stop_if_no_decrease_hook

stop_if_no_increase_hook

stop_if_lower_hook

stop_if_higher_hook



```python
def stop_if_no_decrease_hook(estimator,
    metric_name,
    max_steps_without_decrease,
    eval_dir=None,
    min_steps=0,
    run_every_secs=60,
    run_every_steps=None):
  """Creates hook to stop if metric does not decrease within given max steps.
 
   Usage example:
  ```python
  estimator = ...
  # Hook to stop training if loss does not decrease in over 100000 steps.
  hook = early_stopping.stop_if_no_decrease_hook(estimator, "loss", 100000)
  train_spec = tf.estimator.TrainSpec(..., hooks=[hook])
  tf.estimator.train_and_evaluate(estimator, train_spec, ...)
  ```
  
    max_steps_without_decrease: `int`, maximum number of training steps with no
      decrease in the given metric.
    min_steps: `int`, stop is never requested if global step is less than this
      value. Defaults to 0.
    eval_dir: If set, directory containing summary files with eval metrics. By default, 			  estimator.eval_dir() will be used.
   
  Returns:
    An early-stopping hook of type `SessionRunHook` that periodically checks
    if the given metric shows no decrease over given maximum number of
    training steps, and initiates early stopping if true.
  """
  return _stop_if_no_metric_improvement_hook(
      estimator=estimator,
      metric_name=metric_name,
      max_steps_without_improvement=max_steps_without_decrease,
      higher_is_better=False,
      eval_dir=eval_dir,
      min_steps=min_steps,
      run_every_secs=run_every_secs,
      run_every_steps=run_every_steps)
```

The early-stopping hook uses the evaluation results to decide when it's time to cut the training, **but** you need to pass in the number of training steps you want to monitor and keep in mind how many evaluations will happen in that number of steps.

For example:

max_steps_without_decrease = 10K

run_every_steps = 1K

the evaluations happening in a range of 10k steps, and running 1 eval every 1k steps, this boils down to early-stopping if there's a sequence of 10 consecutive evals without any improvement.



refer:

<https://stackoverflow.com/questions/52641737/tensorflow-1-10-custom-estimator-early-stopping-with-train-and-evaluate/52642619#52642619>



##### save the best model

more refer:

<https://github.com/tensorflow/tensorflow/issues/8658>

Keeping the best model can be done very easily defining a [`tf.estimator.BestExporter`](https://www.tensorflow.org/api_docs/python/tf/estimator/BestExporter) in your `EvalSpec` (snippet taken from the link):

```python
  serving_input_receiver_fn = ... # define your serving_input_receiver_fn
  exporter = tf.estimator.BestExporter(
      name="best_exporter",
      serving_input_receiver_fn=serving_input_receiver_fn,
      exports_to_keep=5) # this will keep the 5 best checkpoints

  eval_spec = [tf.estimator.EvalSpec(
    input_fn=eval_input_fn,
    steps=100,
    exporters=exporter,
    start_delay_secs=0,
    throttle_secs=5)]
```

If you don't know how to define the `serving_input_fn` [have a look here](https://www.tensorflow.org/guide/saved_model#using_savedmodel_with_estimators)







refer:

<https://stackoverflow.com/questions/52641737/tensorflow-1-10-custom-estimator-early-stopping-with-train-and-evaluate/52642619#52642619>







#### accuracy

+ Training accuracy falling down after some epoches

  https://github.com/tensorflow/tensorflow/issues/1997

  https://stackoverflow.com/questions/37044600/sudden-drop-in-accuracy-while-training-a-deep-neural-net







#### batch size

how to set the batch size?

For classification, the batch size is the multiple times of class labels ?



#### buffer size

https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle

The `buffer_size` in `Dataset.shuffle()` can affect the randomness of your dataset, and hence the order in which elements are produced. Instead of shuffling the entire dataset, it maintains a buffer of `buffer_size` elements, and randomly selects the next element from that buffer (replacing it with the next input element, if one is available). Changing the value of `buffer_size` affects how uniform the shuffling is: if `buffer_size` is greater than the number of elements in the dataset, you get a uniform shuffle; if it is `1` then you get no shuffling at all. <u>For very large datasets, a typical "good enough" approach is to randomly shard the data into multiple files once before training, then shuffle the filenames uniformly, and then use a smaller shuffle buffer.</u> However, the appropriate choice will depend on the exact nature of your training job.

The `buffer_size` in `Dataset.prefetch()` only affects the time it takes to produce the next element.


















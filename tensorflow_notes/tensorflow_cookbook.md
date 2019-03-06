https://github.com/taki0112/Tensorflow-Cookbook
https://github.com/nfmcclure/tensorflow_cookbook
refer:<br>https://hanxiao.github.io/2017/11/08/Optimizing-Contrastive-Rank-Triplet-Loss-in-Tensorflow-for-Neural/





#### loss

##### rank loss

define the objective function:

1. $\min \ g(q,d^-) - g(q,d^+)​$ 

   backwards: it's unbounded hence won't converge

2. $\min \  \max(0, \epsilon + g(q,d^-) - g(q,d^+))​$ 

   The max function can ignore large negative values, and $\epsilon$ is to avoid $ g(q,d^-) = g(q,d^+)$.

   It is basically a hinge loss. In fact, we could use any loss function besides the hinge loss, e.g. logistic loss, exponential loss. As for the metric, we also have plenty of options, e.g. cosine, $\ell_1/\ell_2$-norm. We could even parametrize the metric function with a multi-layer perceptron and learn it from the data.

   Given a training set of $N​$ triplets $\{ q_i, \{d^{+}_{i,j},\ldots\}, \{d^{-}_{i,k},\ldots\} \}​$, the final loss function of the model has the following general form:
   $$
   min \sum_{i=1}^{N} \sum_{j=1}^{∣d_{i}^{+}∣} \sum_{k=1}^{∣d_{i}^{−}∣} w_{i,j}ℓ(g(q_i,d_{i,j}^{+}),g(q_i,d_{i,k}^{−}))
   $$
   where $w_{i,j}$ is the weight of the positive query-document pair. In practice, this could be the click-through rate, or log number of clicks mined from the query-log. 

   (You can use the weight to balance those few-shot/long-tail queries. You may also use something like `log1p(num_clicks_queryi_on_productj)` to value more on those popular (q,d) pairs.) 

   

   For functions $\ell$ and $g​$, the options are:

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


















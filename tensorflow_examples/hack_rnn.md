cs20si - **Recurrent Neural Networks**



more refer: 

https://stackoverflow.com/questions/37901047/what-is-num-units-in-tensorflow-basiclstmcell

https://stats.stackexchange.com/questions/241985/understanding-lstm-units-vs-cells

https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/ good - detail explain!

https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html good - need to update code!



### Question

- [ ] 模型的容量

  

- [ ] 计算过程中，维度变化

  

- [ ] 源码

   https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/rnn_cell_impl.py

   https://tensorflow.google.cn/api_docs/python/tf/nn/rnn_cell/RNNCell

- [ ] RNNCell

   单个RNNCell，没有state size属性

- [ ] BasicRNNCell/BasicLSTMCell

   state size等同于num units

   relationship between time steps, num units ? One time step is correspond to one cell, one cell contain multiple num units.

- [ ] The parameter sharing strategy is each num units share the parameters or each time steps share the parameters ?

- [ ] The difference using initial_state or sequence_length parameter in dynamic_rnn ?

- [x] Is there necessary setting the sequence length for dynamic rnn ?

- [ ] What's the dropout operation exact doing on the bilstm representations (such as the QA representations before cosine similarity matching) ?



### rnn

https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html

How to construct graph for rnn? How many time steps of input should our graph accept at once?
method 1 - one graph to represent a single time step.
there is a problem for backpropagation - the gradients computed during backpropagation are graph-bound. We would only be able to backpropagate errors to the current timestep; we could not backpropagate the error to time step *t-1*. This means our network will not be able to learn how to store long-term dependencies (such as the two in our data) in its state.

method 2 - multi graph as wide as our data sequence.

there is a problem for backpropagation -  for long input sequence, backpropagation is not only (often prohibitively) expensive, but also ineffective, due to the vanishing / exploding gradient problem

solution: The usual pattern for dealing with very long sequences is therefore to <u>“truncate” our backpropagation by backpropagating errors a maximum of n steps</u>. We choose n as a hyperparameter to our model, keeping in mind the trade-off: higher n lets us capture longer term dependencies, but is more expensive computationally and memory-wise.

#### truncated_backpropagation

https://r2rt.com/styles-of-truncated-backpropagation.html

https://stats.stackexchange.com/questions/219914/rnns-when-to-apply-bptt-and-or-update-weights



Two style of truncated_backpropagation



Conclusion:

The n-step Tensorflow-style truncated backprop (i.e., with num_steps = n) does not effectively backpropagate errors the full n-steps. Thus, if you are using Tensorflow-style truncated backpropagation and need to capture n-step dependencies, you may benefit from using a num_steps that is appreciably higher than n in order to effectively backpropagate errors the desired n steps.



True style:

???

Tensorflow style:

https://tensorflow.google.cn/tutorials/sequences/recurrent#truncated_backpropagation

???



Example:

To understand why these two approaches are qualitatively different, consider how they differ on sequences of length 49 with backpropagation of errors truncated to 7 steps. In both, every error is backpropagated to the weights at the current timestep. However, in <u>Tensorflow-style truncated backpropagation</u>, the sequence is broken into 7 subsequences, each of length 7, and only 7 over the errors are backpropagated 7 steps. In <u>“true” truncated backpropagation</u>, 42 of the errors can be backpropagated for 7 steps, and 42 are. This may lead to different results because the ratio of 7-step to 1-step errors used to update the weights is significantly different. 

(Tensorflow-style: 将长度为49的序列截断为步长为7的子序列，7个步长只有7个误差得到传播；

True-style: 不将序列截断，当做一个完整的序列进行误差传播，每次最多传播7个误差，但7个步长可以传播42个误差:question:)

### Cell (tf.nn.rnn_cell)

https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/rnn_cell_impl.py

- BasicRNNCell: The most basic RNN cell.
- RNNCell: Abstract object representing an RNN cell.
- BasicLSTMCell: Basic LSTM recurrent network cell.
- LSTMCell: LSTM recurrent network cell.
- GRUCell: Gated Recurrent Unit cell 

#### BasicRNNCell

图片及公式

`num_units` in TensorFlow is the number of hidden states, i.e. the dimension of $h_t$ in the equations.

The number of hidden units is a direct representation of the learning capacity of a neural network -- it reflects the number of *learned parameters*. 

#### BasicLSTMCell

图片及公式

`num_units` can be interpreted as the analogy of hidden layer from the feed forward neural network.The number of nodes in hidden layer of a feed forward neural network is equivalent to `num_units` number of LSTM units in a LSTM cell at every time step of the network.

#### RNNCell

https://tensorflow.google.cn/api_docs/python/tf/nn/rnn_cell/RNNCell



图片及公式

call方法

```python
 class RNNCell(base_layer.Layer):
    """Abstract object representing an RNN cell.
      Every `RNNCell` must have the properties below and implement `call` with
      the signature `(output, next_state) = call(input, state)`.  

      This definition of cell differs from the definition used in the literature.
      In the literature, 'cell' refers to an object with a single scalar output.
      This definition refers to a horizontal array of such units.
```

#### LSTMCell



#### GRUCell



cell = tf.nn.rnn_cell.GRUCell(hidden_size)



### static_rnn 



#### bidirectional

https://tensorflow.google.cn/api_docs/python/tf/nn/static_bidirectional_rnn

static_bidirectional_rnn

### dynamic_rnn

tf.nn.dynamic_rnn: uses a tf.While loop to dynamically construct the graph when it is executed. Graph creation is faster and you can feed batches of variable size.

https://tensorflow.google.cn/api_docs/python/tf/nn/dynamic_rnn

https://stackoverflow.com/questions/43341374/tensorflow-dynamic-rnn-lstm-how-to-format-input

dynamic_rnn

```python
# create a BasicRNNCell
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)

# 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]

# defining initial state
initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)

# 'state' is a tensor of shape [batch_size, cell_state_size]
outputs, state = tf.nn.dynamic_rnn(rnn_cell, input_data,
                                   initial_state=initial_state,
                                   dtype=tf.float32)
```

```python
# create 2 LSTMCells
rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [128, 256]]

# create a RNN cell composed sequentially of a number of RNNCells
multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

# 'outputs' is a tensor of shape [batch_size, max_time, 256]
# 'state' is a N-tuple where N is the number of LSTMCells containing a
# tf.contrib.rnn.LSTMStateTuple for each cell
outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                   inputs=data,
                                   dtype=tf.float32)
```



#### bidirectional

tf.nn.bidirectional_dynamic_rnn



### stack



```python
layers = [tf.nn.rnn_cell.GRUCell(size) for size in hidden_sizes]
cells = tf.nn.rnn_cell.MultiRNNCell(layers)
```



#### bidirectional

stack_bidirectional_dynamic_rnn

https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/stack_bidirectional_dynamic_rnn



### dealing with variable sequence length 

Method 1:  padded sequence length

Pad all sequences with zero vectors and all labels with zero label (to make them of the same length)

But the padded labels change the total loss, which affects the gradients.

Method 2: Truncated sequence length

Most current models can’t deal with sequences of length larger than 120 tokens, so there is usually a fixed max_length and we truncate the sequences to that max_length



Approach 1: 

- Maintain a mask (True for real, False for padded tokens)
- Run your model on both the real/padded tokens (model will predict labels for the padded tokens as well)
- Only take into account the loss caused by the real elements

```python
full_loss = tf.nn.softmax_cross_entropy_with_logits(preds, labels)

loss = tf.reduce_mean(tf.boolean_mask(full_loss, mask))
```

It does not allow cell clipping, a projection layer, and does not use peep-hole connections: it is the basic baseline. :question:

Approach 2: 

- Let your model know the real sequence length so it only predict the labels for the real tokens

```python
cell = tf.nn.rnn_cell.GRUCell(hidden_size)

rnn_cells = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

tf.reduce_sum(tf.reduce_max(tf.sign(seq), 2), 1)

output, out_state = tf.nn.dynamic_rnn(rnn_cells, seq, length, initial_state)
```

### Vanishing Gradients

Use different activation units:

- tf.nn.relu
- tf.nn.relu6
- tf.nn.crelu
- tf.nn.elu

In addition to:

- tf.nn.softplus
- tf.nn.softsign
- tf.nn.bias_add
- tf.sigmoid
- tf.tanh

### Exploding Gradients

Clip gradients with tf.clip_by_global_norm

```python
gradients = tf.gradients(cost, tf.trainable_variables())
# take gradients of cosst w.r.t. all trainable variables

clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_grad_norm) 
# clip the gradients by a pre-defined max norm

optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.apply_gradients(zip(clipped_gradients, trainables))
# add the clipped gradients to the optimizer
```



### Anneal the learning rate

Optimizers accept both scalars and tensors as learning rate

```python

learning_rate = tf.train.exponential_decay(init_lr, 
										   global_step, 
										   decay_steps, 
										   decay_rate, 
										   staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate)

```



### Overfitting

Use dropout through tf.nn.dropout or DropoutWrapper for cells

- tf.nn.dropout

hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)

- DropoutWrapper

cell = tf.nn.rnn_cell.GRUCell(hidden_size)

cell = tf.nn.rnn_cell.DropoutWrapper(cell,     

​                                    output_keep_prob=keep_prob)





### QA

#### whats-the-difference-between-tensorflow-dynamic-rnn-and-rnn

https://stackoverflow.com/questions/39734146/whats-the-difference-between-tensorflow-dynamic-rnn-and-rnn

`tf.nn.rnn` creates an unrolled graph for a fixed RNN length. That means, if you call `tf.nn.rnn` with inputs having 200 time steps you are creating a static graph with 200 RNN steps. First, graph creation is slow. Second, you’re unable to pass in longer sequences (> 200) than you’ve originally specified.

`tf.nn.dynamic_rnn` solves this. It uses a `tf.While` loop to dynamically construct the graph when it is executed. That means graph creation is faster and you can feed batches of variable size.








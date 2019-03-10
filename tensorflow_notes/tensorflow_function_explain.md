transfer learning:<br>warm_start_from ???



## architecture

https://www.tensorflow.org/guide/extend/architecture

serving

https://www.tensorflow.org/tfx/serving/overview



## key ideas

data/operations/...



### model building step

#### session -based

1. 输入输出、权重
2. 模型（网络结构）
3. 损失函数
4. 优化器
5. 参数初始化
6. 训练（将数据喂入）

#### estimate-based

1. 模型（网络结构）

   tf.layers ...

2. model_fn（model = tf.estimator.Estimator(model_fn)）

   损失函数

   优化器

   评估

   prediction mode

3. input_fn（input_fn = tf.estimator.inputs.numpy_input_fn）

4. 训练（model.train(input_fn））



## basic

### placeholder

tf.placeholder -> parameters: type, dimension, name

~~~python
def placeholder(dtype, shape=None, name=None):
  """Inserts a placeholder for a tensor that will be always fed.

  **Important**: This tensor will produce an error if evaluated. Its value must be fed using the `feed_dict` optional argument to `Session.run()`, `Tensor.eval()`, or `Operation.run()`.

  For example:

  ```python
  x = tf.placeholder(tf.float32, shape=(1024, 1024))
  y = tf.matmul(x, x)

  with tf.Session() as sess:
    print(sess.run(y))  # ERROR: will fail because x was not fed.

    rand_array = np.random.rand(1024, 1024)
    print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
~~~



### variable

When you train a model you use variables to hold and update parameters. Variables are in-memory buffers containing tensors.



tf.Variable

tf.constant





tf.zeros

tf.random_normal



#### trainable_variables

tf.trainable_variables

https://stackoverflow.com/questions/37326002/is-it-possible-to-make-a-trainable-variable-not-trainable





### operation



tf.add,



tf.cast

```python
def cast(x, dtype, name=None):
  """Casts a tensor to a new type.

  The operation casts `x` (in case of `Tensor`) or `x.values`
  (in case of `SparseTensor` or `IndexedSlices`) to `dtype`.
  """
```



tf.argmax

tf.argmax() - axis的用法与numpy.argmax()的axis用法一致



#### Graph

tf.get_default_graph





#### Session

A Session object encapsulates the environment in which Tensor objects are evaluated.

tf.Session() ?

sess.run

tf.get_default_session

tf.InteractiveSession()

##### eval (evaluation)

https://stackoverflow.com/questions/33610685/in-tensorflow-what-is-the-difference-between-session-run-and-tensor-eval<br>https://stackoverflow.com/questions/38987466/eval-and-run-in-tensorflow

eval() 其实就是tf.Tensor的Session.run() 的另外一种写法。<br>eval() 只能用于tf.Tensor类对象，也就是有输出的Operation<br>Session.run()既可以用于有输出的Operation,也可以用于没有输出的Operation

```python
with tf.Session() as sess:
  print(accuracy.eval({x:mnist.test.images,y_: mnist.test.labels}))
  # or
  print(sess.run(accuracy, {x:mnist.test.images,y_: mnist.test.labels}))
```





#### Initialization

tf.initialize_all_variables

tf.global_variables_initializer

tf.variables_initializer 



tf.random_normal_initializer

tf.constant_initializer



#### Optimizer

##### minimize

optimizer.minimize

```python
  def minimize(self, loss, global_step=None, var_list=None,
               gate_gradients=GATE_OP, aggregation_method=None,
               colocate_gradients_with_ops=False, name=None,
               grad_loss=None):
    """Add operations to minimize `loss` by updating `var_list`.

    This method simply combines calls `compute_gradients()` and
    `apply_gradients()`. If you want to process the gradient before applying them call `compute_gradients()` and `apply_gradients()` explicitly instead of using this function.
    
        Args:
      global_step: Optional `Variable` to increment by one after the
        variables have been updated.
    """
```

`minimize()`分为两个步骤：`compute_gradients`和`apply_gradients`，前者用于计算梯度，后者用于使用计算得到的梯度来更新对应的variable。拆分为两个步骤后，在某些情况下我们就可以对梯度做一定的修正，例如为了防止梯度消失(gradient vanishing)或者梯度爆炸(gradient explosion)。

`compute_gradients`和`apply_gradients`具体怎么实现的呢？请看源码 todo

+ compute gradients

  



+ apply gradients

  

###### global_step

https://stackoverflow.com/questions/41166681/what-does-global-step-mean-in-tensorflow<br>https://blog.csdn.net/leviopku/article/details/78508951

`global_step` refers to the number of batches seen by the graph. Every time a batch is provided, the weights are updated in the direction that minimizes the loss. `global_step` just keeps track of the number of batches seen so far. When it is passed in the `minimize()` argument list, the variable is increased by one. 

You can get the `global_step` value using [`tf.train.global_step()`](https://www.tensorflow.org/api_docs/python/tf/train/global_step). Also handy are the utility methods [`tf.train.get_global_step`](https://www.tensorflow.org/api_docs/python/tf/train/get_global_step) or [`tf.train.get_or_create_global_step`](https://www.tensorflow.org/api_docs/python/tf/train/get_or_create_global_step).

`0` is the initial value of the global step in this context.

global_setp为什么能够自动加1？因为在minimize()函数中，global_steps是通过apply_gradients更新变量的同时进行更新的，每调用一次apply_gradients就会调用更新global_steps：

apply_updates = state_ops.assign_add(global_step, 1, name=name)

global_step在滑动平均、优化器、指数衰减学习率等方面都有用到，这个变量的实际意义非常好理解：代表全局步数，比如在多少步该进行什么操作，现在神经网络训练到多少轮等等，类似于一个钟表。

Example: 指数衰减的学习率是伴随global_step的变化而衰减

```python

import tensorflow as tf
import numpy as np
 
x = tf.placeholder(tf.float32, shape=[None, 1], name='x')
y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
w = tf.Variable(tf.constant(0.0))
 
global_steps = tf.Variable(0, trainable=False)
 
 
learning_rate = tf.train.exponential_decay(0.1, global_steps, 10, 2, staircase=False)
loss = tf.pow(w*x-y, 2)
 
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_steps)
 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        sess.run(train_step, feed_dict={x:np.linspace(1,2,10).reshape([10,1]),
            y:np.linspace(1,2,10).reshape([10,1])})
        print(sess.run(learning_rate))
        print(sess.run(global_steps))
```

tf.train.get_global_step

```python
def get_global_step(graph=None):
  """Get the global step tensor.

  The global step tensor must be an integer variable. We first try to find it
  in the collection `GLOBAL_STEP`, or by name `global_step:0`.

```

tf.train.get_or_create_global_step

```python


```





##### clip by * (gradient)

```python
def clip_by_value(t, clip_value_min, clip_value_max,
                  name=None):
  """Clips tensor values to a specified min and max.

  Any values less than `clip_value_min` are set to `clip_value_min`. Any values greater than `clip_value_max` are set to `clip_value_max`.
```



```python
def clip_by_norm(t, clip_norm, axes=None, name=None):
  """Clips tensor values to a maximum L2-norm.

  Given a tensor `t`, and a maximum clip value `clip_norm`, this operation normalizes `t` so that its L2-norm is less than or equal to `clip_norm`, along the dimensions given in `axes`. 
  Specifically, in the default case where all dimensions are used for calculation, if the L2-norm of `t` is already less than or equal to `clip_norm`, then `t` is not modified. If the L2-norm is greater than `clip_norm`, then this operation returns a tensor of the same type and shape as `t` with its values set to:
  `t * clip_norm / l2norm(t)`
```



```python
def clip_by_average_norm(t, clip_norm, name=None):
  """Clips tensor values to a maximum average L2-norm.

  Given a tensor `t`, and a maximum clip value `clip_norm`, this operation normalizes `t` so that its average L2-norm is less than or equal to `clip_norm`. 
  Specifically, if the average L2-norm is already less than or equal to `clip_norm`, then `t` is not modified. If the average L2-norm is greater than `clip_norm`, then this operation returns a tensor of the same type and shape as `t` with its values set to:
  `t * clip_norm / l2norm_avg(t)`
```



```python
def clip_by_global_norm(t_list, clip_norm, use_norm=None, name=None):
  """Clips values of multiple tensors by the ratio of the sum of their norms.

  Given a tuple or list of tensors `t_list`, and a clipping ratio `clip_norm`, this operation returns a list of clipped tensors `list_clipped` and the global norm (`global_norm`) of all tensors in `t_list`. 
  Optionally, if you've already computed the global norm for `t_list`, you can specify the global norm with `use_norm`.

  To perform the clipping, the values `t_list[i]` are set to:
      t_list[i] * clip_norm / max(global_norm, clip_norm)
  where:
      global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))

  If `clip_norm > global_norm` then the entries in `t_list` remain as they are, otherwise they're all shrunk by the global ratio.


  This is the correct way to perform gradient clipping (for example, see [Pascanu et al., 2012](http://arxiv.org/abs/1211.5063)
  ([pdf](http://arxiv.org/pdf/1211.5063.pdf))).

  However, it is slower than `clip_by_norm()` because all the parameters must be ready before the clipping operation can be performed.
```





##### question

+ 梯度裁剪的应用场景？

+ 怎么保证梯度裁剪的合理性？（数学或直观）

  梯度裁剪，实际上是剔除了噪声样本？

  分析被裁剪过的样本 todo

+ 梯度裁剪可能产生的问题？

  不收敛？迭代时间长？



#### loss



##### cross entropy

more:https://www.jianshu.com/p/cf235861311b

todo:<br>公式 

```python
def softmax_cross_entropy_with_logits(
    _sentinel=None,  # pylint: disable=invalid-name
    labels=None,
    logits=None,
    dim=-1,
    name=None):
  """Computes softmax cross entropy between `logits` and `labels`.

  Measures the probability error in discrete classification tasks in which the classes are mutually exclusive (each entry is in exactly one class).  
  For example, each CIFAR-10 image is labeled with one and only one label: an image can be a dog or a truck, but not both.

  **NOTE:**  While the classes are mutually exclusive, their probabilities need not be.  All that is required is that each row of `labels` is a valid probability distribution.  If they are not, the computation of the gradient will be incorrect.

  **WARNING:** This op expects unscaled logits, since it performs a `softmax` on `logits` internally for efficiency.  Do not call this op with the output of `softmax`, as it will produce incorrect results.

  A common use case is to have logits and labels of shape
  `[batch_size, num_classes]`, but higher dimensions are supported, with the `dim` argument specifying the class dimension.

  Backpropagation will happen only into `logits`.
  """
  _ensure_xent_args("softmax_cross_entropy_with_logits", _sentinel, labels,
                    logits)

  with ops.name_scope(name, "softmax_cross_entropy_with_logits_sg",
                      [logits, labels]) as name:
    labels = array_ops.stop_gradient(labels, name="labels_stop_gradient")

  return softmax_cross_entropy_with_logits_v2(
      labels=labels, logits=logits, dim=dim, name=name)
  
```

1. the classes are mutually exclusive, their probabilities need not be. <br>

2. Backpropagation will happen only into `logits`.

   最后一层的softmax输出计算没有在inference里进行，那么原网络inference岂不是不完整了？

   只要不影响到输出值的大小顺序，对最终结果也不会有影响。

3. labels must have the shape [batch_size, num_classes] and dtype float32 or float64.

```python
def softmax_cross_entropy_with_logits_v2(
    _sentinel=None,  # pylint: disable=invalid-name
    labels=None,
    logits=None,
    dim=-1,
    name=None):
  """Computes softmax cross entropy between `logits` and `labels`.

...

  Backpropagation will happen into both `logits` and `labels`.  To disallow backpropagation into `labels`, pass label tensors through `tf.stop_gradient` before feeding it to this function.
  """
```

1. Backpropagation will happen into both `logits` and `labels`.

2. **Soft** classes are allowed, **soft** softmax classification with a probability distribution for each entry.

```python
def sigmoid_cross_entropy_with_logits(  # pylint: disable=invalid-name
    _sentinel=None,
    labels=None,
    logits=None,
    name=None):
    """Computes sigmoid cross entropy given `logits`.

  Measures the probability error in discrete classification tasks in which each class is independent and not mutually exclusive.  
  For instance, one could perform multilabel classification where a picture can contain both an elephant and a dog at the same time.

  For brevity, let `x = logits`, `z = labels`.  The logistic loss is
        z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
      = (1 - z) * x + log(1 + exp(-x))
      = x - x * z + log(1 + exp(-x))

  For x < 0, to avoid overflow in exp(-x), we reformulate the above
        x - x * z + log(1 + exp(-x))
      = log(exp(x)) - x * z + log(1 + exp(-x))
      = - x * z + log(1 + exp(x))

  Hence, to ensure stability and avoid overflow, the implementation uses this equivalent formulation
      max(x, 0) - x * z + log(1 + exp(-abs(x)))
      """
```

each class is independent and not mutually exclusive

```python
def sparse_softmax_cross_entropy_with_logits(
    _sentinel=None,  # pylint: disable=invalid-name
    labels=None,
    logits=None,
    name=None):
  """Computes sparse softmax cross entropy between `logits` and `labels`.

  Measures the probability error in discrete classification tasks in which the classes are mutually exclusive (each entry is in exactly one class).  
  For example, each CIFAR-10 image is labeled with one and only one label: an image can be a dog or a truck, but not both.

  **NOTE:**  For this operation, the probability of a given label is considered exclusive. That is, soft classes are not allowed, and the `labels` vector must provide a single specific index for the true class for each row of
  `logits` (each minibatch entry). For soft softmax classification with a probability distribution for each entry, see `softmax_cross_entropy_with_logits_v2`.

  **WARNING:** This op expects unscaled logits, since it performs a `softmax`
  on `logits` internally for efficiency.  Do not call this op with the
  output of `softmax`, as it will produce incorrect results.

  A common use case is to have logits and labels of shape `[batch_size, num_classes]`, but higher dimensions are supported, in which case the `dim`-th dimension is assumed to be of size `num_classes`. `logits` must have the dtype of `float16`, `float32`, or `float64`, and `labels` must have the dtype of `int32` or `int64`.
  """
    
```

1. the classes are mutually exclusive, the probability of a given label is considered exclusive.<br>
2. labels must have the shape [batch_size] and the dtype int32 or int64. Each label is an int in range `[0, num_classes-1]`.
3. Labels used in `softmax_cross_entropy_with_logits` are the **one hot version** of labels used in `sparse_softmax_cross_entropy_with_logits`.
4. `softmax_cross_entropy_with_logits` and `sparse_softmax_cross_entropy_with_logits`, both functions computes the same results.

```python
def weighted_cross_entropy_with_logits(targets, logits, pos_weight, name=None):

```



##### accuracy

tf.metrics.accuracy

```python
def accuracy(labels,
             predictions,
             weights=None,
             metrics_collections=None,
             updates_collections=None,
             name=None):
  """Calculates how often `predictions` matches `labels`.

  The `accuracy` function creates two local variables, `total` and
  `count` that are used to compute the frequency with which `predictions`
  matches `labels`. This frequency is ultimately returned as `accuracy`: an
  idempotent operation that simply divides `total` by `count`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the `accuracy`.
  Internally, an `is_correct` operation computes a `Tensor` with elements 1.0
  where the corresponding elements of `predictions` and `labels` match and 0.0
  otherwise. Then `update_op` increments `total` with the reduced sum of the
  product of `weights` and `is_correct`, and it increments `count` with the
  reduced sum of `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

```







##### question

+ Backpropagation will happen into both `logits` and `labels` ?
+ 

#### next_batch

more: https://blog.csdn.net/appleml/article/details/57413615



```python
class DataSet(object):
   def __init__(self, images, labels, fake_data=False):
    if fake_data:
      self._num_examples = 10000
    else:
      assert images.shape[0] == labels.shape[0], (
          "images.shape: %s labels.shape: %s" % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]
      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      assert images.shape[3] == 1
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2])
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(numpy.float32)
      images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
    
  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
...
	# 如果所有batch_size之和超过样本量，则对数据进行shuffle并开始新一轮(epoch)遍历
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]
```





### useful for debug

#### get_operations 

tf.get_default_graph().get_operations()



#### name_scope

tf.name_scope



#### variable_scope

tf.variable_scope() provides simple name-spacing to avoid clashes
tf.get_variable() creates/accesses variables from within a variable scope

```python
with tf.variable_scope("foo"):
	with tf.variable_scope("bar"):
		v = tf.get_variable("v", [1])
assert v.name == "foo/bar/v:0"
```



```python
with tf.variable_scope("foo"):
	v = tf.get_variable("v", [1])
	tf.get_variable_scope().reuse_variables()
	v1 = tf.get_variable("v", [1])
assert v1 == v
```



```python
with tf.variable_scope("foo"):
	v = tf.get_variable("v", shape=[1]) # v.name == "foo/v:0"
with tf.variable_scope("foo", reuse=True):
	v1 = tf.get_variable("v") # Shared variable found!
with tf.variable_scope("foo", reuse=False):
	v1 = tf.get_variable("v") # CRASH foo/v:0 already exists!
```



#### get_variable

Behavior depends on whether variable reuse enabled

Case 1: reuse set to false
○ Create and return new variable

```python
with tf.variable_scope("foo"):
	v = tf.get_variable("v", [1])
assert v.name == "foo/v:0"
```

Case 2: Variable reuse set to true
○ Search for existing variable with given name. Raise ValueError if none found.

```python
with tf.variable_scope("foo"):
	v = tf.get_variable("v", [1])
with tf.variable_scope("foo", reuse=True):
	v1 = tf.get_variable("v", [1])
assert v1 == v
```





## tensorboard

https://github.com/tensorflow/tensorboard



motivation: visualization metrics and graph, Weights, Gradients and Activations 



step1 encapsulating all ops into scopes (tf.name_scope) and create a summary to monitor metrics (tf.summary)

step2 write logs to Tensorboard (tf.summary.FileWriter)

step3 run the following command, then open http://0.0.0.0:6006/ into your web browser

```bash
tensorboard --logdir=/tmp/tensorflow_logs 
```



## Debugger

motivation:



https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/debug



## Eager

motivation: Eager execution is an imperative, define-by-run interface where operations are executed immediately as they are called from Python.



https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/g3doc/guide.md



tfe.enable_eager_execution

## layers

tf.layers.dense

```python
def dense(
    inputs, units,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=init_ops.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    trainable=True,
    name=None,
    reuse=None):
  """Functional interface for the densely-connected layer.

  This layer implements the operation:
  `outputs = activation(inputs * kernel + bias)`
  where `activation` is the activation function passed as the `activation` argument (if not `None`), `kernel` is a weights matrix created by the layer, and `bias` is a bias vector created by the layer (only if `use_bias` is `True`).

  Arguments:
    activation: Activation function (callable). Set it to None to maintain a linear activation.
```



## estimator

### Estimator

tf.estimator.Estimator

TF Estimator input is a dict, in case of multiple inputs



### EstimatorSpec

tf.estimator.EstimatorSpec

TF Estimators requires to return a EstimatorSpec, that specify the different ops for training, evaluating, predicting



tf.estimator.ModeKeys.PREDICT



### multi_head

tf.contrib.estimator.multi_head

https://www.tensorflow.org/api_docs/python/tf/contrib/estimator/multi_head

multi-objective learning



## other

### tf.implicit_gradients



### tf.stop_gradient

https://www.tensorflow.org/api_docs/python/tf/stop_gradient

This is useful any time you want to compute a value with TensorFlow but need to pretend that the value was a constant. Some examples include:

- The *EM* algorithm where the *M-step* should not involve backpropagation through the output of the *E-step*.
- Contrastive divergence training of Boltzmann machines where, when differentiating the energy function, the training must not backpropagate through the graph that generated the samples from the model.
- Adversarial training, where no backprop should happen through the adversarial example generation process.



## Question

### [What is the meaning of the word logits in TensorFlow?](https://stackoverflow.com/questions/41455101/what-is-the-meaning-of-the-word-logits-in-tensorflow)

such as:

```python
loss_function = tf.nn.softmax_cross_entropy_with_logits(
     logits = last_layer,
     labels = target_output
)
```

A:<br>Logits is an overloaded term which can mean many different things.<br>**In ML**, it [can be](https://developers.google.com/machine-learning/glossary/#logits)

> the vector of raw (non-normalized) predictions that a classification model generates, which is ordinarily then passed to a normalization function. If the model is solving a multi-class classification problem, logits typically become an input to the softmax function. The softmax function then generates a vector of (normalized) probabilities with one value for each possible class.

For Tensorflow: It's a name that it is thought to imply that this Tensor is the quantity that is being mapped to probabilities by the Softmax.

...


















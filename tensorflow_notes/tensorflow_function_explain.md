transfer learning:<br>warm_start_from ???



## architecture

https://www.tensorflow.org/guide/extend/architecture

serving

https://www.tensorflow.org/tfx/serving/overview



## key ideas

data/operations/...



## basic

### variable

tf.constant



tf.placeholder -> parameters: type, dimension, name

```python
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
  ```
```



tf.Variable



tf.zeros

tf.random_normal



tf.name_scope ?



### operation

tf.add, tf.multiply, tf.matmul, 

tf.cast



tf.trainable_variables



tf.implicit_gradients



#### Session

tf.Session() ?

sess.run

tf.get_default_session



#### Graph

tf.get_default_graph



#### Initialization

tf.variables_initializer 

tf.global_variables_initializer



#### Optimizer

optimizer.minimize

```python
  def minimize(self, loss, global_step=None, var_list=None,
               gate_gradients=GATE_OP, aggregation_method=None,
               colocate_gradients_with_ops=False, name=None,
               grad_loss=None):
    """Add operations to minimize `loss` by updating `var_list`.

    This method simply combines calls `compute_gradients()` and
    `apply_gradients()`. If you want to process the gradient before applying them call `compute_gradients()` and `apply_gradients()` explicitly instead of using this function.
```

`minimize()`分为两个步骤：`compute_gradients`和`apply_gradients`，前者用于计算梯度，后者用于使用计算得到的梯度来更新对应的variable。拆分为两个步骤后，在某些情况下我们就可以对梯度做一定的修正，例如为了防止梯度消失(gradient vanishing)或者梯度爆炸(gradient explosion)。

`compute_gradients`和`apply_gradients`具体怎么实现的呢？请看源码 todo

+ compute gradients

  



+ apply gradients

  



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

1. the classes are mutually exclusive<br>
2. Backpropagation will happen only into `logits`.

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

Backpropagation will happen into both `logits` and `labels`.

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
    
```



```python
def weighted_cross_entropy_with_logits(targets, logits, pos_weight, name=None):

```



##### question

+ Backpropagation will happen into both `logits` and `labels` ?
+ 



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



## estimator

### tf.contrib.estimator.multi_head

https://www.tensorflow.org/api_docs/python/tf/contrib/estimator/multi_head

multi-objective learning



## other

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


















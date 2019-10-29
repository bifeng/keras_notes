transfer learning:<br>warm_start_from ???



## architecture

https://www.tensorflow.org/guide/extend/architecture

serving

https://www.tensorflow.org/tfx/serving/overview



## key ideas

data/operations/...



### model building step

#### session-based

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

4. 训练（model.train(input_fn)）

5. 评估（model.evaluate(input_fn)）



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

```python
  def __init__(self,  # pylint: disable=super-init-not-called
               initial_value=None,
               trainable=True,
               collections=None,
               validate_shape=True,
               caching_device=None,
               name=None,
               variable_def=None,
               dtype=None,
               expected_shape=None,
               import_scope=None,
               constraint=None,
               use_resource=None,
               synchronization=VariableSynchronization.AUTO,
               aggregation=VariableAggregation.NONE):
    """Creates a new variable with value `initial_value`.

    The new variable is added to the graph collections listed in `collections`,
    which defaults to `[GraphKeys.GLOBAL_VARIABLES]`.

    If `trainable` is `True` the variable is also added to the graph collection
    `GraphKeys.TRAINABLE_VARIABLES`.

    This constructor creates both a `variable` Op and an `assign` Op to set the
    variable to its initial value.

    Args:
      trainable: If `True`, the default, also adds the variable to the graph
        collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
        the default list of variables to use by the `Optimizer` classes.
      collections: List of graph collections keys. The new variable is added to
        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
```

默认创建全局变量，该变量在tf.GraphKeys.GLOBAL_VARIABLES之中。局部变量则在tf.GraphKeys.LOCAL_VARIABLES之中：

```python
z = tf.Variable(initial_value=tf.constant(1.0), name='z', collections=[tf.GraphKeys.LOCAL_VARIABLES])
```

局部变量的应用场景：<br>	在使用saver的时候，局部变量不存于模型文件





tf.constant



tf.zeros

tf.random_normal



#### trainable

Specify if a variable should be trained or not. By default, all variables are trainable. Session looks at all trainable variables that loss depends on and update them.

For example, global_steps shouldn’t be trainable. Or in double q-learning, you want to alternate which q-value functions to update.



tf.trainable_variables

https://stackoverflow.com/questions/37326002/is-it-possible-to-make-a-trainable-variable-not-trainable



##### how to make a trainable variable not trainable ?

https://stackoverflow.com/questions/37326002/is-it-possible-to-make-a-trainable-variable-not-trainable

https://stackoverflow.com/questions/35298326/freeze-some-variables-scopes-in-tensorflow-stop-gradient-vs-passing-variables

Set the variable's trainable property to `False`. But the variable `'weights'` is still in the output of `tf.trainable_variables`.

```
weight_var = tf.get_variable('weights', trainable = False)
```

1. first solution

   When calling the `minimize` method of the optimizer (see [doc.](https://www.tensorflow.org/api_guides/python/train#Optimizer)), you can pass a `var_list=[...]` as argument with the variables you want to optimizer.

   For instance, if you want to freeze all the layers of VGG except the last two, you can pass the weights of the last two layers in `var_list`.

   ```python
   tvars = tf.trainable_variables()
   g_vars = [var for var in tvars if 'g_' in var.name]
   g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)
   # or
   optimizer = tf.train.AdamOptimizer()
   train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         "scope/prefix/for/second/vars")  
   train_op = optimizer.minimize(cost, var_list=train_vars)
   ```

2. Second solution

   You can use a `tf.train.Saver()` to save variables and restore them later (see [this tutorial](https://www.tensorflow.org/programmers_guide/variables)).

   - First you train your entire VGG model with **all trainable** variables. You save them in a checkpoint file by calling `saver.save(sess, "/path/to/dir/model.ckpt")`.
   - Then (in another file) you train the second version with **non trainable** variables. You load the variables previously stored with `saver.restore(sess, "/path/to/dir/model.ckpt")`.

   Optionally, you can decide to save only some of the variables in your checkpoint file. See the [doc](https://www.tensorflow.org/api_guides/python/state_ops#Saver) for more info.

3. Third solution

   You can do `trainable_collection = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)` to get a reference for the collection of trainable variables, which is a python list, and then use `pop` with the right index to remove the variable from there.

   ```
   #gets a reference to the list containing the trainable variables
   trainable_collection = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
   variables_to_remove = list()
   for vari in trainable_collection:
       #uses the attribute 'name' of the variable
       if vari.name=="batch_normalization/gamma:0" or vari.name=="batch_normalization/beta:0":
           variables_to_remove.append(vari)
   for rem in variables_to_remove:
       trainable_collection.remove(rem)
   ```





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

tf.get_default_graph()   # default graph

tf.Graph()  # user created graph



##### graph finalize (memory leak)

more: https://dantkz.github.io/How-To-Debug-A-Memory-Leak-In-TensorFlow/

tf.Graph.finalize()

```python
  def finalize(self):
    """Finalizes this graph, making it read-only.

    After calling `g.finalize()`, no new operations can be added to
    `g`.  This method is used to ensure that no operations are added
    to a graph when it is shared between multiple threads, for example
    when using a `tf.train.QueueRunner`.
    """
```



##### control dependencies

tf.Graph.control_dependencies(control_inputs)

Used it to declare the control dependencies. For example, a variable can only be used after being initialized. 

```python
# defines which ops should be run first
# your graph g have 5 ops: a, b, c, d, e
g = tf.get_default_graph()
with g.control_dependencies([a, b, c]):
	# 'd' and 'e' will only run after 'a', 'b', and 'c' have executed.
	d = ...
	e = …
```



##### feedable tensor

**You can feed_dict any feedable tensor. Placeholder is just a way to indicate that something must be fed**

```python
tf.Graph.is_feedable(tensor) 
# True if and only if tensor is feedable.
```





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

tf.global_variables_initializer

tf.initialize_local_variables



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

`minimize()`分为两个步骤：`compute_gradients`和`appy_gradients`，前者用于计算梯度，后者用于使用计算得到的梯度来更新对应的variable。拆分为两个步骤后，在某些情况下我们就可以对梯度做一定的修正，例如为了防止梯度消失(gradient vanishing)或者梯度爆炸(gradient explosion)。

`compute_gradients`和`appy_gradients`具体怎么实现的呢？请看源码 todo

+ compute gradients

  



+ apply gradients

  

###### global_step

https://stackoverflow.com/questions/41166681/what-does-global-step-mean-in-tensorflow<br>https://blog.csdn.net/leviopku/article/details/78508951

TensorFlow uses the `global_step` parameter to count the number of training steps that have been processed (to know when to end a training run). Furthermore, the `global_step` is essential for TensorBoard graphs to work correctly.

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

```python
global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
learning_rate = 0.01 * 0.99 ** tf.cast(global_step, tf.float32)
increment_step = global_step.assign_add(1)
optimizer = tf.train.GradientDescentOptimizer(learning_rate) # learning rate can be a tensor
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

##### gradient

###### gradient

tf.gradients

This is especially useful when training only parts of a model. For example, we can use tf.gradients()  to take the derivative G of the loss w.r.t. to the middle layer. Then we use an optimizer to minimize the difference between the middle layer output M and M + G. This only updates the lower half of the network.

```python
tf.gradients(ys, xs, grad_ys=None, name='gradients', colocate_gradients_with_ops=False, gate_gradients=False, aggregation_method=None)
```

tf.gradients(ys, [xs]) with [xs] stands for a list of tensors with respect to those you’re trying to compute the gradient of ys. 

###### stop gradient

tf.stop_gradient

https://www.tensorflow.org/api_docs/python/tf/stop_gradient

This is useful any time you want to compute a value with TensorFlow but need to pretend that the value was a constant ( freeze certain variables during training). Some examples include:

- The *EM* algorithm where the *M-step* should not involve backpropagation through the output of the *E-step*.
- Contrastive divergence training of Boltzmann machines where, when differentiating the energy function, the training must not backpropagate through the graph that generated the samples from the model.
- Adversarial training (GAN (Generative Adversarial Network)), where no backprop should happen through the adversarial example generation process.

###### clip by * (gradient)

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

###### question

- 梯度裁剪的应用场景？

- 怎么保证梯度裁剪的合理性？（数学或直观）

  梯度裁剪，实际上是剔除了噪声样本？

  分析被裁剪过的样本 todo

- 梯度裁剪可能产生的问题？

  不收敛？迭代时间长？



#### loss



##### cross entropy

more:https://www.jianshu.com/p/cf235861311b

refer: https://stackoverflow.com/questions/47034888/how-to-choose-cross-entropy-loss-in-tensorflow?rq=1

todo:<br>公式 



In functional sense, the [sigmoid is a partial case of the softmax function](https://stats.stackexchange.com/q/233658/130598), when the number of classes equals 2. 

###### sigmoid functions family

sigmoid allows to deal with non-exclusive labels (a.k.a. *multi-labels*)

- [`tf.nn.sigmoid_cross_entropy_with_logits`](https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits)

- [`tf.nn.weighted_cross_entropy_with_logits`](https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits)

  It allows to set *class weights* (remember, the classification is binary), i.e. make positive errors larger than negative errors. This is useful when the training data is unbalanced.

- [`tf.losses.sigmoid_cross_entropy`](https://www.tensorflow.org/api_docs/python/tf/losses/sigmoid_cross_entropy)

  It allows to set the *in-batch weights*, i.e. make some examples more important than others.

Application:

binary classification

multi-label classification - The labels must be <u>one-hot encoded</u> or can contain soft class probabilities.



###### softmax functions family

softmax deals with exclusive classes 

- [`tf.nn.softmax_cross_entropy_with_logits_v2`](https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits_v2)

- [`tf.losses.softmax_cross_entropy`](https://www.tensorflow.org/api_docs/python/tf/losses/softmax_cross_entropy)

  It allows to set the *in-batch weights*, i.e. make some examples more important than others. 

Application:

binary classification

multi-class classification - The labels must be <u>one-hot encoded</u> or can contain soft class probabilities: a particular example can belong to class A with 50% probability and class B with 50% probability. Note that strictly speaking it doesn't mean that it belongs to both classes, but one can interpret the probabilities this way. (labels must have the shape [batch_size, num_classes] and dtype float32 or float64.)

###### sparse functions family

- [`tf.nn.sparse_softmax_cross_entropy_with_logits`](https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits)

- [`tf.losses.sparse_softmax_cross_entropy`](https://www.tensorflow.org/api_docs/python/tf/losses/sparse_softmax_cross_entropy)

  It allows to set the in-batch weights

Application:

binary classification

multi-classification - The difference is in labels encoding: the classes are specified as integers (class index), not one-hot vectors. Obviously, <u>this doesn't allow soft classes (???)</u>, but it can save some memory when there are thousands or millions of classes. However, note that `logits` argument must still contain logits per each class, thus it consumes at least `[batch_size, classes]` memory. (labels must have the shape [batch_size] and the dtype int32 or int64. Each label is an int in range `[0, num_classes-1]`.)

###### sampled softmax functions family

- [`tf.nn.sampled_softmax_loss`](https://www.tensorflow.org/api_docs/python/tf/nn/sampled_softmax_loss)
- [`tf.contrib.nn.rank_sampled_softmax_loss`](https://www.tensorflow.org/api_docs/python/tf/contrib/nn/rank_sampled_softmax_loss)
- [`tf.nn.nce_loss`](https://www.tensorflow.org/api_docs/python/tf/nn/nce_loss)

These functions provide another alternative for dealing with huge number of classes. Instead of computing and comparing an exact probability distribution, they compute a loss estimate from a random sample.

The arguments `weights` and `biases` specify a separate fully-connected layer that is used to compute the logits for a chosen sample.

Like above, `labels` are not one-hot encoded, but have the shape `[batch_size, num_true]`.

Sampled functions are only suitable for training. In test time, it's recommended to use a standard `softmax` loss (either sparse or one-hot) to get an actual distribution.

Another alternative loss is `tf.nn.nce_loss`, which performs *noise-contrastive estimation* (if you're interested, see this [very detailed discussion](https://datascience.stackexchange.com/q/13216/18375)). I've included this function to the softmax family, because NCE guarantees approximation to softmax in the limit.



###### example

多分类的几种方式

1. softmax_cross_entropy_with_logits

   输入

   ```
   self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
   ```

   输出

   ```
   self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
   self.predictions = tf.argmax(self.scores, 1, name="predictions")
   ```

   loss

   ```python
   losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
   self.loss = tf.reduce_mean(losses)
   ```

   accuracy

   ```
   correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
   self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
   ```

2. softmax_cross_entropy_with_logits

   输入

   ```
   self.y = tf.placeholder(name='y', shape=(None,), dtype=tf.int32)
   ```

   输出

   ```
   self.logits = tf.matmul(pool_drop, self.w) + self.b
   self.probs = tf.nn.softmax(self.logits)
   self.predict = tf.argmax(name="predictions", input=self.probs, axis=1)
   ```

   loss

   ```
   labels_one_hot = tf.one_hot(self.y, self.class_num)
   loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot,
                                                  logits=self.logits)
   ```

   accuracy

   ```
   correct_prediction = tf.equal(tf.cast(self.predict, tf.int32), self.y)
   self.accuracy = tf.reduce_mean(name="accuracy", input_tensor=tf.cast(correct_prediction, tf.float32))
   ```

3. sparse_softmax_cross_entropy_with_logits

   输入

   ```
   self.input_y = tf.placeholder(tf.int32, [None,],name="input_y")
   ```

   输出

   ```
   self.predictions = tf.argmax(self.logits, 1, name="predictions")
   ```

   loss

   ```
   losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
   loss=tf.reduce_mean(losses)
   ```

   accuracy

   ```
   correct_prediction = tf.equal(tf.cast(self.predictions,tf.int32), self.input_y)
   self.accuracy =tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
   ```



多标签方式：

1. sigmoid_cross_entropy_with_logits

   输入

```
self.input_y_multilabel = tf.placeholder(tf.float32,[None,self.num_classes], name="input_y_multilabel")
```

​	输出

```
self.possibility=tf.nn.sigmoid(self.logits)
```

​	loss

```
losses=tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel, logits=self.logits)
losses=tf.reduce_sum(losses,axis=1)   #shape=(?,). loss for all data in the batch
loss=tf.reduce_mean(losses)   #shape=().   average loss in the batch
```



注意：

多标签分类，通常label需要二值化处理：

```python
from sklearn.preprocessing import MultiLabelBinarizer
y = [[2,3,4],[2],[0,1,3],[0,1,2,3,4],[0,1,2],[0,1,2]]
MultiLabelBinarizer().fit_transform(y)
array([[0, 0, 1, 1, 1],
       [0, 0, 1, 0, 0],
       [1, 1, 0, 1, 0],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 0, 0],
       [1, 1, 1, 0, 0]])
```



示例：y_true/y_pred是每个样本的类标签

```python
y_true = [0,1,3]
y_pred = [1,2,1]

from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder

one_hot = OneHotEncoder(n_values=4, sparse=False)

y_true = one_hot.fit_transform([0,1,3])
y_pred = one_hot.fit_transform([1,2,1])
log_loss(y_true, y_pred)
```



示例：label是每个样本的类标签，prob是每个样本属于各类的概率

```python
from tensorboard import summary as summary_lib
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

def pr_curve(label_list, prob_list, class_number):
    truth = np.array(label_list).reshape(-1,1)

    lb = OneHotEncoder(n_values=class_number, sparse=False)
    lb.fit(truth)

    labels = lb.transform(truth)

    summary_proto = summary_lib.pr_curve_pb(
        name='PR_Curve',
        predictions=np.array(prob_list),
        labels=labels)
    return summary_proto


def multiclass_roc_auc_score(truth, prob, average, class_number):
    truth = np.array(truth).reshape(-1,1)
    prob = np.array(prob)

    lb = OneHotEncoder(n_values=class_number, sparse=False)
    lb.fit(truth)

    truth = lb.transform(truth)

    return roc_auc_score(truth, prob, average=average)
```



###### source code



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



##### metric

###### accuracy

https://stackoverflow.com/questions/46409626/how-to-properly-use-tf-metrics-accuracy



tf.metrics.accuracy

<https://github.com/tensorflow/docs/blob/r1.0/site/en/api_docs/python/tf/metrics/accuracy.md>

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







###### auc

the idea of streaming

https://www.tensorflow.org/api_docs/python/tf/metrics/auc

https://tensorflow.google.cn/api_docs/python/tf/contrib/metrics/streaming_auc

https://stackoverflow.com/questions/39435341/how-to-calculate-auc-with-tensorflow

https://stackoverflow.com/questions/44422508/tensorflow-attempting-to-use-uninitialized-value-auc-auc-auc-false-positives



###### multi-class

calculate metric with streaming data:

<https://github.com/guillaumegenthial/tf_metrics/blob/master/tf_metrics/__init__.py>

<https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/metrics_impl.py>



[Support for multi-class roc_auc scores](https://github.com/scikit-learn/scikit-learn/issues/3298#) #3298

https://stackoverflow.com/questions/39685740/calculate-sklearn-roc-auc-score-for-multi-class



```python
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

def multiclass_roc_auc_score(truth, pred, average="macro"):

    lb = LabelBinarizer()
    lb.fit(truth)

    truth = lb.transform(truth)
    pred = lb.transform(pred)

    return roc_auc_score(truth, pred, average=average)
```

```python
import pandas as pd
import numpy as np
from scipy import interp

from  sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

def class_report(y_true, y_pred, y_score=None, average='micro'):
    if y_true.shape != y_pred.shape:
        print("Error! y_true %s is not the same shape as y_pred %s" % (
              y_true.shape,
              y_pred.shape)
        )
        return

    lb = LabelBinarizer()

    if len(y_true.shape) == 1:
        lb.fit(y_true)

    #Value counts of predictions
    labels, cnt = np.unique(
        y_pred,
        return_counts=True)
    n_classes = len(labels)
    pred_cnt = pd.Series(cnt, index=labels)

    metrics_summary = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels)

    avg = list(precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred,
            average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index,
        columns=labels)

    support = class_report_df.loc['support']
    total = support.sum() 
    class_report_df['avg / total'] = avg[:-1] + [total]

    class_report_df = class_report_df.T
    class_report_df['pred'] = pred_cnt
    class_report_df['pred'].iloc[-1] = total

    if not (y_score is None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for label_it, label in enumerate(labels):
            fpr[label], tpr[label], _ = roc_curve(
                (y_true == label).astype(int), 
                y_score[:, label_it])

            roc_auc[label] = auc(fpr[label], tpr[label])

        if average == 'micro':
            if n_classes <= 2:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                    lb.transform(y_true).ravel(), 
                    y_score[:, 1].ravel())
            else:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                        lb.transform(y_true).ravel(), 
                        y_score.ravel())

            roc_auc["avg / total"] = auc(
                fpr["avg / total"], 
                tpr["avg / total"])

        elif average == 'macro':
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([
                fpr[i] for i in labels]
            ))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in labels:
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr

            roc_auc["avg / total"] = auc(fpr["macro"], tpr["macro"])

        class_report_df['AUC'] = pd.Series(roc_auc)

    return class_report_df


report_with_auc = class_report(
    y_true=y_test, 
    y_pred=model.predict(X_test), 
    y_score=model.predict_proba(X_test))

print(report_with_auc)
```

```py
             precision    recall  f1-score  support    pred       AUC
0             0.262774  0.553846  0.356436    130.0   274.0  0.766477
1             0.405405  0.333333  0.365854    135.0   111.0  0.773974
2             0.367347  0.150000  0.213018    120.0    49.0  0.817341
3             0.350993  0.424000  0.384058    125.0   151.0  0.803364
4             0.379310  0.447154  0.410448    123.0   145.0  0.802436
5             0.525000  0.182609  0.270968    115.0    40.0  0.680870
6             0.362573  0.488189  0.416107    127.0   171.0  0.855768
7             0.330189  0.299145  0.313901    117.0   106.0  0.766526
8             0.328571  0.407080  0.363636    113.0   140.0  0.754812
9             0.571429  0.248276  0.346154    145.0    63.0  0.769100
avg / total   0.390833  0.354400  0.345438   1250.0  1250.0  0.776071
```



##### question

+ Why the backpropagation happens into both logits and labels is useful ?

  In supervised learning one doesn't need to backpropagate to labels. They are considered fixed ground truth and only the weights need to be adjusted to match them.

  But in some cases, the labels themselves *may* come from a differentiable source, another network. One example might be [adversarial learning](https://en.wikipedia.org/wiki/Generative_adversarial_network). In this case, both networks might benefit from the error signal. That's the reason why `tf.nn.softmax_cross_entropy_with_logits_v2` was [introduced](https://github.com/tensorflow/tensorflow/commit/d60f6513232bd49b658c188c0597dd119e9a52d8). Note that when the labels are the placeholders (which is also typical), there is no difference if the gradient through flows or not, because there are no variables to apply gradient to.

  https://stats.stackexchange.com/questions/327348/how-is-softmax-cross-entropy-with-logits-different-from-softmax-cross-entropy-wi

+ how to calculate accuracy for multi-label classification ?

  https://stackoverflow.com/questions/37746670/tensorflow-multi-label-accuracy-calculation?rq=1

+ nce loss

  https://datascience.stackexchange.com/questions/13216/intuitive-explanation-of-noise-contrastive-estimation-nce-loss

  





### useful functions

#### embedding_lookup/gather

https://stackoverflow.com/questions/34870614/what-does-tf-nn-embedding-lookup-function-do

In its simplest form, it is similar to `tf.gather`. It returns the elements of `params` according to the indexes specified by `ids`.

But `embedding_lookup` is more than that. The `params` argument can be a **list** of tensors, rather than a single tensor.

...

#### matmul/multiply/tensordot/einsum

https://tensorflow.google.cn/versions/r1.9/api_docs/python/tf/matmul?hl=en

matmul

三维矩阵乘法： tf.matmul(a,b)

a -> (W, H, L)  b-> (W, L, H)



multiply





tensordot

https://tensorflow.google.cn/versions/r1.9/api_docs/python/tf/tensordot?hl=en

```python
tf.tensordot(
    a,
    b,
    axes,
    name=None
)
```

Tensor contraction of a and b along specified axes.

This operation corresponds to `numpy.tensordot(a, b, axes)`.

Example 1: When `a` and `b` are matrices (order 2), the case `axes = 1` is equivalent to matrix multiplication.

Example 2: When `a` and `b` are matrices (order 2), the case `axes = [[1], [0]]` is equivalent to matrix multiplication.

Example 3: Suppose that aijk and blmn represent two tensors of order 3. Then, `contract(a, b, [[0], [2]])` is the order 4 tensor cjklm whose entry corresponding to the indices (j,k,l,m) is given by:

cjklm=∑iaijkblmi.

In general, `order(c) = order(a) + order(b) - 2*len(axes[0])`.



einsum

https://tensorflow.google.cn/versions/r1.8/api_docs/python/tf/einsum?hl=en

```python
tf.einsum(
    equation,
    *inputs,
    **kwargs
)
```

A generalized contraction between tensors of arbitrary dimension.



#### slice

https://tensorflow.google.cn/versions/r1.9/api_docs/python/tf/slice?hl=en

```python
tf.slice(
    input_,
    begin,
    size,
    name=None
)
input_: A Tensor.
begin: An int32 or int64 Tensor.
size: An int32 or int64 Tensor.
name: A name for the operation (optional).
```

This operation extracts a slice of size `size` from a tensor `input` starting at the location specified by `begin`. 

#### transpose

https://tensorflow.google.cn/versions/r1.11/api_docs/python/tf/transpose?hl=en

```python
tf.transpose(
    a,
    perm=None,
    name='transpose',
    conjugate=False
)
a: A Tensor.
perm: A permutation of the dimensions of a.
name: A name for the operation (optional).
conjugate: Optional bool. Setting it to True is mathematically equivalent to tf.conj(tf.transpose(input)).
```

Transposes `a`. Permutes the dimensions according to `perm`.



#### squeeze

https://tensorflow.google.cn/versions/r1.8/api_docs/python/tf/keras/backend/squeeze?hl=en

```python
tf.keras.backend.squeeze(
    x,
    axis
)
x: A tensor or variable.
axis: Axis to drop.
```

Removes a 1-dimension from the tensor at index "axis".



#### get_operations 

tf.get_default_graph().get_operations()



#### name_scope/variable_scope

https://stackoverflow.com/questions/35919020/whats-the-difference-of-name-scope-and-a-variable-scope-in-tensorflow?rq=1

Summary:

`tf.variable_scope()` adds a prefix to the names of all variables (no matter how you create them -`tf.get_variable()` or `tf.Variable()`), ops, constants. 

`tf.name_scope()` ignores variables created with `tf.get_variable()` because it assumes that you know which variable and in which scope you wanted to use.



This allows us to easily share variables across different parts of the program, even within different name scopes:

```python
with tf.name_scope("foo"):
    with tf.variable_scope("var_scope"):
        v = tf.get_variable("var", [1])
with tf.name_scope("bar"):
    with tf.variable_scope("var_scope", reuse=True):
        v1 = tf.get_variable("var", [1])
assert v1 == v
print(v.name)   # var_scope/var:0
print(v1.name)  # var_scope/var:0
```

With variable scope, one can define separate scopes for re-usable variables that are not affected by the current name scope used to define operations.



A good documentation on [Sharing variables](https://www.tensorflow.org/programmers_guide/variable_scope) tells you that

> `tf.variable_scope()`: Manages namespaces for names passed to `tf.get_variable()`.

The same documentation provides a more details how does Variable Scope work and when it is useful.





name scope

**Group nodes together** **with tf.name_scope(name)**





variable scope

Variable scope facilitates variable sharing.

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

tf.get_variable(<name>, <shape>, <initializer>)

If a variable with <name> already exists, reuse it; If not, initialize it with <shape> using <initializer>.

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



#### get_collection (GraphKeys)

By default, all variables are placed in tf.GraphKeys.GLOBAL_VARIABLES. 

If you set trainable=True (which is always set by default) when you create your variable, that variable will be in the collection tf.GraphKeys.TRAINABLE_VARIABLES. 

```python
tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='my_scope')
```

You can create your own collections with tf.add_to_collection(name, value). For example, you can create a collection of initializers and  add all init ops to that. 

The standard library uses various well-known names to collect and retrieve values associated with a graph. For example, the tf.train.Optimizer subclasses default to optimizing the variables collected under tf.GraphKeys.TRAINABLE_VARIABLES if none is specified, but it is also possible to pass an explicit list of variables. For the list of predefined graph keys, please see [the official documentation](https://www.tensorflow.org/api_docs/python/tf/GraphKeys).



#### How to get tensor or op by name ?

```py
import tensorflow as tf

c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
e = tf.matmul(c, d, name='example')

with tf.Session() as sess:
    test =  sess.run(e)
    print e.name #example:0
    test = tf.get_default_graph().get_tensor_by_name("example:0")
    print test #Tensor("example:0", shape=(2, 2), dtype=float32)
```



 You can use the [`tf.Graph.get_operation_by_name()`](https://www.tensorflow.org/api_docs/python/tf/Graph#get_operation_by_name) method to get a `tf.Operation` by name. For example, to get an operation called `"enqueue"` from the default graph:

```py
op = tf.get_default_graph().get_operation_by_name("enqueue")
```

https://stackoverflow.com/questions/36612512/tensorflow-how-to-get-a-tensor-by-name?rq=1

https://stackoverflow.com/questions/42685994/how-to-get-a-tensorflow-op-by-name

## Debugger

motivation:



https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/debug



## Eager

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/g3doc/guide.md

motivation: Eager execution is an imperative, define-by-run interface where operations are executed immediately as they are called from Python.

You no longer need to worry about ...

1. placeholders
2. sessions
3. control dependencies
4. "lazy loading"
5. {name, variable, op} scopes



when eager execution is enabled  …

- - prefer **tfe**.Variable under eager execution (compatible with graph construction)

  - manage your own variable storage — variable collections are not supported!

  - use tf.contrib.summary

  - use **tfe**.Iterator to iterate over datasets under eager execution

  - prefer object-oriented layers (e.g., tf.layers.Dense) 

  - - functional layers (e.g., tf.layers.dense) only work if wrapped in **tfe**.make_template

- + prefer tfe.py_func over tf.py_func



tfe.enable_eager_execution



tfe.Variable



### gradients

There are 4 ways to automatically compute gradients when eager execution is enabled (actually, they also work in graph mode)... 

https://stackoverflow.com/questions/50098971/whats-the-difference-between-gradienttape-implicit-gradients



tfe.gradients_function()
tfe.value_and_gradients_function()
tfe.implicit_gradients()
tfe.implicit_value_and_gradients()



#### tf.implicit_gradients

https://tensorflow.google.cn/api_docs/python/tf/contrib/eager/implicit_gradients



## feature_column

tf.feature_column.input_layer



## nn

### tf.nn.conv2d/tf.layers.conv2d

tf.nn.conv2d

[Possibly buffer overflow in tf.nn.conv2d on GPU #24196](https://github.com/tensorflow/tensorflow/issues/24196)

https://tensorflow.google.cn/api_docs/python/tf/nn/conv2d

```python
tf.nn.conv2d(
    input,
    filter,
    strides,
    padding,
    use_cudnn_on_gpu=True,  # why need explicit this arguments?
    data_format='NHWC',
    dilations=[1, 1, 1, 1],
    name=None
)
```

Defined in generated file: `tensorflow/python/ops/gen_nn_ops.py`.

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn_ops.py

https://stackoverflow.com/questions/34835503/tensorflow-where-is-tf-nn-conv2d-actually-executed

https://stackoverflow.com/questions/34619177/what-does-tf-nn-conv2d-do-in-tensorflow?rq=1

[【TensorFlow】tf.nn.conv2d是怎样实现卷积的？](https://blog.csdn.net/mao_xiao_feng/article/details/53444333)



tf.layers.conv2d





tf.nn.conv2d vs tf.layers.conv2d

https://stackoverflow.com/questions/42785026/tf-nn-conv2d-vs-tf-layers-conv2d

tf.nn.conv2d

  `filter`: A Tensor. Must have the same type as input. A 4-D tensor of shape [filter_height, filter_width,              in_channels, out_channels]

```
def conv2d(input, filter, strides, padding, use_cudnn_on_gpu=True, data_format="NHWC", dilations=[1, 1, 1, 1], name=None):
```

tf.layers.conv2d

  `filters`: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).

```
def conv2d(inputs,
           filters,
           kernel_size,
           strides=(1, 1),
           padding='valid',
           data_format='channels_last',
           dilation_rate=(1, 1),
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
```

I would use tf.nn.conv2d when loading a pretrained model (example code: <https://github.com/ry/tensorflow-vgg16>), and tf.layers.conv2d for a model trained from scratch.







### tf.nn.dropout

Defined in [`tensorflow/python/ops/nn_ops.py`](https://tensorflow.google.cn/code/stable/tensorflow/python/ops/nn_ops.py).

```python
tf.nn.dropout(
    x,
    keep_prob=None,  # DEPRECATED
    noise_shape=None,
    seed=None,
    name=None,
    rate=None  # rate = 1 - keep_prob
)
```



### tf.nn.xw_plus_b

Defined in [`tensorflow/python/ops/nn_ops.py`](https://tensorflow.google.cn/code/stable/tensorflow/python/ops/nn_ops.py).

```python
tf.nn.xw_plus_b(
    x,
    weights,
    biases,
    name=None
)
```



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



## Dataset

### from_generator

tf.data.Dataset.from_generator



Case 1: ValueError: `generator` yielded an element of shape (1, 28, 28, 1) where an element of shape (11, 28, 28, 1) was expected. 

When specifying Tensor shapes in `from_generator`, you can use `None` as an element to specify variable-sized dimensions. This way you can accommodate batches of different sizes, in particular "leftover" batches that are a bit smaller than your requested batch size. So you would use

```
def make_fashion_dset(file_name, batch_size, shuffle=False):
    dgen = _make_fashion_generator_fn(file_name, batch_size)
    features_shape = [None, 28, 28, 1]
    labels_shape = [None, 10]
    ds = tf.data.Dataset.from_generator(
        dgen, (tf.float32, tf.uint8),
        (tf.TensorShape(features_shape), tf.TensorShape(labels_shape))
    )
    ...
```

refer: [TensorFlow DataSet `from_generator` with variable batch size](https://stackoverflow.com/questions/52121347/tensorflow-dataset-from-generator-with-variable-batch-size)

### next_batch

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



### map_and_batch

https://tensorflow.google.cn/api_docs/python/tf/data/experimental/map_and_batch

https://tensorflow.google.cn/api_docs/python/tf/contrib/data/map_and_batch

Maps `map_func` across `batch_size` consecutive elements of this dataset and then combines them into a batch. Functionally, it is equivalent to `map` followed by `batch`. However, by fusing the two transformations together, the implementation can be more efficient. 



## estimator

### Estimator

tf.estimator.Estimator

TF Estimator input is a dict, in case of multiple inputs

https://tensorflow.google.cn/api_docs/python/tf/estimator/Estimator

```python
train(
    input_fn,
    hooks=None,
    steps=None,
    max_steps=None,
    saving_listeners=None
)
input_fn: A function that provides input data for training as minibatches. See Premade Estimators for more information. The function should construct and return one of the following: * A tf.data.Dataset object: Outputs of Dataset object must be a tuple (features, labels) with same constraints as below. * A tuple (features, labels): Where features is a tf.Tensor or a dictionary of string feature name to Tensor and labels is a Tensor or a dictionary of string label name to Tensor. Both features and labels are consumed by model_fn. ****They should satisfy the expectation of model_fn from inputs.****

```



```python
predict(
    input_fn,
    predict_keys=None,
    hooks=None,
    checkpoint_path=None,
    yield_single_examples=True
)
hooks: List of tf.train.SessionRunHook subclass instances. Used for callbacks inside the prediction call.
yield_single_examples: If False, yields the whole batch as returned by the model_fn instead of decomposing the batch into individual elements. This is useful if model_fn returns some tensors whose first dimension is not equal to the batch size.
```



#### convert estimator to tpuestimator

https://cloud.google.com/tpu/docs/tutorials/migrating-to-tpuestimator-api

https://github.com/tensorflow/models/tree/master/official/mnist



### EstimatorSpec

tf.estimator.EstimatorSpec

TF Estimators requires to return a EstimatorSpec, that specify the different ops for training, evaluating, predicting

https://tensorflow.google.cn/api_docs/python/tf/estimator/EstimatorSpec

```python
@staticmethod
__new__(
    cls,
    mode,
    predictions=None,
    loss=None,
    train_op=None,
    eval_metric_ops=None,
    export_outputs=None,
    training_chief_hooks=None,
    training_hooks=None,
    scaffold=None,
    evaluation_hooks=None,
    prediction_hooks=None
)

scaffold: A tf.train.Scaffold object that can be used to set initialization, saver, and more to be used in training.
```

https://tensorflow.google.cn/api_docs/python/tf/train/Scaffold



tf.estimator.ModeKeys.PREDICT



#### add extra metrics

You can try to do by creating a logging hook and passing it into to estimator run.

In the body of `model_fn` function for your estimator:

```
logging_hook = tf.train.LoggingTensorHook({"loss" : loss, 
    "accuracy" : accuracy}, every_n_iter=10)

# Rest of the function

return tf.estimator.EstimatorSpec(
    ...params...
    training_hooks = [logging_hook])
```

EDIT:

To see the output you must also set logging verbosity high enough (unless its your default): `tf.logging.set_verbosity(tf.logging.INFO)`

<https://stackoverflow.com/questions/45353389/printing-extra-training-metrics-with-tensorflow-estimator?noredirect=1&lq=1>

### RunConfig

https://tensorflow.google.cn/api_docs/python/tf/estimator/RunConfig

```python
__init__(
    model_dir=None,
    tf_random_seed=None,
    save_summary_steps=100,
    save_checkpoints_steps=_USE_DEFAULT,
    save_checkpoints_secs=_USE_DEFAULT,
    session_config=None,
    keep_checkpoint_max=5,
    keep_checkpoint_every_n_hours=10000,
    log_step_count_steps=100,
    train_distribute=None,
    device_fn=None,
    protocol=None,
    eval_distribute=None,
    experimental_distribute=None
)

session_config: a ConfigProto used to set session parameters, or None.

```

https://tensorflow.google.cn/api_docs/python/tf/ConfigProto

https://tensorflow.google.cn/api_docs/python/tf/distribute/Strategy

https://tensorflow.google.cn/api_docs/python/tf/contrib/distribute/MirroredStrategy  Mirrors vars to distribute across multiple devices and machines.

https://tensorflow.google.cn/api_docs/python/tf/contrib/distribute/CrossDeviceOps

https://tensorflow.google.cn/api_docs/python/tf/contrib/distribute/AllReduceCrossDeviceOps



### multi_head

tf.contrib.estimator.multi_head

https://www.tensorflow.org/api_docs/python/tf/contrib/estimator/multi_head

multi-objective learning

### TPU



```
the TPU only supports tf.int32

# TPU requires a fixed batch size for all batches, therefore the number
# of examples must be a multiple of the batch size, or else examples
# will get dropped. So we pad with fake examples which are ignored
# later on.
```

#### TPUEstimator

https://tensorflow.google.cn/api_docs/python/tf/contrib/tpu/TPUEstimator

```python
__init__(
    model_fn=None,
    model_dir=None,
    config=None,
    params=None,
    use_tpu=True,
    train_batch_size=None,
    eval_batch_size=None,
    predict_batch_size=None,
    batch_axis=None,
    eval_on_tpu=True,
    export_to_tpu=True,
    warm_start_from=None
)
params: An optional dict of hyper parameters that will be passed into input_fn and model_fn. Keys are names of parameters, values are basic python types. There are reserved keys for TPUEstimator, including 'batch_size'.
use_tpu: A bool indicating whether TPU support is enabled. Currently, - TPU training and evaluation respect this bit, but eval_on_tpu can override execution of eval. See below. - Predict still happens on CPU.
train_batch_size: An int representing the global training batch size. TPUEstimator transforms this global batch size to a per-shard batch size, as params['batch_size'], when calling input_fn and model_fn. Cannot be None if use_tpu is True. Must be divisible by total number of replicas.
eval_batch_size: An int representing evaluation batch size. Must be divisible by total number of replicas.
predict_batch_size: An int representing the prediction batch size. Must be divisible by total number of replicas.
```

TPUEstimator transforms a global batch size in params to a per-shard batch size when calling the `input_fn` and `model_fn`. Users should specify global batch size in constructor, and then get the batch size for each shard in `input_fn` and `model_fn` by `params['batch_size']`.

One can set `use_tpu` to `False` for testing. All training, evaluation, and predict will be executed on CPU. `input_fn` and `model_fn` will receive `train_batch_size` or `eval_batch_size` unmodified as `params['batch_size']`.

```python
height = 32
width = 32
total_examples = 100

def predict_input_fn(params):
  batch_size = params['batch_size']

  images = tf.random_uniform(
      [total_examples, height, width, 3], minval=-1, maxval=1)

  dataset = tf.data.Dataset.from_tensor_slices(images)
  dataset = dataset.map(lambda images: {'image': images})

  dataset = dataset.batch(batch_size)
  return dataset

def model_fn(features, labels, params, mode):
   # Generate predictions, called 'output', from features['image']

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        predictions={
            'predictions': output,
            'is_padding': features['is_padding']
        })

tpu_est = TPUEstimator(
    model_fn=model_fn,
    ...,
    predict_batch_size=16)

# Fully consume the generator so that TPUEstimator can shutdown the TPU
# system.
for item in tpu_est.predict(input_fn=input_fn):
  # Filter out item if the `is_padding` is 1.
  # Process the 'predictions'
```

#### RunConfig

https://tensorflow.google.cn/api_docs/python/tf/contrib/tpu/RunConfig

https://tensorflow.google.cn/api_docs/python/tf/contrib/tpu/TPUConfig

## model managment

### save/restore

**Saves sessions, not graphs!**

A good practice is to periodically save the model’s parameters after a certain number of steps or epochs so that we can restore/retrain our model from that step if need be. 

```python
# define model

# ceate global_step, initialize it to 0 and set it to be not trainable
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

# pass global_step as a parameter to the optimizer so it knows to increment global_step by one with each training step. This can also help your optimizer know when to decay learning rate.
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss,global_step=global_step)

# create a saver object
saver = tf.train.Saver()

# launch a session to execute the computation
with tf.Session() as sess:
    
    # if a checkpoint exists, restore from the latest checkpoint
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    # actual training loop
    for step in range(training_steps): 
	sess.run([optimizer])
	if (step + 1) % 1000 == 0:
        # it’s helpful to append the number of training steps our model has gone 
	   saver.save(sess, 'checkpoint_directory/model_name', global_step=global_step)
   	
```

### summary (tensorboard)

https://github.com/tensorflow/tensorboard

motivation: visualization metrics and graph, Weights, Gradients and Activations 



Step 1 create summaries - encapsulating all ops into scopes (tf.name_scope) and create a summary to monitor metrics (tf.summary) 

Step 2 run them

Step 3 write summaries to file  (tf.summary.FileWriter)

step 4 run the following command, then open http://0.0.0.0:6006/ into your web browser

```bash
tensorboard --logdir=/tmp/tensorflow_logs 
```



```python
# Step 1 create summaries
with tf.name_scope("summaries"):
    tf.summary.scalar("loss", self.loss)
    tf.summary.scalar("accuracy", self.accuracy)            
    tf.summary.histogram("histogram loss", self.loss)
    # because you have several summaries, we should merge them all
    # into one op to make it easier to manage
    summary_op = tf.summary.merge_all()

saver = tf.train.Saver() # defaults to saving all variables

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    writer = tf.summary.FileWriter('./graphs', sess.graph)
    for index in range(10000):
        ...
        # Step 2 run them
        loss_batch, _, summary = sess.run([loss, optimizer, summary_op])
        # Step 3 write summaries to file
        # Need global step here so the model knows what summary corresponds to what step
        writer.add_summary(summary, global_step=index)

        if (index + 1) % 1000 == 0:
            saver.save(sess, 'checkpoints/skip-gram', index)

```



tf.summary.scalar

tf.summary.histogram

tf.summary.image



### control randomization

1. Set random seed at operation level. All random tensors allow you to pass in seed value in their initialization.

   ```python
   my_var = tf.Variable(tf.truncated_normal((-1.0,1.0), stddev=0.1, seed=0))
   ```

2. Set random seed at graph level with tf.Graph.seed

   ```python
   tf.set_random_seed(seed)
   ```

3. ...



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


















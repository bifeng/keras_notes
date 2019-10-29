refer: CS20si

#### reload computation graph in real time prediction

思路：将计算图只读取一遍后常驻内存

参考：

https://github.com/hanxiao/bert-as-service

https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/rOP4VKcfphg

<https://github.com/marcsto/rl/blob/master/src/fast_predict2.py>

使用tf.data.Dataset.from_generator 来读取数据，代替 tf.data.Dataset.from_tensor_slices等其他方法。 然后维护一个生成器，来不停的yield一个成员变量，期间<u>保持生成器开启</u>。每当要实时预测时，修改该成员变量为待预测数据，即可直接输出预测结果。

```python
"""
    Speeds up estimator.predict by preventing it from reloading the graph on each call to predict.
    加速estimator.predict，防止每次predict时重新加载计算图
    It does this by creating a python generator to keep the predict call open.
    原理：创建一个python生成器，保持predict进程处于一直开启状态
    This version supports tf 1.4 and above and can be used by pre-made Estimators like tf.estimator.DNNClassifier. 
    Author: Marc Stogaitis
 """
import tensorflow as tf


class FastPredict:

    def __init__(self, estimator, input_fn):
        self.estimator = estimator
        self.first_run = True
        self.closed = False
        self.input_fn = input_fn

    def _create_generator(self):
        while not self.closed:
            yield self.next_features

    def predict(self, feature_batch):
        """ Runs a prediction on a set of features. Calling multiple times
            does *not* regenerate the graph which makes predict much faster.
            feature_batch a list of list of features. IMPORTANT: If you're only classifying 1 thing,
            you still need to make it a batch of 1 by wrapping it in a list (i.e. predict([my_feature]), not predict(my_feature) 
        """
        self.next_features = feature_batch
        # 每次输入的batch必须保持一致
        if self.first_run:
            self.batch_size = len(feature_batch)
            self.predictions = self.estimator.predict(
                input_fn=self.input_fn(self._create_generator))
            self.first_run = False
        elif self.batch_size != len(feature_batch):
            raise ValueError("All batches must be of the same size. First-batch:" + str(self.batch_size) + " This-batch:" + str(len(feature_batch)))
	
        results = []
        for _ in range(self.batch_size):
            results.append(next(self.predictions))
        return results

    def close(self):
        self.closed = True
        try:
            next(self.predictions)
        except:
            print("Exception in fast_predict. This is probably OK")


def example_input_fn(generator):
    """ An example input function to pass to predict. It must take a generator as input """

    def _inner_input_fn():
        dataset = tf.data.Dataset().from_generator(generator, output_types=(tf.float32)).batch(1)
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()
        return {'x': features}

    return _inner_input_fn
```





#### LazyAdamOptimizer

more: https://www.zhihu.com/question/265357659/answer/580469438 比LazyAdamOptimizer实现更好?

https://tensorflow.google.cn/versions/r1.12/api_docs/python/tf/contrib/opt/LazyAdamOptimizer?hl=en

Variant of the Adam optimizer that handles sparse updates more efficiently.

The original Adam algorithm maintains two moving-average accumulators for each trainable variable; the accumulators are updated at every step. 

This class provides lazier handling of gradient updates for sparse variables. It only updates moving-average accumulators for sparse variable indices that appear in the current batch, rather than updating the accumulators for all indices. 



##### Application

通常自然语言处理模型的输入是非常稀疏的。对于包含几十万上百万词的词表，在训练的每个 Batch 中能出现的独立词数不超过几万个。也就是说，在每一轮梯度计算过程中，只有几万个词的 embedding 的梯度是非 0 的，其它 embedding 的梯度都是 0。

对于momentum-based的Optimizer，它会用当前的momentum去更新所有词的embedding，所以对低频词的 embedding，每次梯度下降的等效学习率是非常大的，容易引起类似过拟合的问题（？）。



##### Question

+ 为什么LazyAdamOptimizer比AdamOptimizer效果好？

  AdamOptimizer对于当前batch没有采样到的词，更新其embedding会导致过拟合？





#### Avoid lazy loading

more: https://www.zhihu.com/question/58577743

```python
# Normal loading
# Node “Add” added once to the graph definition
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = tf.add(x, y) 		# create the node before executing the graph

writer = tf.summary.FileWriter('./graphs/normal_loading', tf.get_default_graph())
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for _ in range(10):
		sess.run(z)
    print(sess.graph.as_graph_def())
    # or
    print(tf.get_default_graph().as_graph_def())
writer.close()
```

```python
# Lazy loading
# Node “Add” added 10 times to the graph definition Or as many times as you want to compute z
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')

writer = tf.summary.FileWriter('./graphs/normal_loading', tf.get_default_graph())
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for _ in range(10):
		sess.run(tf.add(x, y)) # someone decides to be clever to save one line of code
    print(sess.graph.as_graph_def())
    # or
    print(tf.get_default_graph().as_graph_def())
writer.close()
```

Imagine you want to compute an op thousands, or millions of times! For example, you might want to compute the same loss function or make the same prediction every batch of training samples. If you aren’t careful, you can add thousands of unnecessary nodes to your graph. Your graph gets bloated, Slow to load, Expensive to pass around. 

Solution:

1. **Separate definition** of ops from computing/running ops 
2. Use <u>Python @property (?)</u> to ensure function is also loaded once the first time it is called* (more refer: https://danijar.com/structuring-your-tensorflow-models/)



#### control dependencies

tf.Graph.control_dependencies(control_inputs)

\# defines which ops should be run first

```python
# your graph g have 5 ops: a, b, c, d, e
g = tf.get_default_graph()
with g.control_dependencies([a, b, c]):
	# 'd' and 'e' will only run after 'a', 'b', and 'c' have executed.
	d = ...
	e = …
```



 

#### distributed computation with subgraph

Possible to break graphs into several chunks and run them parallelly across multiple CPUs, GPUs, TPUs, or other devices.

```python
To put part of a graph on a specific CPU or GPU:
# Creates a graph.
with tf.device('/gpu:2'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='b')
  c = tf.multiply(a, b)

# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# Runs the op.
print(sess.run(c))
```





#### 限制显存

method 1 预先分配（每个GPU占用百分比）

```python
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333)))
```



method 2 动态分配

```python
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
```


















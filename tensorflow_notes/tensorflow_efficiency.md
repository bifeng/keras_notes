refer: CS20si



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


















refer: CS20si



#### data types

https://www.tensorflow.org/api_docs/python/tf/DType





```python
t_0 = 19 			         			# scalars are treated like 0-d tensors
tf.zeros_like(t_0)                  			# ==> 0
tf.ones_like(t_0)                    			# ==> 1

t_1 = [b"apple", b"peach", b"grape"] 	# 1-d arrays are treated like 1-d tensors
tf.zeros_like(t_1)                   			# ==> [b'' b'' b'']
tf.ones_like(t_1)                    			# ==> TypeError: Expected string, got 1 of type 'int' instead.

t_2 = [[True, False, False],
  [False, False, True],
  [False, True, False]]         		# 2-d arrays are treated like 2-d tensors

tf.zeros_like(t_2)                   			# ==> 3x3 tensor, all elements are False
tf.ones_like(t_2)                    			# ==> 3x3 tensor, all elements are True

```







#### constants



tf.zeros(shape, dtype=tf.float32, name=None)
tf.zeros_like(input_tensor, dtype=None, name=None, optimize=True)

tf.ones(shape, dtype=tf.float32, name=None)
tf.ones_like(input_tensor, dtype=None, name=None, optimize=True)

```python
tf.fill(dims, value, name=None)
tf.fill([2, 3], 8) ==> [[8, 8, 8], [8, 8, 8]]
```



#### sequences



```python
tf.lin_space(start, stop, num, name=None) 
tf.lin_space(10.0, 13.0, 4) ==> [10. 11. 12. 13.]

tf.range(start, limit=None, delta=1, dtype=None, name='range')
tf.range(3, 18, 3) ==> [3 6 9 12 15]
tf.range(5) ==> [0 1 2 3 4]
```

tf.random_normal
tf.truncated_normal  :star::star::star::star::star:
tf.random_uniform
tf.random_shuffle
tf.random_crop
tf.multinomial
tf.random_gamma

tf.set_random_seed(seed)  :star::star::star::star::star:

#### operations



| Category                             | Examples                                            |
| ------------------------------------ | --------------------------------------------------- |
| Element-wise operations              | Add,Sub,Mul,Div,Exp,Log,Greater,Less,Equal,...      |
| Array operations                     | Concat, Slice,Split,Constant,Rank,Shape,Shuffle,... |
| Matrix operations                    | MatMul, MatrixInverse,MatrixDeterminant,...         |
| Stateful operations                  | Variable, Assign,AssignAdd,...                      |
| Neural network buiding blocks        | SoftMax, Sigmoid,ReLU,Convolution2D,MaxPool,...     |
| Checkpointing operations             | Save, Restore                                       |
| Queue and synchronization operations | Enqueue, Dequeue, MutexAcquire, MutexRelease,...    |
| Control flow operations              | Merge,Switch, Enter, Leave, NextIteration,...       |



##### Element-wise operations



```python
a = tf.constant([2, 2], name='a')
b = tf.constant([[0, 1], [2, 3]], name='b')
with tf.Session() as sess:
	print(sess.run(tf.div(b, a)))             ⇒ [[0 0] [1 1]]
	print(sess.run(tf.divide(b, a)))          ⇒ [[0. 0.5] [1. 1.5]]
	print(sess.run(tf.truediv(b, a)))         ⇒ [[0. 0.5] [1. 1.5]]
	print(sess.run(tf.floordiv(b, a)))        ⇒ [[0 0] [1 1]]
	print(sess.run(tf.realdiv(b, a)))         ⇒ # Error: only works for real values
	print(sess.run(tf.truncatediv(b, a)))     ⇒ [[0 0] [1 1]]
	print(sess.run(tf.floor_div(b, a)))       ⇒ [[0 0] [1 1]]

```



#### variables

A constant's value is stored in the graph and replicated wherever the graph is loaded. <u>Constants are stored in the graph definition</u>. When constants are memory expensive, such as a weight matrix with millions of entries, it will be slow each time you have to load the graph. To see what’s stored in the graph's definition, simply print out the graph's protobuf.

```python
import tensorflow as tf

my_const = tf.constant([1.0, 2.0], name="my_const")
print(tf.get_default_graph().as_graph_def())
```



A variable is stored separately, and may <u>live on a parameter server</u> (Sessions allocate memory to store variable values).



tf.constant is an op<br>tf.Variable is a class with many ops:

```python
x = tf.Variable(...) 

x.initializer # init op
x.value() # read op
x.assign(...) # write op
x.assign_add(...)
x.assign_sub(...)
...
```

##### create variables

```python
# create variables with tf.Variable
s = tf.Variable(2, name="scalar") 
m = tf.Variable([[0, 1], [2, 3]], name="matrix") 
W = tf.Variable(tf.zeros([784,10]))

# create variables with tf.get_variable
s = tf.get_variable("scalar", initializer=tf.constant(2)) 
m = tf.get_variable("matrix", initializer=tf.constant([[0, 1], [2, 3]]))
W = tf.get_variable("big_matrix", shape=(784, 10), initializer=tf.zeros_initializer())
```

##### initialize

So, you have to initialize your variables before using it.

```python
# The easiest way is initializing all variables at once:
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

# Initialize only a subset of variables:
with tf.Session() as sess:
	sess.run(tf.variables_initializer([a, b]))

# Initialize a single variable
W = tf.Variable(tf.zeros([784,10]))
with tf.Session() as sess:
	sess.run(W.initializer)

```

##### assign

You don’t need to initialize variable because assign_op does it for you. In fact, initializer op is the assign op that assigns the variable’s initial value to the variable itself.

```python
import tensorflow as tf

W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
    # sess.run(W.initializer)
    sess.run(assign_op)
    print(W.eval())
```

```python
# create a variable whose original value is 2
my_var = tf.Variable(2, name="my_var") 

# assign a * 2 to a and call that op a_times_two
my_var_times_two = my_var.assign(2 * my_var)

with tf.Session() as sess:
	sess.run(my_var.initializer)
	sess.run(my_var_times_two) 				# >> the value of my_var now is 4
	sess.run(my_var_times_two) 				# >> the value of my_var now is 8
	sess.run(my_var_times_two) 				# >> the value of my_var now is 16
```


















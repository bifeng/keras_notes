refer:<br>[Deep Multi-Task Learning – 3 Lessons Learned](https://engineering.taboola.com/deep-multi-task-learning-3-lessons-learned/)



### hard parameter sharing

using Estimators with multiple heads

### task as feature for another

forward-pass:

The estimate is a Tensor, so we can wire it just like any other layer’s output.

backprop:

We probably wouldn’t want to propagate the gradients between tasks.

TensorFlow’s API has [tf.stop_gradient](https://www.tensorflow.org/api_docs/python/tf/stop_gradient) just for that reason. When computing the gradients, it lets you pass a list of Tensors you wish to treat as constants, which is exactly what we need.

```python
all_gradients = tf.gradients(loss, all_variables, stop_gradients=stop_tensors)
```



#### Question

+ In the forward-pass, which layer should combine this feature?
+ 

### loss function

1. simply sum the different losses

   backwards: While one task converges to good results, the others look pretty bad. When taking a closer look, we could easily see why. The losses’ scales were so different, that one task dominated the overall loss, while the rest of the tasks didn’t have a chance to affect the learning process of the shared layers.

2. weighted sum the different losses

   a weighted sum, that brought all losses to approximately the same scale. 

   backwards: This solution involves another hyperparameter that might need to be tuned every once in a while.

3. use the homoscedastic uncertainty to weigh losses

   The way it is done, is by learning another noise parameter that is integrated in the loss function for each task. This allows having multiple tasks, possibly regression and classification, and bringing all losses to the same scale.

   Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics, Alex Kendall, Yarin Gal, Roberto Cipolla, CVPR 2018 [arxiv](https://arxiv.org/abs/1705.07115) | [code](https://github.com/yaringal/multi-task-learning-example) 

4. ...

#### optimization

refer:[site](https://hanxiao.github.io/2017/07/07/Get-10x-Speedup-in-Tensorflow-Multi-Task-Learning-using-Python-Multiprocessing/) 

+ alternatively optimizing each task-specific loss.

  

+ jointly optimizing total loss.

  It require training data to be aligned across tasks, and you can add **adaptive weight** on each task to obtain more task-sensitive learning. 

  

### learning rate

Choosing the higher rate caused [dying Relu’s](https://www.quora.com/What-is-the-dying-ReLU-problem-in-neural-networks) on one of the tasks, while using the lower one brought a slow convergence on the other task.

1. We could tune a separate learning rate for each of the “heads” (task-specific subnets), and another rate for the shared subnet.

   ```python
   optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
   
   # replace with the following code:
   
   all_variables = shared_vars + a_vars + b_vars
   all_gradients = tf.gradients(loss, all_variables)
   
   shared_subnet_gradients = all_gradients[:len(shared_vars)]
   a_gradients = all_gradients[len(shared_vars):len(shared_vars + a_vars)]
   b_gradients = all_gradients[len(shared_vars + a_vars):]
   
   shared_subnet_optimizer = tf.train.AdamOptimizer(shared_learning_rate)
   a_optimizer = tf.train.AdamOptimizer(a_learning_rate)
   b_optimizer = tf.train.AdamOptimizer(b_learning_rate)
   
   train_shared_op = shared_subnet_optimizer.apply_gradients(zip(shared_subnet_gradients, shared_vars))
   train_a_op = a_optimizer.apply_gradients(zip(a_gradients, a_vars))
   train_b_op = b_optimizer.apply_gradients(zip(b_gradients, b_vars))
   
   train_op = tf.group(train_shared_op, train_a_op, train_b_op)
   ```

   

   backwards: ...

2. ...




















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



tf.Variable



tf.zeros

tf.random_normal



tf.name_scope ?



### operation

tf.add, tf.multiply, tf.matmul, 

tf.cast



tf.Session() ?

sess.run



tf.get_default_graph



tf.trainable_variables



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



## QA

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


















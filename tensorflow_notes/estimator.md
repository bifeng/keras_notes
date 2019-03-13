refer: https://tensorflow.google.cn/guide/custom_estimators

### input_fn

input: features, labels

output: features, labels  ???



```python
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset
```



#### numpy_input_fn

tf.estimator.inputs.numpy_input_fn

```python
def numpy_input_fn(x,
                   y=None,
                   batch_size=128,
                   num_epochs=1,
                   shuffle=None,
                   queue_capacity=1000,
                   num_threads=1):
  """Returns input function that would feed dict of numpy arrays into the model.

  This returns a function outputting `features` and `targets` based on the dict of numpy arrays. The dict `features` has the same keys as the `x`. The dict `targets` has the same keys as the `y` if `y` is a dict.
  
    Args:
    x: numpy array object or dict of numpy array objects. If an array, the array will be treated as a single feature.
    y: numpy array object or dict of numpy array object. `None` if absent.
    
```





### model_fn

input: features, labels, mode

output: EstimatorSpec

```python
def my_model_fn(
   features, # This is batch_features from input_fn
   labels,   # This is batch_labels from input_fn
   mode,     # An instance of tf.estimator.ModeKeys
   params):  # Additional configuration
    # Define the model
   
    # Specify additional calculations for each of the three different modes: predict, evaluate, train
    
    ## predict
    # Compute predictions.
    predictions = {
        'class_ids': predicted_classes[:, tf.newaxis],
        'probabilities': tf.nn.softmax(logits),
        'logits': logits,
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    ## evaluate
    metrics = {'accuracy': accuracy}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)
    
    ## train
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
	
```



| Estimator method                                             | Estimator Mode                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`tf.estimator.Estimator.train`](https://tensorflow.google.cn/api_docs/python/tf/estimator/Estimator#train) | [`tf.estimator.ModeKeys.TRAIN`](https://tensorflow.google.cn/api_docs/python/tf/estimator/ModeKeys#TRAIN) |
| [`tf.estimator.Estimator.evaluate`](https://tensorflow.google.cn/api_docs/python/tf/estimator/Estimator#evaluate) | [`tf.estimator.ModeKeys.EVAL`](https://tensorflow.google.cn/api_docs/python/tf/estimator/ModeKeys#EVAL) |
| [`tf.estimator.Estimator.predict`](https://tensorflow.google.cn/api_docs/python/tf/estimator/Estimator#predict) | [`tf.estimator.ModeKeys.PREDICT`](https://tensorflow.google.cn/api_docs/python/tf/estimator/ModeKeys#PREDICT) |

For each mode value, your code must return an instance of [`tf.estimator.EstimatorSpec`](https://tensorflow.google.cn/api_docs/python/tf/estimator/EstimatorSpec), which contains the information the caller requires.



tf.estimator.EstimatorSpec

```python

class EstimatorSpec(
    collections.namedtuple('EstimatorSpec', [
        'mode', 'predictions', 'loss', 'train_op', 'eval_metric_ops',
        'export_outputs', 'training_chief_hooks', 'training_hooks', 'scaffold',
        'evaluation_hooks', 'prediction_hooks'
    ])):
  """Ops and objects returned from a `model_fn` and passed to an `Estimator`.

  `EstimatorSpec` fully defines the model to be run by an `Estimator`.
  """

  def __new__(cls,
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
              prediction_hooks=None):
    """Creates a validated `EstimatorSpec` instance.

    Depending on the value of `mode`, different arguments are required. Namely

    * For `mode == ModeKeys.TRAIN`: required fields are `loss` and `train_op`.
    * For `mode == ModeKeys.EVAL`: required field is `loss`.
    * For `mode == ModeKeys.PREDICT`: required fields are `predictions`.

    model_fn can populate all arguments independent of mode. In this case, some
    arguments will be ignored by an `Estimator`. E.g. `train_op` will be
    ignored in eval and infer modes. Example:

    ```python
    def my_model_fn(features, labels, mode):
      predictions = ...
      loss = ...
      train_op = ...
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions,
          loss=loss,
          train_op=train_op)
    ```

    Alternatively, model_fn can just populate the arguments appropriate to the
    given mode. Example:

    ```python
    def my_model_fn(features, labels, mode):
      if (mode == tf.estimator.ModeKeys.TRAIN or
          mode == tf.estimator.ModeKeys.EVAL):
        loss = ...
      else:
        loss = None
      if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = ...
      else:
        train_op = None
      if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = ...
      else:
        predictions = None

      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions,
          loss=loss,
          train_op=train_op)
    ```

    Args:
      mode: A `ModeKeys`. Specifies if this is training, evaluation or
        prediction.
      predictions: Predictions `Tensor` or dict of `Tensor`.
      loss: Training loss `Tensor`. Must be either scalar, or with shape `[1]`.
      train_op: Op for the training step.
      eval_metric_ops: Dict of metric results keyed by name.
        The values of the dict can be one of the following:
        (1) instance of `Metric` class.
        (2) Results of calling a metric function, namely a
        `(metric_tensor, update_op)` tuple. `metric_tensor` should be
        evaluated without any impact on state (typically is a pure computation
        results based on variables.). For example, it should not trigger the
        `update_op` or requires any input fetching.
      export_outputs: Describes the output signatures to be exported to
        `SavedModel` and used during serving.
        A dict `{name: output}` where:
        * name: An arbitrary name for this output.
        * output: an `ExportOutput` object such as `ClassificationOutput`,
            `RegressionOutput`, or `PredictOutput`.
        Single-headed models only need to specify one entry in this dictionary.
        Multi-headed models should specify one entry for each head, one of
        which must be named using
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY.
        If no entry is provided, a default `PredictOutput` mapping to
        `predictions` will be created.
      training_chief_hooks: Iterable of `tf.train.SessionRunHook` objects to
        run on the chief worker during training.
      training_hooks: Iterable of `tf.train.SessionRunHook` objects to run
        on all workers during training.
      scaffold: A `tf.train.Scaffold` object that can be used to set
        initialization, saver, and more to be used in training.
      evaluation_hooks: Iterable of `tf.train.SessionRunHook` objects to
        run during evaluation.
      prediction_hooks: Iterable of `tf.train.SessionRunHook` objects to
        run during predictions.
        
      """
    

```












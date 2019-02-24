refer:<br>[high level apis 2018](https://www.youtube.com/watch?v=4oNdaQk0Qv4)<br>[tf.data 2018](https://www.youtube.com/watch?v=uIcqeP7MFH0)<br>[Eager Execution 2018](https://www.youtube.com/watch?v=T8AW0fKP0Hs)



#### Eager Execution

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/g3doc/guide.md

![graph](https://github.com/bifeng/deep_coding_notes/raw/master/image/why_graphs.png)

#### tf.data

https://www.tensorflow.org/guide/performance/datasets

![data](https://github.com/bifeng/deep_coding_notes/raw/master/image/data.png)

+ Fast

![data etl](https://github.com/bifeng/deep_coding_notes/raw/master/image/data_etl.png)

![data etl new](https://github.com/bifeng/deep_coding_notes/raw/master/image/data_etl_new.png)

+ Flexibility

  custom python code via Dataset.from_generator()



+ Ease of use

  use python for loops in eager execution mode

  standard recipes for tf.train.Example and CSV

  



#### high level apis

##### Estimator

![estimator](https://github.com/bifeng/deep_coding_notes/raw/master/image/estimator.png)

tf.estimator<br>tf.contrib.estimator



TensorFlow Estimators: Managing Simplicity vs. Flexibility in High-Level Machine Learning Frameworks, KDD 2017 [arxiv](https://arxiv.org/abs/1708.02637) 

##### Features

https://www.tensorflow.org/guide/feature_columns

tf.feature_column<br>tf.contrib.feature_column

##### Head

![head](https://github.com/bifeng/deep_coding_notes/raw/master/image/head.png)

multi_class_head: multi-task learning

##### Scaling

multi-gpu training

distributed training

##### Serving

![serving](https://github.com/bifeng/deep_coding_notes/raw/master/image/serving.png)

```python
estimator = ...
estimator.train(...)

receive_fn = build_parsing_serving_input_receive_fn(make_parse_example_spec(feature_columns))

estimator.export_savedmodel('/tmp/awesome_model', receiver_fn)
```
































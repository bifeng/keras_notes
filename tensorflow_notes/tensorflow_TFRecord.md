#### FixedLenFeature vs. FixedLenSequenceFeature 

<https://stackoverflow.com/questions/49588382/how-to-convert-float-array-list-to-tfrecord>

If your feature is a fixed 1-d array then using tf.FixedLenSequenceFeature is not correct at all. As the documentation mentioned, the tf.FixedLenSequenceFeature is for a input data with dimension 2 and higher. 

In this example you need to flatten your price array (prices is an array of shape(1,288)) to become (288,) and then for decoding part you need to mention the array dimension.

Encode:

```py
example = tf.train.Example(features=tf.train.Features(feature={
                                       'prices': _floats_feature(prices.tolist()),
                                       'label': _int64_feature(label[0]),
                                       'pip': _floats_feature(pip)
```

Decode:

```py
keys_to_features = {'prices': tf.FixedLenFeature([288], tf.float32),
                'label': tf.FixedLenFeature([], tf.int64)}
```
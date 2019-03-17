refer: CS20si

tf.data module is faster than placeholders and easier to use than queues, and doesn’t crash. (queues are notorious for being difficult to use and prone to crashing.)





#### tf.data.Dataset

**motivate - Store data in tf.data.Dataset**

- tf.data.Dataset.from_tensor_slices((features, labels))
- tf.data.Dataset.from_generator(gen, output_types, output_shapes)



**create Dataset from files**

- tf.data.TextLineDataset(filenames): each of the line in those files will become one entry. It’s good for datasets whose entries are delimited by newlines such as data used for machine translation or data in csv files.
- tf.data.FixedLengthRecordDataset(filenames): each of the data point in this dataset is of the same length. It’s good for datasets whose entries are of a fixed length, such as CIFAR or ImageNet.
- tf.data.TFRecordDataset(filenames): it’s good to use if your data is stored in tfrecord format.



#### tf.data.Iterator

**motivate - Create an iterator to iterate through samples in Dataset**

- iterator = dataset.make_one_shot_iterator()
  Iterates through the dataset exactly once. No need to initialization.

+ iterator = dataset.make_initializable_iterator()

  Iterates through the dataset as many times as we want. Need to initialize with each epoch.

```python
iterator = dataset.make_one_shot_iterator()
X, Y = iterator.get_next()         # X is the birth rate, Y is the life expectancy
with tf.Session() as sess:
	print(sess.run([X, Y]))		# >> [1.822, 74.82825]
	print(sess.run([X, Y]))		# >> [3.869, 70.81949]
	print(sess.run([X, Y]))		# >> [3.911, 72.15066]
```

```python
iterator = dataset.make_initializable_iterator()
X, Y = iterator.get_next()
...
for i in range(100): 
        sess.run(iterator.initializer) 
        total_loss = 0
        try:
            while True:
                sess.run([optimizer]) 
        except tf.errors.OutOfRangeError:
            pass
```

**Creating an iterator with different initializers!**

```python
iterator = tf.data.Iterator.from_structure(train_data.output_shapes, train_data.output_types)
img, label = iterator.get_next()
train_iter = iterator.make_initializer(train_data)
test_iter = iterator.make_initializer(test_data)
```



#### data operations

dataset = dataset.shuffle(1000)

dataset = dataset.repeat(100)

dataset = dataset.batch(128)

dataset = dataset.map(lambda x: tf.one_hot(x, 10)) 

\# convert each elem of dataset to one_hot vector






refer: cs20si



###### TFRecord

TFRecord is a Binary file format  -  a serialized tf.train.Example protobuf object (protobuf: google’s xml-like format)

- make better use of disk cache
- faster to move around 
- can handle data of different types, e.g. you can put both images and labels in one place

```python
# Write to TFRecord

# Step 1: create a writer to write tfrecord to that file
writer = tf.python_io.TFRecordWriter(out_file)

# Step 2: get serialized shape and values of the image
shape, binary_image = get_image_binary(image_file)

# Step 3: create a tf.train.Features object
# Serialize different data type into byte strings
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
features = tf.train.Features(feature={'label': _int64_feature(label),
                                    'shape': _bytes_feature(shape),
                                    'image': _bytes_feature(binary_image)})

# Step 4: create a sample containing of features defined above
sample = tf.train.Example(features=features)

# Step 5: write the sample to the tfrecord file
writer.write(sample.SerializeToString())
writer.close()
```

```python
# Read from TFRecord

dataset = tf.data.TFRecordDataset(tfrecord_files)
dataset = dataset.map(_parse_function)

# Parse each tfrecord_file into different features that we want. In this case, a tuple of (label, shape, image)
def _parse_function(tfrecord_serialized):
    features={'label': tf.FixedLenFeature([], tf.int64),
              'shape': tf.FixedLenFeature([], tf.string),
              'image': tf.FixedLenFeature([], tf.string)}

    parsed_features = tf.parse_single_example(tfrecord_serialized, features)

    return parsed_features['label'], parsed_features['shape'], parsed_features['image']

```










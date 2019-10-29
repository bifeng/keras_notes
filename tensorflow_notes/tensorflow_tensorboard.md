查看events.out.tfevents文件的目录（在model目录下）：

`tensorboard --logdir=/model/ --port=6006`

本地访问：

`http://localhost:6006`

服务器启动，本地访问：

`http://服务器ip:6006`

查看端口是否被占用：

`lsof -i:6006`





## graph

变量

tf.variable_scope

```python
with tf.variable_scope('Inputs'):
    tf_x = tf.placeholder(tf.float32, x.shape, name='x')
    tf_y = tf.placeholder(tf.float32, y.shape, name='y')
```



## histogram

查看activations、bias、weights、layer等分布

It turns out histogram is very useful for debugging as well.

1. distribution

   such as for `layer1/weights` initialized using a uniform distribution with zero mean and value range `-0.15..0.15`

   `layer1/activations` is taken as the distribution over all layer outputs in a batch.

   You can see that the distribution do change over time.

2. ...

   

```python
with tf.variable_scope('Net'):
    l1 = tf.layers.dense(tf_x, 10, tf.nn.relu, name='hidden_layer')
    output = tf.layers.dense(l1, 1, name='output_layer')

    # add to histogram summary
    tf.summary.histogram('h_out', l1)
    tf.summary.histogram('pred', output)
```

refer:

https://stackoverflow.com/questions/42315202/understanding-tensorboard-weight-histograms



embedding

```
                from tensorboard.plugins import projector
                
                
                # Embedding visualization config
                config = projector.ProjectorConfig()
                embedding_conf = config.embeddings.add()
                embedding_conf.tensor_name = "embedding"
                embedding_conf.metadata_path = FLAGS.metadata_file

                projector.visualize_embeddings(train_summary_writer, config)
                projector.visualize_embeddings(validation_summary_writer, config)

                # Save the embedding visualization
                saver.save(sess, os.path.join(out_dir, "embedding", "embedding.ckpt"))
```

refer: https://github.com/RandolphVI/Multi-Label-Text-Classification/blob/master/CNN/train_cnn.py



## events

标量

```python
loss = tf.losses.mean_squared_error(tf_y, output, scope='loss')
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)
tf.summary.scalar('loss', loss)     # add loss to scalar summary
```





## QA

### how to transpose the embedding matrix in tensorboard ?

Such as the relation embedding matrix is (35, 300), but in tensorboard is showing 300 points with 35 dimension.



### how to plot multi-roc or multi-pr curve in one plot using tensorboard ?



### overlapping when restore from checkpoint

https://stackoverflow.com/questions/38636848/whats-the-right-way-for-summary-write-to-avoid-overlapping-on-tensorboard-when?rq=1

i just restore the model and extra the global_step save as (start_step), after then i immediately add summary_writer.add_session_log(SessionLog(status=SessionLog.START), global_step=start_step), so all the events after this start_step will be discarded and no overlap.

### PR curves in tensorboard

pr_curve very detail (详细列举了各种情况) !!! :star::star::star:

https://github.com/tensorflow/tensorboard/tree/master/tensorboard/plugins/pr_curve

https://github.com/tensorflow/tensorboard/issues/894

示例：

```python
from tensorboard import summary as summary_lib
import numpy as np

labels = np.array([False, True, True, False, True])
predictions = np.array([0.2, 0.4, 0.5, 0.6, 0.8])
summary_proto = summary_lib.pr_curve_pb(
    name='foo',
    predictions=predictions,
    labels=labels,
    num_thresholds=11)
```



add non tensor metrics to summary

```python
tf.Summary(value=[tf.Summary.Value(tag="auc", simple_value=auc)])
# if you need update this value, then don't forget to add a step
```



```python
from tensorboardX import SummaryWriter
       # log pr_curve
       writer = SummaryWriter()
       writer.add_pr_curve('precision_recall_curve', labels, outputs, step)  
       writer.close() 
```

```python
summary.op(
        tag='pr_curve',
        labels=tf.cast(labels, tf.bool),
        predictions=tf.cast(scores, tf.float32),
        num_thresholds=num_thresholds,
        display_name='Precision - Recall',
)
merged_op = tf.summary.merge_all()

event_dir = os.path.join(logdir, run_name)
writer = tf.summary.FileWriter(event_dir)

with tf.Session() as session:
    writer.add_summary(session.run(merged_op), global_step=step)

# *** This line would allow to multiple pr curves in one plot
tf.reset_default_graph()
writer.close()
```

### add confusion matrix to tensorbard

https://stackoverflow.com/questions/41617463/tensorflow-confusion-matrix-in-tensorboard

https://github.com/tensorflow/tensorboard/issues/227



### visualize train/test log

https://stackoverflow.com/questions/34471563/logging-training-and-validation-loss-in-tensorboard?rq=1

设置两个`writer`，一个用于写训练的数据，一个用于写测试数据，并且这两个`writer`分别存在train和test路径中，注意测试的`writer`不能加`sess.graph`.

```python
...
train_log_dir = 'logs/train/'
test_log_dir = 'logs/test/'   # 两者路径不同
megred = tf.summary.merge_all()
with tf.Session() as sess:
    writer_train = tf.summary.FileWriter(train_log_dir,sess.graph)
    writer_test = tf.summary.FileWriter(test_log_dir)    # 注意此处不需要sess.graph
    ...other code...
    writer_train.add_summary(summary_str_train,step)
    writer_test.add_summary(summary_str_test,step)
```



### visualize multiple events log

Save the events in current timestamp dir:

```python
from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
...
train_log_dir = 'logs/train/' + TIMESTAMP
test_log_dir = 'logs/test/'   + TIMESTAMP
megred = tf.summary.merge_all()
with tf.Session() as sess:
    writer_train = tf.summary.FileWriter(train_log_dir,sess.graph)
    writer_test = tf.summary.FileWriter(test_log_dir)    
    ...other code...
    writer_train.add_summary(summary_str_train,step)
    writer_test.add_summary(summary_str_test,step)
```

### [multiple scalar summaries in one plot](https://github.com/tensorflow/tensorflow/issues/7089#) #7089

```python
import tensorflow as tf
from numpy import random

writer_1 = tf.summary.FileWriter("./logs/plot_1")
writer_2 = tf.summary.FileWriter("./logs/plot_2")

log_var = tf.Variable(0.0)
tf.summary.scalar("loss", log_var)

write_op = tf.summary.merge_all()

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

for i in range(100):
    # for writer 1
    summary = session.run(write_op, {log_var: random.rand()})
    writer_1.add_summary(summary, i)
    writer_1.flush()

    # for writer 2
    summary = session.run(write_op, {log_var: random.rand()})
    writer_2.add_summary(summary, i)
    writer_2.flush()
```




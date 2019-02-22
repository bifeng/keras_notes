refer:
[BERT中文实战踩坑](https://zhuanlan.zhihu.com/p/51762599)


code:
https://github.com/google-research/bert


### Annotated code
run_classifier.py

notes:对于training, 会进行shuffling.


### fine tuning
run_classifier.py
Step1：自定义processor
Step2：main函数新增processor


### feature based
extract_features.py



### QA
1, oom
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
2, TPU - GPU
源码采用的estimator是tf.contrib.tpu.TPUEstimator，虽然TPU的estimator同样可以在gpu和cpu上运行，但若想在gpu上更高效地做一些提升，可以考虑将其换成tf.estimator.Estimator, 同时model_fn里一些tf.contrib.tpu.TPUEstimatorSpec也需要修改成tf.estimator.EstimatorSpec的形式，以及相关调用参数也需要做一些调整。
3, early stopping
```python
early_stopping_hook = tf.contrib.estimator.stop_if_no_decrease_hook(
            estimator=estimator,
            metric_name='eval_loss',
            max_steps_without_decrease=FLAGS.max_steps_without_decrease,
            eval_dir=None,
            min_steps=0,
            run_every_secs=None,
            run_every_steps=FLAGS.save_checkpoints_steps)
# early_stopping_hook加入estimator.train
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps, hooks=[early_stopping_hook])
```
4, Train and Evaluate
```python
train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps,
                                                hooks=[early_stopping_hook])
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, throttle_secs=60)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
```
5, max_seq_length/train_batch_size等参数调优

6, model save
```python
# 加到开头
flags.DEFINE_bool("do_export", False, "Whether to export the model.")
flags.DEFINE_string("export_dir", None, "The dir where the exported model will be written.")

# 加到结尾
def serving_input_fn():
    label_ids = tf.placeholder(tf.int32, [None], name='label_ids')
    input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_ids')
    input_mask = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='segment_ids')
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'label_ids': label_ids,
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
    })()
    return input_fn

if FLAGS.do_export:
        estimator._export_to_tpu = False
        estimator.export_savedmodel(FLAGS.export_dir, serving_input_fn)
```

7, 速度优化
需要移动设备部署的用TensorFlow Lite的post_training_quantization函数
有GPU的用NAVIDIA的TensorRT
其他的Tensorflow自带optimize_for_inference函数，参考[bert-as-service](https://github.com/hanxiao/bert-as-service)

8, service
bert-as-service-如果不要求后端语言的话用这个性能就很好了。
如果要用java，可以参考bert-as-service/server/graph.py的代码把模型保存为pb文件，用java调用。


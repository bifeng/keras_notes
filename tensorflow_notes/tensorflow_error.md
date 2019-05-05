source: stackoverflow/github issues



#### ValueError: Variable w already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?

1. maybe the tensorflow graph confict, such as train and predict without separate.

   before build the graph,  reset the graph using `tf.reset_default_graph() `.



#### convnet problem

##### [Crash: Could not create cuDNN handle when convnets are used](https://github.com/tensorflow/tensorflow/issues/6698#) #6698

I believe these issues are all related to GPU memory allocation and have nothing to do with the errors being reported. There were other errors before this indicating some sort of memory allocation problem but the program continued to progress, eventually giving the cudnn errors that everyone is getting.

I have resolved this issue by changing the default behavior of TF to allocate a minimum amount of memory and grow as needed as detailed in the webpage.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config, ...)

I have also tried the alternate way and was able to get it to work and fail with experimentally choosing a percentage that worked. In my case it ended up being about .7.

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config, ...)

##### [Blas SGEMM launch failed](https://github.com/tensorflow/tensorflow/issues/9105#) #9105

...

##### [tensorflow crashes when using large image with 3d convolutional network](https://github.com/tensorflow/tensorflow/issues/5688#) #5688

os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
or type:
set TF_CUDNN_USE_AUTOTUNE=0
This will disable the autotune on convolutions, and force all cudnn convolution calls to use the default algorithm.
If this doesn't solve the problem, that means it is not related to conv3d backprop op and must be other CUDA issues. I can tell you where to go from there. If this solves the problem, then it means that during the profiling of one of the internal cudnn kernels, there was some kernel launch error. I'm working on an extensive test suite for cudnn's convolution algorithms to help us spot which internal algorithm caused the problem.



##### Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above. 

https://github.com/tensorflow/tensorflow/issues/24828



solution 0: This error may be related to installation TF with `conda`.

`conda list cudnn`
It will print:
`Name Version Build Channel`

If the result is not empty as the above, so it means you used conda installed TF, when using conda for installing TF, then it will install all the dependencies even CUDA and cuDNN, but the cuDNN version is very **low** for TF, so it will bring compatibility problem. 

So you can uninstall the cuDNN and the CUDA which was installed by conda, and then run TF, then it will work.

solution 1: Update cuDNN 

solution 2: 

```python
# Initialize your scripts/notebook (in the training/evaluation script where you call tf.session and train/eval your model) with the following code:

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# tensorflow 2.0 alpha
from tensorflow.compat.v1 import ConfigProto
config = tensorflow.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tensorflow.compat.v1.Session(config=config)
sess.as_default()

# tensorflow 2.0 alpha
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
```

solution 3:

Downgrade TensorFlow to 1.8.0 using:
`pip uninstall tensorflow-gpu`
`pip install --upgrade tensorflow-gpu==1.8.0`





#### GPU

https://tensorflow.google.cn/guide/using_gpu

https://tensorflow.google.cn/install/gpu

##### estimator with GPU

https://github.com/JayYip/bert-multitask-learning/blob/master/main.py

```python
  dist_trategy = tf.contrib.distribute.MirroredStrategy(
    num_gpus=int(FLAGS.gpu),
    cross_device_ops=tf.contrib.distribute.AllReduceCrossDeviceOps(
      'nccl', num_packs=int(FLAGS.gpu)))

  run_config = tf.estimator.RunConfig(
      train_distribute=dist_trategy,
      eval_distribute=dist_trategy,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps)

  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config)
```



https://github.com/hanxiao/bert-as-service/server/bert_serving/server/__init__.py

```python
  device_id = 1
  gpu_memory_fraction = 0.6
  config = tf.ConfigProto(device_count={'GPU': 0 if device_id < 0 else 1})
  config.gpu_options.allow_growth = True
  config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
  config.log_device_placement = False

  run_config = tf.estimator.RunConfig(
      session_config=config,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps)
  
  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config)
```



##### Using multiple GPUs

https://tensorflow.google.cn/tutorials/images/deep_cnn

```python
# Creates a graph.
c = []
for d in ['/device:GPU:2', '/device:GPU:3']:
  with tf.device(d):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    c.append(tf.matmul(a, b))
with tf.device('/cpu:0'):
  sum = tf.add_n(c)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(sum))
```



##### Tensorflow: executing an ops with a specific core of a GPU

If you would like TensorFlow to automatically choose an existing and supported device to run the operations in case the specified one doesn't exist, you can set `allow_soft_placement` to `True` in the configuration option when creating the session.

```python
# Creates a graph.
with tf.device('/device:GPU:2'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# Creates a session with allow_soft_placement and log_device_placement set
# to True.
sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True))
# Runs the op.
print(sess.run(c))
```



##### Allowing GPU memory growth

https://tensorflow.google.cn/guide/using_gpu

1. only allocate a subset of the available memory to the process

   ```python
   config = tf.ConfigProto()
   config.gpu_options.per_process_gpu_memory_fraction = 0.4
   session = tf.Session(config=config, ...)
   ```

2. only grow the memory usage as is needed by the process

   ```python
   config = tf.ConfigProto()
   config.gpu_options.allow_growth = True
   session = tf.Session(config=config, ...)
   ```

   

##### GPU环境配置

windows 10 GPU配置及安装（cuda/cudnn）：<br>https://blog.csdn.net/qilixuening/article/details/77503631

tensorflow-gpu==1.12.0

cuda_9.0.176_win10.exe

cudnn-9.0-windows10-x64-v7.1



-- 查看CUDA版本

`cat /usr/local/cuda/version.txt` 

-- 查看cuDNN版本
`cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2`



- tensorflow ImportError: DLL load failed: 找不到指定的模块。

  tensorflow-gpu==1.13.1版本太高，降低版本

```python
import tensorflow as tf
tf.test.is_gpu_available()  # tells if the gpu is available

tf.test.gpu_device_name()  # returns the name of the gpu device

```

```python
# give you all devices available to tensorflow
import os 
from tensorflow.python.client import device_lib
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "99"

if __name__ == "__main__":
    print(device_lib.list_local_devices())
```

https://stackoverflow.com/questions/38009682/how-to-tell-if-tensorflow-is-using-gpu-acceleration-from-inside-python-shell

https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow



###### upgrade NVIDIA Driver

`nvidia-smi`  # 查看 NVIDIA Driver Version

`sudo service lightdm.stop` or `sudo /etc/init.d/lightdm stop`  # 停止图形界面

`sudo /usr/bin/nvidia-uninstall` or `sudo apt-get install autoremove --purge nvidia*`  # 卸载已有NVIDIA Driver

```
# ubuntu
sudo apt-get update
apt-cache search nvidia-driver
sudo apt-get install nvidia-418
sudo reboot
```

[Centos7/RedHat7安装NVIDIA显卡驱动](https://blog.csdn.net/sunny_future/article/details/83500788)



#### CPU

##### [Tensorflow: executing an ops with a specific core of a CPU](https://stackoverflow.com/questions/37864081/tensorflow-executing-an-ops-with-a-specific-core-of-a-cpu)

There's no API for pinning ops to a particular core at present, though this would make a good [feature request](https://github.com/tensorflow/tensorflow/issues). You could approximate this functionality by creating multiple CPU devices, each with a single-threaded threadpool, but this isn't guaranteed to maintain the locality of a core-pinning solution:

```python
with tf.device("/cpu:4"):
  # ...

with tf.device("/cpu:7"):
  # ...

with tf.device("/cpu:0"):
  # ...

config = tf.ConfigProto(device_count={"CPU": 8},
                        inter_op_parallelism_threads=1,
                        intra_op_parallelism_threads=1)
sess = tf.Session(config=config)
```

##### Does TensorFlow view all CPUs of one machine as ONE device?

By default all CPUs available to the process are aggregated under `cpu:0` device.

There's answer by mrry [here](https://stackoverflow.com/a/37864489/419116) showing how to create logical devices like `/cpu:1`, `/cpu:2`

There doesn't seem to be working functionality to pin logical devices to specific physical cores or be able to use NUMA nodes in tensorflow.

A possible work-around is to use distributed TensorFlow with multiple processes on one machine and use `taskset` on Linux to pin specific processes to specific cores

https://stackoverflow.com/questions/38836269/does-tensorflow-view-all-cpus-of-one-machine-as-one-device

##### Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2

**If you have a GPU**, you shouldn't care about AVX support, because most expensive ops will be dispatched on a GPU device (unless explicitly set not to). In this case, you can simply ignore this warning by

```
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

... or by setting `export TF_CPP_MIN_LOG_LEVEL=2` if you're on Unix. 

**If you don't have a GPU** and want to utilize CPU as much as possible, **you should build tensorflow from the source optimized for your CPU** with AVX, AVX2, and FMA enabled if your CPU supports them. It's been discussed in [this question](https://stackoverflow.com/q/41293077/712995) and also [this GitHub issue](https://github.com/tensorflow/tensorflow/issues/8037). Tensorflow uses an ad-hoc build system called [bazel](https://bazel.build/) and building it is not that trivial, but is certainly doable. After this, not only will the warning disappear, tensorflow performance should also improve.

It's worth mentioning that TensorFlow Serving has separate installs for non-optimized CPU and optimized CPU (AVX, SSE4.1, etc). the details are here: [github.com/tensorflow/serving/blob/…](https://github.com/tensorflow/serving/blob/8ab0e9aeaff33d44798d6bc429195012483d48cb/tensorflow_serving/g3doc/setup.md#available-binaries) 

https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u










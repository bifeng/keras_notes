more refer:

[基于TensorFlow Serving的深度学习在线预估](https://mp.weixin.qq.com/s/DpqI4AfjiygCh8dqq_Kgmw)

https://www.tensorflow.org/tfx/serving/setup



### Question

tensorflow serving有哪些优势？

tensorflow serving怎么解决并发问题？

tensorflow serving is more security than session ?



优势：

自动检测并加载新版本模型或回退旧版本模型，无需重启服务？ 怎么做到的？（热更新）

可以同时挂载多个模型 ，无需启动多个服务？

无需写任何部署代码 ？



refer: [TensorFlow Serving + Docker + Tornado机器学习模型生产级快速部署](https://zhuanlan.zhihu.com/p/52096200)



### TensorFlow Serving

motivation: 通过启动多个python进程方式，加载多个模型，首先-需要为模型指定相应的GPU，其次-...，随着模型的增加，由于进程的数量也受到GPU数量的限制、并发？...等原因，最终导致资源浪费，运行效率低下。



#### Architecture

https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/architecture.md







##### gRPC

gRPC是一种高性能、通用的远程过程调用（Remote Procedure Call，RPC）框架。

RPC协议包含了编码协议和传输协议:

1. 编码协议是Protocol Buffers（ProtoBuf），它是Google开发的一种二进制格式数据描述语言，支持众多开发语言和平台。与JSON、XML相比，ProtoBuf的优点是体积小、速度快，其序列化与反序列化代码都是通过代码生成器根据定义好的数据结构生成的，使用起来也很简单。
2. 传输协议是HTTP/2，相比于HTTP/1.1，HTTP/2引入了头部压缩算法（HPACK）等新特性，并采用了二进制而非明文来打包、传输客户端——服务器间的数据，性能更好，功能更强。

gRPC提供了一种简单的方法来精确地定义服务，并自动为客户端生成可靠性很强的功能库。

在使用gRPC进行通信之前，我们需要完成两步操作：

1. 定义服务
2. 生成服务端和客户端代码

TensorFlow Serving已经定义好了

```markdown
$ tree serving
serving
├── tensorflow
│   ├── ...
├── tensorflow_serving
│   ├── apis
│   │   ├── model.proto
│   │   ├── model_service.py
│   │   ├── predict.proto
│   │   ├── prediction_service.proto
│   │   ├── ...
│   │   ├── model_service_pb2.py
│   │   ├── model_service_pb2_grpc.py
│   │   ├── prediction_service_pb2.py
│   │   ├── prediction_service_pb2_grpc.py
│   │   ├── ...
│   ├── ...
├── ...
```





##### tensorflow_model_server





#### Docker + tensorflow serving + Web

step 1 : export tensorflow model

step 2: Docker

step 3: TensorFlow Serving Docker

Tensorflow Serving使用gRPC协议

step 4: Client

客户端主要发送预测请求并接收结果，也需要安装使用gRPC协议的API。

step 5: Web

![client_web_tf_serving](https://github.com/bifeng/deep_coding_notes/raw/master/image/client_web_tf_serving.jpg)

Web主要承担所有的数据预处理、后处理任务。

`engine.py`负责构建一个Request用于与TensorFlow Serving交互。

`serving.py`负责接收和处理请求。

在生产环境中一般使用反向代理软件如Nginx实现负载均衡。

Tornado是一个支持异步非阻塞的高性能Web框架，可以接收多个用户的并发请求，然后向TensorFlow Serving并发请求结果。





##### export tensorflow model

https://www.tensorflow.org/tfx/serving/serving_basic



##### Docker

查看正在运行的容器

```bash
sudo docker ps
```

杀死服务

```bash
sudo docker kill IMAGE_NAME
```





##### TensorFlow Serving Docker

https://hub.docker.com/r/tensorflow/serving/tags/



###### 直接启动

```bash
sudo nvidia-docker run -p 8500:8500 \
  --mount type=bind,source=/you/local/models,target=/models \
  -t --entrypoint=tensorflow_model_server tensorflow/serving:latest-gpu \
  --port=8500 --per_process_gpu_memory_fraction=0.5 \
  --enable_batching=true --model_name=east --model_base_path=/models/east_model &
```

- -p 8500:8500 ：指的是开放8500这个gRPC端口。
- --mount type=bind, source=/your/local/models, target=/models：把你导出的本地模型文件夹挂载到docker container的/models这个文件夹，tensorflow serving会从容器内的/models文件夹里面找到你的模型。
- -t --entrypoint=tensorflow_model_server tensorflow/serving:latest-gpu：如果使用非devel版的docker，启动docker之后是不能进入容器内部bash环境的，--entrypoint的作用是允许你“间接”进入容器内部，然后调用tensorflow_model_server命令来启动TensorFlow Serving，这样才能输入后面的参数。紧接着指定使用tensorflow/serving:latest-gpu 这个镜像，可以换成你想要的任何版本。
- --port=8500：开放8500这个gRPC端口（需要先设置上面entrypoint参数，否则无效。下面参数亦然）
- --per_process_gpu_memory_fraction=0.5：只允许模型使用多少百分比的显存，数值在[0, 1]之间。
- --enable_batching：允许模型进行批推理，提高GPU使用效率。
- --model_name：模型名字，在导出模型的时候设置的名字。
- --model_base_path：模型所在容器内的路径，前面的mount已经挂载到了/models文件夹内，这里需要进一步指定到某个模型文件夹，例如/models/east_model指的是使用/models/east_model这个文件夹下面的模型。



###### 内部启动

1）进入容器内部，修改配置环境

```bash
sudo nvidia-docker run -it tensorflow/serving:latest-devel-gpu bash
```

-it  以交互方式进入容器内部

bash 进入容器的shell

2）本地文件夹复制到容器内部的文件夹

```bash
sudo docker cp /your/local/file YOUR_CONTAINER_ID:/your/container/dir
```

3）容器内部启动

```bash
tensorflow_model_server --port=8500 --rest_api_port=8501 \
  --model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME}
```

4）保存镜像

先在容器外部的shell执行`commit`，再退出容器.

```bash
sudo docker commit $(sudo docker ps --last 1 -q) YOUR_IMAGE_NAME:VERSION
```

查看镜像

```bash
sudo docker images
```



##### Client

```bash
pip install tensorflow-serving-api
```

   

##### Web



### Application

https://github.com/pakrchen/text-antispam/tree/master/serving

https://github.com/tobegit3hub/simple_tensorflow_serving

https://github.com/tobegit3hub/tensorflow_template_application

https://github.com/sthalles/deeplab_v3/tree/master/serving

https://medium.freecodecamp.org/how-to-deploy-tensorflow-models-to-production-using-tf-serving-4b4b78d41700











serving

bert serving






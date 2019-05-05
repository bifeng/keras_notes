查看events.out.tfevents文件的目录（在model目录下）：

`tensorboard --logdir=/model/ --port=6006`

本地访问：

`http://localhost:6006`

服务器启动，本地访问：

`http://服务器ip:6006`

查看端口是否被占用：

`lsof -i:6006`


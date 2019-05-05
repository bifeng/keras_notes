##### 查看22端口现在运行的情况 
lsof -i :22

```text
COMMAND  PID USER   FD   TYPE DEVICE SIZE NODE NAME
sshd    1409 root    3u  IPv6   5678       TCP *:ssh (LISTEN)
```



##### 显示进程的运行脚本及工作路径

显示进程22521的运行脚本

`ps 22521`

显示进程22521的工作路径

`pwdx 22521`



#### lsof

https://blog.csdn.net/guoguo1980/article/details/2324454








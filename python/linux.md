

##### Linux CPU个数，核心，线程数

CPU 个数

grep 'physical id' /proc/cpuinfo | sort -u

CPU 核心数

grep 'core id' /proc/cpuinfo | sort -u | wc -l

CPU 线程数

grep 'processor' /proc/cpuinfo | sort -u | wc -l

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

通过 ll /proc/{进程号}/cwd查看运行目录

`ll /proc/22521/cwd`



#### top命令参数详解

more: <https://blog.csdn.net/yjclsx/article/details/81508455>

在top基本视图中，按键盘数字“1”可以监控每个逻辑CPU的状况

按键盘‘b’（打开关闭加亮效果）

按键盘‘x’（打开/关闭排序列的加亮效果）





#### ll /proc/PID

ll /proc/PID/cwd、ll /proc/PID/exe...

cwd符号链接进程运行目录；

exe符号链接执行程序的绝对路径；

cmdline就是程序运行时输入的命令行命令；

environ记录了进程运行时的环境变量；

fd目录下是进程打开或使用的文件的符号连接。



#### lsof

https://blog.csdn.net/guoguo1980/article/details/2324454



#### ps -ef | grep ...

ps（Process Status）-查看进程

ps命令的参数： 
-e : 显示所有进程 
-f : 全格式 <br>...

| 是管道符号，表示ps 和 grep 命令同时执行

grep （Global Regular Expression Print）- 使用正则表达式搜索文本，然后把匹配的行显示出来。

示例：ps -e|grep dae

字段含义如下：
UID       PID       PPID      C     STIME    TTY       TIME         CMD

zzw      14124   13991      0     00:38      pts/0      00:00:00    grep --color=auto dae

UID      ：程序被该 UID 所拥有

PID      ：就是这个程序的 ID 

PPID    ：则是其上级父程序的ID

C          ：CPU使用的资源百分比

STIME ：系统启动时间

TTY     ：登入者的终端机位置

TIME   ：使用掉的CPU时间。

CMD   ：所下达的是什么指令



#### conda in linux

source activate ...








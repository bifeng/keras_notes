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
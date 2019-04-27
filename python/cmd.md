



### `2>&1`

```bat
mycommand > my.log 2>&1
```

命令的结果-通过“%>”的形式来定向输出. 

其中，%表示文件描述符。系统默认%值是1，也就是“1>”，而1>可以简写为>，也就是默认为>。

1为标准输出stdout、2为标准错误stderr，stdout的默认目标是终端，stderr的默认目标为也是终端。

```text
重定向操作符 描述 
> 将命令输出写入到文件或设备（如打印机），而不是命令提示符窗口或句柄。
< 从文件而不是从键盘或句柄读入命令输入。
>> 将命令输出添加到文件末尾而不删除文件中已有的信息。
>& 将一个句柄的输出写入到另一个句柄的输入中。
<& 从一个句柄读取输入并将其写入到另一个句柄输出中。
| 从一个命令中读取输出并将其写入另一个命令的输入中。也称作管道。
```

### conda in bat

https://stackoverflow.com/questions/24678144/conda-environments-and-bat-files

Use the 'call' command when activating/deactivating the environment.

```py
call activate [my_env]
python my_script.py
call deactivate
```






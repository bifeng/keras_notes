more refer:

<https://note.qidong.name/2018/11/python-multiprocessing/>

http://doc.codingdict.com/python_352/library/multiprocessing.html

https://www.liaoxuefeng.com/wiki/897692888725344/923057623066752



### case

https://www.cnblogs.com/Xjng/p/4902514.html



#### apply_async with multiple args

```
from multiprocessing import Pool

def f(x, *args, **kwargs):
    print(x, args, kwargs)

args, kw = (1,2,3), {'cat': 'dog'}

def main():
    print("# Normal call")
    f(0, *args, **kw)
    print("# Multicall")
    P = Pool()
    sol = [P.apply_async(f, (x,) + args, kw) for x in range(2)]
    P.close()
    P.join()

    for s in sol: s.get()

if __name__ == '__main__':
    main()
```

[multiprocess.apply_async How do I wrap *args and **kwargs?](https://stackoverflow.com/questions/16224600/multiprocess-apply-async-how-do-i-wrap-args-and-kwargs) 

#### starmap_async with multiple args

```
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from multiprocessing.pool import Pool


def f(x):
    return x * x


def g(x, y):
    return x**y


def main():
    with Pool(4) as pool:
        result = pool.map_async(f, [1, 2, 3, 4, 5])
        print(type(result))
        print(result.get())

    with Pool(4) as pool:
        result = pool.starmap_async(g, [(1, 3), (2, 4), (3, 5), (4, 6), (5, 7)])
        print(type(result))
        print(result.get())


if __name__ == '__main__':
    main()
```



#### 多进程cpu及内存监控(pid)

```python
#先下载psutil库:pip install psutil
import psutil
import os,datetime,time
import multiprocessing


def     getMemCpu():
	   pid=multiprocessing.current_process().pid
	   process = psutil.Process(pid)
       memory =  'Used Memory:',process.memory_info().rss / 1024 / 1024,'MB'
       cpu = "CPU:%0.2f"%process.cpu_percent()+"%"
       return memory+cpu

def     main():

        while(True):
            info = getMemCpu()
            time.sleep(0.2)
            print info+"\b"*(len(info)+1),

def     main():
	    po = multiprocessing.Pool(6)
        po.starmap_async(getMemCpu(),)
        po.close()
        po.join()
     
            
if      __name__=="__main__":
        main()
```








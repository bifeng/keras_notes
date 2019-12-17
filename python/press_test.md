more:

[**python实现多线程post方法进行压测脚本模板**](https://blog.csdn.net/henni_719/article/details/73188918)





#### 多线程压力测试

创建50个线程，每个线程请求1次

```python
# coding=utf-8
import requests
from time import ctime
import threading
import json

# 定义需要进行发送的数据
params = {"sentences": "['今天天气不错','你好啊！']",
          "tokenizer": "jieba"}

ip = "127.0.0.1"
port = "8000"
req_url = "http://" + ip + ":" + port + "/nlp/api/base/word_segment"


TOTAL = 0  # 总数
SUCC = 0  # 响应成功数
FAIL = 0  # 响应失败数


# 创建请求函数
def Clean(req_url, params):
    global TOTAL
    global SUCC
    global FAIL

    params = json.dumps(params)

    # 发送请求,返回响应
    try:
        response = requests.post(url=req_url, data=params)
        if response.status_code == 200:
            SUCC += 1
            TOTAL += 1
        else:
            FAIL += 1
            TOTAL += 1
    except:
        FAIL += 1
        TOTAL += 1


# 创建数组存放线程
threads = []
# 创建50个线程
for i in range(50):
    # 针对函数创建线程
    t = threading.Thread(target=Clean, args=(req_url, params))
    # 把创建的线程加入线程组
    t.setName("sss1:{}".format(i))
    # t.setDaemon(False)  # 是否守护线程
    threads.append(t)

print('start:', ctime())

if __name__ == '__main__':
    # 当前线程是否主线程
    assert threading.current_thread() == threading.main_thread()
    
    # 启动子线程
    for i in threads:
        i.start()
        print(i.name)
        print(i.current_thread().name)    
    # keep thread
    for i in threads:
        i.join()
    
    # 主线程
    print(threading.current_thread().name)
    for i in range(5):  
        print i
            
    print("total:%d,succ:%d,fail:%d" % (TOTAL, SUCC, FAIL))

print('end:', ctime())
```

创建50个线程，每个线程请求10次

```python
# coding=utf-8
import requests
import time
import threading
import json


# 定义需要进行发送的数据
params={"sentences":"['今天天气不错','你好啊！']",
        "tokenizer":"jieba"}

ip = "127.0.0.1"
port = "8000"
req_url="word_segment":"http://"+ip+":"+port+"/nlp/api/base/word_segment"


SUCC = 0  # 响应成功数
# 创建请求函数
def Clean(req_url,params):
    global SUCC
    global POST_NUM

    params = json.dumps(params)
    # 发送请求,返回响应
    for i in range(POST_NUM):
        response = requests.post(url=req_url,data=params)
        print(response.status_code)
        if response.status_code == 200:
            lock = threading.Lock()  # 创建锁
            lock.acquire()  # 获取锁
            SUCC += 1
            lock.release()  # 释放锁

# 创建数组存放线程
threads = []
# 创建50个线程
TREAD_NUM=50
for i in range(TREAD_NUM):
    # 针对函数创建线程
    t = threading.Thread(target=Clean, args=(req_url,params))
    # 把创建的线程加入线程组
    threads.append(t)

start = time.time()

if __name__ == '__main__':
    # 每个线程发送的请求数目
    POST_NUM = 10
    # 启动线程
    for i in threads:
        i.start()
        # keep thread
    for i in threads:
        i.join()
    TOTAL = POST_NUM * TREAD_NUM
    print("total:%d,succ:%d,fail:%d" %(TOTAL,SUCC,TOTAL-SUCC))

print('cost_time:', time.time()-start,"s")
```






#### Redis

注意：所有操作最好是<u>脚本</u>在<u>服务器</u>执行。本地执行速度慢，而且会遇到连接超时、中文字符等问题。

##### basic

https://www.jianshu.com/p/2639549bedc8

###### string

set()

```
#在Redis中设置值，默认不存在则创建，存在则修改
r.set('name', 'zhangsan')
'''参数：
     set(name, value, ex=None, px=None, nx=False, xx=False)
     ex，过期时间（秒）
     px，过期时间（毫秒）
     nx，如果设置为True，则只有name不存在时，当前set操作才执行,同setnx(name, value)
     xx，如果设置为True，则只有name存在时，当前set操作才执行'''
```



```
setex(name, value, time)
#设置过期时间（秒）

psetex(name, time_ms, value)
#设置过期时间（豪秒）
```

mset()

```
#批量设置值
r.mset(name1='zhangsan', name2='lisi')
#或
r.mget({"name1":'zhangsan', "name2":'lisi'})
```

get(name)

　　获取值

mget(keys, *args)

```
#批量获取
print(r.mget("name1","name2"))
#或
li=["name1","name2"]
print(r.mget(li))
```

...



###### hash

hset(name, key, value)

```
#name对应的hash中设置一个键值对（不存在，则创建，否则，修改）
r.hset("dic_name","a1","aa")
```

hget(name,key)

```
r.hset("dic_name","a1","aa")
#在name对应的hash中根据key获取value
print(r.hget("dic_name","a1"))#输出:aa
```

hgetall(name)

```
#获取name对应hash的所有键值
print(r.hgetall("dic_name"))
```

hmset(name, mapping)

```
#在name对应的hash中批量设置键值对,mapping:字典
dic={"a1":"aa","b1":"bb"}
r.hmset("dic_name",dic)
print(r.hget("dic_name","b1"))#输出:bb
```

hmget(name, keys, *args)

```
# 在name对应的hash中获取多个key的值
li=["a1","b1"]
print(r.hmget("dic_name",li))
print(r.hmget("dic_name","a1","b1"))
```

...

###### transaction

http://www.cnblogs.com/kangoroo/p/7535405.html



##### pipeline

refer:https://www.cnblogs.com/kangoroo/p/7647052.html



```python
import redis
pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
conn = redis.Redis(connection_pool=pool)
pipe =conn.pipeline(transaction=False)
pipe.set('name', 'jack')
pipe.set('role', 'sb')
pipe.sadd('faz', 'baz')
pipe.incr('num')    # 如果num不存在则vaule为1，如果存在，则value自增1
pipe.execute()
or
pipe.set('hello', 'redis').sadd('faz', 'baz').incr('num').execute()
```



```python
pipe_size = 10000
len = 0
key_list = []
vocab_dict = {}
for name in redis.scan_iter(match=self.key + '_freq' + "*"):
    key_list.append(name)
    pipe.get(name)
    if len < pipe_size:
        len += 1
    else:
        result = pipe.execute()
        vocab_dict.update({self.extract_word(key_list[index]): Vocab(v) for index, v in enumerate(result)})
        len = 0
        key_list = []
# finally - for the rest sth smaller than pipe_size
if len(key_list):
    result = pipe.execute()
    vocab_dict.update({self.extract_word(key_list[index]): Vocab(v) for index, v in enumerate(result)})
```





test pipeline:

```python
# -*- coding:utf-8 -*-

import redis
import time
from concurrent.futures import ProcessPoolExecutor

r = redis.Redis(host='10.93.84.53', port=6379, password='bigdata123')


def try_pipeline():
    start = time.time()
    with r.pipeline(transaction=False) as p:
        p.sadd('seta', 1).sadd('seta', 2).srem('seta', 2).lpush('lista', 1).lrange('lista', 0, -1)
        p.execute()
    print(time.time() - start)


def without_pipeline():
    start = time.time()
    r.sadd('seta', 1)
    r.sadd('seta', 2)
    r.srem('seta', 2)
    r.lpush('lista', 1)
    r.lrange('lista', 0, -1)
    print(time.time() - start)


def worker():
    while True:
        try_pipeline()

with ProcessPoolExecutor(max_workers=12) as pool:
    for _ in range(10):
        pool.submit(worker)
```





##### serving word vectors with redis

Question:

1. hash方式存储，可以保证数据分配到redis各个子节点？
2. 删除等操作，redis无法及时清空内存？

![serving_word_vectors_with_redis](https://github.com/bifeng/deep_coding_notes/raw/master/image/serving_word_vectors_with_redis.png)



refer:

https://engineering.talentpair.com/serving-word-vectors-for-distributed-computations-c5065cbaa02f



Load word vectors from redis:

```python
import bz2
import numpy as np
import pickle

from django.conf import settings
from django_redis import get_redis_connection
from gensim.models.keyedvectors import KeyedVectors

from .constants import GOOGLE_WORD2VEC_MODEL_NAME
from .redis import load_word2vec_model_into_redis, query_redis


class RedisKeyedVectors(KeyedVectors):
    """
    Class to imitate gensim's KeyedVectors, but instead getting the vectors from the memory, the vectors
    will be retrieved from a redis db
    """
    def __init__(self, key=GOOGLE_WORD2VEC_MODEL_NAME):
        self.rs = get_redis_connection(alias='word2vec')
        self.syn0 = []
        self.syn0norm = None
        self.check_vocab_len()
        self.index2word = []
        self.key = key

    @classmethod
    def check_vocab_len(cls, key=GOOGLE_WORD2VEC_MODEL_NAME, **kwargs):
        rs = get_redis_connection(alias='word2vec')
        return len(list(rs.scan_iter(key + "*")))
    
    @classmethod
    def load_word2vec_format(cls, **kwargs):
        raise NotImplementedError("You can't load a word model that way. It needs to pre-loaded into redis")

    def save(self, *args, **kwargs):
        raise NotImplementedError("You can't write back to Redis that way.")

    def save_word2vec_format(self, **kwargs):
        raise NotImplementedError("You can't write back to Redis that way.")

    def word_vec(self, word, **kwargs):
        """
        This method is mimicking the word_vec method from the Gensim KeyedVector class. Instead of
        looking it up from an in memory dict, it
        - requests the value from the redis instance, where the key is a combination between the word vector
        model key and the word itself
        - decompresses it
        - and finally unpickles it
        :param word: string
        
        :returns: numpy array of dim of the word vector model (for Google: 300, 1)
        """
        
        try:
            return pickle.loads(bz2.decompress(query_redis(self.rs, word)))
        except TypeError:
            return None

    def __getitem__(self, words):
        """
        returns numpy array for single word or vstack for multiple words
        """
        if isinstance(words, str):
            # allow calls like trained_model['Chief Executive Officer']
            return self.word_vec(words)
        return np.vstack([self.word_vec(word) for word in words])

    def __contains__(self, word):
        """ build in method to quickly check whether a word is available in redis """
        return self.rs.exists(self.key + word)
```

Set word vectors to redis:

```python
import bz2
import pickle

from django.conf import settings
from djang_redis import get_redis_connection
from tqdm import tqdm

from .constants import GOOGLE_WORD2VEC_MODEL_NAME


def load_word2vec_into_redis(rs, wvmodel, key=GOOGLE_WORD2VEC_MODEL_NAME):
    """ This function loops over all available words in the loaded word2vec model and loads
    them into the redis instance via the rs object.
    :param rs: redis connection object from django_redis 
    :param wvmodel: word vector model loaded into the memory of this machine. 
      Once the loading is completed, the memory will be available again.
    :param key: suffix for the redis keys
    """

    print("Update Word2Vec model in redis ...")
    for word in tqdm(list(wvmodel.vocab.keys())):
        rs.set(key + word, bz2.compress(pickle.dumps(wvmodel[word])))
```







#### Oracle

本地连接Oracle服务器数据库

https://blog.csdn.net/guimaxingmc/article/details/80360840

乱码问题：

```python
# 正式执行查询前添加：
import os
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.AL32UTF8'
```






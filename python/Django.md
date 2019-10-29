#### setup

refer:

https://blog.csdn.net/will5451/article/details/53861092

settings文件修改：

```python
ALLOWED_HOSTS = ['*'] # note1
```

note1 - 避免`Invalid HTTP_HOST header: '10.211.55.6:8000'. You may need to add u'10.211.55.6' to ALLOWED_HOSTS.`



python manage.py runserver 0.0.0.0:8000



#### 127.0.0.1/0.0.0.0/本机IP

[what-is-the-difference-between-0-0-0-0-127-0-0-1-and-localhost](https://stackoverflow.com/questions/20778771/what-is-the-difference-between-0-0-0-0-127-0-0-1-and-localhost) :star:

当我们在服务器搭建了一个web服务器的时候，如果监听的端口是`127.0.0.1:端口号`的时候，那么这个web服务器只可以在服务器本地访问了，在别的地方进行访问是不行的。

一个服务有多个IP地址（192.168.1.2和10.1.1.12），如果设置的监听地址是0.0.0.0，那么我们无论是通过192.168.1.2还是10.1.1.12都是可以访问该服务的。

如果我们监听的是本地IP的话，那么只有通过监听的IP才可以访问我们的对应的服务。

```markdown
0.0.0.0:9999 外部可以通过本机ip访问，这种方式最保险 
192.168.0.105:9999 外部可以通过这个ip访问9999 
127.0.0.1：9999 这种方式外部访问不了，本机可以访问
```

记得多用telnet测试，端口是否开放或占用



#### rest/restful api

[REST API规范](https://www.liaoxuefeng.com/wiki/001434446689867b27157e896e74d51a89c25cc8b43bdb3000/0014735914606943e2866257aa644b4bdfe01d26d29960b000)

[RESTful API 最佳实践](http://www.ruanyifeng.com/blog/2018/10/restful-api-best-practices.html)



REST规范定义了资源的通用访问格式

1. REST请求

   仍然是标准的HTTP请求，但是，除了**GET请求**外，**POST、PUT等请求**的body是JSON数据格式，请求的`Content-Type`为`application/json`；

2. REST响应

   返回的结果是JSON数据格式，因此，响应的`Content-Type`也是`application/json`。



示例：商品Product就是一种资源

获取所有Product的URL如下：

```
GET /api/products
```

而获取某个指定的Product，例如，id为`123`的Product，其URL如下：

```
GET /api/products/123
```

新建一个Product使用POST请求，<u>JSON数据包含在body</u>中，URL如下：

```
POST /api/products
```

更新一个Product使用PUT请求，例如，更新id为`123`的Product，其URL如下：

```
PUT /api/products/123
```

删除一个Product使用DELETE请求，例如，删除id为`123`的Product，其URL如下：

```
DELETE /api/products/123
```





POST接受及测试：

```python
        body = request.body
        params = json.loads(body.decode())
        sentence = params.get('sentence', 'hello!')
```

```python
        sentence = request.POST.get('sentence', '')
```



测试：

```bash
curl -H "Content-Type: application/json" -X POST -d "{"name":"XBox","price":3999}" http://localhost:3000/api/products
```

Attention: <br>引号 - 一定要用双引号`""`

中文编码问题 : `params = json.loads(body.decode())`

`UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc4 in position 11: invalid continuation byte`

解码问题：`params = json.loads(body.decode())`

`json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)`



Postman:

```json
{
	"pretrained_embeddings":["elmo"],
	"model_mix_strategy":"AVERAGE",
	"model_mix_weight":[],
	"weight_strategy":"MEAN",
	"sentences":"[['天气', '真好'],['hello', 'nlp', 'word']]",    # 内部单引号，外部双引号
	"mannual_keyword_weight":{}
}
```

Attention:

引号 - key及value内容一定要用双引号`""`！！！

返回空值 ：`sentence = request.POST.get('sentence', '')` 







#### POST/GET

more refer: 

[what-is-the-difference-between-post-and-get](https://stackoverflow.com/questions/3477333/what-is-the-difference-between-post-and-get)

[**HTTP POST GET 本质区别详解**](https://blog.csdn.net/gideal_wang/article/details/4316691)



post

```python
# api
def api(request):
    print(request.body)
    request.POST.get()
    ...

# test api
import requests
requests.post(apiurl, data)
```

get

```python
# api
def api(request):
    print(request.body)
    request.GET.get()
    ...

# test api
import requests
requests.get(apiurl, data)
```



#### Cache support share cross-process ?

[Local-memory caching](https://docs.djangoproject.com/en/2.2/topics/cache/#local-memory-caching)

Note that each process will have its own private cache instance, which means no cross-process caching is possible. This obviously also means the local memory cache isn’t particularly memory-efficient, so it’s probably not a good choice for production environments.



Cache support share cross-process ?

http://www.grantjenks.com/docs/diskcache/tutorial.html

It provide an efficient and safe means of cross-thread and cross-process communication.



#### csrf

https://docs.djangoproject.com/en/2.1/ref/csrf/

http://www.cnblogs.com/lianzhilei/p/6364061.html

```python
MIDDLEWARE = [
...
    'django.middleware.csrf.CsrfViewMiddleware',
...
]
```



#### csrf_exempt

https://khalsalabs.com/csrf-exempt-in-django/

By default, django check for csrf token with each POST request, it verifies csrf token before rendering the view. Its a very good security practice to verify csrf of post requests as we know django can’t be compromised in case of security.

In some cases we do not need csrf validations, e.g for public APIs, common AJAX request, REST APIs. To suppress csrf verification message, we can use **@csrf_exempt** decorator for specific view.

```python
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse

@csrf_exempt
def public_api(request):
    if request.method=='POST':
       return HttpResponse('API hit with post method')
```

Above API will allow a post call without adding csrf parameter in it. Otherwise you have to send csrf token for API calls in django.

#### uwsgi



https://uwsgi-docs-additions.readthedocs.io/en/latest/Options.html



##### [使用uWSGI的spooler做异步任务](https://www.kawabangga.com/posts/3101)

[使用uwsgi实现异步任务](https://knktc.com/2018/07/24/uwsgi-spooler-as-async-queue/)



##### [worker dies by signal 6 + Fatal Python error: Inconsistent interned string state.](https://github.com/unbit/uwsgi/issues/1111#) #1111



##### [DAMN ! worker 1 (pid: 108) died, killed by signal 9 :( trying respawn ...](https://github.com/unbit/uwsgi/issues/1779#) #1779

more refer: [使用uWSGI和nginx如何设置连接超时时间](https://www.jianshu.com/p/f5ee6b6b7e54)



##### [uWSGI request timeout in Python](https://stackoverflow.com/questions/24127601/uwsgi-request-timeout-in-python)



##### Start uWSGI without preload some models?

uwsgi.log - `spawned uWSGI http 1 `

Add following lines in `app/wsgi.py`:

```python
import app.urls
```










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

Attention: <br>引号问题`"` or `'`

中文编码问题 : `params = json.loads(body.decode())`

`UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc4 in position 11: invalid continuation byte`

解码问题：`params = json.loads(body.decode())`

`json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)`



Postman:



Attention:

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


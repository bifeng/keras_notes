

#### POST/GET

more refer: [**HTTP POST GET 本质区别详解**](https://blog.csdn.net/gideal_wang/article/details/4316691)



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




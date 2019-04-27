#### Numpy

##### reduce dimension

```python
>>t0
array([[[ 1,  2,  3],
        [ 4,  5,  6],
        [64, 24, 96]],
       [[10, 11, 12],
        [ 7,  8,  9],
        [12, 29, 43]]])
>>t0.reshape(3,6)
array([[ 1,  2,  3,  4,  5,  6],
       [64, 24, 96, 10, 11, 12],
       [ 7,  8,  9, 12, 29, 43]])
```

```python
>>t0
array([[[ 1,  2,  3],
        [ 4,  5,  6],
        [64, 24, 96]],
       [[10, 11, 12],
        [ 7,  8,  9],
        [12, 29, 43]]])
>>np.column_stack(t0)
array([[ 1,  2,  3, 10, 11, 12],
       [ 4,  5,  6,  7,  8,  9],
       [64, 24, 96, 12, 29, 43]])

```

##### high dimension matmul low dimension

https://ldzhangyx.github.io/2017/12/21/%E3%80%90TensorFlow%E9%9A%8F%E7%AC%94%E3%80%91%E5%85%B3%E4%BA%8E%E4%B8%80%E4%B8%AA%E7%9F%A9%E9%98%B5%E4%B8%8E%E5%A4%9A%E4%B8%AA%E7%9F%A9%E9%98%B5%E7%9B%B8%E4%B9%98%E7%9A%84%E9%97%AE%E9%A2%98/

Specifically, I want to do matmul(A,B) where

 ’A’ has shape (m,n)

 ’B’ has shape (k,n,p)

and the result should have shape (k,m,p)

```python
np.matmul(A, B.reshape(n,k*p)).reshape(k,m,p)
```



##### add new row in for loop

https://stackoverflow.com/questions/22392497/how-to-add-a-new-row-to-an-empty-numpy-array

```python
In [210]: %%timeit
   .....: l = []
   .....: for i in xrange(1000):
   .....:     l.append([3*i+1,3*i+2,3*i+3])
   .....: l = np.asarray(l)
   .....: 
1000 loops, best of 3: 1.18 ms per loop

In [211]: %%timeit
   .....: a = np.empty((0,3), int)  # (0,3) 注意-某个维度必须为0
   .....: for i in xrange(1000):
   .....:     a = np.append(a, 3*i+np.array([[1,2,3]]), 0)
   .....: 
100 loops, best of 3: 18.5 ms per loop

In [214]: np.allclose(a, l)
Out[214]: True
```





#### generator

more: https://jeffknupp.com/blog/2013/04/07/improve-your-python-yield-and-generators-explained/



#### decorator

more: 

https://stackoverflow.com/questions/815110/is-there-a-decorator-to-simply-cache-function-return-values

http://code.activestate.com/recipes/363602/

https://blog.apcelent.com/python-decorator-tutorial-with-example.html

https://github.com/GrahamDumpleton/wrapt/tree/develop/blog

详见deep_coding_notes



#### assert vs. exception

https://stackoverflow.com/questions/944592/best-practice-for-python-assert

Asserts should be used to test conditions that *should never happen*. The purpose is to crash early in the case of a corrupt program state.

Exceptions should be used for errors that can conceivably happen, and **you should almost always create your own Exception classes**.



#### access global variable inside class

https://stackoverflow.com/questions/10814452/how-can-i-access-global-variable-inside-class-in-python

```python
g_c = 0

class TestClass():
    def run(self):
        # declaring it global inside the function that accesses it
        global g_c
        for i in range(10):
            g_c = 1
            print g_c
```



#### make a subclass from a superclass

https://stackoverflow.com/questions/1607612/python-how-do-i-make-a-subclass-from-a-superclass

```python
class SuperHero(object): #superclass, inherits from default object
    def getName(self):
        raise NotImplementedError #you want to override this on the child classes

class SuperMan(SuperHero): #subclass, inherits from SuperHero
    def getName(self):
        return "Clark Kent"

class SuperManII(SuperHero): #another subclass
    def getName(self):
       return "Clark Kent, Jr."

if __name__ == "__main__":
    sm = SuperMan()
    print(sm.getName())
    sm2 = SuperManII()
    print(sm2.getName())
```



#### `B.__init__(self)`/`super().__init__()` 

https://stackoverflow.com/questions/21639788/difference-between-super-and-calling-superclass-directly

more: https://stackoverflow.com/questions/5033903/python-super-method-and-calling-alternatives

For **single inheritance**, `super()` is just a fancier way to refer to the base type. That way, you make the code more maintainable, for example in case you want to change the base type’s name. When you are using `super` everywhere, you just need to change it in the `class` line.

The real benefit comes with **multiple inheritance** though. When using `super`, a single call will not only automatically call the method of *all* base types (in the correct inheritance order), but it will also make sure that each method is only called once.

```python
class A (object):
    def __init__ (self):
        super().__init__()
        print('A')

class B (A):
    def __init__ (self):
        super().__init__()
        print('B')

class C (A):
    def __init__ (self):
        super().__init__()
        print('C')

class D (C, B):
    def __init__ (self):
        super().__init__()
        print('D')
        
>>> D()
A
B
C
D
<__main__.D object at 0x000000000371DD30>
```

```python
class A2 (object):
    def __init__ (self):
        print('A2')

class B2 (A2):
    def __init__ (self):
        A2.__init__(self)
        print('B2')

class C2 (A2):
    def __init__ (self):
        A2.__init__(self)
        print('C2')

class D2 (C2, B2):
    def __init__ (self):
        B2.__init__(self)
        C2.__init__(self)
        print('D2')
        
>>> D2()
A2
B2
A2
C2
D2
<__main__.D2 object at 0x0000000003734E48>
```



#### Why need the `@staticmethod`?

The reason to use `staticmethod` is if you have something that could be written as a standalone function (not part of any class), but you want to keep it within the class because it's somehow semantically related to the class. (For instance, it could be a function that doesn't require any information from the class, but whose behavior is specific to the class, so that subclasses might want to override it.) In many cases, it could make just as much sense to write something as a standalone function instead of a staticmethod.

You should choose whether or not to use a staticmethod based on the function's conceptual relation with a class (or lack thereof).

https://stackoverflow.com/questions/23508248/why-do-we-use-staticmethod

`@staticmethod` might help organize your code by being overridable by subclasses. Without it you'd have variants of the function floating around in the module namespace.

[what-is-the-difference-between-staticmethod-and-classmethod](https://stackoverflow.com/questions/136097/what-is-the-difference-between-staticmethod-and-classmethod) 

#### `@classmethod`/`@staticmethod`

summary:

1. 应用场景：1）重写子类继承的方法；2）...
2. `@staticmethod` 与类有关的函数，但不需要用类里面的方法 
3. `@classmethod` 需要用类里面的方法



more refer: [what-is-the-difference-between-staticmethod-and-classmethod](https://stackoverflow.com/questions/136097/what-is-the-difference-between-staticmethod-and-classmethod) :star::star::star::star::star:



---

The distinction between `"self"` and `"cls"` is defined in [`PEP 8`](http://www.python.org/dev/peps/pep-0008/#function-and-method-arguments) . As Adrien said, this is not a mandatory. It's a coding style. `PEP 8` says:

> *Function and method arguments*:
>
> Always use `self` for the first argument to instance methods.
>
> Always use `cls` for the first argument to class methods.



共同点：

一般来说，要使用某个类的方法，需要先实例化一个对象再调用方法。

@staticmethod、@classmethod，不需要实例化，直接类名.方法名()来调用

不同点：

@staticmethod不需要表示自身对象的self和自身类的cls参数，就跟使用函数一样。
@classmethod也不需要self参数，但第一个参数需要是表示自身类的cls参数。
如果在@staticmethod中要调用到这个类的一些属性方法，只能直接类名.属性名或类名.方法名。
而@classmethod因为持有cls参数，可以来调用类的属性，类的方法，实例化对象等。

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
class A(object):
    # 普通成员函数
    def foo(self, x):
        print "executing foo(%s, %s)" % (self, x)
    @classmethod   # 使用classmethod进行装饰
    def class_foo(cls, x):
        print "executing class_foo(%s, %s)" % (cls, x)
    @staticmethod  # 使用staticmethod进行装饰
    def static_foo(x):
        print "executing static_foo(%s)" % x
def test_three_method():
    obj = A()
    # 直接调用普通的成员方法
    obj.foo("para")  # 此处obj对象作为成员函数的隐式参数, 就是self
    obj.class_foo("para")  # 此处类作为隐式参数被传入, 就是cls
    A.class_foo("para")  #更直接的类方法调用
    obj.static_foo("para")  # 静态方法并没有任何隐式参数, 但是要通过对象或者类进行调用
    A.static_foo("para")
if __name__ == '__main__':
    test_three_method()
    
# 函数输出
executing foo(<__main__.A object at 0x100ba4e10>, para)
executing class_foo(<class '__main__.A'>, para)
executing class_foo(<class '__main__.A'>, para)
executing static_foo(para)
executing static_foo(para)
```

https://stackoverflow.com/questions/4613000/what-is-the-cls-variable-used-for-in-python-classes



#### find-incremental-numbered-sequences

https://stackoverflow.com/questions/16315189/python-find-incremental-numbered-sequences-with-a-list-comprehension

```py
>>> from itertools import groupby, count
>>> nums = [1, 2, 3, 4, 8, 10, 11, 12, 17]
>>> [list(g) for k, g in groupby(nums, key=lambda n, c=count(): n - next(c))]
[[1, 2, 3, 4], [8], [10, 11, 12], [17]]
```

`c` is a counter, so it gives each element in the list an index (`0`, `1`, etc.) then groups values on the difference between their index and their actual value. `[1, 2, 3, 4]` all differ from their index by `1`, `[8]` differs from it's index by 4, etc.



#### sorted

https://www.cnblogs.com/sysu-blackbear/p/3283993.html

https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value?rq=1

https://stackoverflow.com/questions/72899/how-do-i-sort-a-list-of-dictionaries-by-a-value-of-the-dictionary?rq=1



For list:

```py
ls=[{'name':'Homer', 'age':39}, {'name':'Bart', 'age':10}]
sorted(ls, key=lambda x:x['name'])  # x，即为字典
>>[{'name': 'Bart', 'age': 10}, {'name': 'Homer', 'age': 39}]
from operator import itemgetter
newlist = sorted(ls, key=itemgetter('name'))  # item，即为字典
>>[{'name': 'Bart', 'age': 10}, {'name': 'Homer', 'age': 39}]
```

For dict:

It is not possible to sort a dictionary, only to get a representation of a dictionary that is sorted. Dictionaries are inherently orderless, but other types, such as lists and tuples, are not. So you need an ordered data type to represent sorted values, which will be a list—probably a list of tuples.

```py
import operator
x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
sorted_x = sorted(x.items(), key=operator.itemgetter(0))  # sort by key - item,即为元组
sorted_x = sorted(x.items(), key=lambda kv: kv[0])  # python3
>>[(0, 0), (1, 2), (2, 1), (3, 4), (4, 3)]
sorted_x = sorted(x.items(), key=operator.itemgetter(1))  # sort by value - item,即为元组
sorted_x = sorted(x.items(), key=lambda kv: kv[1])  # python3
>>[(0, 0), (2, 1), (1, 2), (4, 3), (3, 4)]
```

`sorted_x` will be a list of tuples sorted by the second element in each tuple. `dict(sorted_x) == x`.

If you want the output as a dict, you can use [`collections.OrderedDict`](https://docs.python.org/3/library/collections.html#collections.OrderedDict):

```py
import collections
sorted_dict = OrderedDict(sorted_x)
```



多级排序：

```python
students = [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]
sorted(students, key=itemgetter(1,2))  # sort by grade then by age  
>>[('john', 'A', 15), ('dave', 'B', 10), ('jane', 'B', 12)] 
sorted(students, key=lambda x: (x[1], -x[2]))  # sort by grade then by age 
>> [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]
```



### Exception

#### [try:except:finally](https://stackoverflow.com/questions/7777456/python-tryexceptfinally)



#### [pass a variable to an exception when raised and retrieve it when excepted?](https://stackoverflow.com/questions/6626816/how-to-pass-a-variable-to-an-exception-when-raised-and-retrieve-it-when-excepted)

Give its constructor an argument, store that as an attribute, then retrieve it in the `except` clause:

```
class FooException(Exception):
    def __init__(self, *args):
        self.args = args
        
try:
    raise FooException("Foo!")
except FooException as e:
    print e.args
    
# or

class FooException(Exception):
    def __init__(self, foo):
        self.foo = foo

try:
    raise FooException("Foo!")
except FooException as e:
    print e.foo
```



#### [Catch multiple exceptions in one line (except block)](https://stackoverflow.com/questions/6470428/catch-multiple-exceptions-in-one-line-except-block)

From [Python Documentation](https://docs.python.org/3/tutorial/errors.html#handling-exceptions):

> An except clause may name multiple exceptions as a parenthesized tuple, for example

```
except (IDontLikeYouException, YouAreBeingMeanException) as e:
    print(e.args)
```





### Installation

#### [using pip according to the requirements.txt file from a local directory?](https://stackoverflow.com/questions/7225900/how-to-install-packages-using-pip-according-to-the-requirements-txt-file-from-a)

```
$ pip install -r requirements.txt --no-index --find-links file:///tmp/packages
```

`--no-index` - Ignore package index (only looking at `--find-links` URLs instead).

`-f, --find-links <URL>` - If a URL or path to an html file, then parse for links to archives. If a local path or `file://` URL that's a directory, then look for archives in the directory listing.

### Database

#### Oracle

本地连接Oracle服务器数据库

https://blog.csdn.net/guimaxingmc/article/details/80360840

乱码问题：

```python
# 正式执行查询前添加：
import os
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.AL32UTF8'
```














more refer:

https://docs.python-guide.org/writing/gotchas/

[Python奇技淫巧](http://andrewliu.in/2015/11/14/Python%E5%A5%87%E6%8A%80%E6%B7%AB%E5%B7%A7/) 







#### @classmethod/@staticmethod

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


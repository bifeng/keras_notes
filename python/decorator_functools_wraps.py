'''
functools wraps decorator - 应用场景：
caching information inside a class based decorator
https://stackoverflow.com/questions/6394511/python-functools-wraps-equivalent-for-classes/6406392

...
'''


def decorator(func):
    """decorator docstring"""

    def inner_function(*args, **kwargs):
        """inner function docstring """
        print(func.__name__ + " was called")
        return func(*args, **kwargs)

    return inner_function


@decorator
def foobar(x):
    """foobar docstring"""
    return x ** 2


# foobar.__name__
# 'inner_function'
# foobar.__doc__
# 'inner function docstring '

'''
The above observation leads us to conclude that the function foobar is being replaced by inner_function. 
This means that we are losing information about the function which is being passed. 

functools.wraps comes to our rescue. 
It takes the function used in the decorator and adds the functionality of copying over the function name, 
docstring, arguemnets etc.
'''
from functools import wraps


def wrapped_decorator(func):
    """wrapped decorator docstring"""

    @wraps(func)
    def inner_function(*args, **kwargs):
        """inner function docstring """
        print(func.__name__ + "was called")
        return func(*args, **kwargs)

    return inner_function


@wrapped_decorator
def foobar(x):
    """foobar docstring"""
    return x ** 2


# foobar.__name__
# 'foobar'
# foobar.__doc__
# 'foobar docstring'

'''
Application: Cached!
'''
import functools


class memoized(object):
    """Decorator that caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned, and
    not re-evaluated.
    """

    def __init__(self, func):
        self.func = func
        self.cache = {}
        functools.update_wrapper(self, func)  ## TA-DA! ##

    def __call__(self, *args):
        pass  # Not needed for this demo.


@memoized
def fibonacci(n):
    """fibonacci docstring"""
    pass  # Not needed for this demo.

# fibonacci
# <__main__.memoized object at 0x0000016F9B3861D0>
# fibonacci.__name__
# 'fibonacci'
# fibonacci.__doc__
# 'fibonacci docstring'

def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class Model(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    @lazy_property
    def calculate(self):
        print('Actually calculating value')
        return self.x * self.y

model = Model(x=3,y=4)
model.calculate
model._cache_calculate

# model.calculate
# Actually calculating value
# 12
# model._cache_calculate
# 12
# model.calculate
# 12

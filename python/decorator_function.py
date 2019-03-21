'''
Python decorator are the function (timetest) that receive a function (input_func) as an argument and
return another function (timed) as return value.
The assumption for a decorator is that we will pass a function (foobar) as argument and the signature of the inner function in
the decorator must match the function to decorate.(Inside decorator, function foobar is referenced as variable input_func. )

'''
'''
function decorator - 应用场景：

'''

import time


def timetest(input_func):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = input_func(*args, **kwargs)
        end_time = time.time()
        print("Method Name - {0}, Args - {1}, Kwargs - {2}, Execution Time - {3}".format(input_func.__name__, args,
                                                                                         kwargs, end_time - start_time))
        return result

    return timed


@timetest
def foobar(*args, **kwargs):
    time.sleep(0.3)
    print("inside foobar")
    print(args, kwargs)


foobar(["hello, world"], foo=2, bar=5)
# inside foobar
# (['hello, world'],) {'foo': 2, 'bar': 5}
# Method Name - foobar, Args - (['hello, world'],), Kwargs - {'foo': 2, 'bar': 5}, Execution Time - 0.3002750873565674

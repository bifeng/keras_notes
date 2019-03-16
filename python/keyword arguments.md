#### Don't use mutable default arguments

Don't use mutable default arguments， except you want specifically “exploit” (read: use as intended) this behavior to maintain state between calls of a function. This is often done when writing a caching function.

https://docs.python-guide.org/writing/gotchas/#mutable-default-arguments

```python
def append_to(element, to=[]):
    to.append(element)
    return to

my_list = append_to(12)
print(my_list)

my_other_list = append_to(42)
print(my_other_list)

[12]
[12, 42]
```

If you use a mutable default argument and mutate it, you *will* and have mutated that object for all future calls to the function as well.

[`None`](https://docs.python.org/3/library/constants.html#None) is often a good choice !

```python
def append_to(element, to=None):
    if to is None:
        to = []
    to.append(element)
    return to
```



  
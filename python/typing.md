#### add default parameter value with type hint

https://stackoverflow.com/questions/38727520/adding-default-parameter-value-with-type-hint-in-python

```python
def foo(opts: dict = {}):
    pass

print(foo.__annotations__)
{'opts': <class 'dict'>}
```

[PEP-3107 The syntax section](https://www.python.org/dev/peps/pep-3107/#syntax) makes it clear that keyword arguments works with function annotations in this way.



#### multiple parameter type or return type with type hint

https://docs.python.org/3/library/typing.html#typing.Union

> class `typing.Union`
>
> Union type; **Union[X, Y] means either X or Y.**

```python
from typing import Union

def foo(client_id: str) -> Union[list,bool]
```














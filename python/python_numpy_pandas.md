#### Numpy

##### replace with np.nan

`np.nan` has type `float`: arrays containing it must also have this datatype (or the `complex` or `object` datatype) so you may need to cast `arr` before you try to assign this value.

```python
>>> arr = arr.astype('float')
>>> arr[arr == 0] = np.nan
```

<https://stackoverflow.com/questions/27778299/replace-the-zeros-in-a-numpy-integer-array-with-nan?noredirect=1>

##### ndindex

<https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndindex.html>

```
>>> for index in np.ndindex(3, 2, 1):
...     print(index)
(0, 0, 0)
(0, 1, 0)
(1, 0, 0)
(1, 1, 0)
(2, 0, 0)
(2, 1, 0)
```



##### Boolean or “mask” index arrays

https://docs.scipy.org/doc/numpy/user/basics.indexing.html

NumPy之四：高级索引和索引技巧 [site](https://blog.csdn.net/wangwenzhi276/article/details/53436694#2-%E4%BD%BF%E7%94%A8%E5%B8%83%E5%B0%94%E5%80%BC%E6%95%B0%E7%BB%84%E8%BF%9B%E8%A1%8C%E7%B4%A2%E5%BC%95) 

```python
import numpy as np
a=np.arange(0,12).reshape(3,4)
b1=[False,True,True]
b2=[False,True,False,True]
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> a[b1,b2]
array([ 5, 11])

>>> a[np.nonzero(b1),np.nonzero(b2)]  # identical 第1行第1列，第2行第3列
array([[ 5, 11]])
>>> np.nonzero(b1)  # 行
(array([1, 2], dtype=int64),)
>>> np.nonzero(b2)  # 列
(array([1, 3], dtype=int64),)

>>> b3=np.ix_(b1,b2)
>>> b3
(array([[1],
       [2]], dtype=int64), array([[1, 3]], dtype=int64))
>>> a[b3]
array([[ 5,  7],
       [ 9, 11]])
```



##### reduce/add dimension

np.squeeze()

```
a.squeeze()
np.squeeze(a)
```



np.newaxis

```
a[:,np.newaxis]
a[np.newaxis,:]
```



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

##### random

more: https://blog.csdn.net/u012149181/article/details/78913167

numpy.random.rand(d0,d1,…,dn)

- rand函数根据给定维度生成[0,1)之间的数据，包含0，不包含1
- dn表示每个维度

numpy.random.randn(d0,d1,…,dn)

- randn函数返回一个或一组样本，具有标准正态分布。
- dn表示每个维度

...



#### Pandas

##### types introspection

<https://pandas.pydata.org/pandas-docs/stable/reference/general_utility_functions.html>

```python
import pandas.api.types as ptypes
```





##### compare dataframe and get the difference

<https://stackoverflow.com/questions/20225110/comparing-two-dataframes-and-getting-the-differences>

If I got you right, you want not to find changes, but symmetric difference. For that, one approach might be concatenate dataframes:

```
>>> df = pd.concat([df1, df2])
>>> df = df.reset_index(drop=True)
```

group by

```
>>> df_gpby = df.groupby(list(df.columns))
```

get index of unique records

```
>>> idx = [x[0] for x in df_gpby.groups.values() if len(x) == 1]
```

filter

```
>>> df.reindex(idx)
         Date   Fruit   Num   Color
9  2013-11-25  Orange   8.6  Orange
8  2013-11-25   Apple  22.1     Red
```


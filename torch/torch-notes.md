##### bugs

###### PyTorch(总)——PyTorch遇到令人迷人的BUG与记录

https://blog.csdn.net/u011276025/article/details/73826562



###### THCudaCheck FAIL file=/pytorch/aten/src/THC/THCGeneral.cpp line=405 error=11 : invalid argument

<https://github.com/pytorch/pytorch/issues/15797>







##### operation



###### one-hot for label



```python
>>>class_num = 10
>>>batch_size = 4
>>>label = torch.LongTensor(batch_size, 1).random_() % class_num
 3
 0
 0
 8

>>>one_hot = torch.zeros(batch_size, class_num).scatter_(1, label, 1)
    0     0     0     1     0     0     0     0     0     0
    1     0     0     0     0     0     0     0     0     0
    1     0     0     0     0     0     0     0     0     0
    0     0     0     0     0     0     0     0     1     0

```





##### function

###### .item



###### .tolist



###### torch.view

类似numpy的resize()



###### .permute

将tensor的维度换位

```python
import torch
import numpy    as np

a=np.array([[[1,2,3],[4,5,6]]])

unpermuted=torch.tensor(a)
print(unpermuted.size())  #  ——>  torch.Size([1, 2, 3])

permuted=unpermuted.permute(2,0,1)
print(permuted.size())     #  ——>  torch.Size([3, 1, 2])
```



###### torch.max

torch.max()[0]， 只返回最大值的每个数

troch.max()[1]， 只返回最大值的每个索引

torch.max()[1].data 只返回variable中的数据部分（去掉Variable containing:）

torch.max()[1].data.numpy() 把数据转化成numpy ndarry

torch.max()[1].data.numpy().squeeze() 把数据条目中维度为1 的删除掉

torch.max(tensor1,tensor2) element-wise 比较tensor1 和tensor2 中的元素，返回较大的那个值

##### install

安装包来源：
1)pypi 下载，带有manylinux标志

2)官网 下载，带有linux标志，且针对不同cuda版本也有编译

https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl  -> cuda 10.0

https://download.pytorch.org/whl/cu90/torch-1.0.0-cp36-cp36m-linux_x86_64.whl  -> cuda 9.0


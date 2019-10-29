refer: https://www.cnblogs.com/databingo/p/9339175.html

more refer:

[Keras自定义实现带masking的meanpooling层](https://blog.csdn.net/songbinxu/article/details/80148856)

[Keras实现支持masking的Flatten层](https://blog.csdn.net/songbinxu/article/details/80254122)



mask 主要对padding部分进行遮蔽，避免其参与计算。

keras是用一个`mask`矩阵来参与到计算当中, 决定在计算中屏蔽哪些位置的值. 因此`mask`矩阵其中的值就是`True/False`, 其形状一般与对应的`Tensor`相同. 同样与`Tensor`相同的是, `mask`矩阵也会在每层`Layer`被处理, 得到传入到下一层的`mask`情况.

当前的应用场景：

1. 初始化`Embedding`层，指定参数`mask_zero`为`True`，即屏蔽padding的0值

   `Embedding`的`compute_mask`方法中, 会计算得到`mask`矩阵. 虽然在`Embedding`层中不会使用这个`mask`矩阵, 即0值还是会根据其对应的向量进行查找, 但是这个`mask`矩阵会被传入到下一层中, 如果下一层, 或之后的层会对`mask`进行考虑, 那就会起到对应的作用.

2. 引用`keras.layers`包中`Masking`层, 使用`mask_value`指定固定的值被屏蔽. 在调用`call`方法时, 就会输出屏蔽后的结果.

   `Masking`的`compute_mask`方法:

   ```python
   def compute_mask(self, inputs, mask=None):
       output_mask = K.any(K.not_equal(inputs, self.mask_value), axis=-1)
       return output_mask
   ```

   这一层输出的`mask`矩阵, 是根据这层的输入得到的, 具体的说是输入的第一个维度, 这是因为最后一个维度被`K.any(axis=-1)`给去掉了. 在使用时需要注意这种操作的意义以及维度的变化.

3. 自定义`mask`方法

   首先, 如果我们希望自定义的这个层支持`mask`操作, 就需要在`__init__`方法中指定:

   ```
   self.supports_masking = True
   ```

   如果在本层计算中需要使用到`mask`, 则`call`方法需要多传入一个`mask`参数, 即:

   ```
   def call(self, inputs, mask=None):
       pass
   ```

   然后, 如果还要继续输出mask, 供之后的层使用, 如果不对`mask`矩阵进行变换, 这不用进行任何操作, 否则就需要实现`compute_mask`函数:

   ```
   def compute_mask(self, inputs, mask=None):
       pass
   ```

   这里的`inputs`就是输入的`Tensor`, 与`call`方法中接收到的一样, `mask`就是上层传入的`mask`矩阵.

   如果希望`mask`到此为止, 之后的层不再使用, 则该函数直接返回`None`即可:

   ```
   def compute_mask(self, inputs, mask=None):
       return None
   ```



### Question



 
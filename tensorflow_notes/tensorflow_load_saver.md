在不定义模型的情况下，直接加载出模型结构和模型参数值。

```python
# 加载 结构，即 模型参数 变量等
new_saver = tf.train.import_meta_graph("model_v/model.ckpt.meta")
print "ModelV construct"
all_vars = tf.trainable_variables()
for v in all_vars:
    print v.name
    #print v.name,v.eval(self.sess) # v 都还未初始化，不能求值
# 加载模型 参数变量 的 值
new_saver.restore(self.sess, tf.train.latest_checkpoint('model_v/'))
print "ModelV restored."
all_vars = tf.trainable_variables()
for v in all_vars:
    print v.name,v.eval(self.sess)
```


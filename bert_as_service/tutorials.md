How to get probability from a finetuned classify model #213
https://github.com/hanxiao/bert-as-service/issues/213

1. bert-as-service 仅对句子进行encoding.
   It is a feature extraction service, no prediction is provided. To do prediction, you need to build your own downstream network using this service.
   所以，如果task是sentence pairs，需要对两个句子的type_ids进行encoding...

2. ##### **Q:** Could I use other pooling techniques?

   **A:** For sure. But if you introduce new `tf.variables` to the graph, then you need to train those variables before using the model. You may also want to check [some pooling techniques I mentioned in my blog post](https://hanxiao.github.io/2018/06/24/4-Encoding-Blocks-You-Need-to-Know-Besides-LSTM-RNN-in-Tensorflow/#pooling-block).

3. 

**论文入选要求：<br>1. 理论 - basic<br>2. 技巧 - feature/tricks<br>**



### tricks

+ <https://github.com/Conchylicultor/Deep-Learning-Tricks>

+ <https://zhpmatrix.github.io/2019/06/30/model-debug-tips/>

  Troubleshooting Deep Neural Networks-A Field Guide to Fixing Your Model

  1. [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)
  2. 《Troubleshooting Deep Neural Networks》
  3. [《writing code for nlp research》](https://github.com/allenai/writing-code-for-nlp-research-emnlp2018/blob/master/writing_code_for_nlp_research.pdf) allennlp

  

+ Writing code for NLP Research, Tutorial at EMNLP 2018, AllenNLP, [site](https://github.com/allenai/writing-code-for-nlp-research-emnlp2018) 

+ Must Know Tips/Tricks in Deep Neural Networks, [Xiu-Shen Wei](http://lamda.nju.edu.cn/weixs/) [site](http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html) 

+ [A Few Useful Things to Know about Machine Learning](http://www.cs.washington.edu/homes/pedrod/papers/cacm12.pdf). Communications of the ACM, 55 (10), 78-87, 2012.

  <https://towardsml.com/2019/04/09/12-key-lessons-from-ml-researchers-and-practitioners/>

  

  

+ [文本分类](https://www.zhihu.com/question/265357659) 
+ [神经网络中，设计loss function有哪些技巧?](https://www.zhihu.com/question/268105631)



+ TensorFlow tricks [site](https://www.zhihu.com/question/268375146) 
+ Pytorch tricks [site](https://www.zhihu.com/question/274635237)
+ 训练效率低？GPU利用率上不去？快来看看别人家的tricks吧～ [site](https://mp.weixin.qq.com/s/zpEVU1E5DfElAnFqHCqHOw) 

+ TFSEQ 系列 (tensorflow seq2seq) [site](https://zhuanlan.zhihu.com/p/50071442) 
+ 

### blogs

+ 37 Reasons why your Neural Network is not working [site](<https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607?gi=cd8c336b3c3b>) done



### book

+ MACHINE LEARNING YEARNING - Andrew Ng - continue
  http://www.mlyearning.org/  

  https://www.deeplearning.ai/machine-learning-yearning/

  https://github.com/ajaymache/machine-learning-yearning

  https://github.com/AcceptedDoge/machine-learning-yearning-cn

+ 

### course

[Practical Deep Learning for coders](http://course.fast.ai/)



### paper

- [ ] FRAGE: Frequency-Agnostic Word Representation, Chengyue Gong, Di He, Xu Tan, Tao Qin, Liwei Wang, Tie-Yan Liu, NIPS 2018 [arxiv](https://arxiv.org/abs/1809.06858) 
- [x] A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification, Ye Zhang, Byron Wallace, 2016 [arxiv](https://arxiv.org/abs/1510.03820) 
- [ ] Bag of Tricks for Image Classification with Convolutional Neural Networks, Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, Mu Li, 2018, [arxiv](<https://arxiv.org/abs/1812.01187>) | [code](<https://github.com/weiaicunzai/Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks>) 



### Basic Step

1. Start with a simple model that is known to work for this type of data (for example, VGG for images). Use a standard loss if possible.

2. Turn off all bells and whistles, e.g. regularization and data augmentation.

3. If finetuning a model, double check the preprocessing, for it should be the same as the original model’s training.

   一般像word2vec、glove、fasttext这些官方release的预训练词向量都会公布相应训练语料的信息，包括预处理策略如分词等，这种情况真是再好不过了，不用纠结，如果你决定了使用某一份词向量，那么直接使用训练该词向量所使用的分词器叭！此分词器在下游任务的表现十之八九会比其他花里胡哨的分词器好用。

4. Verify that the input data is correct.

5. Start with a really small dataset (2–20 samples). Overfit on it and gradually add more data.

6. Check the loss at start.

   *Initialize with small parameters, without regularization. For example, if we have 10 classes, at chance means we will get the correct class 10% of the time, and the Softmax loss is the negative log probability of the correct class so: -ln(0.1) = 2.302.*

   After this, try increasing the regularization strength which should increase the loss.

7. Start gradually adding back all the pieces that were omitted: augmentation/regularization, custom loss functions, try more complex models.



### gradient check

[DebuggingGradientChecking](http://ufldl.stanford.edu/tutorial/supervised/DebuggingGradientChecking/) [gradcheck](http://cs231n.github.io/neural-networks-3/#gradcheck) [gradient-checking](https://www.coursera.org/learn/machine-learning/lecture/Y3s6r/gradient-checking) 

https://towardsdatascience.com/why-default-cnn-are-broken-in-keras-and-how-to-fix-them-ce295e5e5f2



### weight and bias

[Deeplearning4j](https://deeplearning4j.org/visualization#usingui) points out what to expect in histograms of weights and biases:

*“For weights, these histograms should have an* **approximately Gaussian (normal)** *distribution, after some time. For biases, these histograms will generally start at 0, and will usually end up being* **approximately Gaussian** *(One exception to this is for LSTM). Keep an eye out for parameters that are diverging to +/- infinity. Keep an eye out for biases that become very large. This can sometimes occur in the output layer for classification if the distribution of classes is very imbalanced.”*



### layer update and activation

Check layer updates, they should have a Gaussian distribution.

Check layer updates, as very large values can indicate exploding gradients. Gradient clipping may help.

Check layer activations. From [Deeplearning4j](https://deeplearning4j.org/visualization#usingui) comes a great guideline: *“A good standard deviation for the activations is on the order of 0.5 to 2.0. Significantly outside of this range may indicate vanishing or exploding activations.”*



### loss

refer: 训练loss不下降原因集合 [site](https://blog.csdn.net/jacke121/article/details/79874555) 



1. train loss 与 test loss的结果分析

   train loss 不断下降，test loss不断下降，说明网络仍在学习;

   train loss 不断下降，test loss趋于不变，说明网络**过拟合**;

   train loss 不断下降，test loss不断上升，说明样本不平衡；

   ​	譬如在损失函数赋予了少类样本极大的权重。在训练过程中，分错一个少类样本带来的loss值远远超过分错大类样本时的loss值，所以即使正确识别了大部分的大类样本，只要模型分错了一个少类样本，还是会使得validation的loss不断上升。 为了能够令模型真正学习到少类样本，即使loss在上升，还是应该加大epoch，继续训练下去，loss有可能在某个节点开始会出现下降。

   train loss 趋于不变，test loss不断下降，说明数据集100%有问题;

   train loss 趋于不变，test loss趋于不变，说明学习遇到瓶颈，需要减小学习率，增大批量数目;

   train loss 不断上升，test loss不断上升，说明网络结构设计不当，训练超参数设置不当，数据集经过清洗等问题。

   

2. ...

   [深度学习与计算机视觉系列(8)_神经网络训练与注意点](https://blog.csdn.net/han_xiaoyang/article/details/50521064) 

   1.梯度检验2.训练前检查，3.训练中监控4.首层可视化5.模型融合和优化等

3. ...

   https://www.zhihu.com/question/38937343

4. ...

   https://blog.csdn.net/u010911921/article/details/71079367

5. ...

   https://www.zhihu.com/question/68603783

6. 



#### loss is constant without change or converge

检查该常数是否loss的最大值或最小值。

可能的原因：

1. softmax在计算的过程中得到了概率值出现了零，由于softmax是用指数函数计算的，指数函数的值都是大于0的，所以应该是计算过程中出现了float溢出的异常，也就是出现了inf，nan等异常值导致softmax输出为0. 当softmax之前的feature值过大时，由于softmax先求指数，会超出float的数据范围，成为inf。inf与其他任何数值的和都是inf，softmax在做除法时任何正常范围的数值除以inf都会变成0.

solution - 由于softmax输入的feature由两部分计算得到：一部分是输入数据，另一部分是各层的权值等组成.

1) 减小初始化权重，以使得softmax的输入feature处于一个比较小的范围

2) 降低学习率，这样可以减小权重的波动范围

3) 如果有BN(batch normalization)层，finetune时最好不要冻结BN的参数，否则数据分布不一致时很容易使输出值变得很大(注意将batch_norm_param中的use_global_stats设置为false )。

4) 观察数据中是否有异常样本或异常label导致数据读取异常



#### loss fluctuation 

波动剧烈

gradient检查？sgd优化器/adam优化器？



#### loss decreasing but accuracy stable

It means the model is not learning at all !

A decrease in binary cross-entropy loss does not imply an increase in accuracy. Consider label 1, predictions 0.2, 0.4 and 0.6 at timesteps 1, 2, 3 and classification threshold 0.5. timesteps 1 and 2 will produce a decrease in loss but no increase in accuracy.



<https://stackoverflow.com/questions/43499199/tensorflow-loss-decreasing-but-accuracy-stable>





### learning rate

Play around with your current learning rate by multiplying it by 0.1 or 10.



### NaNs

 [how to deal with NaNs](http://russellsstewart.com/notes/0.html)



### tensorflow

#### Optimizer

+ 和图像等领域不同，对 NLU 之类的任务，每个 batch 采样到的词有限，每次更新对 Embedding 的估计都是**梯度稀疏**的。非 momentum-based 的 Optimizer 每步只会更新采样到的词，而对于 momentum-based 的 Optimizer，现在所有框架的实现都会用当前的 momentum 去更新所有的词，即使这些词在连续的几十步更新里都没有被采样到。这可能会使 Embedding 过拟合。







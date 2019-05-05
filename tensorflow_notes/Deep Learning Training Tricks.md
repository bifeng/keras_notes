**论文入选要求：<br>1. 理论 - basic<br>2. 技巧 - feature/tricks<br>**



### tricks

+ Writing code for NLP Research, Tutorial at EMNLP 2018, AllenNLP, [site](https://github.com/allenai/writing-code-for-nlp-research-emnlp2018) 
+ Must Know Tips/Tricks in Deep Neural Networks, [Xiu-Shen Wei](http://lamda.nju.edu.cn/weixs/) [site](http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html) 
+ [文本分类](https://www.zhihu.com/question/265357659) 
+ [神经网络中，设计loss function有哪些技巧?](https://www.zhihu.com/question/268105631)
+ 





+ TensorFlow tricks [site](https://www.zhihu.com/question/268375146) 
+ Pytorch tricks [site](https://www.zhihu.com/question/274635237)
+ 训练效率低？GPU利用率上不去？快来看看别人家的tricks吧～ [site](https://mp.weixin.qq.com/s/zpEVU1E5DfElAnFqHCqHOw) 

+ TFSEQ 系列 (tensorflow seq2seq) [site](https://zhuanlan.zhihu.com/p/50071442) 
+ 



### book

+ MACHINE LEARNING YEARNING - Andrew Ng - continue
  http://www.mlyearning.org/  

  https://www.deeplearning.ai/machine-learning-yearning/

  https://github.com/ajaymache/machine-learning-yearning

  https://github.com/AcceptedDoge/machine-learning-yearning-cn

+ 

### paper

- [ ] FRAGE: Frequency-Agnostic Word Representation, Chengyue Gong, Di He, Xu Tan, Tao Qin, Liwei Wang, Tie-Yan Liu, NIPS 2018 [arxiv](https://arxiv.org/abs/1809.06858) 

- [x] A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification, Ye Zhang, Byron Wallace, 2016 [arxiv](https://arxiv.org/abs/1510.03820) 



### nlp

#### word segment

+ 一般像word2vec、glove、fasttext这些官方release的预训练词向量都会公布相应训练语料的信息，包括预处理策略如分词等，这种情况真是再好不过了，不用纠结，如果你决定了使用某一份词向量，那么直接使用训练该词向量所使用的分词器叭！此分词器在下游任务的表现十之八九会比其他花里胡哨的分词器好用。





### tensorflow

#### Optimizer

+ 和图像等领域不同，对 NLU 之类的任务，每个 batch 采样到的词有限，每次更新对 Embedding 的估计都是**梯度稀疏**的。非 momentum-based 的 Optimizer 每步只会更新采样到的词，而对于 momentum-based 的 Optimizer，现在所有框架的实现都会用当前的 momentum 去更新所有的词，即使这些词在连续的几十步更新里都没有被采样到。这可能会使 Embedding 过拟合。







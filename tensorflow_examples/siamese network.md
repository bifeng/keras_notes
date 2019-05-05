refer: https://www.slideshare.net/NicholasMcClure1/siamese-networks

code: https://github.com/nfmcclure/tensorflow_cookbook/tree/master/09_Recurrent_Neural_Networks/06_Training_A_Siamese_Similarity_Measure



motivation:

Change the classification learning problem to distance learning problem, especially for rare cases in classification, siamese networks allow us to use relationships between data points, which lead to more data points to learning.



Why it is effect?

1. share representation space

   For a shared layer network, the corresponding elements in Q1 and Q2 vectors represent in the same vector space. While for the network with separate Q1 and Q2 parameters, there is no such constraint and the model has doublesized parameters, making it difficult to learn for the optimizer.

2. ...



### training dataset

The label is depends on the similarity metric (?). Such as, if using cosine similarity, then the similar label is 1, dissimilar label is -1. 

Most studies have shown that the ratio of dissimilar to similar is optimal around: between 2:1 and 10:1. This depends on the problem and specificity of the model needed.



### backpropagation

method 1

it is standard to **average** the gradients of the two 'sides' before performing the gradient update step.



method 2





### loss

The loss function is a combination of a similar-loss and dissimilar-loss.

contrastive loss



### paper (important)

+ Signature Verification using a "Siamese" Time Delay Neural Network, Jane Bromley, Isabelle Guyon, Yann LeCun,
  Eduard Sickinger and Roopak Shah, 1993 [paper](https://papers.nips.cc/paper/769-signature-verification-using-a-siamese-time-delay-neural-network.pdf) 
+ ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs, Wenpeng Yin, Hinrich Schutze, Bing Xiang, Bowen Zhou 2016 [paper](https://aclweb.org/anthology/Q16-1019)
+ Siamese Recurrent Architectures for Learning Sentence Similarity, Jonas Mueller, Aditya Thyagarajan AAAI 2016 [paper](http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf) 










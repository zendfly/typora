# 半监督学习—semi supervised learning

使用少量的数据有标签（label），使用Unlabel的数据学习整个数据的潜在分布。



#### 方法

- label propagation(标签传播算法)-----基于图的方法
- S3VM(半监督的SVM)
- Graph(Graph convolutional Network)，图卷积神经网络-----基于图结构的广义神经网络
- Laplacian ELM(自己翻译的：Laplacian 极限学习机)
- Ladder network



#### Pseudo label

在缺乏数据的情况下，对没有标签的数据使用网络进行预测，得到的标签作为训练标签来进行训练。随之损失函数也进行更改，包括真实标签、pseudo label两部分。

**就是把网络对无标签数据的预测，作为无标签数据的标签（即 Pseudo label）**


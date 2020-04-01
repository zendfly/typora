# Faster R-CNN



## RPN(Region Proposal Networks)

![RPN](C:\Users\Fit-hj\Desktop\typora\net\Faster R-CNN.assets\RPN.jpg)

RPN的输入是基础特征提取网络生成的特征图，RPN部分分为两条线，

- 上面一条通过Softmax对anchors产生的矩形框进行positive 和 negative分类。

- 下面一条线是计算anchors的bounding box regression偏移量，以获得精准的proposal。

最后的Proposal则是综合两条线的结果，提出太小和超出边界的proposal。



### Anchors

在RPN阶段，基于来自特征提取网络的特征图进行Anchor生成。实际中的anchors是一些列（长宽比不同）的矩形框，这些矩形框是在输入特征图上的每个点进行生成。

![anchros示意图](C:\Users\Fit-hj\Desktop\anchros示意图.jpg)

```
[[ -84.  -40.   99.   55.]
 [-176.  -88.  191.  103.]
 [-360. -184.  375.  199.]
 [ -56.  -56.   71.   71.]
 [-120. -120.  135.  135.]
 [-248. -248.  263.  263.]
 [ -36.  -80.   51.   95.]
 [ -80. -168.   95.  183.]
 [-168. -344.  183.  359.]]
```

在论文中，对每个点生成9个不同尺寸的矩形框。例如，输入尺寸是**NxMx256**的特征图，每个点都是256的维度，在每个点上生成9个不同尺度的矩形，把生成的9个矩形框作为初始的检测框。用于RPN的positive anchor、negative anchor判断。

当输入特征图的尺寸为50x38时，每个点生成9个矩形框，总共：58x39x9=17100个矩形框。这个多个矩形框，可以有效的包含图像中的所有目标，但数量众多，不可能把所有的矩形框都送入后面进行检测，故，在进行positive和negative判断筛选。

![generate Anchors](C:\Users\Fit-hj\Desktop\generate Anchors.jpg)

在RPN部分，所作的工作就是在原图尺度上设置大量的anchors，然后用cnn进行positive anchor(有目标)和negative anchor(没有目标)二分类判断。



### softmax分类positive、negative

![RPN中softmax进行positive、negative](C:\Users\Fit-hj\Desktop\typora\net\Faster R-CNN.assets\RPN中softmax进行positive、negative.jpg)

对输入的特征图经过一个**3x3**的卷积后，

1. 首先进行一个1x1x18（卷积核尺寸：1x1，深度为18）的卷积，18对应着每个点有9个anchor，且有positive、negative两种分类，9x2=18。
2. 进行Reshape操作，这里reshape操作是为了方便softmax分类。更caffe的数据存储方式有关。（在object detection api中是什么样还不清楚）



### Bounding box regression（待写）



### Proposal Layer

Proposal的输入有三个：

- softmax对positive/negative anchors的分类结果，rpn_cls_prob_reshape
- bounding box regression产生的偏移量，rpn_bbox_pred
- im_info，im_info保存了缩放的所有信息。假设输入尺寸为**PxQ**，faster r-cnn会对输入图像进行缩放到MxN的大小。则im_info = [M,N,scale_factor]。

Proposal layer按照下列顺序进行处理：

1. 生成anchors，并对所有的anchors进行bbox regression回归
2. 对softmax进行positive、negative判断的scores值进行排序，提取Top-N的anchors，以及回顾ihou的positive anchors。
3. 剔除太小的positive anchors，并对超出图像边界的positive anchors进行处理（这里超出图像边界，是指映射会原图后进行判断的）。
4. 对剩余的positive anchors进行NMS



RPN网络结构，总结：

**生成anchors -> softmax分类器提取positvie anchors -> bbox reg回归positive anchors -> Proposal Layer生成proposals**



### Roi pooling

Roi是一种特殊的SPP-net结构，目的是为了将不同尺寸的输入映射为固定尺寸，用于后面的fc层。Roi过程：

- 首先对输入映射回**MXN**尺寸（即，RPN的输入）
- 将每个Proposal对应的feature map区域水平分为**WxH**的网格
- 对网格的每一份都进行max pooling处理

经过这样处理，能够将不同大小的输入映射成**WxH**大小的固定输出，实现了长度统一。

![proposal_roi](C:\Users\Fit-hj\Desktop\typora\net\Faster R-CNN.assets\proposal_roi.jpg)



## Classification

classification利用已经获得的proposal feature map，通过fc和softmax再对每个Proposal进行分类判读，输出cls_score值；同时再利用bounding box regression获得每个proposal的位置偏移量，用于得到更加精准的检测框。

Classification结构图：

![calssification](C:\Users\Fit-hj\Desktop\typora\net\Faster R-CNN.assets\calssification.jpg)

例如，ROi pooling的输出是7x7（即，对RPN的输出映射到原图区域，划分成7x7个格子进行max pooling）大小的proposal feature maps后，做了如下两个操作：

- 通过全连接层进和softmax进行分类
- 对proposal进行bounding box regression，获取更高进度的回归框

fc层示意图：
![fc层示意图](C:\Users\Fit-hj\Desktop\typora\net\Faster R-CNN.assets\fc层示意图.jpg)

计算公式：

![fc层计算公式](C:\Users\Fit-hj\Desktop\typora\net\Faster R-CNN.assets\fc层计算公式.jpg)

其中，w和bias 都是预先训练好，即大小固定。则也就是为什么要使用Roi pooling进行映射成固定大小的原因。

## Faster R-CNN训练

Faster R-CNN的训练是在已经训练好model（基础网络，如：VGG、Resnet等）的基础上进行继续训练，分为两个步骤：

- stage1：首先在已经训练好的基础model上，训练RPN网络，然后并对rpn进行测试；然后训练fast rcnn网络
- stage2：首先训练RPN网络，在此基础上收集proposal；然后在训练 fast rcnn网络

可以看出，训练流程类似一种“迭代”的过程，循环2次的迭代。

流程图：

![faster r cnn训练流程图](C:\Users\Fit-hj\Desktop\typora\net\Faster R-CNN.assets\faster r cnn训练流程图.jpg)





### RPN训练

该部分，首先读取与训练好的model开始迭代训练。结构如图：

![RPN训练流程图](C:\Users\Fit-hj\Desktop\typora\net\Faster R-CNN.assets\RPN训练流程图.jpg)

其中：

- input_data：输入图片
- Conv_layers：特征提取网络
- rpn_cls_socre、rpn_bbox_pred：分别为RPN部分的softmax进行positive、negative判断值和bounding box regression回归的偏移量

|

|

|

|（待补充）



### 通过训练好的RPN收集proposals

利用训练好的RPN网络，得到proposal rois，以及对应的positive softmax probability。结构图：

![RPN生成proposal](C:\Users\Fit-hj\Desktop\typora\net\Faster R-CNN.assets\RPN生成proposal.jpg)

## 训练 faster rcnn

将输入图像输入网络，然后：

1. 将RPN提取的scores和rois传入网络
2. 计算bbox_inside_weights+bbox_outside_weights，计算loss。



结构图：

![训练faster rcnn](C:\Users\Fit-hj\Desktop\typora\net\Faster R-CNN.assets\训练faster rcnn.jpg)


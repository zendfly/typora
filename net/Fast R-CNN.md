# Fast R-CNN



## architecture

### input(输入):

 a images and a set of object proposals.

### processes(过程):

-  the whole image with several convolutional (*conv*) and max pooling layers to produce a conv feature map. (通过一个特征提取网络生成一些列特征图)

- for each object proposal a region of interest (*RoI*) pooling layer extracts a fixed-length feature vector from the feature map.(将输入的建议区域映射到特征图中进行尺度转换（即，使用Roi），将转换后的向量输入fc层)

- Each feature vector is fed into a sequence of fully connected

  (*fc*) layers that finally branch into two sibling output layers: one that produces softmax probability estimates over *K* object classes plus a catch-all “background” class and another layer that outputs four real-valued numbers for each of the *K* object classes. Each set of 4 values encodes refined bounding-box positions for one of the *K* classes.(接收 Roi的全连接层，并行到两个分支：1.进行softmax操作（通过softmax估算一个近似值），得到k+1（1为背景）个类别的近似概率；2.对k类别输出4个值(目标框的位置)。



## Roi(region of pooling)

在卷积的最后一层使用Roi进行维度转换。Roi是Spp-net的一种特殊情况。将区域建议映射到feature map上的window(窗口)划分成一个7x7=49个网格，对每个网格进行max_pooling得到一个7x7的特征图，然后将其送入fc进行后续操作。



## multi-task loss

Fast R-CNN有两个兄妹输出层（sibling output layers）。

- a discrete probability distribution(一个离散的概率).——即，k个类别的概率。
- bounding-box regression offsets(每个类别的坐标).

Each training RoI is labeled with a ground-truth class **u** and a ground-truth bounding-box regression target ***v***. We use a multi-task loss *L* on each labeled RoI to jointly train for classification and bounding-box regression(每一个Roi都会有一个包含类别（u）和真实坐标（v）。我们使用多任务损失L来联合训练分类和回归):
$$
L(p,u,t^u,v)=L_{cls}{(p,u)} + \lambda [u \ge 1]L_{loc}(t_u,v)
$$

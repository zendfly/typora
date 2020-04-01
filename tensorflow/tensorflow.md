# tensorflow

## .ckpt

 .ckpt 二进制文件，存储weights,biases,gradients等变量

新版的保存为：

.chpt.meta

包含元图，即计算图的结构，没有变量值。变量保存在.chpt.data文件

.chpt.data

保存变量，没有结构。

.chpt.index



## .checkpoint

文本文件，记录保存最近的checkpoint文件，及其列表；可以修改这个文件，制定使用哪个Model



## .meta

保存图结构，meta文件是pb格式，包含变量、结合、op等






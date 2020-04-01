# ResNet(Residual Network)

传统网络（也称平原网络）随着层数增加性能下降。Resnet通过建立一种恒等映射使网络层的输入传到输出中，如图：

![residual](C:\Users\Fit-hj\Desktop\typora\net\ResNet(Residual Network).assets\residual.png)

图中，x为输入，H(x)为输出，通常，x输入网络后得到输出H(x)，但残差网络通过‘shortcut connections’的方法，将输入x恒等映射到出中，使得H(x) = F(x) + x，F(x)为x经过卷积层得到的输出；这样的方式，使学习目标不再是一个单独的输出，而是H(x)与x的差值，即残差F(x) = H(x) - x。





## residual V1&V2

![residual_v1_v2](C:\Users\Fit-hj\Desktop\typora\net\ResNet(Residual Network).assets\residual_v1_v2.png)
# Graph、GraphDef

## Graph

tf使用图（graph）来表示计算任务，使用seesion（会话）的上下文管理器执行图。Graph中包含了各种op（操作）和用于计算的tensor，在计算之前需要对Graph进行定义，即定义Graph中的op，这是构件图的第一步。官方定义：一些operation和tensor的集合。

我们使用python代码来表达Graph中op和tensor的关系。但在实际中，python表达的Graph在运行后（Session）不是一沉不变的，因为tf在运行中，真实的计算会被分配的到多cpu或者GPU等异构设备中。实际，tf首先将python表达的图片转换成protocol buffer，再通过c/c++/cuda运行protocol buffer。



## GraphDef

GraphDef是Graph的序列化表示，使用python代码对Graph进行描述表示，会得到一个序列化的图，叫GraphDef。




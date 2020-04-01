# tensorboard



tensorboard通过读取tensorflow的事件文件来运行。事件文件包括了你会在tensorflow运行中涉及到的主要数据（具体还未知）。

tensorboard通过运行所有节点，并附加相关操作来生成所有信息。例如：tensorboard通过向节点使用`scalar_summary`操作来输出学习速度和期望误差。使用`histogram_summary`运算来收集权重变量和梯度输出。

Tf 中，所有操作只有当你执行，或者另一个操作依赖于它的输出时才会运行。我们附加节点都围绕着原始节点：没有任何操作依赖与它（即，只能运行所有节点才能运行我们附加的节点，得到想要的信息）。但这样很乏味，tensorboard使用`tf.merge_all_summaries`将他们合并为一个操作（合并命令）。

使用`tf.merge_all_summaries`操作后，会将所有数据生成一个序列化的summay protobuf对象。并将汇总的protobuf传递给`tf.train.Summarywrite`用于写入磁盘。

**Summarywrite**函数包含了logdir参数，即写入路径。

**tensorboard启动**

```cmd
tensorboard --logdir=<path>
```

在浏览器输出`localhost:6006`来查看tensorboard



## Name scoping(名称域) 和Node(节点)

tensorflow节点数量巨大，为了准确清楚的对其可视化，对变量名划定范围，在可视化中，默认情况下只显示顶层节点。例如：使用tf.name_scope在hidden命名域定义了三个操作：

```python
import tensorflow as tf

with tf.name_scope('hidden') as scope:
	a = tf.constant(5,name='alpha')
	w = tf.Variable(tf.random_uniform([1,2],-1.0,1.0),name='weights')
	b = tf.Variable(tf.zeros([1]), name='biases')
```

进行附加操作后，会得到三个操作名：

```python
hidden/alpha
hidden/weights
hidden/biases
```

在可视化时，默认的会将三个节点进行折叠成一个节点并标注为hidden，但其中的细节没有丢失，只是隐藏了而已。在tensorboard中，名称域设计越好，可视化效果就越好。

tensorflow图标有：数据依赖和控制依赖两种链接关系，在tensorboard中，使用实心箭头表示数据依赖，使用点线表示控制依赖。
# Object Detection Api (ODa)

Google推出的目标识别系统，可以将训练后的模型直接加载使用。感觉是一个平台，将自己训练的模型放上去，可以部署到移动或者pc端。地址：https://github.com/tensorflow/models

ODa提供5个默认预训练模型（可能随着跟新，回加入其它模型）：

-  带有MobileNets的SSD (Single Shot Multibox Detector)

-  带有Iception V2的SSD
-  带有Resnet 101的R-FCN (Region-Based Fully Convolutional Networks)
-  带有Resnet 101的Faster RCNN
-  带有Inception-Resenet v2的Faster RCNN

ODa使用TFRcord格式数据，在训练之前需要将数据进行转换。



## 安装

在resarch路径下，需要编译protoch ，下载地址：https://github.com/protocolbuffers/protobuf/releases/tag/v3.4.0，将bin下的protoch.exe拷贝到research路径下，执行下面命令

```python
protoc object_detection/protos/*.proto --python_out=.
```

执行完后，在object_detection/protos文件下，出现很多.py文件。

然后：

```python
python setup.py build
python setup.py install
```



## g3doc

相关说明文档，建议重点看





## train.py

在models/research/object_detection/**legacy**下的train.py中。

### train.py

flags输入（还未看）

```python
flags = tf.app.flags
flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')
flags.DEFINE_integer('task', 0, 'task id')
flags.DEFINE_integer('num_clones', 1, 'Number of clones to deploy per worker.')
flags.DEFINE_boolean('clone_on_cpu', False,
                     'Force clones to be deployed on CPU.  Note that even if '
                     'set to False (allowing ops to run on gpu), some ops may '
                     'still be run on the CPU if they have no GPU kernel.')
flags.DEFINE_integer('worker_replicas', 1, 'Number of worker+trainer '
                     'replicas.')
flags.DEFINE_integer('ps_tasks', 0,
                     'Number of parameter server tasks. If None, does not use '
                     'a parameter server.')
flags.DEFINE_string('train_dir', '',
                    'Directory to save the checkpoints and training summaries.')

flags.DEFINE_string('pipeline_config_path', '',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')

flags.DEFINE_string('train_config_path', '',
                    'Path to a train_pb2.TrainConfig config file.')
flags.DEFINE_string('input_config_path', '',
                    'Path to an input_reader_pb2.InputReader config file.')
flags.DEFINE_string('model_config_path', '',
                    'Path to a model_pb2.DetectionModel config file.')

FLAGS = flags.FLAGS
```

main()函数

- 解析flags函数（flags函数，解析cmd输入命令）得到模型的configs

  ```python
    assert FLAGS.train_dir, '`train_dir` is missing.'
    if FLAGS.task == 0: tf.gfile.MakeDirs(FLAGS.train_dir)
    if FLAGS.pipeline_config_path:        #如果指定训练配置文件
      configs = config_util.get_configs_from_pipeline_file(       # 读取训练配置文件
          FLAGS.pipeline_config_path)
      if FLAGS.task == 0:
        tf.gfile.Copy(FLAGS.pipeline_config_path,
                      os.path.join(FLAGS.train_dir, 'pipeline.config'),
                      overwrite=True)
    else:         # 没有指定训练配置文件
      configs = config_util.get_configs_from_multiple_files(      # 读取训练配置文件
          model_config_path=FLAGS.model_config_path,          # 模型路径
          train_config_path=FLAGS.train_config_path,          # 训练路径
          train_input_config_path=FLAGS.input_config_path)        # 输入路径
      if FLAGS.task == 0:
        for name, config in [('model.config', FLAGS.model_config_path),
                             ('train.config', FLAGS.train_config_path),
                             ('input.config', FLAGS.input_config_path)]:
          tf.gfile.Copy(config, os.path.join(FLAGS.train_dir, name),
                        overwrite=True)
  ```

  

- 根据configs进行模型构建

  ```python
    model_config = configs['model']
    train_config = configs['train_config']
    input_config = configs['train_input_config']
  
      # 根据model_config创建模型
    model_fn = functools.partial(         # model_builder.build(model_config)
        # functools.partial,返回一个可以调用的partial对象，
        # 使用方法是 partial(func,*args,*kw),func必须存在，*arg、*kw必须存在一个
        model_builder.build,
        model_config=model_config,
        is_training=True)
  
      # 获取下一步内容
    def get_next(config):
      return dataset_builder.make_initializable_iterator(
          dataset_builder.build(config)).get_next()
    # dataset_builder.make_initializable_iterator()  创建一个迭代器，
  
      # 继续调用下一步。.next()
    create_input_dict_fn = functools.partial(get_next, input_config)
  
  ```

  

- 设置tensorflow的一系列分布式参数（后半部分没细看）

  ```python
  
    """
    分布式计算相关操作
      分布式TF使用gRPC(google remote proceduce call)底层进行通信。
      Cluster job task
      job 是 task的集合
      cluster 是 job的集合
      分布式计算中，每台电脑大多数上只允许一个task(主机上的一个进程)。
      在分布式TF中，job分为Parameter 和 worker , Paramter job主要是管理参数的存储和跟新，worker job主要运行ops,
      如果参数太多，就需要多个task，故 job是task的集合
      我们使用集群（分布式），就是用多个job,故cluster是job的集合
    """
      # os.environ.get('TF_CONFIG', '{}') 获取环境变量，如果没有就返回'{}'
    env = json.loads(os.environ.get('TF_CONFIG', '{}'))       # 使用json读取获取的环境变量
    cluster_data = env.get('cluster', None)   # 获取cluster的值，如果没有就返回None
    cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None    #
    task_data = env.get('task', None) or {'type': 'master', 'index': 0}
    task_info = type('TaskSpec', (object,), task_data)
  
    # Parameters for a single worker.单一工作时的参数设置
    ps_tasks = 0
    worker_replicas = 1
    worker_job_name = 'lonely_worker'
    task = 0
    is_chief = True
    master = ''
  
    if cluster_data and 'worker' in cluster_data:
      # Number of total worker replicas include "worker"s and the "master".
      worker_replicas = len(cluster_data['worker']) + 1
    if cluster_data and 'ps' in cluster_data:
      ps_tasks = len(cluster_data['ps'])
  
    if worker_replicas > 1 and ps_tasks < 1:
      raise ValueError('At least 1 ps task is needed for distributed training.')
  
    if worker_replicas >= 1 and ps_tasks > 0:
      # Set up distributed training.
      server = tf.train.Server(tf.train.ClusterSpec(cluster), protocol='grpc',
                               job_name=task_info.type,
                               task_index=task_info.index)
      if task_info.type == 'ps':
        server.join()
        return
  
      worker_job_name = '%s/task:%d' % (task_info.type, task_info.index)
      task = task_info.index
      is_chief = (task_info.type == 'master')
      master = server.target
  
    graph_rewriter_fn = None
    if 'graph_rewriter_config' in configs:
      graph_rewriter_fn = graph_rewriter_builder.build(
          configs['graph_rewriter_config'], is_training=True)
  ```

  

- 开始训练

```python
  trainer.train(
      create_input_dict_fn,
      model_fn,
      train_config,
      master,
      task,
      FLAGS.num_clones,
      worker_replicas,
      FLAGS.clone_on_cpu,
      ps_tasks,
      worker_job_name,
      is_chief,
      FLAGS.train_dir,
      graph_hook_fn=graph_rewriter_fn)
```



开始运行

```python
if __name__ == '__main__':
  tf.app.run()
```



## protobuf

缓冲池的设置（不确定）



## samples

configs文件夹下保存一些列的.config，文件保存模型配置文件。文件中保存着model的结构，例如关键的参数配置、模型选用、图片大小、、、、等等。



## legacy 

训练文件 train.py





## ODa在win10下运行教程：

### 配置环境

1. 深度学习相关环境

   TF、Python、cuda、cudnn等

2. 相关依赖包

   conda install -c anaconda protobuf   #需要使用管理员权限使用cmd，可以使用网址进行下载，

   [https://github.com/protocolbuffers/protobuf/releases](https://links.jianshu.com/go?to=https%3A%2F%2Fgithub.com%2Fprotocolbuffers%2Fprotobuf%2Freleases)，在最新的ODa中，已经包含了protobuf文件，需要编译，（在\research目录下）编译命令：protoc object_detection/protos/*.proto --python_out=.

   conda install pillow

   conda install lxml

   conda install Cython

   conda install contestlib2

   conda install jupyter

   conda install matplotlib

   conda install pandas

   conda install opencv-python

   protobuf使用之前需要编译，

在models-master/research/object_detection/data路径下，有.pbtxt格式文件，用于保存label map

在models-master/research/object_detection/dataset_tools路径下，保存着格式转换相关文件



## 数据转换

search/object_detection/dataset_tools路径下



























### absl库

absl库全称是 Abseil Python Common Libraries。它原本是个C++库，后来被迁移到了Python上。：

- 简单的应用创建
- 分布式的命令行标志系统
- 用户自定义的记录模块，并拥有额外的功能。
- 拥有测试工具

```python
from absl import flags
# 创建Python应用，用于接收参数

FLAGS = flags.FLAGS
flags.DEFIN_string("name",None,"Your name.")
flags.DEFIN_integer('"num_times", 1,
                     "Number of times to print greeting."')
                    
                    
# 指定必须的参数
flags.mark_flag_as_required('name')
                    
def main(argv):
  del argv  # 无用
  for i in range(0, FLAGS.num_times):
    print('Hello, %s!' % FLAGS.name)
 
 
if __name__ == '__main__':
	app.run(main) 	# 和tf.app.run()类似
```

在cmd中输入：

```python
python .\absl_hello.py --name=World --num_times=10  # 这四条命令等价
python .\absl_hello.py --name World --num_times 10
python .\absl_hello.py -name World -num_times 10
python .\absl_hello.py -name=World -num_times=10

```

得到的结果一样,Hello,world！ 重复10次。

```python
Hello, World!
Hello, World!
Hello, World!
Hello, World!
Hello, World!
Hello, World!
Hello, World!
Hello, World!
Hello, World!
Hello, World!
```


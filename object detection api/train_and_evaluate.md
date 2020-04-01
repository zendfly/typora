# train_and_evaluate

TF1.14中，使用

```python
tf.estimator.(
    estimator,
    train_spec,
    eval_spec
) 
```

代替：

```python
tf.contrib.learn.Experiment
```

tf.estimator.train_and_evaluate **API**用来train然后evaluate一个Estimator。**tf.estimator.train_and_evaluate**可以保证 本地 和 分布式环境下行为的一致性，也就是说，使用 `Estimator` 和 `train_and_evaluate` 编写的程序同时支持本地、集群上的训练，而不需要修改任何代码。

参数：

- estimator:

- train_spec:接收一个`tf.estimator.TrainSpec` 实例。

  - TrainSpec的参数：

    - ```python
      __new__(   
      cls, *# 这个参数不用指定，忽略即可。*  input_fn,   
      max_steps=None,   
       hooks=None 
       )
      ```

      其中： input_fn：用来指定输入数据

      ​			max_steps：指定训练步数，这是训练的唯一终止条件

      ​			hooks：参数用来挂一些 `tf.train.SessionRunHook`，用来在 session 运行的时候做一些额外的操作，比如记录一些 TensorBoard 日志什么的。

- eval_spec:接收`tf.estimator.EvalSpec` 实例。

  ```python
  __new__(
      cls, # 这个参数不用指定，忽略即可。
      input_fn,
      steps=100, # 评估的迭代步数，如果为None，则在整个数据集上评估。
      name=None,
      hooks=None,
      exporters=None,
      start_delay_secs=120,
      throttle_secs=600
  )
  
  ```

  其中：steps:指定评估的迭代步数，None为评估整个数据集

  ​			name:如果要在多个不同的数据集上进行评估，通过name保证不同数据集上的日志保存在不同的文件夹中，以作区分

  ​			exporters:`tf.estimator.export` 模块中的类的实例。

  ​			start_delay_sevs:调用train_and_evaluate函数后，多少秒开始评估。第一次评估时间在start_delay_secs + throttle_secs 后发生。

  ​			throttle_secs:多少秒后又开始评估，如果没有有新的 checkpoints 产生，则不评估，所以这个间隔是最小值。

ruturn:

- Estimator.evaluate 
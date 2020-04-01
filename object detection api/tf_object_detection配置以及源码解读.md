 # TF object_detection 配置以及源码解读

## 环境配置以及安装

## 源代码解读

​        目标检测的源代码可以从两个角度开始进行模型训练评估，第一种是采用model_main,第二种是采用training。下面分别从这两个方面解读源代码。限于水平，解读当中可能存在部分错误或者用词不当之处，需要在项目进行过程当中不断更新和改进。

**1. 高阶api的整体架构**

​		model_main 位于object_detection的主目录下，代码采用了tf 的高层api来实现，所以代码封装程度比较高，目前尚未找到采用多GPU运行的方式执行（经文献查找，截止到19年10月，model_main尚不支持多gpu，所以想多gpu只能train.py）。因此要深入理解model_main,需要先大致了解高层的api。在tensorflow当中高层api的结构相互关系大致如下，其中比较主要的是dataset和estimator：

![](./data/tf_object_detection配置以及源码解读.assets/高级api结构.png)

​		

具体地，左边dataset主要负责处理深度学习的输入数据集，也就是读取tfrecord数据并对其进行预处理，最终完成数据的queue输入模型，主要依靠形成batch数据迭代器将数据在训练过程当中输入模型。中间的两个input_fn最终返回的就是一个dataset的迭代器，正如model_main代码当中的所写。hooks主要用于训练过程的当中对session监控以及对图的管理。比如说session开始之后锁定图结构，不允许向图中继续添加节点等等，至于更深入的作用有待进一步研究。这里的核心其实是estimator。一般来说，estimator的定义需要modelfn 和trainstep完成。这里的model_fn并不是一个简单的model定义函数，实际上model_fn返回的是一个 estimatorspec，而这个spec的输入就是一系列的tensorflow的op，或者op字典。主要就是模型的输出节点prediction，模型的loss计算节点，训练的优化器的op以及精度测量的op【或者op字典】等。而model_fn的输入是featues 和labels都是由input_fn返回的tensor节点【或者tensor字典】，从整体意义和作用上看，这个model_fn其实就是模型的核心结构的作用，即给定输入tensor给出输出tensor和loss等等。model_main基本是按照上面的框架构建的。

**2. 从model_main开始的整体逻辑设计**

​		model_main文件前一部分主要是参数设定，这一部分源码说明比较明确，按照参数的说明进行配置就可以，唯一需要指出的是：

```python
flags.DEFINE_integer('sample_1_of_n_eval_examples', 5000, 'Will sample one of '
                     'every n eval input examples, where n is provided.')
```

上述参数的含义是每多少个样本当中取一个进行eval。如果在pip_line_config【所有配置文件都在：\research\object_detection\samples\configs文件夹下，可以参考修改】：

```python
eval_config: {
  num_examples: 5000
  # metrics_set: "coco_detection_metrics"  #采用这种方式更改精度衡量方式，默认pascal_voc
}
```

的num_example设置为5000，那么**sample_1_of_n_eval_examples**就是说5000个样本当中只取一张进行eval，会大大提升eval速度，如果**sample_1_of_n_eval_examples**设置为1 ，那么就是说5000个样本全部进行eval。会比较费时间

​		参数设置完成后是主程序main，由于主程序全部运用tensorflow的高级api：**tf.estimator**运行，所以在讲主程序之前需要了解**tf.estimator**。下面是tf.estimator下的文件以及文件夹。main函数中用到的**tf.estimator.train_and_evaluate**就在training当中。

![](./data/tf_object_detection配置以及源码解读.assets/estimator.png)

​		**tf.estimator.Estimator**是tensorflow高层api当中专门用于训练和测试深度神经网络模型的一个类，其构造函数\___init___如下:

```python
__init__(
    model_fn,
    model_dir=None,
    config=None,
    params=None,
    warm_start_from=None
)
```

可见，estimator主要包括model_fn，model_dir，config，param 几个属性，其中model_fn 就是指object_detection定义的模型函数，但是这个函数的定义实际上是比较复杂的，其中包含了模型输入函数的定义，模型主体架构的定义以及loss的定义等等，这个函数的定义随后再详细讨论，这里先说剩下的参数，首先是model_dir,从[源码](https://github.com/tensorflow/estimator/blob/master/tensorflow_estimator/python/estimator/estimator.py)解释是保存训练完成后的代码的地方，代码可以从其中保存的ckpt接着训练而不需要重新开始。config是模型在train和eval时采用的主要配置参数，都通过**tf.estimator.RunConfig**来定义。RunConfig 的构造函数定义：

```python
  def __init__(self,
               model_dir=None,
               tf_random_seed=None,
               save_summary_steps=100,
               save_checkpoints_steps=_USE_DEFAULT,
               save_checkpoints_secs=_USE_DEFAULT,
               session_config=None,
               keep_checkpoint_max=5,
               keep_checkpoint_every_n_hours=10000,
               log_step_count_steps=100,
               train_distribute=None,
               device_fn=None,
               protocol=None,
               eval_distribute=None,
               experimental_distribute=None,
               experimental_max_worker_delay_secs=None):
```

可以看到在训练过程当中采用了哪些默认的参数。param是超参，至于超参和RunConfig之间的关系，有待进一步研究。下面先说代码：

```python
  tf.logging.set_verbosity(tf.logging.INFO)


  flags.mark_flag_as_required('model_dir')
  flags.mark_flag_as_required('pipeline_config_path')   # 这个配置文件是什么样子？在什么位置
  # 下面这句话是原代码当中的config
  config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir)
    
# 这个字典主要是定义后面的相关参数
  train_and_eval_dict = model_lib.create_estimator_and_inputs(
      run_config=config,
      hparams=model_hparams.create_hparams(FLAGS.hparams_overrides),
      pipeline_config_path=FLAGS.pipeline_config_path,
      train_steps=FLAGS.num_train_steps,
      sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
      sample_1_of_n_eval_on_train_examples=(
          FLAGS.sample_1_of_n_eval_on_train_examples))
  estimator = train_and_eval_dict['estimator']
  train_input_fn = train_and_eval_dict['train_input_fn']
  eval_input_fns = train_and_eval_dict['eval_input_fns']
  eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
  predict_input_fn = train_and_eval_dict['predict_input_fn']
  train_steps = train_and_eval_dict['train_steps']

  if FLAGS.checkpoint_dir:
    if FLAGS.eval_training_data:
      name = 'training_data'
      input_fn = eval_on_train_input_fn
    else:
      name = 'validation_data'
      # The first eval input will be evaluated.
      input_fn = eval_input_fns[0]
    if FLAGS.run_once:
      estimator.evaluate(input_fn,
                         steps=None,
                         checkpoint_path=tf.train.latest_checkpoint(
                             FLAGS.checkpoint_dir))
    else:
      model_lib.continuous_eval(estimator, FLAGS.checkpoint_dir, input_fn,
                                train_steps, name)
  else:

    train_spec, eval_specs = model_lib.create_train_and_eval_specs(
        train_input_fn,
        eval_input_fns,
        eval_on_train_input_fn,
        predict_input_fn,
        train_steps,
        eval_on_train_data=False)

    # Currently only a single Eval Spec is allowed.
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])
```

代码从**tf.estimator.train_and_evaluate**这开始，从源码解释看， 这个函数可以训练、评估和输出训练的模型， 所有和训练相关的东西都由train_spec来定义，包括训练需要的input_fn，和训练的最大步数。所有和评估以及输出模型相关的都由eval_spec来定义，包括eval的input_fn等【这里的input_fn都是为输入做准备的，和dataset相关】。来看train_and_evaluate源码，如下：

```python
_assert_eval_spec(eval_spec)  # fail fast if eval_spec is invalid.

  executor = _TrainingExecutor(
      estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)
  config = estimator.config

  # If `distribute_coordinator_mode` is set and running in distributed
  # environment, we run `train_and_evaluate` via distribute coordinator.
  if distribute_coordinator_training.should_run_distribute_coordinator(config):
    logging.info('Running `train_and_evaluate` with Distribute Coordinator.')
    distribute_coordinator_training.train_and_evaluate(
        estimator, train_spec, eval_spec, _TrainingExecutor)
    return

  if (config.task_type == run_config_lib.TaskType.EVALUATOR and
      config.task_id > 0):
    raise ValueError(
        'For distributed training, there can only be one `evaluator` task '
        '(with task id 0).  Given task id {}'.format(config.task_id))

  return executor.run()
```

训练过程当中的配置参数config都由estimator 的config 来定义，也就是说是实际由RunConfig来定义的。 可以看出，实际运行过程是先创建一个executor，然后又executor执行run运行的。我们看下executor的run方法部分代码：

```python
  def run(self):
    """Executes the run_foo for task type `foo`.

    `_TrainingExecutor` predefines the procedure for task type 'chief',
    'worker', 'ps', and 'evaluator'. For task type `foo`, the corresponding
    procedure is `run_foo'. This `run` method invoke the procedure base on the
    `RunConfig.task_type`.

    Returns:
      A tuple of the result of the `evaluate` call to the `Estimator` and the
      export results using the specified `ExportStrategy`.
      Currently undefined for distributed training mode.

    Raises:
      ValueError: if the estimator.config is mis-configured.
    """
    config = self._estimator.config

    if (not config.cluster_spec and
        config.task_type != run_config_lib.TaskType.EVALUATOR):
      logging.info('Running training and evaluation locally (non-distributed).')
      return self.run_local()
```

该方法主要是两种模式，一种是分布式模式，一种是local模式。在某些条件下，run方法其实调用的是run_local（实际上经过测试，model_main运行的本地模式实际上调用的就是run_local）,那么对于run_local：

```python
  def run_local(self):
    """Runs training and evaluation locally (non-distributed)."""
    _assert_eval_spec(self._eval_spec)

    train_hooks = list(self._train_spec.hooks) + list(self._train_hooks)
    logging.info('Start train and evaluate loop. The evaluate will happen '
                 'after every checkpoint. Checkpoint frequency is determined '
                 'based on RunConfig arguments: save_checkpoints_steps {} or '
                 'save_checkpoints_secs {}.'.format(
                     self._estimator.config.save_checkpoints_steps,
                     self._estimator.config.save_checkpoints_secs))

    evaluator = _TrainingExecutor._Evaluator(self._estimator, self._eval_spec,
                                             self._train_spec.max_steps)

    listener_for_eval = _NewCheckpointListenerForEvaluate(
        evaluator, self._eval_spec.throttle_secs,
        self._continuous_eval_listener)
    saving_listeners = [listener_for_eval]

    self._estimator.train(
        input_fn=self._train_spec.input_fn,
        max_steps=self._train_spec.max_steps,
        hooks=train_hooks,
        saving_listeners=saving_listeners)

    eval_result = listener_for_eval.eval_result or _EvalResult(
        status=_EvalStatus.MISSING_CHECKPOINT)
    return eval_result.metrics, listener_for_eval.export_results
```

注意上述代码的logging_info，这个表明，eval会在每次保存完代码之后运行，而保存代码的频率由RunConfig 当中的save_checkpoints_steps或者save_checkpoints_secs来定义，也就是eval频率是由这参数定义的，基本上是不能动的。这已经到官方estimator的源代码当中了，改这个的话，面临的风险比较大不建议修改。 训练由estimator的train方法实现，具体如何train目前暂不继续跟进。

​		综上，如果train_and_evaluate，必须定义estimator，train_spec以及eval_spec。那么首先看estimator的定义。

```python
estimator = train_and_eval_dict['estimator']
```

代码当中，estimator是train_and_eval_dict字典当中映射来的，而这个字典是由create_estimator_and_inputs获取得到的：

```python
train_and_eval_dict = model_lib.create_estimator_and_inputs(
      run_config=config,
      hparams=model_hparams.create_hparams(FLAGS.hparams_overrides),
      pipeline_config_path=FLAGS.pipeline_config_path,
      train_steps=FLAGS.num_train_steps,
      sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
      sample_1_of_n_eval_on_train_examples=(
          FLAGS.sample_1_of_n_eval_on_train_examples))
```

再看create_estimator_and_inputs：

```python
create_estimator_and_inputs(run_config,
                                hparams,
                                pipeline_config_path,
                                config_override=None,
                                train_steps=None,
                                sample_1_of_n_eval_examples=None,
                                sample_1_of_n_eval_on_train_examples=1,
                                model_fn_creator=create_model_fn,
                                use_tpu_estimator=False,
                                use_tpu=False,
                                num_shards=1,
                                params=None,
                                override_eval_num_epochs=True,
                                save_final_config=False,
                                postprocess_on_cpu=False,
                                export_to_tpu=None,
                                **kwargs):
  """Creates `Estimator`, input functions, and steps.

  Args:
    run_config: A `RunConfig`.
    hparams: A `HParams`.
    pipeline_config_path: A path to a pipeline config file.
    config_override: A pipeline_pb2.TrainEvalPipelineConfig text proto to
      override the config from `pipeline_config_path`.
    train_steps: Number of training steps. If None, the number of training steps
      is set from the `TrainConfig` proto.
    sample_1_of_n_eval_examples: Integer representing how often an eval example
      should be sampled. If 1, will sample all examples.
    sample_1_of_n_eval_on_train_examples: Similar to
      `sample_1_of_n_eval_examples`, except controls the sampling of training
      data for evaluation.
    model_fn_creator: A function that creates a `model_fn` for `Estimator`.
      Follows the signature:

      * Args:
        * `detection_model_fn`: Function that returns `DetectionModel` instance.
        * `configs`: Dictionary of pipeline config objects.
        * `hparams`: `HParams` object.
      * Returns:
        `model_fn` for `Estimator`.

    use_tpu_estimator: Whether a `TPUEstimator` should be returned. If False,
      an `Estimator` will be returned.
    use_tpu: Boolean, whether training and evaluation should run on TPU. Only
      used if `use_tpu_estimator` is True.
    num_shards: Number of shards (TPU cores). Only used if `use_tpu_estimator`
      is True.
    params: Parameter dictionary passed from the estimator. Only used if
      `use_tpu_estimator` is True.
    override_eval_num_epochs: Whether to overwrite the number of epochs to 1 for
      eval_input.
    save_final_config: Whether to save final config (obtained after applying
      overrides) to `estimator.model_dir`.
    postprocess_on_cpu: When use_tpu and postprocess_on_cpu are true,
      postprocess is scheduled on the host cpu.
    export_to_tpu: When use_tpu and export_to_tpu are true,
      `export_savedmodel()` exports a metagraph for serving on TPU besides the
      one on CPU.
    **kwargs: Additional keyword arguments for configuration override.

  Returns:
    A dictionary with the following fields:
    'estimator': An `Estimator` or `TPUEstimator`.
    'train_input_fn': A training input function.
    'eval_input_fns': A list of all evaluation input functions.
    'eval_input_names': A list of names for each evaluation input.
    'eval_on_train_input_fn': An evaluation-on-train input function.
    'predict_input_fn': A prediction input function.
    'train_steps': Number of training steps. Either directly from input or from
      configuration.
  """
  get_configs_from_pipeline_file = MODEL_BUILD_UTIL_MAP[
      'get_configs_from_pipeline_file']
  merge_external_params_with_configs = MODEL_BUILD_UTIL_MAP[
      'merge_external_params_with_configs']
  create_pipeline_proto_from_configs = MODEL_BUILD_UTIL_MAP[
      'create_pipeline_proto_from_configs']
  create_train_input_fn = MODEL_BUILD_UTIL_MAP['create_train_input_fn']
  create_eval_input_fn = MODEL_BUILD_UTIL_MAP['create_eval_input_fn']
  create_predict_input_fn = MODEL_BUILD_UTIL_MAP['create_predict_input_fn']
  detection_model_fn_base = MODEL_BUILD_UTIL_MAP['detection_model_fn_base']

  configs = get_configs_from_pipeline_file(
      pipeline_config_path, config_override=config_override)
  kwargs.update({
      'train_steps': train_steps,
      'use_bfloat16': configs['train_config'].use_bfloat16 and use_tpu
  })
  if sample_1_of_n_eval_examples >= 1:
    kwargs.update({
        'sample_1_of_n_eval_examples': sample_1_of_n_eval_examples
    })
  if override_eval_num_epochs:
    kwargs.update({'eval_num_epochs': 1})
    tf.logging.warning(
        'Forced number of epochs for all eval validations to be 1.')
  configs = merge_external_params_with_configs(
      configs, hparams, kwargs_dict=kwargs)
  model_config = configs['model']
  train_config = configs['train_config']
  train_input_config = configs['train_input_config']
  eval_config = configs['eval_config']
  eval_input_configs = configs['eval_input_configs']
  eval_on_train_input_config = copy.deepcopy(train_input_config)
  eval_on_train_input_config.sample_1_of_n_examples = (
      sample_1_of_n_eval_on_train_examples)
  if override_eval_num_epochs and eval_on_train_input_config.num_epochs != 1:
    tf.logging.warning('Expected number of evaluation epochs is 1, but '
                       'instead encountered `eval_on_train_input_config'
                       '.num_epochs` = '
                       '{}. Overwriting `num_epochs` to 1.'.format(
                           eval_on_train_input_config.num_epochs))
    eval_on_train_input_config.num_epochs = 1

  # update train_steps from config but only when non-zero value is provided
  if train_steps is None and train_config.num_steps != 0:
    train_steps = train_config.num_steps

  detection_model_fn = functools.partial(
      detection_model_fn_base, model_config=model_config)

  # Create the input functions for TRAIN/EVAL/PREDICT.
  train_input_fn = create_train_input_fn(
      train_config=train_config,
      train_input_config=train_input_config,
      model_config=model_config)
  eval_input_fns = [
      create_eval_input_fn(
          eval_config=eval_config,
          eval_input_config=eval_input_config,
          model_config=model_config) for eval_input_config in eval_input_configs
  ]
  eval_input_names = [
      eval_input_config.name for eval_input_config in eval_input_configs
  ]
  eval_on_train_input_fn = create_eval_input_fn(
      eval_config=eval_config,
      eval_input_config=eval_on_train_input_config,
      model_config=model_config)
  predict_input_fn = create_predict_input_fn(
      model_config=model_config, predict_input_config=eval_input_configs[0])

  # Read export_to_tpu from hparams if not passed.
  if export_to_tpu is None:
    export_to_tpu = hparams.get('export_to_tpu', False)
  tf.logging.info('create_estimator_and_inputs: use_tpu %s, export_to_tpu %s',
                  use_tpu, export_to_tpu)
  model_fn = model_fn_creator(detection_model_fn, configs, hparams, use_tpu,
                              postprocess_on_cpu)
  if use_tpu_estimator:
    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        train_batch_size=train_config.batch_size,
        # For each core, only batch size 1 is supported for eval.
        eval_batch_size=num_shards * 1 if use_tpu else 1,
        use_tpu=use_tpu,
        config=run_config,
        export_to_tpu=export_to_tpu,
        eval_on_tpu=False,  # Eval runs on CPU, so disable eval on TPU
        params=params if params else {})
  else:
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)

  # Write the as-run pipeline config to disk.
  if run_config.is_chief and save_final_config:
    pipeline_config_final = create_pipeline_proto_from_configs(configs)
    config_util.save_pipeline_config(pipeline_config_final, estimator.model_dir)

  return dict(
      estimator=estimator,
      train_input_fn=train_input_fn,
      eval_input_fns=eval_input_fns,
      eval_input_names=eval_input_names,
      eval_on_train_input_fn=eval_on_train_input_fn,
      predict_input_fn=predict_input_fn,
      train_steps=train_steps)
```

可见，create_estimator_and_inputs最终返回一个字典，字典当中包含estimator和对应的input_fn。而每一个input_fn基本都由对应的create函数构建，create函数由MODEL_BUILD_UTIL_MAP这个字典映射，最后由该字典映射的函数来构建。而最基础的model_fn则是由detection_model_fn_base构建（python的偏函数根据config）构造完成，最终由modelfn_creator创建。所以基本上就是detection_model_fn_base，只不过某些基本参数默认给定了。而estimator实际上就是给定model_fn和config之后的标准的estimator类。其他的操作都是拼接和编辑config的。所以上面函数最核心的是给出对应的inputfn和estimator。input_fn都是和神经网络输入相关的，包括数据的预处理等等，最后形成一个dataset（也是tf高级api之一）。estimator是核心，而estimator的核心是model_fn。model_fn才是模型真正构建的地方，其中定义了网络架构和loss的各种计算等等。

具体看traininputfn和modelfn的相关实现，首先看train_input_fn:

```python
def create_train_input_fn(train_config, train_input_config,
                          model_config):
  """Creates a train `input` function for `Estimator`.

  Args:
    train_config: A train_pb2.TrainConfig.
    train_input_config: An input_reader_pb2.InputReader.
    model_config: A model_pb2.DetectionModel.

  Returns:
    `input_fn` for `Estimator` in TRAIN mode.
  """

  def _train_input_fn(params=None):
    return train_input(train_config, train_input_config, model_config,
                       params=params)

  return _train_input_fn
```

可以看见返回的是train_input，具体再看：

```python
def train_input(train_config, train_input_config,
                model_config, model=None, params=None):
  """Returns `features` and `labels` tensor dictionaries for training.

  Args:
    train_config: A train_pb2.TrainConfig.
    train_input_config: An input_reader_pb2.InputReader.
    model_config: A model_pb2.DetectionModel.
    model: A pre-constructed Detection Model.
      If None, one will be created from the config.
    params: Parameter dictionary passed from the estimator.

  Returns:
    A tf.data.Dataset that holds (features, labels) tuple.

    features: Dictionary of feature tensors.
      features[fields.InputDataFields.image] is a [batch_size, H, W, C]
        float32 tensor with preprocessed images.
      features[HASH_KEY] is a [batch_size] int32 tensor representing unique
        identifiers for the images.
      features[fields.InputDataFields.true_image_shape] is a [batch_size, 3]
        int32 tensor representing the true image shapes, as preprocessed
        images could be padded.
      features[fields.InputDataFields.original_image] (optional) is a
        [batch_size, H, W, C] float32 tensor with original images.
    labels: Dictionary of groundtruth tensors.
      labels[fields.InputDataFields.num_groundtruth_boxes] is a [batch_size]
        int32 tensor indicating the number of groundtruth boxes.
      labels[fields.InputDataFields.groundtruth_boxes] is a
        [batch_size, num_boxes, 4] float32 tensor containing the corners of
        the groundtruth boxes.
      labels[fields.InputDataFields.groundtruth_classes] is a
        [batch_size, num_boxes, num_classes] float32 one-hot tensor of
        classes.
      labels[fields.InputDataFields.groundtruth_weights] is a
        [batch_size, num_boxes] float32 tensor containing groundtruth weights
        for the boxes.
      -- Optional --
      labels[fields.InputDataFields.groundtruth_instance_masks] is a
        [batch_size, num_boxes, H, W] float32 tensor containing only binary
        values, which represent instance masks for objects.
      labels[fields.InputDataFields.groundtruth_keypoints] is a
        [batch_size, num_boxes, num_keypoints, 2] float32 tensor containing
        keypoints for each box.

  Raises:
    TypeError: if the `train_config`, `train_input_config` or `model_config`
      are not of the correct type.
  """
  if not isinstance(train_config, train_pb2.TrainConfig):
    raise TypeError('For training mode, the `train_config` must be a '
                    'train_pb2.TrainConfig.')
  if not isinstance(train_input_config, input_reader_pb2.InputReader):
    raise TypeError('The `train_input_config` must be a '
                    'input_reader_pb2.InputReader.')
  if not isinstance(model_config, model_pb2.DetectionModel):
    raise TypeError('The `model_config` must be a '
                    'model_pb2.DetectionModel.')

  if model is None:
    model_preprocess_fn = INPUT_BUILDER_UTIL_MAP['model_build'](
        model_config, is_training=True).preprocess
  else:
    model_preprocess_fn = model.preprocess

  def transform_and_pad_input_data_fn(tensor_dict):
    """Combines transform and pad operation."""
    data_augmentation_options = [
        preprocessor_builder.build(step)
        for step in train_config.data_augmentation_options
    ]
    data_augmentation_fn = functools.partial(
        augment_input_data,
        data_augmentation_options=data_augmentation_options)

    image_resizer_config = config_util.get_image_resizer_config(model_config)
    image_resizer_fn = image_resizer_builder.build(image_resizer_config)
    transform_data_fn = functools.partial(
        transform_input_data, model_preprocess_fn=model_preprocess_fn,
        image_resizer_fn=image_resizer_fn,
        num_classes=config_util.get_number_of_classes(model_config),
        data_augmentation_fn=data_augmentation_fn,
        merge_multiple_boxes=train_config.merge_multiple_label_boxes,
        retain_original_image=train_config.retain_original_images,
        use_multiclass_scores=train_config.use_multiclass_scores,
        use_bfloat16=train_config.use_bfloat16)

    tensor_dict = pad_input_data_to_static_shapes(
        tensor_dict=transform_data_fn(tensor_dict),
        max_num_boxes=train_input_config.max_number_of_boxes,
        num_classes=config_util.get_number_of_classes(model_config),
        spatial_image_shape=config_util.get_spatial_image_size(
            image_resizer_config))
    return (_get_features_dict(tensor_dict), _get_labels_dict(tensor_dict))

  dataset = INPUT_BUILDER_UTIL_MAP['dataset_build'](
      train_input_config,
      transform_input_data_fn=transform_and_pad_input_data_fn,
      batch_size=params['batch_size'] if params else train_config.batch_size)
  return dataset
```

可见，其实际上最终返回的是一个dataset。根据解释，这个dataset带着features和labels。features是一个字典，包含了fields.InputDataFields.image，也就是预处理过后的图像；HASH_KEY，即图像的唯一标识符；fields.InputDataFields.true_image_shape，真实的图像形状，为padding等处理用；以及fields.InputDataFields.original_image，原始图像，这一项是可选的。同理，labels 也是一个字典，包含fields.InputDataFields.num_groundtruth_boxes，标识每张图像当中的groundtruth的数量；fields.InputDataFields.groundtruth_boxes标识groundtruth 的坐标信息；fields.InputDataFields.groundtruth_classes即groundtrhtu对应的类别以及对应的权重；当然在做语义分割当中还包含了fields.InputDataFields.groundtruth_instance_masks即实例分割的mask。这就是dataset最终的tuple所包含的信息，主要是构建了两个字典，字典当中包含了输入图像全部信息。具体的图像预处理方式和最终如何形成标签，后面解释（TODO：如何进行预处理的）。

​		再来看model_fn：从estimator 的model_fn的定义来看，model_fn接收之前input_fn给出的features和labels字典，以及其他config和hparam参数，最终返回的实际是一个tf.estimator.EstimatorSpec。

```python
def create_model_fn(detection_model_fn, configs, hparams, use_tpu=False,
                    postprocess_on_cpu=False):
  """Creates a model function for `Estimator`.

  Args:
    detection_model_fn: Function that returns a `DetectionModel` instance.
    configs: Dictionary of pipeline config objects.
    hparams: `HParams` object.
    use_tpu: Boolean indicating whether model should be constructed for
        use on TPU.
    postprocess_on_cpu: When use_tpu and postprocess_on_cpu is true, postprocess
        is scheduled on the host cpu.

  Returns:
    `model_fn` for `Estimator`.
  """
  train_config = configs['train_config']
  eval_input_config = configs['eval_input_config']
  eval_config = configs['eval_config']

  def model_fn(features, labels, mode, params=None):
    """Constructs the object detection model.

    Args:
      features: Dictionary of feature tensors, returned from `input_fn`.
      labels: Dictionary of groundtruth tensors if mode is TRAIN or EVAL,
        otherwise None.
      mode: Mode key from tf.estimator.ModeKeys.
      params: Parameter dictionary passed from the estimator.

    Returns:
      An `EstimatorSpec` that encapsulates the model and its serving
        configurations.
    """
    params = params or {}
    total_loss, train_op, detections, export_outputs = None, None, None, None
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    # Make sure to set the Keras learning phase. True during training,
    # False for inference.
    tf.keras.backend.set_learning_phase(is_training)
    detection_model = detection_model_fn(
        is_training=is_training, add_summaries=(not use_tpu))   #这个就是detection_model_fun的base函数

    scaffold_fn = None

    if mode == tf.estimator.ModeKeys.TRAIN:
      labels = unstack_batch(
          labels,
          unpad_groundtruth_tensors=train_config.unpad_groundtruth_tensors)
    elif mode == tf.estimator.ModeKeys.EVAL:
      # For evaling on train data, it is necessary to check whether groundtruth
      # must be unpadded.
      boxes_shape = (
          labels[fields.InputDataFields.groundtruth_boxes].get_shape()
          .as_list())
      unpad_groundtruth_tensors = boxes_shape[1] is not None and not use_tpu
      labels = unstack_batch(
          labels, unpad_groundtruth_tensors=unpad_groundtruth_tensors)

    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
      provide_groundtruth(detection_model, labels)

    preprocessed_images = features[fields.InputDataFields.image]
    if use_tpu and train_config.use_bfloat16:
      with tf.contrib.tpu.bfloat16_scope():
        prediction_dict = detection_model.predict(
            preprocessed_images,
            features[fields.InputDataFields.true_image_shape])
        prediction_dict = ops.bfloat16_to_float32_nested(prediction_dict)
    else:
      prediction_dict = detection_model.predict(
          preprocessed_images,
          features[fields.InputDataFields.true_image_shape])

    def postprocess_wrapper(args):
      return detection_model.postprocess(args[0], args[1])

    if mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT):
      if use_tpu and postprocess_on_cpu:
        detections = tf.contrib.tpu.outside_compilation(
            postprocess_wrapper,
            (prediction_dict,
             features[fields.InputDataFields.true_image_shape]))
      else:
        detections = postprocess_wrapper((
            prediction_dict,
            features[fields.InputDataFields.true_image_shape]))

    if mode == tf.estimator.ModeKeys.TRAIN:
      if train_config.fine_tune_checkpoint and hparams.load_pretrained:
        if not train_config.fine_tune_checkpoint_type:
          # train_config.from_detection_checkpoint field is deprecated. For
          # backward compatibility, set train_config.fine_tune_checkpoint_type
          # based on train_config.from_detection_checkpoint.
          if train_config.from_detection_checkpoint:
            train_config.fine_tune_checkpoint_type = 'detection'
          else:
            train_config.fine_tune_checkpoint_type = 'classification'
        asg_map = detection_model.restore_map(
            fine_tune_checkpoint_type=train_config.fine_tune_checkpoint_type,
            load_all_detection_checkpoint_vars=(
                train_config.load_all_detection_checkpoint_vars))    # 从model 当中直接定义了 图的恢复方式
        available_var_map = (
            variables_helper.get_variables_available_in_checkpoint(
                asg_map,
                train_config.fine_tune_checkpoint,
                include_global_step=False))           # 从ckpt 图当中恢复于元数据  全部是用于finetune的部分
        if use_tpu:

          def tpu_scaffold():
            tf.train.init_from_checkpoint(train_config.fine_tune_checkpoint,
                                          available_var_map)
            return tf.train.Scaffold()

          scaffold_fn = tpu_scaffold
        else:
          tf.train.init_from_checkpoint(train_config.fine_tune_checkpoint,
                                        available_var_map)

    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
      losses_dict = detection_model.loss(
          prediction_dict, features[fields.InputDataFields.true_image_shape])  # detection model当中定义了loss 的计算方式。

      losses = [loss_tensor for loss_tensor in losses_dict.values()]
      if train_config.add_regularization_loss:
        regularization_losses = detection_model.regularization_losses()
        if use_tpu and train_config.use_bfloat16:
          regularization_losses = ops.bfloat16_to_float32_nested(
              regularization_losses)
        if regularization_losses:
          regularization_loss = tf.add_n(
              regularization_losses, name='regularization_loss')
          losses.append(regularization_loss)
          losses_dict['Loss/regularization_loss'] = regularization_loss
      total_loss = tf.add_n(losses, name='total_loss')      # tf.add_n 可以把loss 加入到新的图当中 相当于加入了一个节点。列表相加？


      losses_dict['Loss/total_loss'] = total_loss

      if 'graph_rewriter_config' in configs:
        graph_rewriter_fn = graph_rewriter_builder.build(
            configs['graph_rewriter_config'], is_training=is_training)
        graph_rewriter_fn()

      # TODO(rathodv): Stop creating optimizer summary vars in EVAL mode once we
      # can write learning rate summaries on TPU without host calls.
      global_step = tf.train.get_or_create_global_step()
      training_optimizer, optimizer_summary_vars = optimizer_builder.build(
          train_config.optimizer)

    if mode == tf.estimator.ModeKeys.TRAIN:
      if use_tpu:
        training_optimizer = tf.contrib.tpu.CrossShardOptimizer(
            training_optimizer)

      # Optionally freeze some layers by setting their gradients to be zero.
      trainable_variables = None
      include_variables = (
          train_config.update_trainable_variables
          if train_config.update_trainable_variables else None)
      exclude_variables = (
          train_config.freeze_variables
          if train_config.freeze_variables else None)
      trainable_variables = tf.contrib.framework.filter_variables(
          tf.trainable_variables(),
          include_patterns=include_variables,
          exclude_patterns=exclude_variables)

      clip_gradients_value = None
      if train_config.gradient_clipping_by_norm > 0:
        clip_gradients_value = train_config.gradient_clipping_by_norm

      if not use_tpu:
        for var in optimizer_summary_vars:
          tf.summary.scalar(var.op.name, var)
      summaries = [] if use_tpu else None
      if train_config.summarize_gradients:
        summaries = ['gradients', 'gradient_norm', 'global_gradient_norm']
      train_op = tf.contrib.layers.optimize_loss(
          loss=total_loss,
          global_step=global_step,
          learning_rate=None,
          clip_gradients=clip_gradients_value,
          optimizer=training_optimizer,
          update_ops=detection_model.updates(),
          variables=trainable_variables,
          summaries=summaries,
          name='')  # Preventing scope prefix on all variables.

    if mode == tf.estimator.ModeKeys.PREDICT:
      exported_output = exporter_lib.add_output_tensor_nodes(detections)
      export_outputs = {
          tf.saved_model.signature_constants.PREDICT_METHOD_NAME:
              tf.estimator.export.PredictOutput(exported_output)
      }

    eval_metric_ops = None
    scaffold = None
    if mode == tf.estimator.ModeKeys.EVAL:
      class_agnostic = (
          fields.DetectionResultFields.detection_classes not in detections)
      groundtruth = _prepare_groundtruth_for_eval(
          detection_model, class_agnostic,
          eval_input_config.max_number_of_boxes)
      use_original_images = fields.InputDataFields.original_image in features
      if use_original_images:
        eval_images = features[fields.InputDataFields.original_image]
        true_image_shapes = tf.slice(
            features[fields.InputDataFields.true_image_shape], [0, 0], [-1, 3])
        original_image_spatial_shapes = features[fields.InputDataFields
                                                 .original_image_spatial_shape]
      else:
        eval_images = features[fields.InputDataFields.image]
        true_image_shapes = None
        original_image_spatial_shapes = None

      eval_dict = eval_util.result_dict_for_batched_example(
          eval_images,
          features[inputs.HASH_KEY],
          detections,
          groundtruth,
          class_agnostic=class_agnostic,
          scale_to_absolute=True,
          original_image_spatial_shapes=original_image_spatial_shapes,
          true_image_shapes=true_image_shapes)

      if class_agnostic:
        category_index = label_map_util.create_class_agnostic_category_index()
      else:
        category_index = label_map_util.create_category_index_from_labelmap(
            eval_input_config.label_map_path)
      vis_metric_ops = None
      if not use_tpu and use_original_images:
        eval_metric_op_vis = vis_utils.VisualizeSingleFrameDetections(
            category_index,
            max_examples_to_draw=eval_config.num_visualizations,
            max_boxes_to_draw=eval_config.max_num_boxes_to_visualize,
            min_score_thresh=eval_config.min_score_threshold,
            use_normalized_coordinates=False)
        vis_metric_ops = eval_metric_op_vis.get_estimator_eval_metric_ops(
            eval_dict)

      # Eval metrics on a single example.
      eval_metric_ops = eval_util.get_eval_metric_ops_for_evaluators(   #  这个地方也是eval metric
          eval_config, list(category_index.values()), eval_dict)       #这个地方是精度测量，也就是计算训练精度的函数  eval_dict 是在这用的，给他用的
      for loss_key, loss_tensor in iter(losses_dict.items()):
        eval_metric_ops[loss_key] = tf.metrics.mean(loss_tensor)
      for var in optimizer_summary_vars:
        eval_metric_ops[var.op.name] = (var, tf.no_op())
      if vis_metric_ops is not None:
        eval_metric_ops.update(vis_metric_ops)
      eval_metric_ops = {str(k): v for k, v in eval_metric_ops.items()}

      if eval_config.use_moving_averages:
        variable_averages = tf.train.ExponentialMovingAverage(0.0)
        variables_to_restore = variable_averages.variables_to_restore()
        keep_checkpoint_every_n_hours = (
            train_config.keep_checkpoint_every_n_hours)
        saver = tf.train.Saver(
            variables_to_restore,
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)
        scaffold = tf.train.Scaffold(saver=saver)

    # EVAL executes on CPU, so use regular non-TPU EstimatorSpec.
    if use_tpu and mode != tf.estimator.ModeKeys.EVAL:
      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          scaffold_fn=scaffold_fn,
          predictions=detections,
          loss=total_loss,
          train_op=train_op,
          eval_metrics=eval_metric_ops,
          export_outputs=export_outputs)
    else:
      if scaffold is None:
        keep_checkpoint_every_n_hours = (
            train_config.keep_checkpoint_every_n_hours)
        saver = tf.train.Saver(
            sharded=True,
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
            save_relative_paths=True)
        tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
        scaffold = tf.train.Scaffold(saver=saver)
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=detections,
          loss=total_loss,
          train_op=train_op,
          eval_metric_ops=eval_metric_ops,
          export_outputs=export_outputs,
          scaffold=scaffold)

  return model_fn
```

所以上面函数写的类似于装饰器，最终调用自定义的model_fn返回tf.estimator.EstimatorSpec。EstimatorSpec就完全定义了estimator运行时定义的模型。estimatorspec当中要求按照下面的方式定义modelfun：

```python
def my_model_fn(features, labels, mode):
      predictions = ...
      loss = ...
      train_op = ...
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions,
          loss=loss,
          train_op=train_op)
```

或者按照下面的方式定义：

```python 
def my_model_fn(features, labels, mode):
      if (mode == tf.estimator.ModeKeys.TRAIN or
          mode == tf.estimator.ModeKeys.EVAL):
        loss = ...
      else:
        loss = None
      if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = ...
      else:
        train_op = None
      if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = ...
      else:
        predictions = None

      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions,
          loss=loss,
          train_op=train_op)
```

这就是上面代码的组织方式。predictions是一个预测的tensor或者一个tensor的字典，loss一定是一个标量tensor或者说shape为【1】的tensor。trainop就是指一般的optimizer.minimize等训练相关的op。eval_metric_ops评估精度的一系列的op，字典形式保存。来看prediction_dict:

```python
prediction_dict = detection_model.predict(
          preprocessed_images,
          features[fields.InputDataFields.true_image_shape])
```

detection_model就是：

```python
 detection_model = detection_model_fn(
        is_training=is_training, add_summaries=(not use_tpu))   #这个就是detection_model_fun的base函数
```

而 detection_model_fn就是：

```python
detection_model_fn = functools.partial(
      detection_model_fn_base, model_config=model_config)
```

而detection_model_fn_base就是：

```python
detection_model_fn_base = MODEL_BUILD_UTIL_MAP['detection_model_fn_base']
```

也就是：

```python
'detection_model_fn_base': model_builder.build
```

这个模型构造函数是：

```python
def build(model_config, is_training, add_summaries=True):
  """Builds a DetectionModel based on the model config.

  Args:
    model_config: A model.proto object containing the config for the desired
      DetectionModel.
    is_training: True if this model is being built for training purposes.
    add_summaries: Whether to add tensorflow summaries in the model graph.
  Returns:
    DetectionModel based on the config.

  Raises:
    ValueError: On invalid meta architecture or model.
  """
  if not isinstance(model_config, model_pb2.DetectionModel):
    raise ValueError('model_config not of type model_pb2.DetectionModel.')
  meta_architecture = model_config.WhichOneof('model')
  if meta_architecture == 'ssd':
    return _build_ssd_model(model_config.ssd, is_training, add_summaries)
  if meta_architecture == 'faster_rcnn':
    return _build_faster_rcnn_model(model_config.faster_rcnn, is_training,
                                    add_summaries)
  raise ValueError('Unknown meta architecture: {}'.format(meta_architecture))
```

可见他会根据参数构建对应的模型，我们的基础模型是_build_faster_rcnn_model，也就是：

```python
def _build_faster_rcnn_model(frcnn_config, is_training, add_summaries):
  """Builds a Faster R-CNN or R-FCN detection model based on the model config.

  Builds R-FCN model if the second_stage_box_predictor in the config is of type
  `rfcn_box_predictor` else builds a Faster R-CNN model.

  Args:
    frcnn_config: A faster_rcnn.proto object containing the config for the
      desired FasterRCNNMetaArch or RFCNMetaArch.
    is_training: True if this model is being built for training purposes.
    add_summaries: Whether to add tf summaries in the model.

  Returns:
    FasterRCNNMetaArch based on the config.

  Raises:
    ValueError: If frcnn_config.type is not recognized (i.e. not registered in
      model_class_map).
  """
  num_classes = frcnn_config.num_classes
  image_resizer_fn = image_resizer_builder.build(frcnn_config.image_resizer)

  is_keras = (frcnn_config.feature_extractor.type in
              FASTER_RCNN_KERAS_FEATURE_EXTRACTOR_CLASS_MAP)

  if is_keras:
    feature_extractor = _build_faster_rcnn_keras_feature_extractor(
        frcnn_config.feature_extractor, is_training,
        inplace_batchnorm_update=frcnn_config.inplace_batchnorm_update)
  else:
    feature_extractor = _build_faster_rcnn_feature_extractor(
        frcnn_config.feature_extractor, is_training,
        inplace_batchnorm_update=frcnn_config.inplace_batchnorm_update)

  number_of_stages = frcnn_config.number_of_stages
  first_stage_anchor_generator = anchor_generator_builder.build(
      frcnn_config.first_stage_anchor_generator)

  first_stage_target_assigner = target_assigner.create_target_assigner(
      'FasterRCNN',
      'proposal',
      use_matmul_gather=frcnn_config.use_matmul_gather_in_matcher)
  first_stage_atrous_rate = frcnn_config.first_stage_atrous_rate
  if is_keras:
    first_stage_box_predictor_arg_scope_fn = (
        hyperparams_builder.KerasLayerHyperparams(
            frcnn_config.first_stage_box_predictor_conv_hyperparams))
  else:
    first_stage_box_predictor_arg_scope_fn = hyperparams_builder.build(
        frcnn_config.first_stage_box_predictor_conv_hyperparams, is_training)
  first_stage_box_predictor_kernel_size = (
      frcnn_config.first_stage_box_predictor_kernel_size)
  first_stage_box_predictor_depth = frcnn_config.first_stage_box_predictor_depth
  first_stage_minibatch_size = frcnn_config.first_stage_minibatch_size
  use_static_shapes = frcnn_config.use_static_shapes and (
      frcnn_config.use_static_shapes_for_eval or is_training)
  first_stage_sampler = sampler.BalancedPositiveNegativeSampler(
      positive_fraction=frcnn_config.first_stage_positive_balance_fraction,
      is_static=(frcnn_config.use_static_balanced_label_sampler and
                 use_static_shapes))
  first_stage_max_proposals = frcnn_config.first_stage_max_proposals
  if (frcnn_config.first_stage_nms_iou_threshold < 0 or
      frcnn_config.first_stage_nms_iou_threshold > 1.0):
    raise ValueError('iou_threshold not in [0, 1.0].')
  if (is_training and frcnn_config.second_stage_batch_size >
      first_stage_max_proposals):
    raise ValueError('second_stage_batch_size should be no greater than '
                     'first_stage_max_proposals.')
  first_stage_non_max_suppression_fn = functools.partial(
      post_processing.batch_multiclass_non_max_suppression,
      score_thresh=frcnn_config.first_stage_nms_score_threshold,
      iou_thresh=frcnn_config.first_stage_nms_iou_threshold,
      max_size_per_class=frcnn_config.first_stage_max_proposals,
      max_total_size=frcnn_config.first_stage_max_proposals,
      use_static_shapes=use_static_shapes)
  first_stage_loc_loss_weight = (
      frcnn_config.first_stage_localization_loss_weight)
  first_stage_obj_loss_weight = frcnn_config.first_stage_objectness_loss_weight

  initial_crop_size = frcnn_config.initial_crop_size
  maxpool_kernel_size = frcnn_config.maxpool_kernel_size
  maxpool_stride = frcnn_config.maxpool_stride

  second_stage_target_assigner = target_assigner.create_target_assigner(
      'FasterRCNN',
      'detection',
      use_matmul_gather=frcnn_config.use_matmul_gather_in_matcher)
  if is_keras:
    second_stage_box_predictor = box_predictor_builder.build_keras(
        hyperparams_builder.KerasLayerHyperparams,
        freeze_batchnorm=False,
        inplace_batchnorm_update=False,
        num_predictions_per_location_list=[1],
        box_predictor_config=frcnn_config.second_stage_box_predictor,
        is_training=is_training,
        num_classes=num_classes)
  else:
    second_stage_box_predictor = box_predictor_builder.build(
        hyperparams_builder.build,
        frcnn_config.second_stage_box_predictor,
        is_training=is_training,
        num_classes=num_classes)
  second_stage_batch_size = frcnn_config.second_stage_batch_size
  second_stage_sampler = sampler.BalancedPositiveNegativeSampler(
      positive_fraction=frcnn_config.second_stage_balance_fraction,
      is_static=(frcnn_config.use_static_balanced_label_sampler and
                 use_static_shapes))
  (second_stage_non_max_suppression_fn, second_stage_score_conversion_fn
  ) = post_processing_builder.build(frcnn_config.second_stage_post_processing)
  second_stage_localization_loss_weight = (
      frcnn_config.second_stage_localization_loss_weight)
  second_stage_classification_loss = (
      losses_builder.build_faster_rcnn_classification_loss(
          frcnn_config.second_stage_classification_loss))
  second_stage_classification_loss_weight = (
      frcnn_config.second_stage_classification_loss_weight)
  second_stage_mask_prediction_loss_weight = (
      frcnn_config.second_stage_mask_prediction_loss_weight)

  hard_example_miner = None    # 查看如何查找难训练的样本
  if frcnn_config.HasField('hard_example_miner'):
    hard_example_miner = losses_builder.build_hard_example_miner(
        frcnn_config.hard_example_miner,
        second_stage_classification_loss_weight,
        second_stage_localization_loss_weight)

  crop_and_resize_fn = (
      ops.matmul_crop_and_resize if frcnn_config.use_matmul_crop_and_resize
      else ops.native_crop_and_resize)
  clip_anchors_to_image = (
      frcnn_config.clip_anchors_to_image)

  common_kwargs = {
      'is_training': is_training,
      'num_classes': num_classes,
      'image_resizer_fn': image_resizer_fn,
      'feature_extractor': feature_extractor,
      'number_of_stages': number_of_stages,
      'first_stage_anchor_generator': first_stage_anchor_generator,
      'first_stage_target_assigner': first_stage_target_assigner,
      'first_stage_atrous_rate': first_stage_atrous_rate,
      'first_stage_box_predictor_arg_scope_fn':
      first_stage_box_predictor_arg_scope_fn,
      'first_stage_box_predictor_kernel_size':
      first_stage_box_predictor_kernel_size,
      'first_stage_box_predictor_depth': first_stage_box_predictor_depth,
      'first_stage_minibatch_size': first_stage_minibatch_size,
      'first_stage_sampler': first_stage_sampler,
      'first_stage_non_max_suppression_fn': first_stage_non_max_suppression_fn,
      'first_stage_max_proposals': first_stage_max_proposals,
      'first_stage_localization_loss_weight': first_stage_loc_loss_weight,
      'first_stage_objectness_loss_weight': first_stage_obj_loss_weight,
      'second_stage_target_assigner': second_stage_target_assigner,
      'second_stage_batch_size': second_stage_batch_size,
      'second_stage_sampler': second_stage_sampler,
      'second_stage_non_max_suppression_fn':
      second_stage_non_max_suppression_fn,
      'second_stage_score_conversion_fn': second_stage_score_conversion_fn,
      'second_stage_localization_loss_weight':
      second_stage_localization_loss_weight,
      'second_stage_classification_loss':
      second_stage_classification_loss,
      'second_stage_classification_loss_weight':
      second_stage_classification_loss_weight,
      'hard_example_miner': hard_example_miner,
      'add_summaries': add_summaries,
      'crop_and_resize_fn': crop_and_resize_fn,
      'clip_anchors_to_image': clip_anchors_to_image,
      'use_static_shapes': use_static_shapes,
      'resize_masks': frcnn_config.resize_masks
  }

  if (isinstance(second_stage_box_predictor,
                 rfcn_box_predictor.RfcnBoxPredictor) or
      isinstance(second_stage_box_predictor,
                 rfcn_keras_box_predictor.RfcnKerasBoxPredictor)):
    return rfcn_meta_arch.RFCNMetaArch(
        second_stage_rfcn_box_predictor=second_stage_box_predictor,
        **common_kwargs)
  else:
    return faster_rcnn_meta_arch.FasterRCNNMetaArch(
        initial_crop_size=initial_crop_size,
        maxpool_kernel_size=maxpool_kernel_size,
        maxpool_stride=maxpool_stride,
        second_stage_mask_rcnn_box_predictor=second_stage_box_predictor,
        second_stage_mask_prediction_loss_weight=(
            second_stage_mask_prediction_loss_weight),
        **common_kwargs)
```

最终返回这个fasterrcnn 的元结构对象faster_rcnn_meta_arch.FasterRCNNMetaArch，这个东西有preprocess，postprocess，loss和predict等等方法，predict方法也就是返回predict的字典，这个方法的输入和输出是：

```python
Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.#代表原始图像的输入
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.# 真实图像的形状

    Returns:
      prediction_dict: a dictionary holding "raw" prediction tensors:#原始预测产生的这些tensors
        1) rpn_box_predictor_features: A 4-D float32 tensor with shape
          [batch_size, height, width, depth] to be used for predicting proposal
          boxes and corresponding objectness scores.# rpn预测bbox对应分值的featuremap，是一个4维的tensor。
            
        2) rpn_features_to_crop: A 4-D float32 tensor with shape
          [batch_size, height, width, depth] representing image features to crop
          using the proposal boxes predicted by the RPN.# rpn裁剪用的featuremap？
            
        3) image_shape: a 1-D tensor of shape [4] representing the input
          image shape. #图像形状的4个值
        
        4) rpn_box_encodings:  3-D float tensor of shape
          [batch_size, num_anchors, self._box_coder.code_size] containing
          predicted boxes.
            
        5) rpn_objectness_predictions_with_background: 3-D float tensor of shape
          [batch_size, num_anchors, 2] containing class
          predictions (logits) for each of the anchors.  Note that this
          tensor *includes* background class predictions (at class index 0).
        # rpn 输出的有么有北京的一个值
        
        6) anchors: A 2-D tensor of shape [num_anchors, 4] representing anchors
          for the first stage RPN (in absolute coordinates).  Note that
          `num_anchors` can differ depending on whether the model is created in
          training or inference mode.# 输出的anchors

        (and if number_of_stages > 1):
        7) refined_box_encodings: a 3-D tensor with shape
          [total_num_proposals, num_classes, self._box_coder.code_size]
          representing predicted (final) refined box encodings, where
          total_num_proposals=batch_size*self._max_num_proposals. If using
          a shared box across classes the shape will instead be
          [total_num_proposals, 1, self._box_coder.code_size].#调整后的bbox
        
        8) class_predictions_with_background: a 3-D tensor with shape
          [total_num_proposals, num_classes + 1] containing class
          predictions (logits) for each of the anchors, where
          total_num_proposals=batch_size*self._max_num_proposals.
          Note that this tensor *includes* background class predictions
          (at class index 0).
        9) num_proposals: An int32 tensor of shape [batch_size] representing the
          number of proposals generated by the RPN.  `num_proposals` allows us
          to keep track of which entries are to be treated as zero paddings and
          which are not since we always pad the number of proposals to be
          `self.max_num_proposals` for each image.
            
        10) proposal_boxes: A float32 tensor of shape
          [batch_size, self.max_num_proposals, 4] representing
          decoded proposal bounding boxes in absolute coordinates.
        11) mask_predictions: (optional) a 4-D tensor with shape
          [total_num_padded_proposals, num_classes, mask_height, mask_width]
          containing instance mask predictions.
```

从之前model_fn的定义当中，train_op就是：

```python
train_op = tf.contrib.layers.optimize_loss(
          loss=total_loss,
          global_step=global_step,
          learning_rate=None,
          clip_gradients=clip_gradients_value,
          optimizer=training_optimizer,
          update_ops=detection_model.updates(),
          variables=trainable_variables,
          summaries=summaries,
          name='')  # Preventing scope prefix on all variables.
```

再看total_loss这个tensor的定义，由于estimatorspec要求loss 必须是输出单值的tensor，不能是tensor字典，所以在model_fn的定义过程当中源码采用了下面的计算方式即tf.add_n把所有的loss加到一起：

```python
    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
      losses_dict = detection_model.loss(
          prediction_dict, features[fields.InputDataFields.true_image_shape])  # detection model当中定义了loss 的计算方式。

      losses = [loss_tensor for loss_tensor in losses_dict.values()]
      if train_config.add_regularization_loss:
        regularization_losses = detection_model.regularization_losses()
        if use_tpu and train_config.use_bfloat16:
          regularization_losses = ops.bfloat16_to_float32_nested(
              regularization_losses)
        if regularization_losses:
          regularization_loss = tf.add_n(
              regularization_losses, name='regularization_loss')
          losses.append(regularization_loss)
          losses_dict['Loss/regularization_loss'] = regularization_loss
      total_loss = tf.add_n(losses, name='total_loss') 
```

而上面loss_dict字典则由FasterRCNNMetaArch这元结构的loss得出。

eval_metric_ops由得到:

```python
def get_eval_metric_ops_for_evaluators(eval_config,
                                       categories,
                                       eval_dict):
  """Returns eval metrics ops to use with `tf.estimator.EstimatorSpec`.

  Args:
    eval_config: An `eval_pb2.EvalConfig`.
    categories: A list of dicts, each of which has the following keys -
        'id': (required) an integer id uniquely identifying this category.
        'name': (required) string representing category name e.g., 'cat', 'dog'.
    eval_dict: An evaluation dictionary, returned from
      result_dict_for_single_example().

  Returns:
    A dictionary of metric names to tuple of value_op and update_op that can be
    used as eval metric ops in tf.EstimatorSpec.
  """
  eval_metric_ops = {}
  evaluator_options = evaluator_options_from_eval_config(eval_config)
  evaluators_list = get_evaluators(eval_config, categories, evaluator_options)
  for evaluator in evaluators_list:
    eval_metric_ops.update(evaluator.get_estimator_eval_metric_ops(
        eval_dict))
  return eval_metric_ops

```

可见，eval_metric_ops也是一个字典，由evaluator得到，而这个evaluator实际上就是由字典映射得到的：

```python
 evaluator_options = evaluator_options or {}
  eval_metric_fn_keys = eval_config.metrics_set
  if not eval_metric_fn_keys:
    eval_metric_fn_keys = [EVAL_DEFAULT_METRIC]
  evaluators_list = []
  for eval_metric_fn_key in eval_metric_fn_keys:   #eval函数的关键字，
    if eval_metric_fn_key not in EVAL_METRICS_CLASS_DICT:
      raise ValueError('Metric not found: {}'.format(eval_metric_fn_key))
    kwargs_dict = (evaluator_options[eval_metric_fn_key] if eval_metric_fn_key
                   in evaluator_options else {})
    evaluators_list.append(EVAL_METRICS_CLASS_DICT[eval_metric_fn_key](
        categories,
        **kwargs_dict))
  return evaluators_list
```

所以说具体采用哪个函数实际上是由eval_config当中的eval_config.metrics_set这个决定的。所有的evalator,并且默认是：coco_detection_metrics。

```py
EVAL_METRICS_CLASS_DICT = {
    'coco_detection_metrics':
        coco_evaluation.CocoDetectionEvaluator,
    'coco_mask_metrics':
        coco_evaluation.CocoMaskEvaluator,
    'oid_challenge_detection_metrics':
        object_detection_evaluation.OpenImagesDetectionChallengeEvaluator,
    'oid_challenge_segmentation_metrics':
        object_detection_evaluation
        .OpenImagesInstanceSegmentationChallengeEvaluator,
    'pascal_voc_detection_metrics':
        object_detection_evaluation.PascalDetectionEvaluator,
    'weighted_pascal_voc_detection_metrics':
        object_detection_evaluation.WeightedPascalDetectionEvaluator,
    'precision_at_recall_detection_metrics':
        object_detection_evaluation.PrecisionAtRecallDetectionEvaluator,
    'pascal_voc_instance_segmentation_metrics':
        object_detection_evaluation.PascalInstanceSegmentationEvaluator,
    'weighted_pascal_voc_instance_segmentation_metrics':
        object_detection_evaluation.WeightedPascalInstanceSegmentationEvaluator,
    'oid_V2_detection_metrics':
        object_detection_evaluation.OpenImagesDetectionEvaluator,
}

EVAL_DEFAULT_METRIC = 'coco_detection_metrics'
```

比较好用的evaluator是Pascalvoc的：

```python
class PascalDetectionEvaluator(ObjectDetectionEvaluator):
  """A class to evaluate detections using PASCAL metrics."""

  def __init__(self, categories, matching_iou_threshold=0.7):   # 这个地方我可以用全局变量来设置 原值 0.5 完美修改 ，那个是后处理的
    super(PascalDetectionEvaluator, self).__init__(
        categories,
        matching_iou_threshold=matching_iou_threshold,
        evaluate_corlocs=False,
        metric_prefix='PascalBoxes',
        use_weighted_mean_ap=False)
```

这个iou阈值是可调的。到这里，基本上model_fn就差不多定义完了。也就是estimator就基本定义完了。当然模型定义的细节和loss计算的细节，还有待进一步深入挖掘，比如如何改模型，从哪里改，如何改loss 的计算等，这里目前是先提供大框架的理解。

​		下面就是是trainspec和evalspec的定义了：

```python
def create_train_and_eval_specs(train_input_fn,
                                eval_input_fns,
                                eval_on_train_input_fn,
                                predict_input_fn,
                                train_steps,
                                eval_on_train_data=False,
                                final_exporter_name='Servo',
                                eval_spec_names=None):
  """Creates a `TrainSpec` and `EvalSpec`s.

  Args:
    train_input_fn: Function that produces features and labels on train data.
    eval_input_fns: A list of functions that produce features and labels on eval
      data.
    eval_on_train_input_fn: Function that produces features and labels for
      evaluation on train data.
    predict_input_fn: Function that produces features for inference.
    train_steps: Number of training steps.
    eval_on_train_data: Whether to evaluate model on training data. Default is
      False.
    final_exporter_name: String name given to `FinalExporter`.
    eval_spec_names: A list of string names for each `EvalSpec`.

  Returns:
    Tuple of `TrainSpec` and list of `EvalSpecs`. If `eval_on_train_data` is
    True, the last `EvalSpec` in the list will correspond to training data. The
    rest EvalSpecs in the list are evaluation datas.
  """
  train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn, max_steps=train_steps)

  if eval_spec_names is None:
    eval_spec_names = [str(i) for i in range(len(eval_input_fns))]

  eval_specs = []
  for index, (eval_spec_name, eval_input_fn) in enumerate(
      zip(eval_spec_names, eval_input_fns)):
    # Uses final_exporter_name as exporter_name for the first eval spec for
    # backward compatibility.
    if index == 0:
      exporter_name = final_exporter_name
    else:
      exporter_name = '{}_{}'.format(final_exporter_name, eval_spec_name)
    exporter = tf.estimator.FinalExporter(
        name=exporter_name, serving_input_receiver_fn=predict_input_fn)
    eval_specs.append(
        tf.estimator.EvalSpec(
            name=eval_spec_name,
            input_fn=eval_input_fn,
            steps=None,
            exporters=exporter,
            throttle_secs=3600,
        ))

  if eval_on_train_data:
    eval_specs.append(
        tf.estimator.EvalSpec(
            name='eval_on_train', input_fn=eval_on_train_input_fn, steps=None))

  return train_spec, eval_specs
```

可见：

```python
 train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn, max_steps=train_steps)
```

train_spec就是标准的tf.estimator.TrainSpec，input_fn就是之前解释的和处理输入数据（包括预处理等）的dataset，最后得到的一个batch的数据等。其实evalspec定义的也是标准的tf.estimator的Spec，只不过这里给定了spec的不同的名字，给定不同的名字和不同的input_fn，其目的是利用不同的输入数据集来评估不同数据集下的精度，最终的结果会分别写在不同的文件夹下。exporter是给tfserving输出模型服务的，就是保存savedmodel用的。因此，train_spec,eval_spec,以及estimator定义完成后，就可以进行训练了。以上就是模型的运行的整体逻辑。（还需要解决的细节有：tfrecord的生成和读取，模型loss 的具体计算，模型结构的具体设计实现以及如何修改）eval的所有精度的参数在根目录下的eval_utils.py文件当中。



1. 从training开始

   

####  coco数据集mask标签的构造

​		实际上如果不想自己构造并转换标注数据的情况下，用vggvia标注图像生成json的标签文件也是可以的，但是需要注意修正两点，vggvia标注完成之后，图像id是字符串，不是整数，而程序在取图时，实际上利用的是整数，所以需要对代码进行修正，如下：

```python
      for annotation in groundtruth_data['annotations']:
        image_id = int(annotation['image_id'])   #我们软件标注的形式是字符串的形式
        if image_id not in annotations_index:
          annotations_index[image_id] = []
        annotations_index[image_id].append(annotation)
```

把字符串转换为整数。另外，***在训练过程当中***必须保证label_map_pbtxt文件里面的类别和标签数量，以及***名称*** 都必须保证和json 文件里的相对应好，保证完全一致。否则会出现loss在迭代一定时间之后爆炸的问题。原因就是标签对不上。create_coco_tfrecord.

​		从coco 数据集转换成的mask标签来看，其mask在进入神经网络之前是全部转换成了【0,1】的标签数据，也就是一张mask标签图只能表示一个类别，如果同一张图像上有多个类别，那么就是说需要生成不同mask标签的图，标签图带有类别编号。并不是说以一个彩色图或者灰度图标识多个类，用png标签图做全部任务。因此，其基本的组织方式是：一张图带有一个图的编号，一张图可能有多个segmentation，每个segmentation都有一个类别编号，每一个segmentation 都将转换成一张0，1的二值图，也就是说一个图像可能会生成多个0,1二值图，最后做列表的append完成一整张图的segmentation的标记。

​		coco 创建数据集主要依靠下面的函数，创建，一个example就是这么组织的，所以说，构建相类似的tfrecord就需要按照下面的形式组织自己的数据集。

```python
_, tf_example, num_annotations_skipped = create_tf_example(
          image, annotations_list, image_dir, category_index, include_masks)
```

首先看上面函数的输入的组织方式：

image的组织形式（因为image实际上是从文件夹当中读取的，所以初步猜测这个里面只有filename有用）：![](./data/tf_object_detection配置以及源码解读.assets/image的组织形式.png)

然后看annotation的组织形式，annotation实际上是一个list（一个图像可以有多个annotation），每个annota的结构：![](./data/tf_object_detection配置以及源码解读.assets/annotation的组织形式.png)

annotation实际上都对应一个图像id，segmentation就是标记的坐标点（或者说语义分割关键点），比较重要的信息有category_id(类别标签)，iscrowd（辅助生成01标签png图的），每一个annotation都是一个字典。image_dir就是存储图像的路径。category_index是一系列的标签映射，最后一个参数就是True或者false。category_index的组织形式：

![](./data/tf_object_detection配置以及源码解读.assets/类别标签.png)

所以构建自己的数据集关键就是组织成上述形式。经过编码测试，按照上述方式构建自己的数据集之后，确实能够正常运行。自己的代码如下：

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import json
import os
import contextlib2
import numpy as np
import PIL.Image

from pycocotools import mask
import tensorflow as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from copy import deepcopy

flags = tf.app.flags
tf.flags.DEFINE_boolean('include_masks', True,
                        'Whether to include instance segmentations masks '
                        '(PNG encoded) in the result. default: False.')
tf.flags.DEFINE_string('train_image_dir', r'C:\Users\TIAN\Desktop\mask_rcnn_mask',
                       'Training image directory.')
tf.flags.DEFINE_string('val_image_dir', r'C:\Users\TIAN\Desktop\mask_rcnn_mask',
                       'Validation image directory.')
tf.flags.DEFINE_string('test_image_dir', '',
                       'Test image directory.')
tf.flags.DEFINE_string('train_annotations_file', r'C:\Users\TIAN\Desktop\mask_rcnn_mask/000000000724.json',
                       'Training annotations JSON file.')
tf.flags.DEFINE_string('val_annotations_file', r'C:\Users\TIAN\Desktop\mask_rcnn_mask/000000000724.json',
                       'Validation annotations JSON file.')
tf.flags.DEFINE_string('testdev_annotations_file', '',
                       'Test-dev annotations JSON file.')
tf.flags.DEFINE_string('output_dir', r'D:\python_program\tf_model_master\models-master\coco\coco_record_test', 'Output data directory.')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def create_tf_example(image,
                      annotations_list,
                      image_dir,
                      category_index,
                      include_masks=False):
  """Converts image and annotations to a tf.Example proto.

  Args:
    image: dict with keys:
      [u'license', u'file_name', u'coco_url', u'height', u'width',
      u'date_captured', u'flickr_url', u'id']
    annotations_list:
      list of dicts with keys:
      [u'segmentation', u'area', u'iscrowd', u'image_id',
      u'bbox', u'category_id', u'id']
      Notice that bounding box coordinates in the official COCO dataset are
      given as [x, y, width, height] tuples using absolute coordinates where
      x, y represent the top-left (0-indexed) corner.  This function converts
      to the format expected by the Tensorflow Object Detection API (which is
      which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
      to image size).
    image_dir: directory containing the image files.
    category_index: a dict containing COCO category information keyed
      by the 'id' field of each category.  See the
      label_map_util.create_category_index function.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
  Returns:
    example: The converted tf.Example
    num_annotations_skipped: Number of (invalid) annotations that were ignored.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  image_height = image['height']
  image_width = image['width']
  filename = image['file_name']
  image_id = image['id']

  full_path = os.path.join(image_dir, filename)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  key = hashlib.sha256(encoded_jpg).hexdigest()

  xmin = []
  xmax = []
  ymin = []
  ymax = []
  is_crowd = []
  category_names = []
  category_ids = []
  area = []
  encoded_mask_png = []
  num_annotations_skipped = 0
  for object_annotations in annotations_list:
    (x, y, width, height) = tuple(object_annotations['bbox'])
    if width <= 0 or height <= 0:
      num_annotations_skipped += 1
      continue
    if x + width > image_width or y + height > image_height:
      num_annotations_skipped += 1
      continue
    xmin.append(float(x) / image_width)
    xmax.append(float(x + width) / image_width)
    ymin.append(float(y) / image_height)
    ymax.append(float(y + height) / image_height)
    is_crowd.append(object_annotations['iscrowd'])
    category_id = int(object_annotations['category_id'])
    category_ids.append(category_id)
    category_names.append(category_index[category_id]['name'].encode('utf8'))
    area.append(object_annotations['area'])

    if include_masks:
      run_len_encoding = mask.frPyObjects(object_annotations['segmentation'],
                                          image_height, image_width)
      binary_mask = mask.decode(run_len_encoding)

      a = object_annotations['iscrowd']
      if a:
          print(a)

      if not object_annotations['iscrowd']:  # 多一个维度 1 维
        binary_mask = np.amax(binary_mask, axis=2)  # 这里是读取mask的，binary吗，
      pil_image = PIL.Image.fromarray(binary_mask)
      output_io = io.BytesIO()
      pil_image.save(output_io, format='PNG') #只是在内存当中保存
      # save_img = PIL.Image.fromarray(binary_mask*255) #要保存查看的图像
      # save_img.save("sada.png",format="PNG")
      encoded_mask_png.append(output_io.getvalue())

  feature_dict = {
      'image/height':
          dataset_util.int64_feature(image_height),
      'image/width':
          dataset_util.int64_feature(image_width),
      'image/filename':
          dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id':
          dataset_util.bytes_feature(str(image_id).encode('utf8')),
      'image/key/sha256':
          dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded':
          dataset_util.bytes_feature(encoded_jpg),
      'image/format':
          dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin':
          dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax':
          dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin':
          dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax':
          dataset_util.float_list_feature(ymax),
      'image/object/class/text':
          dataset_util.bytes_list_feature(category_names),
      'image/object/is_crowd':
          dataset_util.int64_list_feature(is_crowd),
      'image/object/area':
          dataset_util.float_list_feature(area),
  }
  if include_masks:
    feature_dict['image/object/mask'] = (
        dataset_util.bytes_list_feature(encoded_mask_png))
  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))  #这个实例好像直接放在一个列表当中，没有给每个实例标签，
  return key, example, num_annotations_skipped


def _create_tf_record_from_coco_annotations(
    annotations_file, image_dir, output_path, include_masks, num_shards):
  """Loads COCO annotation json files and converts to tf.Record format.

  Args:
    annotations_file: JSON file containing bounding box annotations.
    image_dir: Directory containing the image files.
    output_path: Path to output tf.Record file.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
    num_shards: number of output file shards.
  """
  with contextlib2.ExitStack() as tf_record_close_stack, \
      tf.gfile.GFile(annotations_file, 'r') as fid:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_path, num_shards)
    groundtruth_data = json.load(fid)
    #  for循环 按照图片来
    # 一张一张读，关键是创建字典,原始字典的定义，按图来
    image = {
        "file_name":"",
        "height":0,
        "width":0,
        "id":0,
    }
    annotations_list = []
    category_index_dict ={
         1: {'supercategory': 'person', 'id': 1, 'name': 'person'},
         2: {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'},
         3: {'supercategory': 'vehicle', 'id': 3, 'name': 'car'},
         4: {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'},
         5: {'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'},
         6: {'supercategory': 'vehicle', 'id': 6, 'name': 'bus'},
         7: {'supercategory': 'vehicle', 'id': 7, 'name': 'train'},
         8: {'supercategory': 'vehicle', 'id': 8, 'name': 'truck'},
         9: {'supercategory': 'vehicle', 'id': 9, 'name': 'boat'},
         10: {'supercategory': 'outdoor', 'id': 10, 'name': 'traffic light'},
         11: {'supercategory': 'outdoor', 'id': 11, 'name': 'fire hydrant'},
         }

    _annotation={
        "segmentation":[],
        "area":0,
        "iscrowd":0,
        "image_id":0,
        "bbox":[],
        "category_id":1,
        "id":0,
    }

    image["file_name"]=groundtruth_data["imagePath"]
    image["height"] = groundtruth_data["imageHeight"]
    image["width"] = groundtruth_data["imageWidth"]
    image["id"] = groundtruth_data["imagePath"][:-4]

    # 形成annotation字典
    def match_annotation(two_dimention_list):
        #匹配二位标签到1维
        one_dimention_list=[]
        for content in two_dimention_list:
            one_dimention_list.extend(content)
        return  [one_dimention_list]

    def get_bbox_from_seg(_matrix):
        # 标注的二维标签当中生成bbox
        xmin,ymin = np.min(_matrix,axis=0)
        xmax,ymax = np.max(_matrix,axis=0)
        # coco数据返回的组织方式：[x, y, width, height]所以这个地方要修正，因为后面会再处理，bbox的宽和高
        # 图是一样的，segmentation是不一样的
        return xmin,ymin,xmax-xmin,ymax-ymin  # 注意这个返回的顺序是不是能和coco返回的数据对上，最大最小，x轴y轴

    for inde,segmen in enumerate(groundtruth_data["shapes"]):
        # 这个segmentation 的 list的组织形式（主要是顺序）是不是和coco一样？？？还有bbox 的顺序很有可能影响最终结果的正确性
        _annotation['segmentation']=match_annotation(segmen["points"])
        _annotation["image_id"]=image["id"]
        _annotation["bbox"]=list(get_bbox_from_seg(np.asarray(segmen["points"]))) #这个bbox是根据语义分割的标签计算出来的
        _annotation["area"]=0
        _annotation["iscrowd"]=0
        # _annotation["category_id"]=1   # 类别需要做映射
        _annotation["id"]=int(image["id"]+str(inde))
        annotations_list.append(deepcopy(_annotation)) #因为拼接的是对象，所以不能直拼接（传递指针）导致拼接的字典全部相同


    category_index = category_index_dict
    _, tf_example, num_annotations_skipped = create_tf_example(
          image, annotations_list, image_dir, category_index, include_masks)
    num_of_images =1
    shard_idx = num_of_images % num_shards
    output_tfrecords[shard_idx].write(tf_example.SerializeToString())
    tf.logging.info('Finished writing, skipped %d annotations.')


def main(_):
  assert FLAGS.train_image_dir, '`train_image_dir` missing.'
  assert FLAGS.val_image_dir, '`val_image_dir` missing.'
  # assert FLAGS.test_image_dir, '`test_image_dir` missing.'
  assert FLAGS.train_annotations_file, '`train_annotations_file` missing.'
  assert FLAGS.val_annotations_file, '`val_annotations_file` missing.'
  # assert FLAGS.testdev_annotations_file, '`testdev_annotations_file` missing.'

  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)
  train_output_path = os.path.join(FLAGS.output_dir, 'coco_train.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'coco_val.record')
  # testdev_output_path = os.path.join(FLAGS.output_dir, 'coco_testdev.record')

  _create_tf_record_from_coco_annotations(
      FLAGS.train_annotations_file,
      FLAGS.train_image_dir,
      train_output_path,
      FLAGS.include_masks,
      num_shards=10)
  _create_tf_record_from_coco_annotations(
      FLAGS.val_annotations_file,
      FLAGS.val_image_dir,
      val_output_path,
      FLAGS.include_masks,
      num_shards=10)
  # _create_tf_record_from_coco_annotations(
  #     FLAGS.testdev_annotations_file,
  #     FLAGS.test_image_dir,
  #     testdev_output_path,
  #     FLAGS.include_masks,
  #     num_shards=10)


if __name__ == '__main__':
  tf.app.run()

```

标图生成的json文件如下：

```json
{
  "version": "3.16.2",
  "flags": {},
  "shapes": [
    {
      "label": "label1",
      "line_color": null,
      "fill_color": null,
      "points": [
        [
          185.11570247933884,
          98.17355371900827
        ],
        [
          96.68595041322314,
          122.14049586776859
        ],
        [
          108.25619834710744,
          191.5619834710744
        ],
        [
          246.2727272727273,
          208.0909090909091
        ],
        [
          329.74380165289256,
          168.4214876033058
        ],
        [
          332.22314049586777,
          118.00826446280992
        ],
        [
          241.3140495867769,
          90.73553719008265
        ],
        [
          177.6776859504132,
          91.56198347107438
        ]
      ],
      "shape_type": "polygon",
      "flags": {}
    },
    {
      "label": "labels2",
      "line_color": null,
      "fill_color": null,
      "points": [
        [
          352.88429752066116,
          279.1652892561984
        ],
        [
          176.02479338842977,
          282.4710743801653
        ],
        [
          80.98347107438016,
          346.93388429752065
        ],
        [
          146.27272727272728,
          436.1900826446281
        ],
        [
          242.96694214876032,
          485.77685950413223
        ],
        [
          288.4214876033058,
          386.60330578512395
        ],
        [
          399.9917355371901,
          448.58677685950414
        ],
        [
          403.297520661157,
          383.297520661157
        ],
        [
          430.57024793388433,
          331.2314049586777
        ],
        [
          442.14049586776866,
          301.4793388429752
        ],
        [
          397.5123966942149,
          275.8595041322314
        ]
      ],
      "shape_type": "polygon",
      "flags": {}
    },
    {
      "label": "labels3",
      "line_color": null,
      "fill_color": null,
      "points": [
        [
          315.6942148760331,
          506.43801652892563
        ],
        [
          479.33057851239676,
          474.20661157024796
        ],
        [
          457.84297520661164,
          532.8842975206612
        ],
        [
          322.3057851239669,
          585.7768595041323
        ],
        [
          187.59504132231405,
          570.0743801652893
        ],
        [
          250.40495867768595,
          527.099173553719
        ],
        [
          294.20661157024796,
          504.78512396694214
        ]
      ],
      "shape_type": "polygon",
      "flags": {}
    }
  ],
  "lineColor": [
    0,
    255,
    0,
    128
  ],
  "fillColor": [
    255,
    0,
    0,
    128
  ],
  "imagePath": "000000000285.jpg",
  "imageData": "/9j/4AAQSkZJ",#这个地方是base64编码的图像，太长我删掉了
  "imageHeight": 640,
  "imageWidth": 586
}
```

但是标记生成完成后有几个**需要十分注意的点还没有验证**：如代码注释，标记生成后**多边形的坐标的拼接顺序是不是和coco数据集一样**，另外就是**bbox的坐标位置**是否和coco的一样，这个可以通过保存成图像进行验证，主要就是要确保表完之后经过转换坐标位置确实不变，不会经过变换后图形不一样了，导致label变了。检验的方式就利用coco生成的mask的二维的0,1png图像来进行判断，确保按照coco的label的组织方式进行。

另外需要解决一个问题就是单标签的处理和多标签的出来在coco数据集当中的表现是到底是怎么样的，是不是存在一种情况同一个像素点有两个以上标签的情况。经过查阅maskrcnn的文献，文献当中明确表达了maskrcnn 在训练过程当中是将mask分类和物体分类相区别开的。也就是说mask就是0,1的标识，只标识背景和前景，不标识类别，类别由bbox回归来完成，因为作者发现这样做能够有效地提升模型的精度，防止mask在类之间竞争。

```python
def create_tf_example(image,
                      annotations_list,
                      image_dir,
                      category_index,
                      include_masks=False):
  """Converts image and annotations to a tf.Example proto.

  Args:
    image: dict with keys:
      [u'license', u'file_name', u'coco_url', u'height', u'width',
      u'date_captured', u'flickr_url', u'id']
    annotations_list:
      list of dicts with keys:
      [u'segmentation', u'area', u'iscrowd', u'image_id',
      u'bbox', u'category_id', u'id']
      Notice that bounding box coordinates in the official COCO dataset are
      given as [x, y, width, height] tuples using absolute coordinates where
      x, y represent the top-left (0-indexed) corner.  This function converts
      to the format expected by the Tensorflow Object Detection API (which is
      which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
      to image size).
```

注意上述代码当中的Notice部分，notice部分说官方的coco数据的bbox坐标是按照 [x, y, width, height]这种方式组织的，该函数会把图像的绝对值坐标转换为  [ymin, xmin, ymax, xmax] 组织的相对值坐标，所以说，图像坐标在生成的时候一定要按照官方的原始坐标形式组织，以免在生成数据时产生误差，造成结果偏移。关于mask的生成部分，按照labelme的多边形标注格式，直接输入转换tfrecord的代码，经过实际输出确认正确，也就表明了labelme标注的生成格式与coco标注的生成方法实际上是一样的。生成的方式如下图所示：即从起始点开始（图中红色区域最高点）， 沿逆时针方向排序每个点，每个点的坐标以【x，y】的形式组织，形成一个二维矩阵

![](./data/tf_object_detection配置以及源码解读.assets/mask的组织形式.png)

下图是经过转tfrecord验证之后的mask：

![](./data/tf_object_detection配置以及源码解读.assets/mask正确性验证.png)

可见红色区域与右侧mask区域是相符合的，说明解析过程没有问题，符合输入数据要求。为了验证算法训练是否可行，训练上述单张图像，保存model模型并用该model 预测这张图像，查看运行结果是否符合预期要求，如果这个图像训练经过训练能够过拟合，测出结果符合标记要求，基本能判定算法是符合要求的。下图给出自标数据集训练所得结果（请忽略标签，为了简化把图的标签设置为了1，也就是person）：主要关注图像当中的mask，以及bbox，bbox是根据mask标签计算得到，可见mask数据和bbox 数据在进入tfrecord之后是完全正确的。

![](./data/tf_object_detection 配置以及源码解读.assets/自标数据训练结果.png)

关于maskrcnn的训练方式，和object detection是完全一样的，只是改了配置文件，可以自己修改object detection的配置文件（修改的方式参照instance segmentatio.md），当然最直接的方式是采用官方给定的配置文件，比如我采用了mask_rcnn_inception_v2_coco.config作为pipline文件，采用train.py进行训练即可。

### 模型微调需要固定参数的详细方式

***下面的方式微调适用于legacy当中的train，modelmain的当中的可能是用hook实现吗，后面再具体研究***如果需要利用模型进行finetune，在恢复参数的基础上在训练时需要固定某些参数的权重，进行模型finetune，具体要固定哪些参数也是写到config当中，按照下述方式写：

```python
train_config: {
  freeze_variables:"FirstStageFeatureExtractor,SecondStageFeatureExtractor"
  batch_size: 1
  optimizer {
    momentum_optimizer: {
```

也就是按照上面红色的部分写入即可。代码当中：在legacy下的trainner当中：（model_main的话是在model_lib当中定义的）

```python
     # Optionally freeze some layers by setting their gradients to be zero.
      if train_config.freeze_variables:          # 通过把梯度设置为0 来固定某一层，相当于不让这一层训练
        grads_and_vars = variables_helper.freeze_gradients_matching_regex(
            grads_and_vars, train_config.freeze_variables)

      # Optionally clip gradients
      if train_config.gradient_clipping_by_norm > 0:
        with tf.name_scope('clip_grads'):
          grads_and_vars = slim.learning.clip_gradient_norms(
              grads_and_vars, train_config.gradient_clipping_by_norm)  # 截断梯度，推进训练过程
```

是用variables_helper.freeze_gradients_matching_regex实现的，再utils，variables_helper当中：

```python
 kept_vars = []
  variables_to_ignore_patterns = list([fre for fre in filter_regex_list if fre])
  for var in variables:
    add = True
    for pattern in variables_to_ignore_patterns:
      if re.match(pattern, var.op.name):
        add = False
        break
    if add != invert: # 如果不逆转，那么就保留这些变量，也就是说训练的就是这些变量
      kept_vars.append(var)
  return kept_vars
```

是用re正则匹配的方式去除掉固定不训练的节点的。

***关于图中到底恢复了哪些节点用于finetune***，恢复节点的设定是meta_architectures/faster_rcnn_meta_arch.py中定于（采用fasterRcnn架构），就是说恢复源节点的操作是在原结构的定义当中给出的。具体的：

```python
   if fine_tune_checkpoint_type not in ['detection', 'classification']:
      raise ValueError('Not supported fine_tune_checkpoint_type: {}'.format(
          fine_tune_checkpoint_type))
    if fine_tune_checkpoint_type == 'classification':
      return self._feature_extractor.restore_from_classification_checkpoint_fn(
          self.first_stage_feature_extractor_scope,
          self.second_stage_feature_extractor_scope)

    variables_to_restore = variables_helper.get_global_variables_safely()
    variables_to_restore.append(slim.get_or_create_global_step())
    # Only load feature extractor variables to be consistent with loading from
    # a classification checkpoint.
    include_patterns = None
    if not load_all_detection_checkpoint_vars:
      include_patterns = [
          self.first_stage_feature_extractor_scope,
          self.second_stage_feature_extractor_scope
      ]
    feature_extractor_variables = tf.contrib.framework.filter_variables(
        variables_to_restore, include_patterns=include_patterns)
    return {var.op.name: var for var in feature_extractor_variables}
```

也就是首先恢复原图，然后根据模式对节点进行过滤，模式有两类，self.first_stage_feature_extractor_scope,self.second_stage_feature_extractor_scope这两个变量的值其实就是：***FirstStageFeatureExtractor,SecondStageFeatureExtractor***，也就是计算图节点的前缀，然后根据节点过滤实际所需要恢复的节点，tf.contrib.framework.filter_variables(variables_to_restore, include_patterns=include_patterns)，过滤的这个实现实际上在：\anaconda\envs\tensorflow-gpu-1.12\Lib\site-packages\tensorflow\contrib\framework\python\ops\variables.py，也就是源码当中，也是通过re.match实现的。具体：

```python
def filter_variables(var_list,
                     include_patterns=None,
                     exclude_patterns=None,
                     reg_search=True):
  """Filter a list of variables using regular expressions.

  First includes variables according to the list of include_patterns.
  Afterwards, eliminates variables according to the list of exclude_patterns.

  For example, one can obtain a list of variables with the weights of all
  convolutional layers (depending on the network definition) by:
  
  variables = tf.contrib.framework.get_model_variables()
  conv_weight_variables = tf.contrib.framework.filter_variables(
      variables,
      include_patterns=['Conv'],
      exclude_patterns=['biases', 'Logits'])
  Args:
    var_list: list of variables.
    include_patterns: list of regular expressions to include. Defaults to None,
      which means all variables are selected according to the include rules. A
      variable is included if it matches any of the include_patterns.
    exclude_patterns: list of regular expressions to exclude. Defaults to None,
      which means all variables are selected according to the exclude rules. A
      variable is excluded if it matches any of the exclude_patterns.
    reg_search: boolean. If True (default), performs re.search to find matches
      (i.e. pattern can match any substring of the variable name). If False,
      performs re.match (i.e. regexp should match from the beginning of the
      variable name).

  Returns:
    filtered list of variables.
  """
  if reg_search:
    reg_exp_func = re.search
  else:
    reg_exp_func = re.match

# First include variables.

  if include_patterns is None:
    included_variables = list(var_list)
  else:
    included_variables = []
    for var in var_list:
      if any(reg_exp_func(ptrn, var.name) for ptrn in include_patterns):
        included_variables.append(var)

# Afterwards, exclude variables.

  if exclude_patterns is None:
    filtered_variables = included_variables
  else:
    filtered_variables = []
    for var in included_variables:
      if not any(reg_exp_func(ptrn, var.name) for ptrn in exclude_patterns):
        filtered_variables.append(var)

  return filtered_variables
```

 也就是说，实际上模型已经**默认恢复了两个阶段的特征的提取部分**，其他不必要的节点实际上是没有进行恢复的。说明，如果不想finetune 整个网络结构，那么我们就仅仅需要固定特征提取部分，也就是上面两个前缀。然后提高学习率进行训练。这样应该比用较小的学习率finetune整个网络收到的效果可能要好（没做实验确认，只是猜测。。。）。实际上经过试验之后，如过实际数据量足够多的情况下，微调整个网络的效果比只微调最后的层的效果要好。但是如果实际数据不足的情况下，固定前面的特征提取部分，也是能获得比较好的结果的。

目标检测的第一阶段的特征提取器在object_detection下的models里面。

## FasterRcnn模型整体的架构设计

![](./data/tf_object_detection配置以及源码解读.assets/fasterrcnn架构.png)

借助知乎大佬的图，实际上在object_detection的代码当中，faster网络的设计于上面思路大致相同，但是也存在一定的区别，前面网络特征提取部分可以随便换，所以没啥好说的，中间代码当中到下面3x3卷积部分是一样的，卷积完之后的图分为两个去向，一个去向是做后面裁剪或者roi_pooling（项目代码采用的处理实际上是crop_and_resize，由tf官方代码实现，与roi_pooling实际上有一定的区别），另外一部分用于去做bbox 的处理。比如第一阶段的nms的处理等。代码在第一阶段获取到特征层和encoding的bbox坐标（实际上就是论文当中所说的4个变换，不是真正的坐标，所以称为encoding的坐标，需要经过decode才能形成01之间的相对坐标）之后，做了第一阶段的nms，**所谓第一阶段的nms，实际上包括的内容比较多**：首先是将feature map上产生的超出图像边界的框给去掉，只留下在图像内部的有效的框，其次是利用softmax的结果对所框区域有没有物体进行初步判断（这就是这里为什么会有softmax的处理的原因），对框内没有物体的bbox给拿掉，然后在进一步计算和ground truth的ROI面积，进一步去掉一些没用的框，经过这三个处理之后，最后只留下config当中设置的第一阶段的最多产生的框的数量，**经过这样的处理， 网络的效率就会非常高，正负样本比例就会相对比较均衡，loss计算也就比容易。这是于一阶段算法差异最大的部分，也是拉开两者差距的主要原因**。目标检测最后所做的事情，主要是：经过上面的处理，形成第一次nms处理后的框，根据第一阶段的100（参数设置）个候选框，对该位置的feature map 进行crop_and_resize，就是先crop，裁剪获得到含有目标的区域之后，再进行resize，形成固定大小的包含目标的特征图，这些特征图再进行最后的卷积，也就是上面紫色和绿色部分。实际上代码的实现和上面的图不一样，代码当中是做了bottle_neck结构*3的卷积层的卷积，也就是卷积内容相对还是比较复杂的。卷积完之后的图再去做最后的分支预测，一部分bbox，另一部分是cls。代码当中bbox 实际上就做了全局均值池化，平展拉伸，全连接，dropout（参数设置没有采用），输出encoding的bbox 。完成目标检测的预测过程。其中faster_rcnn_meta_arc其他部分，包括target_assigner（看起来是两个阶段都会用到）等（匹配候选框和ground_truth），主要是用于训练过程的loss计算。

## 模型config文件的配置（包括微调，预处理的写法）

以resnet101为例：

``` python
# Faster R-CNN with Resnet-50 (v1), configuration for MSCOCO Dataset.
# Users should configure the fine_tune_checkpoint field in the train config as
# well as the label_map_path and input_path fields in the train_input_reader and
# eval_input_reader. Search for "PATH_TO_BE_CONFIGURED" to find the fields that
# should be configured.

model {
  faster_rcnn {
    num_classes: 1
    image_resizer {
      keep_aspect_ratio_resizer {
        min_dimension: 1200 #网络训练时真正的图像尺寸
        max_dimension: 1388 #网络训练时真正的图像尺寸
      }
    }
    hard_example_miner{    #困难样本挖掘的算法设置
        loss_type:0      # both cls loc (0,1,2三选一) 默认是both(0) 所以不用填
        max_negatives_per_positive:5   #   正负样本的比例，一个正正样本最多对应多少负样本，默认是0，也就是None，，也就是不限制
        num_hard_examples:64      #每幅图像所选择的最大的hard数量，默认是64
        iou_threshold:0.8          #默认0.7
        # classification_weight 没有这个域，默认0.05 ，实际采用的是config中的second_stage_classification_loss_weight
        #  localization_weight	类似同上
        min_negatives_per_image:2   #默认是0，每幅图的最小的负样本的选择数量，如果这个 值大于0，即使这个图没有正样本anchor，这个图也会采样

    }
    feature_extractor {
      type: 'faster_rcnn_resnet101' #第一阶段特征提取层，可用结构：resnet18,50,101,152，										#inceptionv3  当然结构可以改
      first_stage_features_stride: 16 #第一阶段的卷积之后的图像与缩小为原始图像的倍数，只能是16或									   #8（否则代码会报确认错误） 
    }
    first_stage_anchor_generator {
        #anchor 的生成函数，可用的还有：multiple_grid_anchor_generator
        						#multiscale_grid_anchor_generator
        						#flexible_grid_anchor_generator
      grid_anchor_generator {     
        scales: [0.25, 0.5, 1.0, 2.0]
        aspect_ratios: [0.5, 1.0, 2.0]
        height_stride: 16
        width_stride: 16
      }
    }
    first_stage_box_predictor_conv_hyperparams {
      op: CONV
      regularizer {
        l2_regularizer {
          weight: 0.0
        }
      }
      initializer {
        truncated_normal_initializer {
          stddev: 0.01
        }
      }
    }
    first_stage_nms_score_threshold: 0.0
    first_stage_nms_iou_threshold: 0.7
    first_stage_max_proposals: 300
    first_stage_localization_loss_weight: 2.0
    first_stage_objectness_loss_weight: 1.0  #上述都是第一阶段的nms的处理过程
    initial_crop_size: 14
    maxpool_kernel_size: 2
    maxpool_stride: 2
    second_stage_box_predictor {
      mask_rcnn_box_predictor {
        use_dropout: false      #第二阶段的bbox的坐标预测的，从源码看，不建议开启dropout
        dropout_keep_probability: 1.0
        fc_hyperparams {
          op: FC
          regularizer {
            l2_regularizer {
              weight: 0.0
            }
          }
          initializer {
            variance_scaling_initializer {
              factor: 1.0
              uniform: true
              mode: FAN_AVG
            }
          }
        }
      }
    }
    second_stage_post_processing {
      batch_non_max_suppression {
        score_threshold: 0.0
        iou_threshold: 0.6
        max_detections_per_class: 100 #每个类别最多保存的框
        max_total_detections: 300
      }
      score_converter: SOFTMAX
    }
    second_stage_localization_loss_weight: 2.0
    second_stage_classification_loss_weight: 1.0
  }
}


train_config: {
  freeze_variables:"FirstStageFeatureExtractor,SecondStageFeatureExtractor"# 固定这两个阶段的特征提取器不变
  batch_size: 1
  
    #优化器，在builders/optimizer_builder当中定义，可选的优化器有：
    # adam_optimizer，momentum_optimizer，rms_prop_optimizer，配置
    #参数参考源代码
  optimizer {            								 			
    momentum_optimizer: {
      learning_rate: {
        manual_step_learning_rate {
          initial_learning_rate: 0.0003
          schedule {
            step: 900000
            learning_rate: .00003
          }
          schedule {
            step: 1200000
            learning_rate: .000003
          }
        }
      }
      momentum_optimizer_value: 0.9
    }
    use_moving_average: false
  }
  gradient_clipping_by_norm: 10.0
  fine_tune_checkpoint: "D:/trained_model/baidu_model/saved_model/model.ckpt"
  from_detection_checkpoint: true
  # Note: The below line limits the training process to 200K steps, which we
  # empirically found to be sufficient enough to train the pets dataset. This
  # effectively bypasses the learning rate schedule (the learning rate will
  # never decay). Remove the below line to train indefinitely.
  num_steps: 200000
   
  # 数据增强的配置,可选的方式有： 
    #NormalizeImage normalize_image = 1;
	#RandomHorizontalFlip random_horizontal_flip = 2;
	#RandomPixelValueScale random_pixel_value_scale = 3;
	#RandomImageScale random_image_scale = 4;
	#RandomRGBtoGray random_rgb_to_gray = 5;
	#RandomAdjustBrightness random_adjust_brightness = 6;
	#RandomAdjustContrast random_adjust_contrast = 7;
	#RandomAdjustHue random_adjust_hue = 8;
	#RandomAdjustSaturation random_adjust_saturation = 9;
	#RandomDistortColor random_distort_color = 10;
	#RandomJitterBoxes random_jitter_boxes = 11;
	#RandomCropImage random_crop_image = 12;
	#RandomPadImage random_pad_image = 13;
	#RandomCropPadImage random_crop_pad_image = 14;
	#RandomCropToAspectRatio random_crop_to_aspect_ratio = 15;
	#RandomBlackPatches random_black_patches = 16;
	#RandomResizeMethod random_resize_method = 17;
	#ScaleBoxesToPixelCoordinates scale_boxes_to_pixel_coordinates = 18;
	#ResizeImage resize_image = 19;
	#SubtractChannelMean subtract_channel_mean = 20;
	#SSDRandomCrop ssd_random_crop = 21;
	#SSDRandomCropPad ssd_random_crop_pad = 22;
	#SSDRandomCropFixedAspectRatio ssd_random_crop_fixed_aspect_ratio = 23;
    #其他更多的方法可以参考 proto/preprocessor.proto
    #详细的参数设置可以参考 object_detection/builders/preprocessor_builder_test.py
    #源代码在object_detection/core/preprocessor.py
    # 例如：最牛逼的采用 AutoAugmentImage autoaugment_image = 31;自动全家桶的方式
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    random_vertical_flip {
    }
  }
  data_augmentation_options {
    random_rotation90 {
    }
  }
  data_augmentation_options {
    random_adjust_brightness {
      max_delta: 0.5
    }
  }
  data_augmentation_options {
    random_adjust_contrast {
      min_delta: 0.7
      max_delta: 1.1
    }
  }
  data_augmentation_options {
    random_adjust_saturation {
      min_delta: 0.75
      max_delta: 1.15
    }
  }
    
  data_augmentation_options {
    autoaugment_image {
      policy_name: 'v0' #采用自动全部数据增强的方式（coco数据集表现比较好的方式，还有v1，v2，v3）
    }
  }
    
}

train_input_reader: {
  tf_record_input_reader {
    input_path: "D:/dataset/black_point/tfrecord_ori/coco_train.record-?????-of-00005"
  }
  label_map_path: "D:/dataset/black_point/tfrecord_ori/clm_label_map.pbtxt"
}

eval_config: {
  num_examples: 8000
  # Note: The below line limits the evaluation process to 10 evaluations.
  # Remove the below line to evaluate indefinitely.
  max_evals: 10
   #每隔多长时间eval一次
  eval_interval_secs:600
}

eval_input_reader: {
  tf_record_input_reader {
    input_path: "D:/dataset/black_point/tfrecord_ori/coco_val.record-?????-of-00005"
  }
  label_map_path:"D:/dataset/black_point/tfrecord_ori/clm_label_map.pbtxt"
  shuffle: false
  num_readers: 1
}







```

数据增强可以参考链接[数据增强](https://stackoverflow.com/questions/44906317/what-are-possible-values-for-data-augmentation-options-in-the-tensorflow-object).如果想更方便地复制预处理参数，可以**参考builders/preprocessor_builder_test**另外**关于keep_aspect_ratio_resizer**,这个resizer实际上采用的是core/preprocessor当中的resize_to_range函数处理的输入图像，这个函数的resize方法跟常用的resize方法是不一样的，其解释如下：也就是说，先对图像进行缩放，使得图像的小边到预设定的min_dimension，如果此时图像大边没有超过max_dimension，那么这么缩放，如果缩放小边之后，大边超过了max_dimension,那么就缩放大边到最大尺寸。总之最大尺寸不能超过max_dimension.也就是：min(new_height, new_width) == min_dimension 或者max(new_height, new_width) == max_dimension.就是保持纵横比形式的缩放，从源代码的图像缩放的操作来看，就是保持宽高的最小的纵横比进行缩放（如果没有其他额外操作的话）

```python
def resize_to_range(image,
                    masks=None,
                    min_dimension=None,
                    max_dimension=None,
                    method=tf.image.ResizeMethod.BILINEAR,
                    align_corners=False,
                    pad_to_max_dimension=False,
                    per_channel_pad_value=(0, 0, 0)):
  """Resizes an image so its dimensions are within the provided value.

  The output size can be described by two cases:
  1. If the image can be rescaled so its minimum dimension is equal to the
     provided value without the other dimension exceeding max_dimension,
     then do so.
  2. Otherwise, resize so the largest dimension is equal to max_dimension.

  Args:
    image: A 3D tensor of shape [height, width, channels]
    masks: (optional) rank 3 float32 tensor with shape
           [num_instances, height, width] containing instance masks.
    min_dimension: (optional) (scalar) desired size of the smaller image
                   dimension.
    max_dimension: (optional) (scalar) maximum allowed size
                   of the larger image dimension.
    method: (optional) interpolation method used in resizing. Defaults to
            BILINEAR.
    align_corners: bool. If true, exactly align all 4 corners of the input
                   and output. Defaults to False.
    pad_to_max_dimension: Whether to resize the image and pad it with zeros
      so the resulting image is of the spatial size
      [max_dimension, max_dimension]. If masks are included they are padded
      similarly.
    per_channel_pad_value: A tuple of per-channel scalar value to use for
      padding. By default pads zeros.

  Returns:
    Note that the position of the resized_image_shape changes based on whether
    masks are present.
    resized_image: A 3D tensor of shape [new_height, new_width, channels],
      where the image has been resized (with bilinear interpolation) so that
      min(new_height, new_width) == min_dimension or
      max(new_height, new_width) == max_dimension.
    resized_masks: If masks is not None, also outputs masks. A 3D tensor of
      shape [num_instances, new_height, new_width].
    resized_image_shape: A 1D tensor of shape [3] containing shape of the
      resized image.

  Raises:
    ValueError: if the image is not a 3D tensor.
  """
  if len(image.get_shape()) != 3:
    raise ValueError('Image should be 3D tensor')

  def _resize_landscape_image(image):
    # resize a landscape image
    return tf.image.resize_images(
        image, tf.stack([min_dimension, max_dimension]), method=method,
        align_corners=align_corners, preserve_aspect_ratio=True)

  def _resize_portrait_image(image):
    # resize a portrait image
    return tf.image.resize_images(
        image, tf.stack([max_dimension, min_dimension]), method=method,
        align_corners=align_corners, preserve_aspect_ratio=True)

  with tf.name_scope('ResizeToRange', values=[image, min_dimension]):
    if image.get_shape().is_fully_defined():
      if image.get_shape()[0] < image.get_shape()[1]:
        new_image = _resize_landscape_image(image)
      else:
        new_image = _resize_portrait_image(image)
      new_size = tf.constant(new_image.get_shape().as_list())
    else:
      new_image = tf.cond(
          tf.less(tf.shape(image)[0], tf.shape(image)[1]),
          lambda: _resize_landscape_image(image),
          lambda: _resize_portrait_image(image))
      new_size = tf.shape(new_image)

    if pad_to_max_dimension:
      channels = tf.unstack(new_image, axis=2)
      if len(channels) != len(per_channel_pad_value):
        raise ValueError('Number of channels must be equal to the length of '
                         'per-channel pad value.')
      new_image = tf.stack(
          [
              tf.pad(
                  channels[i], [[0, max_dimension - new_size[0]],
                                [0, max_dimension - new_size[1]]],
                  constant_values=per_channel_pad_value[i])
              for i in range(len(channels))
          ],
          axis=2)
      new_image.set_shape([max_dimension, max_dimension, 3])

    result = [new_image]
    if masks is not None:
      new_masks = tf.expand_dims(masks, 3)
      new_masks = tf.image.resize_images(
          new_masks,
          new_size[:-1],
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
          align_corners=align_corners)
      if pad_to_max_dimension:
        new_masks = tf.image.pad_to_bounding_box(
            new_masks, 0, 0, max_dimension, max_dimension)
      new_masks = tf.squeeze(new_masks, 3)
      result.append(new_masks)

    result.append(new_size)
    return result
```

## train.py和eval.py同时使用（train训练，eval评估）

这里解释如何采用train.py进行训练并同时使用eval.py查看训练效果。实际上两个文件不能同时放在一个gpu上运行，即使设定运行时的显存占用，也不能同时放在一个gpu上运行，因为，没有办法注册op到设备上（除非有两个显卡，各运行各的，但是一个显卡能部署两个模型？这个地方有问题）。按照下面的方式修改gpu占用，还会导致cudnn kernel启动失败(修改trainner.py)：

```python
# 控制gpu option 为eval 留出空间
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95) #从测试看，加这句话会导致cudnn 启动失败
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    # gpu_options=gpu_options,
                                    log_device_placement=False)
```

​		另外，model_main.py虽然看起来像是将两个脚本集合到了一起，能够在train的时候eval，实际上并不是并行执行，在eval的时候train并不执行。这也是modelmain的一大缺点，eval会影响train的速度。所以说，train和eval不能在同一个gpu上进行，对于单卡的机器，想在train的时候eval需要将eval放在cpu上计算。也就是屏蔽掉cudnn。只需要在eval.py文件的开始加上：os.environ["CUDA_VISIBLE_DEVICES"]="-1"，屏蔽掉可见gpu就行，这样tf在运行当中会把计算节点注册到cpu上， 就不会和gpu产生冲突，这样的话，就可以同时运行两个文件，更爽的是，eval.py采用的是监听的形式执行，也就是说，脚本启动后会处于一种监听状态，每隔一段时间就执行一次eval，而不需要手动启动，并且脚本会把eval的结果写到tensorboard的events当中，这样的话，就可以实现tensorboard 监听训练效果了，并且eval实际上只是一个推理过程，不会想训练一样大量占用cpu，所以计算资源占用情况也可以，不是很严重。eval的间隔时间设置写在配置文件的eval config 当中，如上所述，所以说，这样操作的话，整体上就非常方便了，甚至能够**同时监听训练状态的loss下降，eval过程的loss下降以及bbox的iou精度等等**。同时实现训练和eval互不影响，eval也不影响train 的训练速度。

# 为模型加入EfficientNet

efficient net已经加入到模型库当中，在目标检测中作为主干网，模型加入的位置在：object_detection/builders/model_builder.py文件中。代码如下：

```python
 frcnn_inc_res.FasterRCNNInceptionResnetV2FeatureExtractor,
    'faster_rcnn_inception_v2':
    frcnn_inc_v2.FasterRCNNInceptionV2FeatureExtractor,
    'faster_rcnn_resnet18':
        frcnn_resnet_v1.FasterRCNNResnet18FeatureExtractor,

    "faster_rcnn_efficientnet":
        faster_rcnn_efficientnet_feature_extractor.FasterRCNNEfficientnetFeatureExtractor,

    'faster_rcnn_resnet50':
    frcnn_resnet_v1.FasterRCNNResnet50FeatureExtractor,
    'faster_rcnn_resnet101':
    frcnn_resnet_v1.FasterRCNNResnet101FeatureExtractor,
```

具体其他详细的加入方式在代码中体现。不在此赘述。默认的主干网结构是b8。如下(\object_detection\models\faster_rcnn_efficientnet_feature_extractor.py)：

```python
 rpn_feature_map, end_points = efficientnet_builder.build_model(
            images = preprocessed_inputs,
            model_name = "efficientnet-b8",
            training = self._is_training,
            override_params=None,
            model_dir=None,
            fine_tuning=False,
            features_only=True,
            pooled_features_only=False)
```

具体的调用方式，把模型的config 文件的中的特征提取部分改成：faster_rcnn_efficientnet即可。

## 加入随机裁剪之后报错的更改：

```python
def to_absolute_coordinates(boxlist,
                            height,
                            width,
                            check_range=True,
                            maximum_normalized_coordinate=1.1,
                            scope=None):
```

修改\object_detection\core\box_list_ops.py的默认参数check_range为False。这个只是目前的解决方式。根本原因似乎是标注数据里的标签太小导致的（有待进一步验证）。














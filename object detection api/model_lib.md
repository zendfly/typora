# model_lib

## MODEL_BUILD_UTIL_MAP

映射到到utils中，以帮助创建模型

utils --> config_util 

## config_util

用于读取和跟新配置文件的函数（Functions for reading and updating configuration files）

```python
get_configs_from_pipeline_file(pipeline_config_path, config_override=None):
    # 读取pipline_config_path中赌读取配文件，以字典形式返回
 	"""Reads config from a file containing pipeline_pb2.TrainEvalPipelineConfig.

  Args:
    pipeline_config_path: Path to pipeline_pb2.TrainEvalPipelineConfig text
      proto.
      pipline_bps.TrainEvalPipelineConfig 是.proto格式，在researh/object_detection/protos路径下
    config_override: A pipeline_pb2.TrainEvalPipelineConfig text proto to
      override pipeline_config_path.

  Returns:
    Dictionary of configuration objects. Keys are `model`, `train_config`,
      `train_input_config`, `eval_config`, `eval_input_config`. Value are the
      corresponding config objects.
  """
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(pipeline_config_path, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)
  if config_override:
    text_format.Merge(config_override, pipeline_config)
  return create_configs_from_pipeline_proto(pipeline_config)
```

## inputs

创建输入环境，包括 tarin(训练)、evi(环境)、prideted(预测)输入环境
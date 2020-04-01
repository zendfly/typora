# object detection api训练、推理

## 数据转换



## 配置文件设置

在object_detection/samples/configs路径下，以faster_rcnn_resnet50_coco_a.config文件为例。通常需要设置训练、验证数据路径，label_map、预训练模型等设置。

```python
# Faster R-CNN with Resnet-50 (v1), configuration for MSCOCO Dataset.
# Users should configure the fine_tune_checkpoint field in the train config as
# well as the label_map_path and input_path fields in the train_input_reader and
# eval_input_reader. Search for "PATH_TO_BE_CONFIGURED" to find the fields that
# should be configured.

model {
  faster_rcnn {
    num_classes: 5			# 列别数，不包含背景
    image_resizer {
      keep_aspect_ratio_resizer {
        min_dimension: 600
        max_dimension: 1024
      }
    }
    feature_extractor {		# 基础网络
      type: 'faster_rcnn_resnet50'
      first_stage_features_stride: 16
    }
    first_stage_anchor_generator {		# 第一阶段的anchor生成参数
      grid_anchor_generator {
        scales: [0.25, 0.5, 1.0, 2.0]
        aspect_ratios: [0.5, 1.0, 2.0]
        height_stride: 16
        width_stride: 16
      }
    }
    first_stage_box_predictor_conv_hyperparams {		# 卷积层超参数
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
    first_stage_objectness_loss_weight: 1.0
    initial_crop_size: 14
    maxpool_kernel_size: 2
    maxpool_stride: 2
    second_stage_box_predictor {
      mask_rcnn_box_predictor {
        use_dropout: false
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
        max_detections_per_class: 100
        max_total_detections: 300
      }
      score_converter: SOFTMAX
    }
    second_stage_localization_loss_weight: 2.0
    second_stage_classification_loss_weight: 1.0
  }
}

train_config: {
  batch_size: 1
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
  # fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"			# 预训练模型路径
  # from_detection_checkpoint: false			# Ture 则使用预训练，False 则不使用
  # Note: The below line limits the training process to 200K steps, which we
  # empirically found to be sufficient enough to train the pets dataset. This
  # effectively bypasses the learning rate schedule (the learning rate will
  # never decay). Remove the below line to train indefinitely.
  num_steps: 200000			# 训练步数
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
}

train_input_reader: {		# 训练集
  tf_record_input_reader {
    input_path: "D:/models-master/models/research/object_detection/fit_data/pet_train.record"	
  }
  label_map_path: "D:/models-master/models/research/object_detection/fit_data/pet_label_map.pbtxt"
}

eval_config: {		# 验证相关参数
  num_examples: 8000		# 验证集样本数
  # Note: The below line limits the evaluation process to 10 evaluations.
  # Remove the below line to evaluate indefinitely.
  max_evals: 10		
}

eval_input_reader: {		# 验证集
  tf_record_input_reader {
    input_path: "D:/models-master/models/research/object_detection/fit_data/pet_evl.record"
  }
  label_map_path: "D:/models-master/models/research/object_detection/fit_data/pet_label_map.pbtxt"
  shuffle: false	# 验证集是否打乱
  num_readers: 1
}

```



## train.py 设置

在object_detection/legacy/train.py文件，在fiags参数中，需要对train_dir(train place)、pipeline_config_dir(.config路径)路径设置。

```python
This executable is used to train DetectionModels. There are two ways of
configuring the training job:

1) A single pipeline_pb2.TrainEvalPipelineConfig configuration file
can be specified by --pipeline_config_path.

Example usage:
    ./train \
        --logtostderr \
        --train_dir=path/to/train_dir \		# 可以在train.py中直接指定
        --pipeline_config_path=pipeline_config.pbtxt

2) Three configuration files can be provided: a model_pb2.DetectionModel
configuration file to define what type of DetectionModel is being trained, an
input_reader_pb2.InputReader file to specify what training data will be used and
a train_pb2.TrainConfig file to configure training parameters.

Example usage:
    ./train \
        --logtostderr \
        --train_dir=path/to/train_dir \
        --model_config_path=model_config.pbtxt \
        --train_config_path=train_config.pbtxt \
        --input_config_path=train_input_config.pbtxt
"""
```

在train_dir路径中，会生成.ckpt 二进制文件，用于存储weights,biases,gradients等变量

.chpt.meta

包含元图，即计算图的结构，没有变量值。变量保存在.chpt.data文件

.chpt.data

保存变量，没有结构。

.chpt.index



## export_inferenct_detection.py设置

将训练好的模型转换成.bp格式，用于模型部署。在object_detection路径下的export_inference_graph.py

导出模型时，可以根据train_dir中保存的模型进行导出。导出需要.config、.ckpt等文件。

```python
python export_inference_graph \
    --input_type image_tensor \
    --pipeline_config_path path/to/ssd_inception_v2.config \
    --trained_checkpoint_prefix path/to/model.ckpt \
    --output_directory path/to/exported_model_directory
    """
    其中， input_type不需要更改
    	  pipeline_config_path 训练时的.config文件
    	  trained_checkpoint_prefix 需要导出的model_ckpt.mate文件
    	  output_directory 导出的路径
    """
```

生成后，会生frozen_inference_graph.bp和saved_model/saved_model.bp两个bp文件。



## modeled_detection.py 设置




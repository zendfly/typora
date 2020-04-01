# 在object detection api 架构下构建faster rcnn resnet18模型

构建ResNet18的Faster Rcnn网络，就是修改faster Rcnn的特征提取部分(feature_extrator)。

Object Detection Api(后面缩写为，ODA)，在object_detection/builders/model_builder.py文件中指定各种网络的组件。在model_builder.py文件中提供SSD和Faster Rcnn的一些列默认网络模型map，如下所示：

```python

# A map of names to SSD feature extractors.
SSD_FEATURE_EXTRACTOR_CLASS_MAP = {
    'ssd_inception_v2': SSDInceptionV2FeatureExtractor,
    'ssd_inception_v3': SSDInceptionV3FeatureExtractor,
    'ssd_mobilenet_v1': SSDMobileNetV1FeatureExtractor,
    'ssd_mobilenet_v1_fpn': SSDMobileNetV1FpnFeatureExtractor,
    'ssd_mobilenet_v1_ppn': SSDMobileNetV1PpnFeatureExtractor,
    'ssd_mobilenet_v2': SSDMobileNetV2FeatureExtractor,
    'ssd_mobilenet_v2_fpn': SSDMobileNetV2FpnFeatureExtractor,
    'ssd_mobilenet_v3_large': SSDMobileNetV3LargeFeatureExtractor,
    'ssd_mobilenet_v3_small': SSDMobileNetV3SmallFeatureExtractor,
    'ssd_mobilenet_edgetpu': SSDMobileNetEdgeTPUFeatureExtractor,
    'ssd_resnet50_v1_fpn': ssd_resnet_v1_fpn.SSDResnet50V1FpnFeatureExtractor,
    'ssd_resnet101_v1_fpn': ssd_resnet_v1_fpn.SSDResnet101V1FpnFeatureExtractor,
    'ssd_resnet152_v1_fpn': ssd_resnet_v1_fpn.SSDResnet152V1FpnFeatureExtractor,
    'ssd_resnet50_v1_ppn': ssd_resnet_v1_ppn.SSDResnet50V1PpnFeatureExtractor,
    'ssd_resnet101_v1_ppn':
        ssd_resnet_v1_ppn.SSDResnet101V1PpnFeatureExtractor,
    'ssd_resnet152_v1_ppn':
        ssd_resnet_v1_ppn.SSDResnet152V1PpnFeatureExtractor,
    'embedded_ssd_mobilenet_v1': EmbeddedSSDMobileNetV1FeatureExtractor,
    'ssd_pnasnet': SSDPNASNetFeatureExtractor,
}

SSD_KERAS_FEATURE_EXTRACTOR_CLASS_MAP = {
    'ssd_mobilenet_v1_keras': SSDMobileNetV1KerasFeatureExtractor,
    'ssd_mobilenet_v1_fpn_keras': SSDMobileNetV1FpnKerasFeatureExtractor,
    'ssd_mobilenet_v2_keras': SSDMobileNetV2KerasFeatureExtractor,
    'ssd_mobilenet_v2_fpn_keras': SSDMobileNetV2FpnKerasFeatureExtractor,
    'ssd_resnet50_v1_fpn_keras':
        ssd_resnet_v1_fpn_keras.SSDResNet50V1FpnKerasFeatureExtractor,
    'ssd_resnet101_v1_fpn_keras':
        ssd_resnet_v1_fpn_keras.SSDResNet101V1FpnKerasFeatureExtractor,
    'ssd_resnet152_v1_fpn_keras':
        ssd_resnet_v1_fpn_keras.SSDResNet152V1FpnKerasFeatureExtractor,
}

# A map of names to Faster R-CNN feature extractors.
FASTER_RCNN_FEATURE_EXTRACTOR_CLASS_MAP = {
    'faster_rcnn_nas':
    frcnn_nas.FasterRCNNNASFeatureExtractor,
    'faster_rcnn_pnas':
    frcnn_pnas.FasterRCNNPNASFeatureExtractor,
    'faster_rcnn_inception_resnet_v2':
    frcnn_inc_res.FasterRCNNInceptionResnetV2FeatureExtractor,
    'faster_rcnn_inception_v2':
    frcnn_inc_v2.FasterRCNNInceptionV2FeatureExtractor,
    'faster_rcnn_resnet50':
    frcnn_resnet_v1.FasterRCNNResnet50FeatureExtractor,
    'faster_rcnn_resnet101':
    frcnn_resnet_v1.FasterRCNNResnet101FeatureExtractor,
    'faster_rcnn_resnet152':
    frcnn_resnet_v1.FasterRCNNResnet152FeatureExtractor,
}

FASTER_RCNN_KERAS_FEATURE_EXTRACTOR_CLASS_MAP = {
    'faster_rcnn_inception_resnet_v2_keras':
    frcnn_inc_res_keras.FasterRCNNInceptionResnetV2KerasFeatureExtractor,
}

```

本次使用Faster Rcnn系列，故需要在`FASTER_RCNN_FEATURE_EXTRACTOR_CLASS_MAP`中加入自己的特征提取器(feature_extractor)。

train.py中的：

```python
  	# 创建模型
    model_fn = functools.partial(
      model_builder.build,          # 调用model_builder.build 进行创建模型
      model_config=model_config,
      is_training=True)
```

```python
model_builder.build(model_config, is_training, add_summaries=True)：

META_ARCHITECURE_BUILDER_MAP = {
    'ssd': _build_ssd_model,
    'faster_rcnn': _build_faster_rcnn_model,
    'experimental_model': _build_experimental_model
}

def build(model_config, is_training, add_summaries=True):
"""Builds a DetectionModel based on the model config.
  	 根据model_config构建模型

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

  meta_architecture = model_config.WhichOneof('model')	# 

  if meta_architecture not in META_ARCHITECURE_BUILDER_MAP:
    raise ValueError('Unknown meta architecture: {}'.format(meta_architecture))
  else:
    # 根据meta_architecture进行调用_build_faster_rcnn_model，
    build_func = META_ARCHITECURE_BUILDER_MAP[meta_architecture]
    return build_func(getattr(model_config, meta_architecture), is_training,
                      add_summaries)

```

`_build_faster_rcnn_model`中根据`feature_extractor`调用`_build_faster_rcnn_feature_extractor()`

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
    # 根据.feature_extractor特征提取网络调用_build_faster_rcnn_feature_extractor
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
      use_static_shapes=use_static_shapes,
      use_partitioned_nms=frcnn_config.use_partitioned_nms_in_first_stage,
      use_combined_nms=frcnn_config.use_combined_nms_in_first_stage)
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

  hard_example_miner = None
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
      'resize_masks': frcnn_config.resize_masks,
      'return_raw_detections_during_predict': (
          frcnn_config.return_raw_detections_during_predict)
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

在`_build_faster_rcnn_feature_extractor`函数中，根据特征网络名称，调用`FASTER_RCNN_FEATURE_EXTRACTOR_CLASS_MAP`，
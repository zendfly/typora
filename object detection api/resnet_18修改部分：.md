# 修改部分：

1、resnet_v1.py 添加（284行），在bolck4中，将stride更改为2

```python
def resnet_v1_18(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 store_non_strided_activations=False,
                 min_base_depth=8,
                 depth_multiplier=1,
                 reuse=None,
                 scope='resnet_v1_50'):
  """ResNet-18 model of [1]. See resnet_v1() for arg and return description."""
  depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
  blocks = [
      resnet_v1_block('block1', base_depth=depth_func(64), num_units=2,
                      stride=2),
      resnet_v1_block('block2', base_depth=depth_func(128), num_units=2,
                      stride=2),
      resnet_v1_block('block3', base_depth=depth_func(256), num_units=2,
                      stride=2),
      resnet_v1_block('block4', base_depth=depth_func(512), num_units=2,
                      stride=2),
  ]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=store_non_strided_activations,
                   reuse=reuse, scope=scope)
resnet_v1_18.default_image_size = resnet_v1.default_image_size
```

2、resnet_v1.py 中的 `bottleneck()`函数中，注释掉black的重建层，并修改scope值（125行）

```python

@slim.add_arg_scope
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               rate=1,
               outputs_collections=None,
               scope=None,
               use_bounded_activations=False):
  """Bottleneck residual unit variant with BN after convolutions.

  This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
  its definition. Note that we use here the bottleneck variant which has an
  extra bottleneck layer.

  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.
    use_bounded_activations: Whether or not to use bounded activations. Bounded
      activations better lend themselves to quantized inference.

  Returns:
    The ResNet unit's output.
  """
  with tf.compat.v1.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv2d(
          inputs,
          depth, [1, 1],
          stride=stride,
          activation_fn=tf.nn.relu6 if use_bounded_activations else None,
          scope='shortcut')

    residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1,
                           scope='conv1')
    # residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
    #                                     rate=rate, scope='conv2')     # resnet18，black只有两层，故注释掉中间层
    residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                           activation_fn=None, scope='conv2')

    if use_bounded_activations:
      # Use clip_by_value to simulate bandpass activation.
      residual = tf.clip_by_value(residual, -6.0, 6.0)
      output = tf.nn.relu6(shortcut + residual)
    else:
      output = tf.nn.relu(shortcut + residual)

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)

```

3、在model.builder.py的`FASTER_RCNN_FEATURE_EXTRACTOR_CLASS_MAP{}`字典中添加'faster_rcnn_resnet18'key和valu:

```python
FASTER_RCNN_FEATURE_EXTRACTOR_CLASS_MAP = {
    'faster_rcnn_nas':
    frcnn_nas.FasterRCNNNASFeatureExtractor,
    'faster_rcnn_pnas':
    frcnn_pnas.FasterRCNNPNASFeatureExtractor,
    'faster_rcnn_inception_resnet_v2':
    frcnn_inc_res.FasterRCNNInceptionResnetV2FeatureExtractor,
    'faster_rcnn_inception_v2':
    frcnn_inc_v2.FasterRCNNInceptionV2FeatureExtractor,
    'faster_rcnn_resnet18':		# 添加resnet18
    frcnn_resnet_v1.FasterRCNNResnet18FeatureExtractor,
    'faster_rcnn_resnet50':
    frcnn_resnet_v1.FasterRCNNResnet50FeatureExtractor,
    'faster_rcnn_resnet101':
    frcnn_resnet_v1.FasterRCNNResnet101FeatureExtractor,
    'faster_rcnn_resnet152':
    frcnn_resnet_v1.FasterRCNNResnet152FeatureExtractor,
}

```

4、在faster_rcnn_resnet_v1_feature_extractor.py中添加 `FasterRCNNResnet18FeatureExtractor`类：

```python
class FasterRCNNResnet18FeatureExtractor(FasterRCNNResnetV1FeatureExtractor):
  """Faster R-CNN Resnet 18 feature extractor implementation."""

  def __init__(self,
               is_training,
               first_stage_features_stride,
               batch_norm_trainable=False,
               reuse_weights=None,
               weight_decay=0.0,
               activation_fn=tf.nn.relu):
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.
      batch_norm_trainable: See base class.
      reuse_weights: See base class.
      weight_decay: See base class.
      activation_fn: See base class.

    Raises:
      ValueError: If `first_stage_features_stride` is not 8 or 16,
        or if `architecture` is not supported.
    """
    super(FasterRCNNResnet18FeatureExtractor,
          self).__init__('resnet_v1_18', resnet_v1.resnet_v1_18, is_training,
                         first_stage_features_stride, batch_norm_trainable,
                         reuse_weights, weight_decay, activation_fn)

```

5、在faster_rcnn_resnet_v1_feature_extractor.py  第178行
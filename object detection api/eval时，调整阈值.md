# eval时，调整阈值

调整eval时的阈值，在utils/object_detecion_evaluation.py中的：

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

matching_iou_threshold进行设置。





在evaluator.py下的eval_util.visualize_detection_results()中，对min_score_thres的值进行更改（这是调整tensorboard中的可视化的阈值，对整体eval没有任何作用）：

```pyhton
      eval_util.visualize_detection_results(
          result_dict,
          tag,
          global_step,
          categories=categories,
          summary_dir=eval_dir,
          export_dir=eval_config.visualization_export_dir,
          show_groundtruth=eval_config.visualize_groundtruth_boxes,
          groundtruth_box_visualization_color=eval_config.
          groundtruth_box_visualization_color,
          # min_score_thresh=eval_config.min_score_threshold,
          min_score_thresh=0.1,			# eval时的阈值
          max_num_predictions=eval_config.max_num_boxes_to_visualize,
          skip_scores=eval_config.skip_scores,
          skip_labels=eval_config.skip_labels,
          keep_image_id_for_visualization_export=eval_config.
          keep_image_id_for_visualization_export)
```


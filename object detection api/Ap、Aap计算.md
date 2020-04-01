# Ap、Aap计算

在object detection api中，使用utils\object_detection_evaluation.py文件计算模型的ap以及map值。在object_detection_evaluation.py中将整个计算分为4步：

```
1) Add ground truth information of images sequentially.
2) Add detection result of images sequentially.
3) Evaluate detection metrics on already inserted detection results.
4) Write evaluation result into a pickle file for future processing or
   visualization.
```

个人理解，可以看作一下流程：

1. 依次对所有图片进行检测，得到其所有scores和对应的boungding box，且soces和bounding box一一对应
2. 计算所有图片的ground truth
3. 根据检测结果和ground truth计算ap和map



ap、map知识点可以参考：https://blog.csdn.net/qq_41994006/article/details/81051150



具体的代码流程：

## 对每张图片进行detection

### `ObjectDetectionEvaluation.add_single_detected_image_info()`

对每张图片进行计算得到scores和tp_fp_labels（表示该bounding box属于tp还是fp）

```python
scores, tp_fp_labels, is_class_correctly_detected_in_image = (
        self.per_image_eval.compute_object_detection_metrics(
            detected_boxes=detected_boxes,
            detected_scores=detected_scores,
            detected_class_labels=detected_class_labels,
            groundtruth_boxes=groundtruth_boxes,
            groundtruth_class_labels=groundtruth_class_labels,
# 对每张图片进行所有类别的预测，得到一个cxkx1的array，称为scores，c为类别数，k为所有bounding box的属
# 于类别c的预测值。同步得到tp_fp_label（scores对应的bound box属于tp还是fp，使用boolean表示，   
# （true/falase）
            					              groundtruth_is_difficult_list=groundtruth_is_difficult_list,
            groundtruth_is_group_of_list=groundtruth_is_group_of_list,
            detected_masks=detected_masks,
            groundtruth_masks=groundtruth_masks))

# 按照类别进行合并
   for i in range(self.num_class):
      if scores[i].shape[0] > 0:
        self.scores_per_class[i].append(scores[i])
        self.tp_fp_labels_per_class[i].append(tp_fp_labels[i])
"""
合并所有图片的检测结果，按照类别进行合并，得到一个cx(k*n)x1的array，名为scores_per_class[]。c为类别，n为图片数，k为对应图片bounding box的预测值同时也对tp_fp_label进行相同数据结构合并，名为tp_fp_label_per_class[]
"""
```



### `per_image_eval.compute_object_detection_metrics()`

计算scores、tp_fp_labels，具体代码：

```python
    scores, tp_fp_labels = self._compute_tp_fp(
        detected_boxes=detected_boxes,
        detected_scores=detected_scores,
        detected_class_labels=detected_class_labels,
        groundtruth_boxes=groundtruth_boxes,
        groundtruth_class_labels=groundtruth_class_labels,
        groundtruth_is_difficult_list=groundtruth_is_difficult_list,
        groundtruth_is_group_of_list=groundtruth_is_group_of_list,
        detected_masks=detected_masks,
        groundtruth_masks=groundtruth_masks)
```



### `_compute_tp_fp()`具体代码：

```python
    result_scores = []
    result_tp_fp_labels = []
    for i in range(self.num_groundtruth_classes):
      groundtruth_is_difficult_list_at_ith_class = (
          groundtruth_is_difficult_list[groundtruth_class_labels == i])
      groundtruth_is_group_of_list_at_ith_class = (
          groundtruth_is_group_of_list[groundtruth_class_labels == i])
      (gt_boxes_at_ith_class, gt_masks_at_ith_class,
       detected_boxes_at_ith_class, detected_scores_at_ith_class,
       detected_masks_at_ith_class) = self._get_ith_class_arrays(
           detected_boxes, detected_scores, detected_masks,
           detected_class_labels, groundtruth_boxes, groundtruth_masks,
           groundtruth_class_labels, i)
      scores, tp_fp_labels = self._compute_tp_fp_for_single_class(
          detected_boxes=detected_boxes_at_ith_class,
          detected_scores=detected_scores_at_ith_class,
          groundtruth_boxes=gt_boxes_at_ith_class,
          groundtruth_is_difficult_list=groundtruth_is_difficult_list_at_ith_class,
          groundtruth_is_group_of_list=groundtruth_is_group_of_list_at_ith_class,
          detected_masks=detected_masks_at_ith_class,
          groundtruth_masks=gt_masks_at_ith_class)
      result_scores.append(scores)
      result_tp_fp_labels.append(tp_fp_labels)
    return result_scores, result_tp_fp_labels
"""
    Returns:
      result_scores: A list of float numpy arrays. Each numpy array is of
          shape [K, 1], representing K scores detected with object class
          label c
      result_tp_fp_labels: A list of boolean numpy array. Each numpy array is of
          shape [K, 1], representing K True/False positive label of object
          instances detected with class label c
"""
```





## 根据scores、tp_fp_labels 计算recall和precision

```python
      precision, recall = metrics.compute_precision_recall(
          scores, tp_fp_labels, self.num_gt_instances_per_class[class_index])
```

其中，`metrics.compute_precision_recall()`在每个tp出计算一个Recall。

```python
"""
1、降序排序，并调整labels
2、提取tp和fp
3、使用np.cumsum()对每个tp位置向前求和
4、计算precision和recall
"""
  sorted_indices = np.argsort(scores)
  sorted_indices = sorted_indices[::-1]
  true_positive_labels = labels[sorted_indices]
  false_positive_labels = (true_positive_labels <= 0).astype(float)
  cum_true_positives = np.cumsum(true_positive_labels)
  cum_false_positives = np.cumsum(false_positive_labels)
  precision = cum_true_positives.astype(float) / (
      cum_true_positives + cum_false_positives)
  recall = cum_true_positives.astype(float) / num_gt
```



### 根据Recall和precision计算ap

```python
      recall_within_bound_indices = [
          index for index, value in enumerate(recall) if
          value >= self.recall_lower_bound and value <= self.recall_upper_bound
      ]
      recall_within_bound = recall[recall_within_bound_indices]
      precision_within_bound = precision[recall_within_bound_indices]

      self.precisions_per_class[class_index] = precision_within_bound
      self.recalls_per_class[class_index] = recall_within_bound
      average_precision = metrics.compute_average_precision(
          precision_within_bound, recall_within_bound)
      self.average_precision_per_class[class_index] = average_precision
```

其中， `metrics.compute_average_precision()`，

```python
  recall = np.concatenate([[0], recall, [1]])
  precision = np.concatenate([[0], precision, [0]])

  # Preprocess precision to be a non-decreasing array
  for i in range(len(precision) - 2, -1, -1):
    precision[i] = np.maximum(precision[i], precision[i + 1])

  indices = np.where(recall[1:] != recall[:-1])[0] + 1
  average_precision = np.sum(
      (recall[indices] - recall[indices - 1]) * precision[indices])
  return average_precision
```


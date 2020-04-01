# 生成推理图（export inference graph）

将训练模型导出为pb格式，方便调用。

## 使用export_inference_graph

在object_detection路径下的export_inference_graph

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

# 测试


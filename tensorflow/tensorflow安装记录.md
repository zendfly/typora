# tensorflow安装记录

## 电脑配置

i5-9300、8G、GF1660ti

tensorflow-gpu=1.14

cuda 10.0

cudnn 7.5.0



## 坑

cuda 10.0 和 cudnn 7.6.5时，运行object detection api时，出现

```python
Unknown: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.
```

错误。






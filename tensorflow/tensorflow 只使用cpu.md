# tensorflow 只使用cpu

在运行代码的前面加上：

```python
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
```

使用如下代码检测当前tensorflow能使用的设备情况：

```python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())　
```

指定当前使用的GPU设备：

```python
os.environ[“CUDA_DEVICE_ORDER”] = “PCI_BUS_ID” # 按照PCI_BUS_ID顺序从0开始排列GPU设备 
os.environ[“CUDA_VISIBLE_DEVICES”] = “0” #设置当前使用的GPU设备仅为0号设备  设备名称为'/gpu:0'
os.environ[“CUDA_VISIBLE_DEVICES”] = “1” #设置当前使用的GPU设备仅为1号设备  设备名称为'/gpu:0'
os.environ[“CUDA_VISIBLE_DEVICES”] = “0,1” #设置当前使用的GPU设备为0,1号两个设备,名称依次为'/gpu:0'、'/gpu:1'
os.environ[“CUDA_VISIBLE_DEVICES”] = “1,0” #设置当前使用的GPU设备为1,0号两个设备,名称依次为'/gpu:0'、'/gpu:1'。表示优先使用1号设备,然后使用0号设备
```


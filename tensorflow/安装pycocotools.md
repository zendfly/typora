# 安装pycocotools

## 源码安装

在 https://github.com/philferriere/cocoapi 下载源码，并进行解压。

以**管理员身份**打开 CMD 终端，并切换到 `*\cocoapi-master\PythonAPI` 目录。

运行以下指令：

```python
python setup.py build_ext install
```

  运行以上指令时如果出现以下错误提示：

```python
error: Microsoft Visual C++ 14.0 is required.
// 或者
error: Unable to find vcvarsall.bat
```

  解决方法：此种安装方法需要使用 Microsoft Visual C++ 14.0 对 COCO 源码进行编译。如果本地不支持 Microsoft Visual C++ 14.0 或者版本低于 14.0，可以通过安装 Microsoft Visual Studio 2015 及以上版本。


## 使用pip安装

pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI


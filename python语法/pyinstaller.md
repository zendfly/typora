# pyinstaller 

python打包工具，在window下打包的文件不能在maxos系统使用，反之一样。在windows下打包成.exe文件，Macos下打包格式mac app

`pyinstaller demo.py`

相关参数：

| 可选参数 | 格式举例                                             | 功能说明                                                     |
| -------- | ---------------------------------------------------- | ------------------------------------------------------------ |
| -F       | pyinstaller -F demo.py                               | 只在dist中生产一个demo.exe文件                               |
| `-D`     | `pyinstaller -D demo.py`                             | 默认选项，除了demo.exe外，还会在在dist中生成很多依赖文件，推荐使用 |
| `-c`     | `pyinstaller -c demo.py`                             | 默认选项，只对windows有效，使用控制台，就像编译运行C程序后的黑色弹窗 |
| `-w`     | `pyinstaller -w demo.py`                             | 只对windows有效，不使用控制台                                |
| `-p`     | `pyinstaller -p E:\python\Lib\site-packages demo.py` | 设置导入路径，一般用不到                                     |
| `-i`     | `pyinstaller -i D:\file.icon demo.py`                | 将file.icon设置为exe文件的图标，推荐一个icon网站:[icon](https://tool.lu/tinyimage/) |

参数可以组合使用：

比如`pyinstaller -F -i D:\file.icon demo.py`。
能够`from xxx import yyy`就尽量不要`import xxx`,这样可以减少打包后的体积。





参考：https://www.jianshu.com/p/ab497e1ad257
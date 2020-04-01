# anaconda语法

conda install -c  ，适用于 conda不包含某些包的情况下，加上-c就可以解决。

删除添加的源（切换回默认源）

conda config --remove-key channels 



当网速很慢，conda install tensorflow-gpu出现HTTP等错误时，可以使用以下方法

anaconda search -t conda tensorflow-gpu，查找版本，再选取合适的版本

conda install -c https://conda.anaconda.org/nehaljwani tensorflow-gpu



- 创建环境

create -n (环境名) python=*   # * 版本号 

- 检查已有的环境

conda info -e

- 切换环境

activate (环境名) 

deactivate 

- 删除环境

conda remove -n (环境名)  --all



## 直接下载压缩包

https://pypi.tuna.tsinghua.edu.cn/simple/tensorflow-gpu/
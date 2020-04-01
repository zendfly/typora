# 通过xshell远程查看服务器上的tensorboard情况

## windows

在Windows系统装一个Xshell,在文件->属性->ssh->隧道->添加,类型local，源主机填127.0.0.1（意思是本机），端口设置一个，比如12345，目标主机为服务器，目标端口一般是6006，如果6006被占了可以改为其他端口。

在服务器上运行 `tensorboard --logdir='logs' --port=6006`

在本机打开网页`127.0.0.1:12345` ，即可查看远程的tensorboard。

## Mac/Linux

在登录远程服务器的时候使用命令：

```ssh
ssh -L 16006:127.0.0.1:6006 account@server.address
```

（代替一般ssh远程登录命令：`ssh account@server.address`）

训练完模型之后使用如下命令：

```
tensorboard --logdir="/path/to/log-directory"
```

（其中，/path/to/log-directory为自己设定的日志存放路径，因人而异）。

最后，在本地访问地址：`http://127.0.0.1:16006/`

参考链接：https://stackoverflow.com/questions/38513333/is-it-possible-to-see-tensorboard-over-ssh
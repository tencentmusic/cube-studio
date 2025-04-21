# notebook重启问题

关于续期：因为对gpu的占用方式为独占方式，所以对于gpu notebook会定时清理，需要按时续期。

关于清理：可以通过删除config.py中的delete_notebook定时任务，关闭掉定时清理notebook

关于环境：重启后会自动执行/mnt/$USERNAME/init.sh脚本，所以可以将环境写入此脚本，重启后自动安装环境，否则就需要打包到镜像或者离线anaconda文件

# 构建notebook镜像

需要构建新镜像并在生产上替换，才能让用户使用新的notebook镜像。

## 方法1：Dockerfile构建

jupyter镜像的构建在：https://github.com/data-infra/cube-studio/tree/main/images/jupyter-notebook

vscode镜像的构建在：https://github.com/data-infra/cube-studio/tree/main/images/theia

现在默认使用的镜像为
```
# notebook使用的镜像
NOTEBOOK_IMAGES=[
    ['ccr.ccs.tencentyun.com/cube-studio/notebook:vscode-ubuntu-cpu-base', 'vscode（cpu）'],
    ['ccr.ccs.tencentyun.com/cube-studio/notebook:vscode-ubuntu-gpu-base', 'vscode（gpu）'],
    ['ccr.ccs.tencentyun.com/cube-studio/notebook:jupyter-ubuntu-cpu-base', 'jupyter（cpu）'],
    ['ccr.ccs.tencentyun.com/cube-studio/notebook:jupyter-ubuntu-gpu-base','jupyter（gpu）'],
    ['ccr.ccs.tencentyun.com/cube-studio/notebook:jupyter-ubuntu-bigdata', 'jupyter（bigdata）'],
    ['ccr.ccs.tencentyun.com/cube-studio/notebook:jupyter-ubuntu-machinelearning', 'jupyter（machinelearning）'],
    ['ccr.ccs.tencentyun.com/cube-studio/notebook:jupyter-ubuntu-deeplearning', 'jupyter（deeplearning）'],
]
```
## 方法2，直接commit容器

也可以直接run一个容器，然后安装插件后将容器commmit成镜像。
```
# 启动jupyter
docker run --name jupyter -p 3000:3000 -d ccr.ccs.tencentyun.com/cube-studio/notebook:jupyter-ubuntu-cpu-base jupyter lab --notebook-dir=/ --ip=0.0.0.0 --no-browser --allow-root --port=3000 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*'

# 启动vscode
docker run --name vscode -p 3000:3000 -d ccr.ccs.tencentyun.com/cube-studio/notebook:vscode-ubuntu-cpu-base node /home/theia/src-gen/backend/main.js /home/project --hostname=0.0.0.0 --port=3000

```
然后访问 http://xx.xx.xx.xx:3000/ ， web界面操作，安装notebook插件，安装pip/apt环境等。环境完整后，再使用如下命令commit成镜像。
```
docker commit notebook ccr.ccs.tencentyun.com/cube-studio/notebook:jupyter-ubuntu-cpu-base-1
```

# 修改配置文件

config.py中 NOTEBOOK_IMAGES 变量为notebook可选镜像。更新此变量即可。


# 其他类型的notebook

所有可提供在线编辑功能的web服务都可以定义为notebook。开发代码已提供对外，需要满足几个条件，可配置url前缀，用来区分不同的notebook。

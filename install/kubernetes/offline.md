

前置条件：

内网机器需要安装了docker，docker-compose，iptables

# [部署视频](https://cube-studio.oss-cn-hangzhou.aliyuncs.com/video/%E5%86%85%E7%BD%91%E7%A6%BB%E7%BA%BF%E9%83%A8%E7%BD%B2.mp4)

# 完全无法联网的内网机器

## 安装依赖组件和数据

能连接外网的机器上执行下面的命令，拷贝到内网机器上
````bash
mkdir offline
cd offline
# 下载kubectl 和harbor的离线安装包
# amd64版本
wget https://cube-studio.oss-cn-hangzhou.aliyuncs.com/install/kubectl
wget https://githubfast.com/goharbor/harbor/releases/download/v2.11.1/harbor-offline-installer-v2.11.1.tgz
# arm64版本
wget https://cube-studio.oss-cn-hangzhou.aliyuncs.com/install/kubectl-arm64 && mv kubectl-arm64 kubectl
wget https://githubfast.com/wise2c-devops/build-harbor-aarch64/releases/download/v2.13.0/harbor-offline-installer-aarch64-v2.13.0.tgz

# 下载模型
wget https://cube-studio.oss-cn-hangzhou.aliyuncs.com/inference/resnet50.onnx
wget https://cube-studio.oss-cn-hangzhou.aliyuncs.com/inference/resnet50-torchscript.pt
wget https://cube-studio.oss-cn-hangzhou.aliyuncs.com/inference/resnet50.mar
wget https://cube-studio.oss-cn-hangzhou.aliyuncs.com/inference/tf-mnist.tar.gz
wget https://cube-studio.oss-cn-hangzhou.aliyuncs.com/inference/decisionTree_model.pkl

# 训练,标注数据集
wget https://cube-studio.oss-cn-hangzhou.aliyuncs.com/pipeline/coco.zip
wget https://docker-76009.sz.gfp.tencent-cloud.com/github/cube-studio/aihub/deeplearning/cv-tinynas-object-detection-damoyolo/dataset/coco2014.zip

````

offline目录拷贝到内网机器上

连不上网的机器上

1、安装kubectl
```bash
cd offline
chmod +x kubectl  && cp kubectl /usr/bin/ && cp kubectl /usr/local/bin/
```

2、[安装内网镜像仓库](harbor/readme.md)

参考install/kubernetes/harbor/readme.md

并创建cube-studio和rancher项目，分别存放rancher的基础镜像和cube-studio的基础镜像

配置每台机器docker添加这个 insecure-registries内网的私有镜像仓，如果是https可以忽略

参考： install/kubernetes/rancher/install_docker.md

3、将其他前面下载的数据转移到个人目录下

```bash
cp -r offline /data/k8s/kubeflow/pipeline/workspace/admin/
```

## 镜像转移至内网

## 转移rancher镜像

修改install/kubernetes/rancher/all_image.py中内网仓库地址，运行导出推送和拉取脚本.

联网机器上运行 pull_rancher_images.sh将镜像推送到内网仓库 或 rancher_image_save.sh将镜像压缩成文件再导入到内网机器

不能联网机器上运行，每台机器运行 pull_rancher_harbor.sh 从内网仓库中拉取镜像 或 rancher_image_load.sh 从压缩文件中导入镜像

## 内网部署 k8s

使用rancher相同方法可在内网部署k8s

## 转移cube-studio基础镜像

修改all_image.py中内网仓库地址，运行导出推送和拉取脚本.

联网机器上运行 push_harbor.sh 将镜像推送到内网仓库 或 image_save.sh将镜像压缩成文件再导入到内网机器

不能联网机器上运行，每台机器运行 pull_harbor.sh 从内网仓库中拉取镜像 或 image_load.sh 从压缩文件中导入镜像

## 内网部署cube-studio

1、修改init_node.sh中pull_images.sh 修改为pull_harbor.sh，表示从内网拉取镜像，每台机器都要执行。

2、取消start.sh脚本中下载kubectl，注释掉
```bash
ARCH=$(uname -m)

if [ "$ARCH" = "x86_64" ]; then
  wget https://cube-studio.oss-cn-hangzhou.aliyuncs.com/install/kubectl && chmod +x kubectl  && cp kubectl /usr/bin/ && mv kubectl /usr/local/bin/
elif [ "$ARCH" = "aarch64" ]; then
  wget -O kubectl https://cube-studio.oss-cn-hangzhou.aliyuncs.com/install/kubectl-arm64 && chmod +x kubectl  && cp kubectl /usr/bin/ && mv kubectl /usr/local/bin/
fi
```
3、修改cube-studio镜像为内网镜像。
```bash
vi install/kubernetes/cube/overlays/kustomization.yml
修改最底部的newName和newTag
```

4、修改cube-studio的配置文件

```bash
vi install/kubernetes/cube/overlays/config/config.py

下面的值改为内网仓库地址
REPOSITORY_ORG PUSH_REPOSITORY_ORG USER_IMAGE NOTEBOOK_IMAGES DOCKER_IMAGES NERDCTL_IMAGES NNI_IMAGES WAIT_POD_IMAGES OPEN_WEBUI_IMAGE INFERNENCE_IMAGES

其他修改：
SERVICE_EXTERNAL_IP 添加内网ip
DEFAULT_GPU_RESOURCE_NAME 修改为默认的k8s资源名
```

6、复制k8s的config文件，部署cube-studio，部署方式通外网，参考：部署/单机部署

## web界面的部分内网修正

1、web界面hubsecret改为内部仓库的账号密码

2、修改配置文件中的内网仓库信息和内外网ip

3、自带的目标识别pipeline中，第一个数据拉取任务启动命令改为，`cp offline/coco.zip ./ && ...`

4、自带的推理服务启动命令 由`wget https://xxxx/xx/.zip` 部分改为 `cp /mnt/admin/offline/xx.zip ./`

# 内网中有可以联网的机器

##  联网机器设置代理服务器

联网机器上设置nginx代理软件源，参考install/kubernetes/nginx-https/apt-yum-pip-source.conf

启动nginx代理访问

需要监听80和443端口
```bash
docker run --name proxy-repo -d --restart=always --network=host -v $PWD/nginx-https/apt-yum-pip-source.conf:/etc/nginx/nginx.conf nginx 
```

## 在内网机器上配置host

host
```bash
<出口服务器的IP地址>    mirrors.aliyun.com
<出口服务器的IP地址>    ccr.ccs.tencentyun.com
<出口服务器的IP地址>    registry-1.docker.io
<出口服务器的IP地址>    auth.docker.io
<出口服务器的IP地址>    hub.docker.com
<出口服务器的IP地址>    www.modelscope.cn
<出口服务器的IP地址>    modelscope.oss-cn-beijing.aliyuncs.com
<出口服务器的IP地址>    archive.ubuntu.com
<出口服务器的IP地址>    security.ubuntu.com
<出口服务器的IP地址>    cloud.r-project.org
<出口服务器的IP地址>    deb.nodesource.com
<出口服务器的IP地址>    docker-76009.sz.gfp.tencent-cloud.com
```

添加新的host要重启下kubelet   docker restart kubelet

如果代理机器没法占用80和443，需要使用iptable尝试转发。

iptables
```bash
sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -d mirrors.aliyun.com -j DNAT --to-destination <出口服务器的IP地址>:<出口服务器的端口>
```

## k8s配置域名解析

k8s中修改 kube-system命名空间，coredns的configmap，添加 需要访问的地址 的地址映射
```bash
{
	"Corefile": ".:53 {
		    errors
		    health {
		      lameduck 5s
		    }
		    ready
		    kubernetes cluster.local in-addr.arpa ip6.arpa {
		      pods insecure
		      fallthrough in-addr.arpa ip6.arpa
		    }
		    # 自定义host
		    hosts {
		        <出口服务器的IP地址>    mirrors.aliyun.com
                <出口服务器的IP地址>    ccr.ccs.tencentyun.com
                <出口服务器的IP地址>    registry-1.docker.io
                <出口服务器的IP地址>    auth.docker.io
                <出口服务器的IP地址>    hub.docker.com
                <出口服务器的IP地址>    www.modelscope.cn
                <出口服务器的IP地址>    modelscope.oss-cn-beijing.aliyuncs.com
                <出口服务器的IP地址>    archive.ubuntu.com
                <出口服务器的IP地址>    security.ubuntu.com
                <出口服务器的IP地址>    cloud.r-project.org
                <出口服务器的IP地址>    deb.nodesource.com
                <出口服务器的IP地址>    docker-76009.sz.gfp.tencent-cloud.com
		      fallthrough
		    }
		    prometheus :9153
		    forward . \"/etc/resolv.conf\"
		    cache 30
		    loop
		    reload
		    loadbalance
		} # STUBDOMAINS - Rancher specific change
		"
}
```
重启coredns的pod

## 容器里面使用放开的域名

pip配置https源:
```bash
pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple
```

apt配置https源: 修改/etc/apt/source.list

ubuntu 20.04
```bash

deb https://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse
deb-src https://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse

deb https://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse
deb-src https://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse

deb https://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse
deb-src https://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse

deb https://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse
deb-src https://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse

deb https://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse
deb-src https://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse
```

yum 配置https源：下载阿里的源
```bash
wget -O /etc/yum.repos.d/CentOS-Base.repo https://mirrors.aliyun.com/repo/Centos-8.repo
```



# 在线安装

## ubuntu 在线安装

```bash
# 删除所有 NVIDIA 相关的软件源
sudo rm -f /etc/apt/sources.list.d/nvidia*.list
# 删除冲突的 GPG 密钥
sudo rm -f /usr/share/keyrings/nvidia*.gpg

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt update -y
# docker 运行时安装 nvidia-docker2 
apt install -y nvidia-docker2 
# containerd运行时安装nvidia-container-toolkit
apt install -y nvidia-container-toolkit
```

## centos 在线安装

docker运行时安装 nvidia-docker2

```bash
yum install docker-ce -y
#yum install -y yum-utils
#distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
#curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo
cp nvidia-docker.repo /etc/yum.repos.d/nvidia-docker.repo
yum makecache
yum install -y nvidia-docker2
```

containerd运行时安装 nvidia-container-toolkit

```bash
yum install -y yum-utils
sudo yum-config-manager --add-repo https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo
yum makecache
yum install -y nvidia-container-toolkit
```

# 离线安装

```bash
wget https://cube-studio.oss-cn-hangzhou.aliyuncs.com/install/nvidia-docker2.tar.gz  && tar -zxvf nvidia-docker2.tar.gz && rm nvidia-docker2.tar.gz
cd nvidia-docker2
dpkg -i ./*.deb
dpkg -l | grep nvidia-docker2
```

# docker修改配置
```bash

cat > /etc/docker/daemon.json <<EOF
{
    # 镜像加速器，拉取docker官方镜像时需要
    "registry-mirrors": ["https://hub.rat.dev/","https://docker.xuanyuan.me", "https://docker.m.daocloud.io","https://dockerproxy.com"],
    # dns可不配置
    "dns": ["114.114.114.114","8.8.8.8"],
    # k8s集群可以同时拉取多个镜像
    "max-concurrent-downloads": 30,
    # 默认系统根目录下，如果磁盘有限可以改为其他有空间的目录，占用存储会越来越多
    "data-root": "/data/docker",
    # 内部如果有http的镜像仓库，可以添加
    "insecure-registries":["docker.oa.com:8080"],
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF

systemctl stop docker
systemctl daemon-reload
systemctl start docker

```

# containerd修改配置

vi /etc/containerd/config.toml

## 添加nvidia运行时
```bash
      [plugins."io.containerd.grpc.v1.cri".containerd.runtimes]
        # 添加下面的内容
        [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia]
          runtime_type = "io.containerd.runc.v2"
          [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia.options]
            BinaryName = "/usr/bin/nvidia-container-runtime"
            
```
## 修改默认运行时为nvidia
```bash
    [plugins."io.containerd.grpc.v1.cri".containerd]
      default_runtime_name = "nvidia"
```
## 重启配置生效
```bash
systemctl daemon-reload
systemctl restart containerd
```

# 测试docker识别gpu

```bash
docker run --name test --gpus all -it nvidia/cuda:11.8.0-devel-ubuntu22.04 bash

docker run --name test --gpus all -it ccr.ccs.tencentyun.com/cube-studio/ubuntu-gpu:cuda11.8.0-cudnn8-python3.9  bash
```
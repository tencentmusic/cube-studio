
# 在线安装

## ubuntu 在线安装

```bash
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
(
cat << EOF
{
    "registry-mirrors": ["https://docker.1panel.live", "https://hub.rat.dev/", "https://docker.chenby.cn", "https://docker.m.daocloud.io"],
    "data-root": "/data/docker",
    "max-concurrent-downloads": 30,
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
)> /etc/docker/daemon.json

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

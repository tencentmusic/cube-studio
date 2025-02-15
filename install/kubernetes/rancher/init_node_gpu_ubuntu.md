

# 在线安装

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update -y

sudo apt-get install -y nvidia-docker2
```

# 离线安装

```bash
wget https://cube-studio.oss-cn-hangzhou.aliyuncs.com/install/nvidia-docker2.tar.gz  && tar -zxvf nvidia-docker2.tar.gz && rm nvidia-docker2.tar.gz
dpkg -i ./*.deb
dpkg-grep nvidia-docker2
```


# 修改配置
```bash
(
cat << EOF
{
    "registry-mirrors": ["https://docker.1panel.live", "https://hub.rat.dev/", "https://docker.chenby.cn", "https://docker.m.daocloud.io"],
    "default-runtime": "nvidia",
    "data-root": "/data/docker",
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


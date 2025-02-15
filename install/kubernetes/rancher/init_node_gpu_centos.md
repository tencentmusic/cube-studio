
swapoff -a
# 拷贝源
```bash
cp nvidia-docker.repo /etc/yum.repo.d/
yum update -y
```

# 安装
```bash
yum install docker-ce -y
yum list nvidia-docker2 --showduplicate
yum install -y nvidia-docker2
yum install -y htop docker-compose
yum install -y wireshark
```

# 配置
```bash
(
cat << EOF
{
    "registry-mirrors": ["https://docker.1panel.live", "https://hub.rat.dev/", "https://docker.chenby.cn", "https://docker.m.daocloud.io"],
    "insecure-registries":["docker.oa.com:8080"],
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


service docker stop
systemctl daemon-reload
systemctl start docker
```



# ubuntu 安装containerd

##  卸载旧版本containerd
```bash
sudo apt-get remove -y docker docker-engine docker.io containerd runc
sudo rm -rf /var/lib/docker /etc/docker/
sudo rm -rf /var/lib/containerd /etc/containerd/

```

## 安装containerd

```bash

### 国内使用阿里源
curl -fsSL http://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | apt-key add -
sudo add-apt-repository "deb [arch=amd64] http://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable"

# 安装containerd
sudo apt-get update
sudo apt-get install -y containerd.io

# 查看版本
apt-cache madison containerd

# sudo apt-get install containerd=<VERSION>
# 例如 apt-get install containerd=1.7.20-..

# 查看运行状态
systemctl start containerd
systemctl enable containerd
systemctl status containerd

```

# centos 安装containerd

```bash
yum update -y
yum install -y yum-utils device-mapper-persistent-data lvm2
# 使用阿里源
yum-config-manager --add-repo https://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo
yum -y install containerd.io    # 安装 1.7.20相近版本

systemctl start containerd
systemctl enable containerd
systemctl status containerd

```

# 配置containerd

```bash
# 生成默认配置
mkdir /etc/containerd
containerd config default | tee /etc/containerd/config.toml

vi /etc/containerd/config.toml

添加如下配置

# 1、修改sandbox_image的地址
# sandbox_image = "k8s.gcr.io/pause:3.8"
# 注释上面那行，添加下面这行,注意看一下后面的版本号
sandbox_image = "registry.cn-hangzhou.aliyuncs.com/google_containers/pause:3.8"

# 2、centos配置这个参数：配置Containerd直接使用systemd去管理cgroupfs,
[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc.options]
  # 修改下面这行
  SystemdCgroup = true
       
# 3、添加镜像源，拉取dockerhub镜像
[plugins."io.containerd.grpc.v1.cri".registry] #此行下修改
      config_path = "/etc/containerd/certs.d" # 改为此路径，并且在每一个路径下创建hosts.toml文件，用于存放镜像加速信息

# 创建/etc/containerd/certs.d下的hosts文件
mkdir -p /etc/containerd/certs.d/docker.io

tee /etc/containerd/certs.d/docker.io/hosts.toml << 'EOF'
server = "https://docker.io"
[host."https://docker.anyhub.us.kg"]
  capabilities = ["pull", "resolve"]
EOF

# 4、配置私有镜像仓库
# 只需要编辑下面这段配置即可,给config_path添加对应的地址
[plugins."io.containerd.grpc.v1.cri".registry]
  config_path = "/etc/containerd/cert.d"
  
mkdir -p /etc/containerd/certs.d/172.17.0.4:88
tee /etc/containerd/certs.d/172.17.0.4:88/hosts.toml << 'EOF'
server = "http://172.17.0.4:88"

[host."http://172.17.0.4:88"]
  capabilities = ["pull", "resolve", "push"]
  skip_verify = true
EOF

# 重启配置生效
systemctl daemon-reload
systemctl restart containerd
```

# 安装nerdctl

首先查看containerd与nerdctl的对应关系，下载对应的版本

获取下载地址 https://github.com/containerd/nerdctl/releases

version=1.7.6

wget https://githubfast.com/containerd/nerdctl/releases/download/v${version}/nerdctl-${version}-linux-amd64.tar.gz

tar zxvf nerdctl-${version}-linux-amd64.tar.gz -C /usr/local/bin

echo "alias docker='nerdctl --namespace k8s.io'"  >> /etc/profile
echo "alias docker-compose='nerdctl compose'"  >> /etc/profile
source  /etc/profile

#配置nerdctl
mkdir -p /etc/nerdctl/
cat > /etc/nerdctl/nerdctl.toml << 'EOF'
namespace      = "k8s.io"
insecure_registry = true
EOF

# 安装构建器

首先查看containerd与buildkit的对应关系，下载对应的版本

获取下载地址 https://github.com/moby/buildkit/releases
```
version=v0.15.1

wget https://githubfast.com/moby/buildkit/releases/download/${version}/buildkit-${version}.linux-amd64.tar.gz

tar zxvf buildkit-${version}.linux-amd64.tar.gz -C /usr/local/

vi /etc/systemd/system/buildkit.service 

[Unit]
Description=BuildKit
Documentation=https://github.com/moby/buildkit

[Service]
ExecStart=/usr/local/bin/buildkitd --oci-worker=false --containerd-worker=true

[Install]
WantedBy=multi-user.target


# 启动
systemctl enable buildkit --now

```

# 安装 cni 网络插件
```bash
# 创建目录
sudo mkdir -p /opt/cni/bin

# 下载 CNI 插件 amd
sudo wget https://githubfast.com/containernetworking/plugins/releases/download/v1.1.1/cni-plugins-linux-amd64-v1.1.1.tgz
# 下载 CNI 插件 arm
sudo wget https://githubfast.com/containernetworking/plugins/releases/download/v1.1.1/cni-plugins-linux-arm64-v1.1.1.tgz

# 解压对应版本插件到 /opt/cni/bin
mkdir -p /opt/cni/bin
sudo tar -C /opt/cni/bin -xzvf cni-plugins-linux-amd64-v1.1.1.tgz

# 创建一个 CNI 网络配置文件，比如 /etc/cni/net.d/10-bridge.conf，内容如下：
vi /etc/cni/net.d/10-bridge.conf

{
    "cniVersion": "0.4.0",
    "name": "bridge",
    "type": "bridge",
    "bridge": "cni0",
    "isGateway": true,
    "ipMasq": true,
    "ipam": {
        "type": "host-local",
        "ranges": [
            [{"subnet": "10.22.0.0/16"}]
        ],
        "routes": [
            {"dst": "0.0.0.0/0"}
        ]
    }
}

修正containerd 的配置
vi /etc/containerd/config.toml

[plugins."io.containerd.grpc.v1.cri".cni]
  # ConfDir is the directory to search CNI config files
  conf_dir = "/etc/cni/net.d"
  # BinDir is the directory to search CNI plugin binaries
  bin_dir = "/opt/cni/bin"


# 重启配置生效
systemctl daemon-reload
systemctl restart containerd
```

# 注意

1、如果镜像源没有生效，那在拉取dockerhub镜像的前面加上 `docker.anyhub.us.kg/library/`

例如
拉取 nerdctl pull nginx 
本质拉取是 
nerdctl pull docker.io/library/nginx 
替换为 
nerdctl pull docker.anyhub.us.kg/library/nginx   


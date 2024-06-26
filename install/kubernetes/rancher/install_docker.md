
# ubuntu 安装docker

##  卸载旧版本docker
```bash
sudo systemctl stop docker
apt-get --purge remove -y *docker*  
sudo apt-get autoremove -y
dpkg -l | grep docker
```

## 安装docker

```bash
### 设置docker存储库
sudo apt-get update -y 
sudo apt-get install -y ca-certificates curl gnupg lsb-release vim git wget net-tools

### 添加官方秘钥

sudo mkdir -p /etc/apt/keyrings
rm -rf /etc/apt/keyrings/docker.gpg
rm -rf /etc/apt/sources.list.d/docker.list

### 使用docker 官方源
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

### 国内使用阿里源
curl -fsSL http://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | apt-key add -
sudo add-apt-repository "deb [arch=amd64] http://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable"

### 安装docker
sudo apt-get update

### 查看存储库中的可用版本，因为我们需要19.03以上的docker

# 搜索可用版本
apt-cache madison docker-ce
# 安装指定版本，使用安装指定版本
apt install -y docker-ce=5:20.10.24~3-0~ubuntu-focal

# 安装最新版(最好使用指定版本)
# sudo apt install -y docker-ce docker-compose
apt install -y docker-compose
```

# ubuntu 安装 k8s客户端

```bash
apt-get update && apt-get install -y apt-transport-https
# 添加并信任APT证书
curl https://mirrors.aliyun.com/kubernetes/apt/doc/apt-key.gpg | apt-key add - 
# 添加源地址
add-apt-repository "deb https://mirrors.aliyun.com/kubernetes/apt/ kubernetes-xenial main"

apt-get update -y

# 搜索可用版本
apt-cache madison kubectl

# 安装最新版，最好指定版本
# apt install -y kubectl
# 安装执行 版本
apt install -y kubectl=1.24.10-00

# 添加 completion，最好放入 .bashrc 中
apt install -y bash-completion
source <(kubectl completion bash)
```

# centos安装docker

## 安装docker
```bash
# 先卸载原有docker
service docker stop
rpm -qa | grep docker | xargs yum remove -y
rpm -qa | grep docker
rm -rf /usr/lib/systemd/system/docker.service


yum install -y yum-utils
yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
yum-config-manager --add-repo http://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo 

yum update -y
# 查看可用版本
yum list docker-ce --showduplicates
# 安装20.x的版本
yum install -y docker-ce-3:20.10.24-3.el8

systemctl start docker

```

# 配置docker

```bash
vi /etc/docker/daemon.json

添加如下配置

{
	"registry-mirrors": ["https://docker.m.daocloud.io", "https://dockerproxy.com", "https://docker.mirrors.ustc.edu.cn", "https://docker.nju.edu.cn"],
    "dns": ["114.114.114.114","8.8.8.8"],
    "data-root": "/data/docker",
}

systemctl stop docker
systemctl daemon-reload
systemctl start docker
```

# 切换docker根目录

```bash
mkdir -p /data/docker/
# 将源docker目录下文件，复制到新目录下
cp -R /var/lib/docker/* /data/docker/
```
然后按照上面的配置daemon.json，配置根目录为/data/docker/

就可以把之前的目录删掉了
```bash
rm -rf /var/lib/docker
```

## yum安装k8s的源
```bash
cat <<EOF > /etc/yum.repos.d/kubernetes.repo
[kubernetes]
name=Kubernetes
baseurl=https://mirrors.aliyun.com/kubernetes/yum/repos/kubernetes-el7-x86_64/
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://mirrors.aliyun.com/kubernetes/yum/doc/yum-key.gpg https://mirrors.aliyun.com/kubernetes/yum/doc/rpm-package-key.gpg
EOF
setenforce 0
yum install -y kubectl-1.24.0 
source <(kubectl completion bash)
```






# 初始化机器环境(每台机器)，不需要安装，kubekey会自动安装

参考 install/kubernetes/rancher/install_docker.md安装docker

或者

参考 install/kubernetes/rancher/install_containerd.md安装containerd

##### 修改主机名

主机名不要有大写，保持小写主机名
```
hostnamectl set-hostname [新主机名]
```
修改后重新进入终端，主机名才会生效

#### 初始化安装组件
```bash
# 安装时间同步，启动chronyd服务

# yum -y install chrony    # 安装 时间同步
apt install -y chrony

systemctl start chronyd
systemctl enable chronyd

#手动同步一下
timedatectl set-ntp true
chronyc -a makestep
date

# ubuntu安装基础依赖
apt install -y socat conntrack ebtables ipset ipvsadm
# centos安装基础依赖
yum install -y socat conntrack ebtables ipset ipvsadm tar
# 关闭firewalld服务
systemctl stop firewalld
systemctl disable firewalld
# 禁用iptable
systemctl stop iptables
systemctl disable iptables
# 禁用selinux
#apt install selinux-utils && setenforce 1
echo "SELINUX=disabled" > /etc/selinux/config
#临时关闭swap分区
swapoff -a
# 永久关闭swap分区
# 编辑分区配置文件/etc/fstab，注释掉swap分区一行
# 注意修改完毕之后需要重启linux服务
vim /etc/fstab
注释掉 /dev/mapper/centos-swap swap
# /dev/mapper/centos-swap swap
#检查
free -m
#如果swap全部是0，则代表配置完成

```

# 搭建 k8s

> 注意：机器最低规格为：16C32G ；kubectl 版本要1.24 ；之前安装过 KS 要提前清理下环境。

* 下载 KubeKey 

如果下载不成功，可以多执行几次
```shell
# arm64版本
# wget https://cube-studio.oss-cn-hangzhou.aliyuncs.com/install/kubekey-v3.1.10-linux-arm64 -O /usr/bin/kk
# amd64版本
# wget https://cube-studio.oss-cn-hangzhou.aliyuncs.com/install/kubekey-v3.1.10-linux-amd64 -O /usr/bin/kk

export KKZONE=cn
curl -sfL https://get-kk.kubesphere.io | VERSION=v3.1.10 sh -
chmod +x kk
cp kk /usr/bin/
```

* 清理 kubeconfig，不然会导致其他 node 节点 无法使用 kubectl

```shell
rm -rf  /root/.kube/config
```

## 创建集群配置文件

```bash
./kk create config --with-kubernetes v1.25.16 

单机可忽略：根据实际情况配置多台机器的免密登录和config-sample.yaml配置。kubekey会使用这个文件里面的机器的账号密码登录远程机器执行添加命令

示例如下：name写内网ip地址
spec:
  hosts:
  - {name: 172.16.0.2, address: 172.16.0.2, internalAddress: 172.16.0.2, user: ubuntu, password: "Qcloud@123"}
  - {name: 172.16.0.3, address: 172.16.0.3, internalAddress: 172.16.0.3, user: ubuntu, password: "Qcloud@123"}
  roleGroups:
    etcd:
    - 172.16.0.2
    control-plane: 
    - 172.16.0.2
    worker:
    - 172.16.0.2
    - 172.16.0.3
```

*  安装 1.25 版本的 k8s
```bash
export KKZONE=cn
./kk create cluster -f config-cluster.yaml
会自己安装 containerd，kubectl，kubeadm kubecni，helm 等
```

# 部署cube-studio(主节点)

2、如果使用containerd运行时，替换脚本中的docker命令
```bash
# 安装containerd下的cli
export TARGETARCH=amd64
curl -L  https://github.com/containerd/nerdctl/releases/download/v1.7.2/nerdctl-1.7.2-linux-${TARGETARCH}.tar.gz | tar xzv -C /usr/local/bin nerdctl

# 替换拉取文件中的拉取命令
cd install/kubernetes/
sed -i 's/^docker/nerdctl/g' pull_images.sh

```

3、对于kubekey部署的ipvs模式的k8s，

  （1）要将install/kubernetes/start.sh脚本最后面的`kubectl patch svc istio-ingressgateway -n istio-system -p '{"spec":{"externalIPs":["'"$1"'"]}}'`注释掉。取消注释代码`kubectl patch svc istio-ingressgateway -n istio-system -p '{"spec":{"type":"NodePort"}}'`

  （2）将配置文件install/kubernetes/cube/overlays/config/config.py中的 CONTAINER_CLI的值 改为 nerdctl，K8S_NETWORK_MODE的值 改为ipvs


4、将k8s集群的kubeconfig文件（默认位置：~/.kube/config）复制到install/kubernetes/config文件中，然后执行下面的部署命令，其中xx.xx.xx.xx为机器内网的ip（不是外网ip）

```
# 在k8s worker机器上执行

如果只部署了k8s，没有部署kubesphere，执行
sh start.sh xx.xx.xx.xx
```

# 扩缩容节点

新机器执行前面的基础环境准备

修改config-cluster.yaml文件

```bash
export KKZONE=cn
./kk add nodes -f config-cluster.yaml       增加机器
./kk delete node <nodeName> -f config-cluster.yaml     释放机器
```

# 升级配置

```bash
./kk upgrade -f config-cluster.yaml
```

# 可能的问题

参考<<平台单机部署>> 中的 “部署后排查” 环节

# 配置prometheus替换为kubesphere的

grafana中数据源地址替换为http://prometheus-k8s.kubesphere-monitoring-system.svc:9090

配置文件config.py中

PROMETHEUS 修改为 prometheus-k8s.kubesphere-monitoring-system:9090

# 卸载 KubeSphere 和 Kubernetes

```bash
./kk delete cluster -f config-cluster.yaml
```

# 彻底清理
关闭kubelet,etcd和docker
```bash
systemctl stop kubelet
systemctl stop etcd
```
删除k8s相关目录
```bash
sudo rm -rvf ~/.kube
sudo rm -rvf ~/.kube/
sudo rm -rvf /etc/kubernetes/
sudo rm -rvf /etc/systemd/system/kubelet.service.d
sudo rm -rvf /etc/systemd/system/kubelet.service
sudo rm -rvf /usr/bin/kube*
sudo rm -rvf /etc/cni
sudo rm -rvf /opt/cni
sudo rm -rvf /var/lib/etcd
sudo rm -rvf /var/etcd
```
清除Docker所有相关资源
```bash
docker rm -f $(docker ps -aq)
docker rmi -f $(docker images -aq)
docker volume prune
docker network prune
```
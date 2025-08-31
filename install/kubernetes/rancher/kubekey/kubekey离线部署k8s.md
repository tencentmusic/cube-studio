# 所需镜像导入到内网机器

所需镜像列表

```bash
registry.cn-beijing.aliyuncs.com/kubesphereio/kube-apiserver:v1.25.16
registry.cn-beijing.aliyuncs.com/kubesphereio/kube-controller-manager:v1.25.16
registry.cn-beijing.aliyuncs.com/kubesphereio/kube-proxy:v1.25.16
registry.cn-beijing.aliyuncs.com/kubesphereio/kube-scheduler:v1.25.16
registry.cn-beijing.aliyuncs.com/kubesphereio/pause:3.8
registry.cn-beijing.aliyuncs.com/kubesphereio/coredns:1.9.3
registry.cn-beijing.aliyuncs.com/kubesphereio/cni:v3.27.3
registry.cn-beijing.aliyuncs.com/kubesphereio/kube-controllers:v3.27.3
registry.cn-beijing.aliyuncs.com/kubesphereio/node:v3.27.3
registry.cn-beijing.aliyuncs.com/kubesphereio/pod2daemon-flexvol:v3.23.2
registry.cn-beijing.aliyuncs.com/kubesphereio/k8s-dns-node-cache:1.22.20
```

# 制作离线包
在联网机器上执行

### 下载 KubeKey
如果不成功，可以多试几次
```bash
export KKZONE=cn
curl -sfL https://get-kk.kubesphere.io | VERSION=v3.1.10 sh -
chmod +x kk
cp kk /usr/bin/
```
### 配置离线包内容

manifest-sample.yaml详细说明:"https://github.com/kubesphere/kubekey/blob/master/docs/manifest-example.md"

vi manifest-sample.yaml    # 文件夹中文档含有

### 制作离线包
```bash
export KKZONE=cn
./kk artifact export -m manifest-sample.yaml -o kubesphere.tar.gz
```

# 将下载好的包复制到内网机器
scp -r  *  xx.xx.xx.xx:/root/

# 离线启动
##### 修改主机名
主机名不要有大写，保持小写主机名
hostnamectl set-hostname [新主机名]

#### 初始化安装组件(内网机器加入前安装下面的基础组件)
```bash
# 安装时间同步，启动chronyd服务
apt install -y chrony

systemctl start chronyd
systemctl enable chronyd

#手动同步一下
timedatectl set-ntp true
chronyc -a makestep
date

apt update
# ubuntu安装基础依赖
apt install -y socat conntrack ebtables ipset ipvsadm
# 关闭firewalld服务
systemctl stop firewalld
systemctl disable firewalld
# 禁用iptable
systemctl stop iptables
systemctl disable iptables
# 禁用selinux
#apt install selinux-utils && setenforce 1
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

* 清理 kubeconfig，不然会导致其他 node 节点 无法使用 kubectl
rm -rf  /root/.kube/config


### 创建配置文件，修改配置文件中的参数
```bash
vi config-cluster-offline.yaml 
```

# 创建集群
```bash
export KKZONE=cn
./kk create cluster -f config-cluster-offline.yaml -a kubesphere.tar.gz --skip-pull-images
```
# 卸载 KubeSphere 和 Kubernetes

```bash
./kk delete cluster -f config-cluster-offline.yaml
```

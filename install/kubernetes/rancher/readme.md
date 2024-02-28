# 1、内网使用rancher自建k8s集群

如果可以使用公有云k8s，可以直接构建公有云厂商的容器服务。这里介绍如何在内网使用k8s自建k8s集群

# 2、建设前准备

如果内网无法连接到互联网的话，需要在内网申请一个docker仓库，如果内网没有docker仓库，则可以使用Harbor自建一个内网仓库。比如申请一个地址是docker.oa.com的内网仓库

# 3、将基础组件推送到内网仓库，并在idc机器拉取

大多数情况下，内网是无法连接外网的，需要我们提前拉好镜像。如果你的机器可以连接外网，则可以忽略这一部分的操作。

关于镜像的版本，这与rancher和k8s的版本有关。你可以在这里选择一个能够部署k8s 1.21的rancher版本：https://github.com/rancher/rancher/releases

比如我这里使用的是rancher_version=v2.6.2，即2.6.2版本，那么这个版本依赖的镜像，可以在https://github.com/rancher/rancher/releases/tag/$rancher_version  中找到其所依赖的镜像txt文件，也就是 https://github.com/rancher/rancher/releases/download/$rancher_version/rancher-images.txt

之后，将依赖的镜像在开发网中拉取下来，然后重新tag成内网仓库镜像，例如docker.oa.com域名下的镜像，推送到docker.oa.com上，接着需要在idc中的每个机器上拉取下来，再tag成原始镜像名。
参考命令：

## 3.1、可以连接外网的机器上
```bash
docker pull rancher/rancher-agent:$rancher_version
docker tag rancher/rancher-agent:$rancher_version docker.oa.com:8080/public/rancher/rancher-agent:$rancher_version
```

## 3.2、内网idc机器
```bash
docker pull docker.oa.com/public/rancher/rancher-agent:$rancher_version
docker tag docker.oa.com/public/rancher/rancher-agent:$rancher_version rancher/rancher-agent:$rancher_version
```

由于依赖的镜像比较多，我们可以写一个脚本，批量的去拉取和tag。

# 4、初始化节点

想要初始化节点，我们可以批量下发init_node.sh和reset_docker.sh这两个脚本。

init_node.sh 是为了初始化机器，可以把自己要做的初始化任务加入到其中
例如：
修改hostname，修改dns或者host；
卸载已经存在的docker，安装docker并把默认路径放在/data/docker目录下，防止放在/var/lib/docker目录下机器磁盘会满；
为docker添加私有仓库拉取权限；

reset_docker.sh 是为了在机器从rancher集群中踢出以后，把rancher环境清理干净。

# 5、centos8 初始化

```bash
#修改/etc/firewalld/firewalld.conf
#FirewallBackend=nftables
FirewallBackend=iptables

yum install -y yum-utils device-mapper-persistent-data lvm2
yum install -y iptables container-selinux iptables-services
# 加载内核模块
(
cat << EOF

systemctl stop firewalld
systemctl disable firewalld
systemctl stop iptables
systemctl disable iptables
systemctl stop ip6tables
systemctl disable ip6tables
systemctl stop nftables
systemctl disable nftables

modprobe br_netfilter 
modprobe ip_tables 
modprobe iptable_nat 
modprobe iptable_filter 
modprobe iptable_mangle 
modprobe iptable_mangle
modprobe ip6_tables 
modprobe ip6table_nat 
modprobe ip6table_filter 
modprobe ip6table_mangle 
modprobe ip6table_mangle

EOF
)>>  /etc/rc.d/rc.local
chmod +x /etc/rc.d/rc.local
sh /etc/rc.d/rc.local
# 查看加载的内核模块
lsmod
sudo echo 'ip_tables' >> /etc/modules


systemctl status iptables
systemctl status ip6tables
systemctl status nftables
systemctl status firewalld

modinfo iptable_nat
modinfo ip6table_nat

echo "net.bridge.bridge-nf-call-ip6tables = 1" >> /etc/sysctl.conf
echo "net.bridge.bridge-nf-call-iptables=1" >> /etc/sysctl.conf
echo "net.ipv4.ip_forward=1" >> /etc/sysctl.conf
echo "1" >/proc/sys/net/bridge/bridge-nf-call-iptables
sysctl -p

systemctl restart docker

reboot

# ipv6相关错误可以忽略
# 查看模块
# ls /lib/modules/`uname -r`/kernel/net/ipv6/netfilter/
```


# 6、ubuntu 22.04

```bash
vi /etc/default/grub

GRUB_CMDLINE_LINUX="cgroup_memory=1 cgroup_enable=memory swapaccount=1 systemd.unified_cgroup_hierarchy=0"
更新
sudo update-grub

vi  /etc/sysctl.conf

net.bridge.bridge-nf-call-iptables=1

桌面版还要禁用大内存页
设置vm.nr_hugepages=0

重启
```


# 7、部署rancher server

单节点部署rancher server  

```bash
# 清理历史部署痕迹
cd cube-studio/install/kubernetes/rancher/
sh reset_docker.sh

# 需要拉取镜像(非2.6.2版本需要执行wget，2.6.2版本已拉取过了)
wget https://github.com/rancher/rancher/releases/download/v2.6.2/rancher-images.txt

sh pull_rancher_images.sh 

export RANCHER_CONTAINER_TAG=v2.6.2
sudo docker run -d --privileged --restart=unless-stopped -p 443:443 --name=myrancher -e AUDIT_LEVEL=3 rancher/rancher:$RANCHER_CONTAINER_TAG
# 打开 https://xx.xx.xx.xx:443/ 等待web界面可以打开。预计要1~10分钟
# 查看登陆密码
docker logs  myrancher  2>&1 | grep "Bootstrap Password:"
```

# 8、rancher server 启动可能问题

8.1、permission denied

mount 查看所属盘是否有noexec 限制

8.2、重启机器，rancher server无法正常启动。

如果是一直等待k3s 启动，按照下面的高可用方法。  
如果是k3s启动失败，docker exec -it myrancher cat k3s.log > k3s.log  查看k3s的日志  
如果k3s日志报错 iptable的问题，那就按照上面的centos8或者ubuntu22.04配置iptable，  
如果k3s日志报错 containerd的问题，那就 docker exec -it myrancher mv /var/lib/rancher/k3s/agent/containerd /varllib/rancher/k3slagent/_containerd  

# 9、部署k8s集群

部署完rancher server后，进去rancher server的https://xx.xx.xx.xx/ 的web界面，这里的xx取决于你服务器的IP地址。

选择“Set a specific password to use”来配置rancher的密码，不选择"Allow collection of anonymous statistics ......"，选择"I agree to the terms and conditions ......"。
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/bf9eb26c1ee14ef4b18b02fbf3c17f7a.png)

之后选择添加集群->选择自定义集群->填写集群名称
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/235de769236b4643b2c9a2eb1b109100.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/7693842628df47a7b5aefdd36a000f82.png)

然后选择kubernetes的版本（注意：这个版本在第一次打开选择页面时可能刷新不出来，需要等待1~2分钟再刷新才能显示）

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/f67444489647471494fa2d3ed062ee6d.png)

修改Advanced option，主要是禁用nginx ingress，修改端口范围，使用docker info检查服务器上的docker根目录是否和默认的一致，不一致则需要更改。

之后选择编辑yaml文件。 添加kubelet的挂载参数，需要把分布式存储的位置都加入挂载。可以添加一个非根目录的父目录。

```bash
    kube-api:
      ...
    kubelet:
      extra_binds:
        - '/data:/data'
```

这个yaml文件中控制着k8s基础组件的启动方式。比如kubelet的启动参数，api-server的启动参数等等。

有几个需要修改的是k8s使用的网段，由于默认使用的是10.xx，如果和你公司的网段重复，则可以修改为其他网关，例如：
172.16.0.0/16和172.17.0.0/16 两个网段

services部分的示例（注意缩进对齐）

```bash
  services:
    etcd:
      backup_config:
        enabled: true
        interval_hours: 12
        retention: 6
        safe_timestamp: false
      creation: 12h
      extra_args:
        election-timeout: '5000'
        heartbeat-interval: '500'
      gid: 0
      retention: 72h
      snapshot: false
      uid: 0
    kube-api:
      always_pull_images: false
      pod_security_policy: false
      # 服务node port范围
      service_node_port_range: 10-32767
      # 服务的ip范围，如果公司ip网段与k8s网段有冲突，则需要改这里
      service_cluster_ip_range: 172.16.0.0/16
      # 证书 https版本isito需要，k8s在1.21版本以下的，需要加extra_args
      extra_args:     
        service-account-issuer: kubernetes.default.svc
        service-account-signing-key-file: /etc/kubernetes/ssl/kube-service-account-token-key.pem
    kube-controller:
      # 集群pod的ip范围，如果公司ip网段与k8s网段有冲突，则需要改这里
      cluster_cidr: 172.17.0.0/16
      # 集群服务的 ip 范围，如果公司ip网段与k8s网段有冲突，则需要改这里
      service_cluster_ip_range: 172.16.0.0/16
    kubelet:
      # dns服务的ip，如果公司ip网段与k8s网段有冲突，则需要改这里
      cluster_dns_server: 172.16.0.10
      # 主机镜像回收触发门槛，如果机器空间小，可以把这两个参数调高
      extra_args:
        image-gc-high-threshold: 90
        image-gc-low-threshold: 85
      # kubelet挂载主机目录，这样才能使用subpath，所有情况下部署都必加，且仅此处是必须要加的
      extra_binds:
        - '/data:/data'
    kubeproxy: {}
    scheduler: {}
```

如果有其他的参数需要后面修改，我们可以再在这里对yaml文件进行修改，然后升级集群。 

修改后直接进入下一步。
接着可以选择节点的角色：etcd是用来部署k8s的数据库，可以多个节点etcd。control相当于k8s的master，用来部署控制组件，可以在多个节点的部署k8s master节点，实现k8s高可用。worker相当于k8s的工作节点。

我们可以在部署rancher server的这台机器上，添加etcd/control角色。(如果你准备单机部署或者只是简单尝试，可以把所有角色都选上)

最后复制页面中显示的命令，接着在rancher server的终端上粘贴命令，这样就部署了一个没有worker节点的k8s集群。

粘贴后等待部署完成就行了。

部署完成后，集群的状态会变为"Active"，之后就可以继续其他的操作了，比如执行sh start.sh xx.xx.xx.xx等等

# 10、rancher server 高可用
  
 rancher server 有高可用部署方案，可以参考官网https://rancher.com/docs/rancher/v2.x/en/installation/how-ha-works/

## 10.1、单节点的配置高可用

由于官方提供的几种高可用方案，要么需要ssh互联，要么需要跳板机账号密码，这些都无法在idc环境实现。
并且使用单容器模式部署的时候，如果docker service或者机器重启了，rancher server就会报异常。一般会报wait k3s start的错误。
因此下面提供一种方案，能使在单容器模式下，机器重启后，rancher server仍可用。
```bash

systemctl stop firewalld
systemctl disable firewalld
systemctl stop iptables
systemctl disable iptables
systemctl stop ip6tables
systemctl disable ip6tables
systemctl stop nftables
systemctl disable nftables

export RANCHER_CONTAINER_NAME=myrancher
export RANCHER_CONTAINER_TAG=v2.6.2

docker stop $RANCHER_CONTAINER_NAME
docker create --volumes-from $RANCHER_CONTAINER_NAME --name rancher-data rancher/rancher:$RANCHER_CONTAINER_TAG
# 先备份一遍
docker run --volumes-from rancher-data --privileged -v $PWD:/backup alpine tar zcvf /backup/rancher-data-backup.tar.gz /var/lib/rancher
docker run --name myrancher-new -d --privileged --volumes-from rancher-data --restart=unless-stopped -p 443:443 rancher/rancher:$RANCHER_CONTAINER_TAG
# 等到新web界面正常打开
docker rm $RANCHER_CONTAINER_NAME
```

然后就可以把原有容器删除掉了。
这个新启动的容器，在docker service重启后是可以继续正常工作的。

## 10.2、配置认证过期

因为rancher server的证书有效期是一年，在一年后，rancher server会报证书过期。因此，可以通过下面的方式，创建新的证书。

```bash
docker stop $RANCHER_CONTAINER_NAME
docker start $RANCHER_CONTAINER_NAME 
docker exec -it $RANCHER_CONTAINER_NAME sh -c "mv k3s/server/tls k3s/server/tls.bak" 
docker logs --tail 3 $RANCHER_CONTAINER_NAME 

# 将出现类似于以下的内容: 
# 2021/01/03 03:07:01 [INFO] Waiting for server to become available: Get https://localhost:6443/version?timeout=30s: x509: certificate signed by unknown authority 
# 2021/01/03 03:07:03 [INFO] Waiting for server to become available: Get https://localhost:6443/version?timeout=30s: x509: certificate signed by unknown authority 
# 2021/01/03 03:07:05 [INFO] Waiting for server to become available: Get https://localhost:6443/version?timeout=30s: x509: certificate signed by unknown authority 

docker stop $RANCHER_CONTAINER_NAME 
docker start $RANCHER_CONTAINER_NAME
```

# 11、部署完成后需要部分修正

1、因为metric-server默认镜像拉取是Always，所以要修改成imagePullPolicy: IfNotPresent
2、nginx如果不想使用，或者因为端口占用只在部分机器上使用，可以添加亲密度不启动或者在部分机器上启动。
affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              
              - key: ingress-nginx
                operator: In
                values:
                - "true"
 3、由于coredns在资源limits太小了，因此可以取消coredns的limits限制，不然dns会非常慢，整个集群都会缓慢

# 12、机器扩容
现在k8s集群已经有了一个master节点，但还没有worker节点，或者想添加更多的master/worker节点就需要机器扩容了。

在集群主机界面，点击编辑集群

修改docker目录为上面已修改的docker根目录/data/docker，然后选择角色为worker（根据自己的需求选择角色）

之后复制命令到目标主机上运行，等待完成就可以了。

# 13、rancher/k8s 多用户
如果集群部署好了，需要添加多种权限类型的用户来管理，则可以使用rancher来实现k8s的rbac的多用户。

# 14、客户端kubectl
如果你不会使用rancher界面或者不习惯使用rancher界面，可以使用kubectl或者kubernetes-dashboard。

点击Kubeconfig文件可以看到config的内容，通过内容可以看到，kube-apiserver可以使用rancher-server（端口443）的api接口，或者kube-apiserver（端口6443）的接口控制k8s集群。
由于6443端口在idc网络里面并不能暴露到外面，所以主要使用rancher-server的443端口代理k8s-apiserver的控制。
提示，如果你的rancher server 坏了，你可以在内网通过6443端口继续控制k8s集群。

下载安装不同系统办公电脑对应的kubectl，然后复制config到~/.kube/config文件夹，就可以通过命令访问k8s集群了。

# 15、kubernetes-dashboard
如果你喜欢用k8s-dashboard，可以自己安装dashboard。
可以参考这个：https://kuboard.cn/install/install-k8s-dashboard.html

这样我们就完成了k8s的部署。

# 16、节点清理
当安装失败需要重新安装，或者需要彻底清理节点。由于清理过程比较麻烦，我们可以在rancher界面上把node删除，然后再去机器上执行reset_docker.sh，这样机器就恢复了部署前的状态。

如果web界面上删除不掉，我们也可以通过kubectl的命令  

```bash
kubectl delete node node12
```

# 17、rancher server 节点迁移
我们可以实现将rancher server 节点迁移到另一台机器，以防止机器废弃后无法使用的情况。

首先，先在原机器上把数据压缩，不要关闭源集群rancher server 因为后面还要执行kubectl，这里的.tar.gz的文件名称以实际为准

```bash
docker create --volumes-from myrancher-new --name rancher-data-new rancher/rancher:$rancher_version
docker run --volumes-from rancher-data-new  -v $PWD:/backup alpine tar zcvf /backup/rancher-data-backup-20210101.tar.gz /var/lib/rancher
```

之后把tar.gz 文件复制到新的rancher server机器上,这里的.tar.gz的文件名称以实际为准
```
tar -zxvf rancher-data-backup-20210101.tar.gz && mv var/lib/rancher /var/lib/
```

接着在新机器上启动新的rancher server

```
sudo docker run -d --restart=unless-stopped -v /var/lib/rancher:/var/lib/rancher -p 443:443 --privileged --name=myrancher -e AUDIT_LEVEL=3 rancher/rancher:$rancher_version

注意以下几点：
1、新rancher server的web界面上要修改rancher server的url
2、打开地址 https://新rancher的ip/v3/clusters/源集群id/clusterregistrationtokens
例如：
https://100.116.64.86/v3/clusters/c-dfqxv/clusterregistrationtokens
3、在上面的界面上找到最新时间的 insecureCommand 的内容，之后curl --insecure -sfl过去
curl --insecure -sfL https://100.108.176.29/v3/import/d9jzxfz7tmbsnbhf22jbknzlbjcck7c2lzpzknw8t8sd7f6tvz448b.yaml | kubectl apply -f -
```

配置kubeconfig文件为原集群，执行上面的命令。这样旧rancher上的应用就会连接到新的rancher server；

等新集群正常以后，将新机器加入到k8s集群的etcd controller节点；

将老机器踢出集群；

至此完成


# 18、总结

rancher使用**全部容器化**的形式来部署k8s集群，能大幅度降低k8s集群扩部署/缩容的门槛。
你可以使用rancher来扩缩容 etcd，k8s-master，k8s-worker。
k8s集群(包括etcd)的增删节点动作是由rancher server节点控制，由rancher agent来执行的。在新节点上通过运行rancher agent容器，来访问rancher server 获取要执行的部署命令,部署对应的k8s组件容器（包含kubelet，api-server，scheduler，controller等）。

rancher本身并不改变k8s的基础组件和工作原理，k8s的架构依然不变，只不过多了一个认证代理（auth proxy），也就是前面说的config文件中的rancher server中的接口。


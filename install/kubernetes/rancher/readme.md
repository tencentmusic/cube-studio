# 内网使用rancher自建k8s集群

如果可以使用公有云k8s，可以直接构建公有云厂商的容器服务。这里介绍如何在内网使用k8s自建k8s集群

# 建设前准备

如果内网无法连接到互联网的话，需要在内网申请一个docker仓库，如果内网没有docker仓库可以使用Harbor自建一个内网仓库。比如内网的仓库docker.oa.com

# 将基础组件推送到内网仓库，并来idc机器拉取

内网无法连接外网，需要我们提前拉好镜像。如果你的机器可以连接外网，可以忽略这一部分的操作。

关于镜像的版本与rancher版本、k8s版本有关。可以选择一个能够部署k8s 1.18的rancher版本：https://github.com/rancher/rancher/releases

比如我这里使用的是rancher_version=v2.5.2，这个版本依赖的镜像，在官网https://github.com/rancher/rancher/releases/tag/$rancher_version  中找到依赖的镜像txt文件

https://github.com/rancher/rancher/releases/download/$rancher_version/rancher-images.txt

将依赖镜像在开发网拉取下来，然后重新tag成内网仓库镜像，例如docker.oa.com域名下的镜像，推送到docker.oa.com上，后面需要在idc每个机器上拉取下来，再tag成原始镜像名。例如

## 可以连接外网的机器上
docker pull rancher/rancher-agent:$rancher_version
docker tag rancher/rancher-agent:$rancher_version docker.oa.com:8080/public/rancher/rancher-agent:$rancher_version

## 内网idc机器
docker pull docker.oa.com/public/rancher/rancher-agent:$rancher_version
docker tag docker.oa.com/public/rancher/rancher-agent:$rancher_version rancher/rancher-agent:$rancher_version
由于依赖镜像比较多，可以写一个脚本，批量的拉取和tag。

# 初始化节点

可以批量下发init_node.sh和reset_docker.sh的脚本。

init_node.sh 是为了初始化机器，可以把自己的要做的初始化任务加入到其中
修改hostname，修改dns或者host，
卸载已经存在的docker，安装docker并把默认路径放在/data/docker目录下，防止放在/var/lib/docker目录下机器磁盘会满
为docker添加私有仓库拉取权限

reset_docker.sh 是为了在机器从rancher集群踢出以后，把rancher环境清理感觉，回归自由机器。


# 部署rancher server

# 部署k8s集群

单节点部署rancher server
```bash
sudo docker run -d --restart=unless-stopped -p 443:443 --privileged --name=myrancher -e AUDIT_LEVEL=3 rancher/rancher:$rancher_version
```

进去rancher server的https://xx.xx.xx.xx/ 的web界面，选择添加集群，选择自定义集群。填写集群名称

选择kubernetes版本（注意：这个的版本第一次打开时可能刷新不出来，需要等待1~2分钟再刷新才能显示出来）

之后选择编辑yaml文件。

这个yaml文件中控制着k8s基础组件的启动方式。比如kubelet的启动参数，api-server的启动参数。

有几个需要修改的是k8s使用的网段，由于默认使用的是10.xx，如果和公司网段重复，可以修改为其他网关，例如
172.16.0.0/16和172.17.0.0/16 两个网段

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
      service_node_port_range: 10-32767
      service_cluster_ip_range: 172.16.0.0/16
      extra_args:     
        service-account-issuer: kubernetes.default.svc
        service-account-signing-key-file: /etc/kubernetes/ssl/kube-service-account-token-key.pem
    kube-controller:
      cluster_cidr: 172.17.0.0/16
      service_cluster_ip_range: 172.16.0.0/16
    kubelet:
      cluster_dns_server: 172.16.0.10
      extra_args:
        image-gc-high-threshold: 90
        image-gc-low-threshold: 85
      extra_binds:
        - '/data:/data'
    kubeproxy: {}
    scheduler: {}
```

如果有其他的参数需要后面修改，也可以在这里修改yaml文件，然后升级集群。 

修改后直接进入下一步。这里可以选择节点的角色。etcd用来部署k8s的数据库，可以多个节点etcd。control相当于k8s的master，用来部署控制组件，可以在多个节点的部署k8s master节点，实现k8s高可用。worker相当于k8s的工作节点。

我们可以在rancher server这台机器可以添加为etcd/control角色。(如果你只有一台机器或者只是简单尝试，可以把所有角色都选上)

复制web中显示的命令，直接在rancher server上粘贴命令，这样就部署了一个k8s集群。只不过现在还没有worker节点。

粘贴后等待部署就行了。

部署完成的样子


# rancher server 高可用
 rancher server 有高可用部署方案，可以参考官网https://rancher.com/docs/rancher/v2.x/en/installation/how-ha-works/

## 单节点的配置高可用


官方提供的几种高可用方案，要么需要ssh互联，需要需要跳板机账号密码都无法在idc环境实现。并且使用单容器模式部署的时候，docker service或者机器重启，rancher server都会报异常。一直报wait k3s start。下面提供一种方案能使单容器模式下，机器重启rancher server仍可用。
```bash
$ docker stop $RANCHER_CONTAINER_NAME
$ docker create --volumes-from $RANCHER_CONTAINER_NAME --name rancher-data rancher/rancher:$RANCHER_CONTAINER_TAG
# 先备份一遍
$ docker run --volumes-from rancher-data -v $PWD:/backup alpine tar zcvf /backup/rancher-data-backup-$RANCHER_VERSION-$DATE.tar.gz /var/lib/rancher
$ docker run -d --volumes-from rancher-data --restart=unless-stopped -p 80:80 -p 443:443 rancher/rancher:latest
```

然后就可以把原有容器删除掉了。这个新启动的容器，在docker service重启后是可以继续正常工作的。

## 配置认证过期

rancher server的证书有效期是一年，在一年后，rancher server会报证书过期。通过下面的方式你可以创建新的证书。
```bash
docker stop $RANCHER_CONTAINER_NAME
docker start $RANCHER_CONTAINER_NAME 
docker exec -it $RANCHER_CONTAINER_NAME sh -c "mv k3s/server/tls k3s/server/tls.bak" 
docker logs --tail 3 $RANCHER_CONTAINER_NAME 
# Something similar to the below will appear: 
# 2021/01/03 03:07:01 [INFO] Waiting for server to become available: Get https://localhost:6443/version?timeout=30s: x509: certificate signed by unknown authority 
# 2021/01/03 03:07:03 [INFO] Waiting for server to become available: Get https://localhost:6443/version?timeout=30s: x509: certificate signed by unknown authority 
# 2021/01/03 03:07:05 [INFO] Waiting for server to become available: Get https://localhost:6443/version?timeout=30s: x509: certificate signed by unknown authority 
docker stop $RANCHER_CONTAINER_NAME 
docker start $RANCHER_CONTAINER_NAME
```



部署完成后需要部分修正。

1、metric-server默认镜像拉取是Always，修改成imagePullPolicy: IfNotPresent
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
 3、coredns在资源limits太小了，可以取消coredns的limits限制，不然dns非常慢，整个集群都会缓慢

# 机器扩容
现在k8s集群已经有了一个master节点，还没有worker节点，或者想添加更多master/worker节点就要机器扩容了。

在集群主机界面，点击编辑集群

修改docker目录为上面修改的docker根目录/data/docker，然后选择角色为worker（根据自己的需求选择角色）

复制命令到目标主机上运行，等待安装就可以了。

# rancher/k8s 多用户
集群部署好了，需要添加多种权限类型的用户来管理。可以使用rancher来实现k8s的rbac的多用户。

# 客户端kubectl
如果你不会使用rancher界面或者不习惯使用rancher界面，可以使用kubectl或者kubernetes-dashboard。

点击Kubeconfig文件可以看到config的内容，通过内容可以看到，kube-apiserver可以使用rancher-server（端口443）的api接口，或者kube-apiserver（端口6443）的接口控制k8s集群。由于6443端口在idc网络里面并不能暴露到外面，所以主要使用rancher-server的443端口代理k8s-apiserver的控制。提示，如果你的rancher server 坏了，你可以在内网通过6443端口继续控制k8s集群。

下载安装不同系统办公电脑对应的kubectl，然后复制config到~/.kube/config文件。就可以通过命令访问k8s集群了。

这样我们就完成k8s的部署。

# 节点清理
如果安装失败需要重新安装，有时需要彻底清理节点。清理过程比较麻烦。我们可以在rancher界面上把node删除，然后再去机器上执行reset_docker.sh，这样机器就恢复了自由状态。

如果web界面上删除不掉，我们也可以通过kubectl命令  
```bash
kubectl delete node node12
```

# rancher server 节点迁移
可以实现rancher server 节点迁移到另一台机器，以防止机器废弃的情况。

先在原机器上把数据压缩，不要关闭源集群rancher server 因为后面还要执行kubectl
```bash
docker create --volumes-from myrancher-new --name rancher-data-new rancher/rancher:$rancher_version
docker run --volumes-from rancher-data-new -v $PWD:/backup alpine tar zcvf /backup/rancher-data-backup-20210101.tar.gz /var/lib/rancher
```

把tar.gz 文件复制到新的rancher server机器上
```
tar -zxvf rancher-data-backup-20210101.tar.gz && mv var/lib/rancher /var/lib/
```
新机器上启动新的rancher server
```
sudo docker run -d --restart=unless-stopped -v /var/lib/rancher:/var/lib/rancher -p 443:443 --privileged --name=myrancher -e AUDIT_LEVEL=3 rancher/rancher:$rancher_version

1、新rancher server的web界面上修改rancher server的url
2、打开地址 https://新rancher的ip/v3/clusters/源集群id/clusterregistrationtokens
类似
https://100.116.64.86/v3/clusters/c-dfqxv/clusterregistrationtokens
3、在上面的界面上找到最新时间的 insecureCommand 的内容
curl --insecure -sfL https://100.108.176.29/v3/import/d9jzxfz7tmbsnbhf22jbknzlbjcck7c2lzpzknw8t8sd7f6tvz448b.yaml | kubectl apply -f -
```
配置kubeconfig文件为原集群，执行上面的命令。这样让旧rancher上的应用就会连接新rancher server

4、等新集群正常以后，将新机器加入到k8s集群，etcd controller节点。

5、将老机器踢出集群。

至此完成


# 总结

rancher使用全部容器化的形式来部署k8s集群，能大幅度降低k8s集群扩部署/缩容的门槛。可以使用rancher来扩缩容 etcd，k8s-master，k8s-worker。k8s集群(包括etcd)的增删节点动作是由rancher server节点控制，由rancher agent来执行的。在新节点上通过运行rancher agent容器，来访问rancher server 获取要执行的部署命令。部署对应的k8s组件容器（包含kubelet，api-server，scheduler，controller等）。

rancher本身并不改变k8s的基础组件和工作原理，k8s的架构依然是下面图形中的工作模式。只不过多了一个认证代理（auth proxy），也就是前面说的config文件中的rancher server中的接口。


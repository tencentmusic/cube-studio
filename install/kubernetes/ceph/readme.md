
参考：https://www.tangyuecan.com/2020/02/17/%E5%9F%BA%E4%BA%8Ek8s%E6%90%AD%E5%BB%BAceph%E5%88%86%E9%83%A8%E7%BD%B2%E5%AD%98%E5%82%A8/
https://blog.51cto.com/u_14034751/2542998
# 前提比较清楚首先需要如下东西：

正常运行的多节点K8S集群，可以是两个节点也可以是更多。
每一个节点需要一个没有被分区的硬盘，最好大小一致不然会浪费。
没错其实就是一个要求，必须有集群才能进行容器管理，必须有硬盘才能做存储这些都是基础。


#载入rook项目的rook/cluster/examples/kubernetes/ceph下面

#所有的pod都会在rook-ceph命名空间下创建
kubectl create -f common.yaml
 
#所有的pod都会在rook-ceph命名空间下创建
kubectl create -f crd.yaml
 
#部署Rook操作员
kubectl create -f operator.yaml
 
 
#创建Rook Ceph集群
kubectl create -f cluster.yaml
 
 # 部署文件系统

目前Ceph支持块存储、文件系统存储、对象存储三种方案，我们就选择文件系统存储

kubectl create -f filesystem.yaml 
kubectl create -f storageclass.yaml

#部署Ceph toolbox 命令行工具
#默认启动的Ceph集群，是开启Ceph认证的，这样你登陆Ceph组件所在的Pod里，是没法去获取集群状态，以及执行CLI命令，这时需要部署Ceph toolbox，命令如下
kubectl create -f toolbox.yaml
 
 
#进入ceph tool容器
kubectl exec -it pod/rook-ceph-tools-545f46bbc4-qtpfl -n rook-ceph bash
 
 
#查看ceph状态
ceph status


# dashboard没有启用SSL的   代理mgr的一个端口
kubectl create -f dashboard-external-http.yaml

默认用户名为
admin
 
 
密码获取方式执行如下命令
kubectl -n rook-ceph get secret rook-ceph-dashboard-password -o jsonpath="{['data']['password']}" | base64 --decode && echo
 

# 清理ceph集群

每个节点都需要执行
rm -rf /var/lib/rook/*
rm -rf /var/lib/kubelet/plugins/rook-ceph.*
rm -rf /var/lib/kubelet/plugins_registry/rook-ceph.*
https://rook.io/docs/rook/v1.4/ceph-teardown.html

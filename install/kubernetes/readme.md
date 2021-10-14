# 机器/k8s环境/ceph环境准备  
机器环境准备：准备docker，准备rancher，部署k8s。如果已经有可以忽略，没有可以参考rancher/readme.md  

如果使用tke选择独立集群(非托管集群)，这样方便自行管理api server启动参数，部署istio

# 分布式存储

目前机器学习平台依赖强io性能的分布式存储。  建议使用ssd的ceph作为分布式存储。并注意配置好开机自动挂载避免在机器重启后挂载失效

 ！！！重要：分布式文件系统需要挂载到每台机器的/data/k8s/下面，当然也可以挂载其他目录下，以软链的形式链接到/data/k8s/下 

需要每台机器都有对应的目录/data/k8s为分布式存储目录
```bash  
mkdir -p /data/k8s/kubeflow/minio  
mkdir -p /data/k8s/kubeflow/global  
mkdir -p /data/k8s/kubeflow/pipeline/workspace  
mkdir -p /data/k8s/kubeflow/pipeline/archives  
```  
  
# gpu环境的准备  
1、找运维同学在机器上安装gpu驱动  
2、安装nvidia docker2（k8s没有支持新版本docker的--gpu）  
3、修改docker配置文件  
```abash  
cat /etc/docker/daemon.json  
  
{  
    "insecure-registries":["docker.oa.com:8080"],  
    "default-runtime": "nvidia",  
    "runtimes": {  
        "nvidia": {  
            "path": "/usr/bin/nvidia-container-runtime",  
            "runtimeArgs": []  
        }  
    }  
}  
```  
# 如果是内网部署，需要先把镜像传递到私有仓库，再从私有仓库拉取到每台机器上  
  
```bash  
sh pull_image_kubeflow.sh  
```  
  
# 创建命名空间、秘钥(有私有仓库，自己添加)  
将本地的~/.kube/config文件换成将要部署的集群的kubeconfig  
  
```bash  
修改里面的docker hub拉取账号密码  
sh create_ns_secret.sh  
```  
  
# 部署k8s-dashboard  
新版本的k8s dashboard 可以直接开放免登陆的http，且可以设置url前缀代理  
```bash  
kubectl apply -f dashboard/v2.2.0-cluster.yaml  
kubectl apply -f dashboard/v2.2.0-user.yaml  
```  
# 部署kube-batch  
kube-batch用来实现gang调度  
```bash  
kubectl apply -f kube-batch/deploy.yaml  
```  
  
# 部署元数据组件mysql  
参考mysql/readme.md  

创建所需要的各类数据库
```bash
CREATE DATABASE IF NOT EXISTS kubeflow DEFAULT CHARACTER SET utf8 DEFAULT COLLATE utf8_general_ci;
CREATE DATABASE IF NOT EXISTS mlpipeline DEFAULT CHARACTER SET utf8 DEFAULT COLLATE utf8_general_ci;
CREATE DATABASE IF NOT EXISTS katib DEFAULT CHARACTER SET utf8 DEFAULT COLLATE utf8_general_ci;
CREATE DATABASE IF NOT EXISTS metadb DEFAULT CHARACTER SET utf8 DEFAULT COLLATE utf8_general_ci;
CREATE DATABASE IF NOT EXISTS cachedb DEFAULT CHARACTER SET utf8 DEFAULT COLLATE utf8_general_ci;
```
  
# 部署缓存组件redis  
参考redis/readme.md  
  
# 部署kubeflow  
参考kubeflow/v1.2.0/readme.md  
  
# 部署prometheus生态组件  
参考prometheus/readme.md  
  
# 部署efk生态组件  
参考efk/readme.md  
  
# 部署frameworkercontroller组件  
参考frameworkercontroller/readme.md  
  
# 部署volcano组件  
```bash
kubectl apply -f volcano/volcano-development.yaml
```

# 部署 管理平台  

如果涉及到多集群部署，修改kubeconfig中的文件，文件名为$cluster-config，并在每个集群中部署configmap

```bash
kubectl delete configmap kubernetes-config -n infra
kubectl create configmap kubernetes-config --from-file=kubeconfig -n infra

kubectl delete configmap kubernetes-config -n pipeline
kubectl create configmap kubernetes-config --from-file=kubeconfig -n pipeline

kubectl delete configmap kubernetes-config -n katib
kubectl create configmap kubernetes-config --from-file=kubeconfig -n katib
```

  
组件说明  
 - cube/base/deploy.yaml为myapp的前后端代码  
 - cube/base/deploy-schedule.yaml 为任务产生器  
 - cube/base/deploy-worker.yaml 为任务执行器  
 - cube/base/deploy-watch.yaml 任务监听器  

配置文件说明  
 - cube/overlays/config/entrypoint.sh 镜像启动脚本  
 - cube/overlays/config/config.py  配置文件，需要将其中的配置项替换为自己的  
  
部署入口  
cube/overlays/kustomization.yml    
  
修改kustomization.yml中需要用到的环境变量。例如HOST为平台的域名，需要指向istio ingressgate的服务(本地调试可以写入到/etc/hosts文件中)  
  
部署执行命令  
```bash  
为部署控制台容器的机器添加lable,  kubeflow-dashboard=true
kubectl apply -k cube/overlays  
```  
  
## 部署pv-pvc.yaml  
  
```bash  
kubectl create -f pv-pvc-infra.yaml  
kubectl create -f pv-pvc-jupyter.yaml  
kubectl create -f pv-pvc-katib.yaml  
# kubectl create -f pv-pvc-kubeflow.yaml  
kubectl create -f pv-pvc-pipeline.yaml  
kubectl create -f pv-pvc-service.yaml  
```  

# 部署平台入口  
```bash  
# 创建8080网关服务  
kubectl apply -f gateway.yaml  
# 创建新的账号需要  
kubectl apply -f sa-rbac.yaml          
# 修改并创建virtual。需要将其中的域名kubeflow.local.com批量修改为平台的域名
kubectl apply -f virtual.yaml  
```  
  
# 通过label进行机器管理  

- 对于cpu的train/notebook/service会选择cpu=true的机器  
- 对于gpu的train/notebook/service会选择gpu=true的机器  

- 训练任务会选择train=true的机器  
- notebook会选择notebook=true的机器  
- 服务化会选择service=true的机器  
- 不同项目的任务会选择对应org=xx的机器。默认为org=public 
  

# kube-scheduler调度策略
可以参考 https://docs.qq.com/doc/DWFVqcFd0S29WUGxD


# 版本升级
数据库升级，数据库记录要批量添加默认值到原有记录上，不然容易添加失败



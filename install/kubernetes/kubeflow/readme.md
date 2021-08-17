
# 基础环境需求
k8s集群，分布式存储，这些都假设在前面的步骤已经完成

# 先创建官方kubeflow
参考v1.2.0/readme.md

# 部署kubeflow-pipeline

kubeflow每一部分相应的可以独立升级
下载github官方版本
在官方github的pipelines/manifests/kustomize/readme.md中都可以看到Install it to any K8s cluster的方法

修改pipeline运行实例到pipeline命名空间，而不是官方的kubeflow命名空间
```bash
修改ml的 NAMESPACE_TO_WATCH NAMESPACE POD_NAMESPACE ARGO_NAMESPACE 变量的值为pipeline
把资源的limit改大或者去掉

```


1.6.0 版本替换为自己的mysql数据库（未测试）
```
注释pipeline/1.6.0/kustomize/cluster-scoped-resources/kustomization.yaml中的 cache部署 ,namespace的部署
注释pipeline/1.6.0/kustomize/base/installs/generic/kustomization.yaml中的cache部署
注释pipeline/1.6.0/kustomize/env/platform-agnostic/kustomization.yaml 中的 mysql部署
修改pipeline/1.6.0/kustomize/base/installs/generic/mysql-secret.yaml 中的mysql账号密码
修改pipeline/1.6.0/kustomize/base/installs/generic/pipeline-install-config.yaml 中mysql的地址信息


部署1.6.0版本
cd pipeline/1.6.0/kustomize
kubectl apply -k cluster-scoped-resources/
kubectl wait crd/applications.app.k8s.io --for condition=established --timeout=60s
kubectl apply -k env/platform-agnostic/
```

1.0.4版本替换为自己的mysql
```
注释pipeline/1.0.4/kustomize/cluster-scoped-resources/kustomization.yaml中的 cache部署,namespace的部署
注视pipeline/1.0.4/kustomize/base/kustomization.yaml中的 cache部署
注释pipeline/1.0.4/kustomize/env/platform-agnostic/kustomization.yaml 中的 mysql部署
修改pipeline/1.0.4/kustomize/base/params-db-secret.env 中的mysql账号密码
修改pipeline/1.0.4/kustomize/base/params.env 中mysql的地址信息

部署1.0.4版本
cd pipeline/1.0.4/kustomize
kubectl apply -k cluster-scoped-resources/
kubectl wait crd/applications.app.k8s.io --for condition=established --timeout=60s
kubectl apply -k env/platform-agnostic/
kubectl wait applications/pipeline -n kubeflow --for condition=Ready --timeout=1800s

```

##  部署minio-pv
按照自己集群的分布式存储方式创建pv
```bash
# pipline需要的minio
kubectl create -f pipeline/minio-pv-hostpath.yaml        
```
1、原有账号argo要集群角色，才能获取到pipeline中的信息,pipeline中的所有role都改成集群角色
2、workflow的环境变量，启动命令，configmap都改成
```
kubectl apply -f pipeline/minio-artifact-secret.yaml
kubectl apply -f pipeline/pipeline-runner-rolebinding.yaml
```



# 部署xgboost-operator
下载github官方版本
在官方github的xgboost-operator/manifests/中是部署文件

```bash
kubectl kustomize  xgboost-operator/manifests/base | kubectl apply -f -
```

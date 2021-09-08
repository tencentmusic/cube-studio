
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


1.6.0 版本更改

	为所有的Deployment打上节点亲和度补丁
	修改流水线执行的命名空间
	修改mysql服务host和密码为自己部署的
	minio不要用subPath,rancher部署的集群不支持
	minio-pvc 加一个selector
	修改 metadata-writer 的 NAMESPACE_TO_WATCH 环境变量
	修改 ml-pipeline 的 POD_NAMESPACE 环境变量
	修改 ml-pipeline-persistenceagent 的 NAMESPACE 环境变量
	所有镜像拉取策略改为IfNotPresent
	修改某些跨命名空间的serviceaccount权限
	不部署cache模块和mysql模块

部署1.6.0版本

	cd pipeline/1.6.0/kustomize  
	kustomize build cluster-scoped-resources/ | kubectl apply -f -
	kubectl wait crd/applications.app.k8s.io --for condition=established --timeout=60s  
	kustomize build env/platform-agnostic/  | kubectl apply -f -
	kubectl wait applications/pipeline -n kubeflow --for condition=Ready --timeout=1800s  
	# 注意：部署前需要在env/platform-agnostic/kustomize.yaml第41、49、51行分别填写mysql的账号、密码、host。如果是用教程默认部署的mysql，则不用修改。
	# 注意：1.6.0版本初始化时如果检测到mysql里有mlpipeline库，不会在里面建表。所以部署前保证mlpipeline库已经建好表
	#或者没有mlpipeline库
	# 注意：需要kustomize版本大于v3.0.0，安装可下载releases：https://github.com/kubernetes-sigs/kustomize/releases/tag/kustomize%2Fv4.3.0
	#如果kubectl版本大于等于v1.22.1，也可以直接用kubectl apply -k 安装。

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


# 基础环境需求
k8s集群，分布式存储，这些都假设在前面的步骤已经完成

# 创建角色权限
```bash
kubectl apply -f sa-rbac.yaml
```
# 部署客户端准备
下载客户端到根目录  https://github.com/kubeflow/kfctl/releases/tag/v1.2.0

# 部署文件修改

可以自己修改kfctl_k8s_istio.v1.2.0.yaml中的部署文件。这里文件定义了要去运行和部署哪些文件。真正的部署文件都在kustomize文件夹中。

若私有网络部署需要
 - 把knative里面的所有@sha256去掉
 - imagePullPolicy: Always 更改为 IfNotPresent

修改自定义配置
 - 修改v1.2.0/.cache/manifests/manifests-1.2.0/katib/installs/katib-external-db/secrets.env 中配置的katib连接的数据库
 - 修改v1.2.0/.cache/manifests/manifests-1.2.0/metadata/overlays/external-mysql/params.env  中配置的metadata连接的数据库

# 修改k8s apiserver启动参数
修改k8s api server启动参数才能正常安装istio

rancher部署的k8s修改(选中集群-升级-编辑yaml)
```
services:
    ...
    kube-api:
      ...
      extra_args:     
        service-account-issuer: kubernetes.default.svc
        service-account-signing-key-file: /etc/kubernetes/ssl/kube-service-account-token-key.pem

```

tke联系腾讯云添加
```bash
    kube-api:
      extra_args:
        service-account-issuer: kubernetes.default.svc
        service-account-signing-key-file=/etc/kubernetes/files/apiserver/service-account.key
```

# 创建mysql数据库
在自己的mysql数据库中创建kubeflow需要的各类db
```bash
CREATE DATABASE IF NOT EXISTS kubeflow DEFAULT CHARACTER SET utf8 DEFAULT COLLATE utf8_general_ci;
CREATE DATABASE IF NOT EXISTS mlpipeline DEFAULT CHARACTER SET utf8 DEFAULT COLLATE utf8_general_ci;
CREATE DATABASE IF NOT EXISTS katib DEFAULT CHARACTER SET utf8 DEFAULT COLLATE utf8_general_ci;
CREATE DATABASE IF NOT EXISTS metadb DEFAULT CHARACTER SET utf8 DEFAULT COLLATE utf8_general_ci;
CREATE DATABASE IF NOT EXISTS cachedb DEFAULT CHARACTER SET utf8 DEFAULT COLLATE utf8_general_ci;

```

# 配置机器label

配置机器label，为机器打上
```bash
区分不同的机型
cpu=true   或者  gpu=true

运行不同类型任务容器所在的机器
train=true
notebook=true
service=true

部署不同的组件所在的机器
istio=true 
knative=true 
kubeflow=true 
logging=true 
monitoring=true 
```


# 部署kubeflow
```bash
# 创建，一次没有成功，可以修正后多次执行
./kfctl apply -V -f kfctl_k8s_istio.v1.2.0.yaml
# 删除（可能清理不干净）
./kfctl delete -V -f kfctl_k8s_istio.v1.2.0.yaml
```
这里相较于官方的部署做了如下修改。可忽略

0、取消了centraldashboard/jupyter/notebook/xgboost/pipeline/argo/metadata等相关组件的部署

1、修改控制容器到专门的控制节点上

2、镜像拉取策略改成IfNotPresent

3、knative controller容器添加特殊的内网环境 host
在knative-serving-install/base/deployment.yaml中添加内网特殊的host，例如
hostAliases:
- hostnames:
- local.example.com
ip: xx.xx.xx.xx

4、knative中镜像名中的@sha256去掉
5、网关入口改为
externalIPs:
- xx.xx.xx.xx

6、notebook controller 添加环境变量
- name: ENABLE_CULLING
  value: "true"


如果要取消istio的自动注入
```bash
可以修改configmap修改policy=disabled

namespace的自动注入
label istio-injection: enabled
label istio-injection: disabled

pod的注入
annotation sidecar.istio.io/inject=true
```

注意：  
1、上面的运行会花费一定的时间。  
2、如果因为网络问题有些镜像没法拉取，可以在机器上执行pull.sh命令，把国内镜像拉取到机器上

# 暴露入口

全都运行正常以后就可以暴露服务入口了。

如果istio-system/istio-ingressgate服务的LoadBalancer无法自动生成公有云ip，需要自己手动配置一个可以访问的ip
```bash
  externalIPs:
  - xx.xx.xx.xx
```


# 平台基础架构

![image](https://user-images.githubusercontent.com/20157705/167534673-322f4784-e240-451e-875e-ada57f121418.png)

完整的平台包含
 - 1、机器的标准化
 - 2、分布式存储(单机可忽略)、k8s集群、监控体系(prometheus/efk/zipkin)
 - 3、基础能力(tf/pytorch/mxnet/valcano/ray等分布式，nni/ray超参搜索)
 - 4、平台web部分(oa/权限/项目组、在线构建镜像、在线开发、pipeline拖拉拽、超参搜索、推理服务管理等)

# 组件说明

| 命名空间           | 组件名                               | 组件说明                              |
|:---------------|:----------------------------------|:----------------------------------|
| infra          | kubeflow-dashboard-frontend       | cube-studio平台的web前端               |
| infra          | kubeflow-dashboard                | cube-studio平台的web后端               |
| infra          | kubeflow-dashboard-schedule       | 用来调度cube-studio系统自带的调度任务，比如定时清理   |
| infra          | kubeflow-dashboard-worker         | 用来执行cube-studio系统自带的调度任务，比如定时清理   |
| infra          | 	kubeflow-watch                   | 用来监控cube-studio平台中的任务，发起通知和信息更新   |
| infra          | 	mysql                            | 平台元数据的存储                          |
| infra          | 	redis                            | 平台缓存，和异步任务对接                      |
| kube-system    | kubernetes-dashboard-cluster      | k8s中pod的管理界面                      |
| kube-system    | dashboard-cluster-metrics-scraper | k8s中pod的管理界面上的pod资源使用情况的插件        |
| kube-system    | nvidia-device-plugin-daemonset    | k8s中使用机器gpu驱动和设备的插件               |
| kube-system    | metrics-server                    | 集群资源使用情况的指标采集，用来在hpa时使用           |
| kube-system    | kubeflow-prometheus-adapter       | 用来将prometheus采集的指标转化为可以用来控制hpa的指标 |
| kubeflow       | minio                             | 对象存储                              |
| kubeflow       | train-operator                    | 分布式训练                             |
| kubeflow       | workflow-controller               | argo 云原生调度                        |
| istio-system   | istio-ingressgateway              | 入口网关，用来代理所有外部访问                   |
| istio-system   | 其他                                | istio基础组件                         |
| monitoring     | dcgm-exporter                     | gpu机器资源监控                         |
| monitoring     | node-exporter                     | cpu机器资源监控                         |
| monitoring     | prometheus-k8s                    | 监控数据存储服务                          |
| monitoring     | grafana                           | 监控数据可视化                           |
| volcano-system | 全部                                | volcano分布式和批调度                    |
| jupyter        | docker-*                          | 用户创建的在线构建镜像的pod                   |
| jupyter        | 其他                                | 用户创建的在线notebook                   |
| service        | 全部                                | 用户创建的内部服务和推理服务                    |
| pipeline       | 全部                                | 用户创建的pipeline任务                   |
| automl         | 全部                                | 用户创建的超参搜索任务                       |

# 平台部署流程

基础环境依赖
 - docker >= 19.03  
 - kubernetes = 1.21 ~ 1.25 
 - kubectl ==1.24
 - cfs/ceph  挂载到每台机器的 /data/k8s/  
 - 单机 磁盘>=200G   单机磁盘容量要求不大，仅做镜像容器的存储  
 - 控制端机器 cpu>=16 mem>=32G 
 - 任务端机器，根据需要自行配置  

本平台依赖k8s/kubeflow/prometheus/efk相关组件，请参考install/kubenetes/README.md 部署依赖组件。

平台完成部署之后如下:

<img width="100%" alt="167874734-5b1629e0-c3bb-41b0-871d-ffa43d914066" src="https://user-images.githubusercontent.com/20157705/168214806-b8aceb3d-e1b4-48f0-a079-903ef8751f40.png">


# 本地开发

管理平台web端可连接多个k8s集群用来在k8s集群上调度发起任务实例。同时管理多个k8s集群时，或本地调试时，可以将k8s集群的config文件存储在kubeconfig目录中，按照$ENVIRONMENT-kubeconfig 的规范命名

./docker 目录包含本地开发方案，涉及镜像构建，docker-compose启动，前端构建，后端编码等过程

参考install/docker/README.md


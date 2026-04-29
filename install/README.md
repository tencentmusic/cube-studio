
# 平台基础架构

![输入图片说明](https://cube-studio.oss-cn-hangzhou.aliyuncs.com/docs/image/infra.png) 

完整的平台包含
 - 1、机器的标准化
 - 2、分布式存储(单机可忽略)、k8s集群、监控体系(prometheus/grafana)
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

 - docker >= 1.19.x docker 存储目录>1T
 - kubernetes = 1.25~1.31，建议1.28
 - nfs/ceph等共享文件系统： 挂载到每台机器的 /data/k8s/，共享文件系统按需扩容，起步建议SSD 5T （单机可忽略）
 - 数据库接口地址： mysql，没有可忽略使用cube-studio自带的
 - 控制端机器：cpu>=16 mem>=32G，至少1台。生产配置：32核64G*2
 - 任务端cpu/gpu机器：
   - 根据需要自行配置,gpu安装对应厂商的要求安装好机器驱动，机器数量按需扩容
   - 机器学习场景：选择cpu机器即可，>32核64G*2
   - 深度学习镜像：训练机器选择V100机器，推理机器选择T4机器，每张gpu卡配置大于20核cpu，比如4张卡的服务器建议大于80核cpu
   - 大模型镜像：训练机器学习H800机器，推理机器A100机器，每张gpu卡配置大于20核cpu，比如4张卡的服务器建议大于80核cpu
   - 不要只有gpu训练机器，推荐配置纯cpu服务器
 - 所有机器磁盘：>=1T 单机磁盘容量要求不大，仅做镜像容器的的存储  
 - IB/RDMA网络：自动安装机器驱动和IB卡，若无可忽略
 - 系统：ubuntu 20.04 ubuntu 22.04 ubuntu 24.04 或者centos7.9或者centos8

平台完成部署之后如下:

![在这里插入图片描述](https://cube-studio.oss-cn-hangzhou.aliyuncs.com/docs/image/danjibushuxiaoguo.png)

# 本地开发

管理平台web端可连接多个k8s集群用来在k8s集群上调度发起任务实例。同时管理多个k8s集群时，或本地调试时，可以将k8s集群的config文件存储在kubeconfig目录中，按照$ENVIRONMENT-kubeconfig 的规范命名

./docker 目录包含本地开发方案，涉及镜像构建，docker-compose启动，前端构建，后端编码等过程

参考install/docker/README.md


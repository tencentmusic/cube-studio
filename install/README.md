
# 平台基础架构

![image](../docs/example/pic/infra.jpg)

完整的平台包含
 - 1、机器的标准化
 - 2、分布式存储(单机可忽略)、k8s集群、监控体系(prometheus/efk/zipkin)
 - 3、基础能力(tf/pytorch/mxnet/valcano/ray等分布式，nni/katib超参搜索)
 - 4、平台web部分(oa/权限/项目组、在线构建镜像、在线开发、pipeline拖拉拽、超参搜索、推理服务管理等)


# 平台部署流程

基础环境依赖
 - docker >= 19.03  
 - kubernetes = 1.18  
 - kubectl >=1.18  
 - ssd ceph > 10T  挂载到每台机器的 /data/k8s/  
 - 单机 磁盘>=1T   单机磁盘容量要求不大，仅做镜像容器的的存储  
 - 控制端机器 cpu>=32 mem>=64G * 2  
 - 任务端机器，根据需要自行配置  

本平台依赖k8s/kubeflow/prometheus/efk相关组件，请参考install/kubenetes/README.md 部署依赖组件。

平台完成部署之后如下:

![image](../docs/example/pic/pipeline.png)


# 本地开发web部分

web端可以连接多个k8s集群用来在k8s集群上调度发起任务实例。可以将k8s集群的config文件存储在kubeconfig目录中，按照$ENVIRONMENT-kubeconfig 的规范命名

docker 文件夹包含本地开发方案，涉及镜像构建，docker-compose启动，前端构建，后端编码等过程

参考install/docker/README.md



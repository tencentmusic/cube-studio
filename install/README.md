
# 平台基础架构

![image](https://user-images.githubusercontent.com/20157705/167534673-322f4784-e240-451e-875e-ada57f121418.png)

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
 - cfs/ceph  挂载到每台机器的 /data/k8s/  
 - 单机 磁盘>=500G   单机磁盘容量要求不大，仅做镜像容器的的存储  
 - 控制端机器 cpu>=16 mem>=32G 
 - 任务端机器，根据需要自行配置  

本平台依赖k8s/kubeflow/prometheus/efk相关组件，请参考install/kubenetes/README.md 部署依赖组件。

平台完成部署之后如下:

<img width="100%" alt="167874734-5b1629e0-c3bb-41b0-871d-ffa43d914066" src="https://user-images.githubusercontent.com/20157705/168214806-b8aceb3d-e1b4-48f0-a079-903ef8751f40.png">


# 本地开发

管理平台web端可连接多个k8s集群用来在k8s集群上调度发起任务实例。同时管理多个k8s集群时，或本地调试时，可以将k8s集群的config文件存储在kubeconfig目录中，按照$ENVIRONMENT-kubeconfig 的规范命名

./docker 目录包含本地开发方案，涉及镜像构建，docker-compose启动，前端构建，后端编码等过程

参考install/docker/README.md



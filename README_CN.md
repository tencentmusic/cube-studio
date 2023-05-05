# Cube Studio

### 整体架构

<img width="1437" alt="image" src="https://user-images.githubusercontent.com/20157705/182564530-2c965f5f-407d-4baa-8772-73cb2645901b.png">


cube studio是 腾讯音乐 开源的一站式云原生机器学习平台，目前主要包含

![231695906-5b1da227-8455-4274-8857-624093cf574b](https://user-images.githubusercontent.com/20157705/231781297-4eb49101-0997-4a2b-ac21-f3e92602d6ea.png)


# 帮助文档

https://github.com/tencentmusic/cube-studio/wiki

# 开源共建

 学习、部署、体验、开源建设、商业合作 欢迎来撩。或添加微信luanpeng1234，备注<开源建设>， [共建指南](https://github.com/tencentmusic/cube-studio/blob/master/CONTRIBUTING.md)

 <img border="0" width="20%" src="https://user-images.githubusercontent.com/20157705/219829986-66384e34-7ae9-4511-af67-771c9bbe91ce.jpg" />
 
# 支持模板

提示：
- 1、可自由定制任务插件，更适用当前业务需求

| 模板  | 类型 | 组件说明 |
| :----- | :---- | :---- |
| 自定义镜像 | 基础命令 | 完全自定义单机运行环境，可自由实现所有自定义单机功能 | 
| datax | 导入导出 | 异构数据源导入导出 | 
| hadoop | 数据处理 | hadoop大数据组件，hdfs,hbase,sqoop,spark |
| sparkjob | 数据处理 | spark serverless 分布式数据计算 |
| ray | 数据处理 | python ray框架 多机分布式功能，适用于超多文件在多机上的并发处理 |
| volcano | 数据处理 | volcano框架的多机分布式，可自由控制代码，利用环境变量实现多机worker的工作与协同  | 
| ray-sklearn | 机器学习 | 基于ray框架的sklearn支持算法多机分布式并行计算  |
| xgb | 机器学习 | xgb模型训练 |
| tfjob | 深度学习 | tf分布式训练，k8s云原生方式 | 
| pytorchjob | 深度学习 | 	pytorch的多机多卡分布式训练  | 
| horovod | 深度学习 | horovod 的多机多卡分布式训练 | 
| paddle | 深度学习 | 	paddle的多机多卡分布式训练  | 
| mxnet | 深度学习 | mxnet的多机多卡分布式训练  | 
| kaldi | 深度学习 | 	kaldi的多机多卡分布式训练  | 
| tfjob-train | tf分布式 | tf分布式训练，内部支持plain和runner两种方式  | 
| tfjob-runner | tf分布式 | tf分布式-runner方式  | 
| tfjob-plain | tf分布式 | tf分布式-plain方式  | 
| tf-model-evaluation | tf分布式 | tensorflow2.3分布式模型评估  | 
| tf-offline-predict | tf分布式 | tf模型离线推理  | 
| model-register | 模型服务化 | 注册模型 | 
| model-offline-predict | 模型服务化 | 	所有框架的分布式模型离线推理  | 
| deploy-service | 模型服务化 | 部署云原生推理服务 | 
| media-download | 多媒体处理 | 	分布式媒体文件下载  | 
| video-audio | 多媒体处理 | 	分布式视频提取音频  | 
| video-img | 多媒体处理 | 	分布式视频提取图片  | 
| object-detection | 机器视觉 | 基于darknet yolov3 的目标识别|
 
# 平台部署

[参考wiki](https://github.com/tencentmusic/cube-studio/wiki/%E5%B9%B3%E5%8F%B0%E5%8D%95%E6%9C%BA%E9%83%A8%E7%BD%B2) 平台完成部署之后如下:

![cube](https://user-images.githubusercontent.com/20157705/174762561-29b18237-7d45-417e-b7c0-14f5ef96a0e6.gif)

# 公司

![图片 1](https://user-images.githubusercontent.com/20157705/223387901-1b922d96-0a79-4542-b53b-e70938404b2e.png)



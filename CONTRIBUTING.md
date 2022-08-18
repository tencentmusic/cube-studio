![image](https://user-images.githubusercontent.com/20157705/174476217-125e7fb8-0b4d-4921-91ed-38935b8013c7.png)


# 第一阶段发展目标：

1、算法：视觉、文本、语音、搜广推、金融 开源 算法的集成，能更方便的体验使用开源算法。包括算法任务模板(job-template)，算法全自动建模流程(pipeline)，算法推理服务(service)

2、平台开发：数据管理的集成(数据标注/特征平台)，平台公有化/私有化saas版，数据闭环(离线/实时训练闭环)

2、推广、答疑、运营，在相关知识平台发布文章/视频、交流群的答疑、专业社区的分享，git仓库的文案管理，wiki文档等

# 第二阶段发展目标

1、平台本身，算法模板，pipeline等，商业化。通过边缘计算，区块链技术 将算法/数据 价值商业化。



# 算法贡献：

技能要求：[了解模板开发流程，了解平台使用](https://github.com/tencentmusic/cube-studio/wiki/%E5%BC%80%E5%8F%91%E7%AE%97%E6%B3%95%E6%A8%A1%E6%9D%BF)

视觉、文本、语音、搜广推、金融等开源算法的集成，能让使用者更方便的体验使用开源算法。包括算法任务模板(job-template)，算法全自动建模流程(pipeline)，算法推理服务(service)

# 平台开发贡献：

技能要求：[了解平台架构，了解平台代码](https://github.com/tencentmusic/cube-studio/wiki)

主要涉及前后端的开发，平台架构，新功能设计。比如数据管理的集成(数据标注)，平台公有化/私有化saas版，数据闭环(离线/实时训练闭环)


# 运营贡献：

技能要求：[了解平台架构，了解平台使用，熟悉wiki文档](https://github.com/tencentmusic/cube-studio/wiki)

比如：推广、答疑、运营，在相关知识平台发布文章/视频、交流群的答疑、专业社区的分享，git仓库的文案管理，wiki文档等

# 汇总社区需求

算法：

 - 视觉：yolo相关模型、darknet相关模型、PaddleSeg 图像分割，orc相关模型，等训练和推理支持

 - 语音：wenet语音识别的训练和推理支持。

 - 推荐：bin算法，deepfm，ple等算法的训练和推理服务支持

 - 文本： bert框架模型的训练和推理支持

平台：

 - 去除对kubernetes dashboard的依赖，提供服务支持pod，搜索，日志的查看，删除，执行命令界面。

 - jupyter支持链接spark，支持spark任务模板

 - 特征平台，标注系统的支持

 - 数据ETL pipeline对接开源调度平台airflow/azkaban/argo等

 - kubeflow-pipeline依赖去除

 - ceph或其他分布式存储部署方式的开源支持

 - 边缘集群k8s部署方式的支持 KubeEdge/k3s等部署边缘k8s

# Cube Studio

### 整体架构

<img width="1437" alt="image" src="https://user-images.githubusercontent.com/20157705/182564530-2c965f5f-407d-4baa-8772-73cb2645901b.png">


cube studio是 腾讯音乐 开源的一站式云原生机器学习平台，目前主要包含

![231695906-5b1da227-8455-4274-8857-624093cf574b](https://user-images.githubusercontent.com/20157705/231781297-4eb49101-0997-4a2b-ac21-f3e92602d6ea.png)


# 帮助文档

https://github.com/tencentmusic/cube-studio/wiki

# 开源共建

 学习、部署、体验、开源建设、商业合作 欢迎来撩。或添加微信luanpeng1234，备注<开源建设>

 <img border="0" width="20%" src="https://user-images.githubusercontent.com/20157705/219829986-66384e34-7ae9-4511-af67-771c9bbe91ce.jpg" />
 
# 支持模板

提示：
- 1、可自由定制任务插件，更适用当前业务需求

| 模块  | 模板 | 类型 | 文档地址 |
| :----- | :---- | :---- |:---- |
| 数据导入导出 | datax | 单机 | job-template/job/datax/README.md
| 数据导入导出 | 数据集导入 | 单机 | job-template/job/dataset/README.md
| 数据导入导出 | 模型导入 | 单机 | job-template/job/model_download/README.md
| 数据预处理 | data-process | 单机 | job-template/job/data-process/README.md
| 数据处理 | hadoop | 单机 | job-template/job/hadoop/README.md
| 数据处理 | spark | 分布式 | job-template/job/spark/README.md
| 数据处理 | ray | 分布式 | job-template/job/ray/README.md
| 数据处理 | volcanojob | 分布式 | job-template/job/volcano/README.md
| 特征工程 | feature-process | 单机 | job-template/job/feature-process/README.md
| 机器学习框架 | ray-sklearn | 分布式 | job-template/job/ray_sklearn/README.md
| 机器学习算法 | random_forest | 单机 | job-template/job/random_forest/README.md
| 机器学习算法 | lr | 单机 | job-template/job/lr/README.md
| 机器学习算法 | lightgbm | 单机 | job-template/job/lightgbm/README.md
| 机器学习算法 | knn | 单机 | job-template/job/knn/README.md
| 机器学习算法 | kmeans | 单机 | job-template/job/kmeans/README.md
| 机器学习算法 | nni | 单机 | job-template/job/hyperparam-search-nni/README.md
| 机器学习算法 | xgb | 单机 | job-template/job/xgb/README.md
| 机器学习算法 | gbdt | 单机 | job-template/job/gbdt/README.md
| 机器学习算法 | decision-tree | 单机 | job-template/job/decision_tree/README.md
| 机器学习算法 | bayesian | 单机 | job-template/job/bayesian/README.md
| 机器学习算法 | adaboost | 单机 | job-template/job/adaboost/README.md
| 深度学习 | tfjob | 分布式 | job-template/job/tf/README.md
| 深度学习 | pytorchjob | 分布式 | job-template/job/pytorch/README.md
| 深度学习 | paddle | 分布式 | job-template/job/paddle/README.md
| 深度学习 | mxnet | 分布式 | job-template/job/mxnet/README.md
| 深度学习 | mindspore | 分布式 | job-template/job/mindspore/README.md
| 深度学习 | horovod | 分布式 | job-template/job/horovod/README.md
| 深度学习 | mpi | 分布式 | job-template/job/mpi/README.md
| 深度学习 | colossalai | 分布式 | job-template/job/colossalai/README.md
| 深度学习 | deepspeed | 分布式 | job-template/job/deepspeed/README.md
| 深度学习 | megatron | 分布式 | job-template/job/megatron/README.md
| 模型处理 | model-evaluation | 单机 | job-template/job/model_evaluation/README.md
| 模型服务化 | model-convert | 单机 | job-template/job/model_convert/README.md
| 模型服务化 | model-register | 单机 | job-template/job/model_register/README.md
| 模型服务化 | deploy-service | 单机 | job-template/job/deploy-service/README.md
| 模型服务化 | model-offline-predict | 分布式 | job-template/job/model_offline_predict/README.md
| 多媒体类 | media-download | 分布式 | job-template/job/video-audio/README.md
| 多媒体类 | video-img | 分布式 | job-template/job/video-audio/README.md
| 多媒体类 | video-audio | 分布式 | job-template/job/video-audio/README.md
| 大模型 | llama2 | 单机多卡 | job-template/job/llama2/README.md
| 大模型  | chatglm2 | 单机多卡 | job-template/job/chatglm2/README.md
| 大模型  | baichuan2 | 单机多卡 | job-template/job/baichuan2/README.md


# 公司

![图片 1](https://user-images.githubusercontent.com/20157705/223387901-1b922d96-0a79-4542-b53b-e70938404b2e.png)

# 平台简介


完整的平台包含
 - 1、机器的标准化
 - 2、分布式存储(单机可忽略)、k8s集群、监控体系(prometheus/efk/zipkin)
 - 3、基础能力(tf/pytorch/mxnet/valcano/ray等分布式，nni/katib超参搜索)
 - 4、平台web部分(oa/权限/项目组、在线构建镜像、在线开发、pipeline拖拉拽、超参搜索、推理服务管理等)

![image](https://github.com/tencentmusic/cube-studio/assets/20157705/a07b1742-3413-4957-bd15-0f2b3c30f66f)


# 算力/存储/用户管理

算力：
 - 云原生统筹平台cpu/gpu等算力
 - 支持划分多资源组，支持多k8s集群，多地部署
 - 支持T4/V100/A100/昇腾/dcu/VGPU等异构GPU/NPU环境
 - 支持边缘集群模式，支持边缘节点上开发/训练/推理
 - 支持鲲鹏芯片arm64架构，RDMA

存储：
 - 自带分布式存储，支持多机分布式下文件处理
 - 支持外部存储挂载，支持项目组挂载绑定
 - 支持个人存储空间/组空间等多种形式
 - 平台内存储空间不需要迁移

用户权限：
 - 支持sso登录，对接公司账号体系
 - 支持项目组划分，支持配置相应项目组用户的权限
 - 管理平台用户的基本信息，组织架构，rbac权限体系

# 多集群管控

cube支持多集群调度，可同时管控多个训练或推理集群。在单个集群内，不仅能做到一个项目组内对在线开发、训练、推理的隔离，还可以做到一个k8s集群下多个项目组算力的隔离。另外在不同项目组下的算力间具有动态均衡的能力，能够在多项目间共享公共算力池和私有化算力池，做到成本最低化。

![image](https://user-images.githubusercontent.com/20157705/167534695-d63b8239-e85e-42c4-bc7b-5999b9eff882.png)

# 分布式存储

cube会自动为用户挂载用户的个人目录，同一个用户在平台任何地方启动的容器，其用户个人子目录均为/mnt/$username。可以将pvc/hostpath/memory/configmap等挂载成容器目录。同时可以在项目组中配置项目组的默认挂载，进而实现一个项目组共享同一个目录等功能。

![image](https://user-images.githubusercontent.com/20157705/167534724-733ad796-745e-47e1-9224-9e749f918cf2.png)

# 在线开发

 - 系统多租户/多实例管理，在线交互开发调试，无需安装三方控件，只需浏览器就能完成开发。
 - 支持vscode，jupyter，Matlab，Rstudio等多种在线IDE类型
 - Jupyter支持cube-studio sdk，Julia，R，python，pyspark多内核版本，

![image](https://github.com/tencentmusic/cube-studio/assets/20157705/0819214b-d7c1-421a-8978-465f20b8d594)


 - 支持c++，java，conda等多种开发语言，以及tensorboard/git/gpu监控等多种插件
 - 支持ssh remote与notebook互通，本地进行代码开发
 - 在线镜像构建，通过Web Shell方式在浏览器中完成构建；并提供各种版本notebook，inference，gpu，python等基础镜像

![image](https://github.com/tencentmusic/cube-studio/assets/20157705/5793beea-715d-40e2-a01f-8d36939e35bd)


# 标注平台：

 - 支持图/文/音/多模态/大模型多种类型标注功能，用户管理，工作任务分发
 - 对接aihub模型市场，支持自动化标注；对接数据集，支持标注数据导入；对接pipeline，支持标注结果自动化训练

![image](https://github.com/tencentmusic/cube-studio/assets/20157705/a70eb024-77b8-4fe9-9b3e-0f0187470c23)


# 拖拉拽pipeline编排

1、Ml全流程

数据导入，数据预处理，超惨搜索，模型训练，模型评估，模型压缩，模型注册，服务上线，ml算法全流程

2、灵活开放

支持单任务调试、分布式任务日志聚合查看，pipeline调试跟踪，任务运行资源监控，以及定时调度功能(包含补录，忽略，重试，依赖，并发限制，过期淘汰等功能)

![image](https://github.com/tencentmusic/cube-studio/assets/20157705/17d8fc5c-8c13-48ed-934b-bdaffceab4e9)


# 分布式框架

1、训练框架支持分布式（协议和策略）  
2、代码识别分布式角色（有状态）  
3、控制器部署分布式训练集群（operator）  
4、配置分布式训练集群的部署（CRD）  


# 多层次多类型算子

以k8s为核心，  
1、支持tf分布式训练、pytorch分布式训练、spark分布式数据处理、ray分布式超参搜索、mpi分布式训练、horovod分布式训练、nni分布式超参搜索、mxnet分布式训练、volcano分布式数据处理、kaldi分布式语音训练等，  
2、 以及在此衍生出来的分布式的数据下载，hdfs拉取，cos上传下载，视频采帧，音频抽取，分布式的训练，例如推荐场景的din算法，ComiRec算法，MMoE算法，DeepFM算法，youtube dnn算法，ple模型，ESMM模型，双塔模型，音视频的wenet，containAI等算法的分布式训练。

![image](https://github.com/tencentmusic/cube-studio/assets/20157705/b88580a2-a8bb-47e4-9701-008be2960f73)


# 功能模板化

 - 和非模板开发相比，使用模板建立应用成本会更低一些，无需开发平台。
 - 迁移更加容易，通过模板标准化后，后续应用迁移迭代只需迁移配置模板，简化复杂的配置操作。
 - 配置复用，通过简单的配置就可以复用这些能力，算法与工程分离避免重复开发。


为了避免重复开发，对pipeline中的task功能进行模板化开发。平台开发者或用户可自行开发模板镜像，将镜像注册到平台，这样其他用户就可以复用这些功能。平台自带模板在job-template目录下

![image](https://user-images.githubusercontent.com/20157705/167534770-505ffce8-8172-49be-9506-b265cd6ed465.png)

# 流水线调试

 - Pipeline调试支持定时执行，支持，补录，并发限制，超时，实例依赖等。
 - Pipeling运行，支持变量在任务间输入输出，全局变量，流向控制，模板变量，数据时间等
 - Pipeling运行，支持任务结果可视化，图片、csv/json，echart源码可视化

![image](https://github.com/tencentmusic/cube-studio/assets/20157705/3d24ac7c-24d8-4192-9575-477665836da0)

# nni超参搜索


界面化呈现训练各组数据，通过图形界面进行直观呈现。
减少以往开发调参过程的枯燥感，让整个调参过程更加生动具有趣味性，完全无需丰富经验就能实现更精准的参数控制调节。

```bash
# 上报当前迭代目标值
nni.report_intermediate_result(test_acc)
# 上报最终目标值
nni.report_final_result(test_acc)

# 接收超参数为输入参数
parser.add_argument('--batch_size', type=int)
```

![image](https://user-images.githubusercontent.com/20157705/167534784-255f101a-3273-4eea-9254-f2df6879ddf1.png)


# 推理服务

0代码发布推理服务从底层到上层，包含服务网格，serverless，pipeline，http框架，模型计算。

 - 服务网格阶段：主要工作是代理流量的中转和管控，例如分流，镜像，限流，黑白名单之类的。

 - serverless阶段：主要为服务的智能化运维，例如服务的激活，伸缩容，版本管理，蓝绿发布。

 - pipeline阶段：主要为请求在各数据处理/推理之间的流动。推理的前后置处理逻辑等。

 - http/grpc框架：主要为处理客户端的请求，准备推理样本，推理后作出响应。

 - 模型计算：模型在cpu/gpu上对输入样本做前向计算。

主要功能：

 - 支持模型管理注册，灰度发布，版本回退，模型指标可视化，以及在piepline中进行模型注册
 - 推理服务支持多集群，多资源组，异构gpu环境，平台资源统筹监控，VGPU，服务流量分流，复制，sidecar
 - 支持0代码的模型发布，gpu推理加速，支持训练推理混部，服务优先级，自定义指标弹性伸缩。
 
![image](https://user-images.githubusercontent.com/20157705/167534820-9202851a-a97c-41f7-8d63-900d73e4c57e.png)

# 监控和推送

监控：cube-studio集成prometheus生态，可以监控包括主机，进程，服务流量，gpu等相关负载，并配套grafana进行可视化

推送：cube-studio开放推送接口，可自定义推送给企业oa系统

![image](https://github.com/tencentmusic/cube-studio/assets/20157705/8e767e8f-b7ef-4015-907f-95cb46b37ed8)

# AIHub

 - 系统自带通用模型数量400+，覆盖绝大数行业场景，根据需求可以不断扩充。
 - 模型开源、按需定制，方便快速集成，满足用户业务增长及二次开发升级。
 - 模型标准化开发管理，大幅降低使用门槛，开发周期时长平均下降30%以上。

![image](https://github.com/tencentmusic/cube-studio/assets/20157705/03923858-49b4-4430-90e0-94e90735f8b4)


 - AIHub模型可一键部署为WEB端应用，手机端/PC端皆可，实时查看模型应用效果
 - 点击模型开发即可进入notebook进行模型代码的二次开发，实现一键开发
 - 点击训练即可加入自己的数据进行一键微调，使模型更贴合自身场景

![image](https://github.com/tencentmusic/cube-studio/assets/20157705/561b670d-797c-47b5-93fc-de0de7e4b915)

# GPT训练微调

 - cube-studio支持deepspeed/colossalai等分布式加速框架，可一键实现大模型多机多卡分布式训练
 - AIHub包含gpt/AIGC大模型，可一键转为微调pipeline，修改为自己的数据后，便可以微调并部署

![image](https://github.com/tencentmusic/cube-studio/assets/20157705/d3589c4b-afca-44bd-8270-2b542ec4ceaa)


# GPT-RDMA

rdma插件部署后，k8s机器可用资源
```bash
capacity:
  cpu: '128'
  memory: 1056469320Ki
  nvidia.com/gpu: '8'
  rdma/hca: '500'
```
代码分布式训练中使用IB设备
```bash
export NCCL_IB_HCA=mlx5
export MLP_WORKER_GPU=$GPU_NUM
export MLP_WORKER_NUM=$WORLD_SIZE
export MLP_ROLE_INDEX=$RANK
export MLP_WORKER_0_HOST=$MASTER_ADDR
export MLP_WORKER_0_PORT=$MASTER_PORT
```

![image](https://github.com/tencentmusic/cube-studio/assets/20157705/d53e6a99-4053-4931-9456-cb857dc69723)


# gpt私有知识库

 - 数据智能模块可配置专业领域智能对话，快速敏捷使用llm
 - 可为某个聊天场景配置私有知识库文件，支持主题分割，语义embedding，意图识别，概要提取，多路召回，排序，多种功能融合

![image](https://github.com/tencentmusic/cube-studio/assets/20157705/f207cb09-1b5e-486c-91c0-a9ce31863e34)


# gpt智能聊天

 - 可以将智能会话与AIHub相结合，例如下面AIGC模型与聊天会话
 - 可使用Autogpt方式串联所有aihub模型，进行图文音智能化处理
 - 智能会话与公共直接打通，可在微信公众号中进行图文音对话

![image](https://github.com/tencentmusic/cube-studio/assets/20157705/eb62c8a9-7f89-4898-90d8-d544dd73251c)


# 数据中台对接
  
为了加速AI算法平台的使用，cube-studio支持对接公司原有数据中台，包括数据计算引擎sqllab，元数据管理，指标管理，维表管理，数据ETL，数据集管理

![image](https://github.com/tencentmusic/cube-studio/assets/20157705/a9a0b399-8d02-4d19-8198-4e3681074f2f)


### 实时训练

tmeps支持tf框架实时训练，秒级上线，能应对embedding稀疏大模型推荐场景

![image](https://user-images.githubusercontent.com/20157705/167534836-418855cf-daef-45a5-85c9-3bb1b7135f4f.png)

# 三种方式部署

针对企业需求，根据不同场景对计算实时性的不同需求，可以提供三种建设模式

模式一：私有化部署——对数据安全要求高、预算充足、自己有开发能力  
模式二：边缘集群部署——算力分散，多个子网环境的场景，或边缘设备场景  
模式三：serverless集群——成本有限，按需申请算力的场景  

## 边缘计算

通过边缘集群的形式，在中心节点部署平台，并将边缘节点加入调度，每个私有网用户，通过项目组，将notebook，pipeline，service部署在边缘节点  
 - 1、避免数据到中心节点的带宽传输  
 - 2、避免中心节点的算力成本，充分利用边缘节点算力
 - 3、避免边缘节点的运维成本

![图片 1](https://user-images.githubusercontent.com/20157705/170262037-12ad086a-c427-4746-a0fa-ce3bc1586729.png)


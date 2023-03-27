管理员管理注意事项：

# 权限管理

 - oa登录开发。登录方式可通过project.py文件定义的方式自行修改，或自行添加管理员专用接口。
 - rbac的权限管理方式。所有菜单或者模块的增删改查以及所有的后端接口都是有权限控制的，可以在Security中自由控制
 - 用户登录后会自动创建以及绑定用户名同名的角色。同时绑定gamma角色，加入public项目组

# 项目组->模板分类

expand字段
```bash
{
  "index":2   # index 控制该分组下面的job模板在pipeline编排界面左侧的显示顺序
}
```

# 项目组->项目分组
### 跨k8s集群调度的支持：

需要在config.py文件中配置CLUSTERS，并通过ENVIRONMENT环境变量指定控制台所在的集群。其他集群为训练集群。

项目组/项目分组中，可通过项目组的expand字段控制项目组的调度集群
	
	{
		"cluster": "dev"
	}

### 资源划分项目组
机器通过label进行管理，所有的调度机器由平台控制，不由用户直接控制。

- 对于cpu的train/notebook/service会选择cpu=true的机器  
- 对于gpu的train/notebook/service会选择gpu=true的机器  

- 训练任务会选择train=true的机器  
- notebook会选择notebook=true的机器  
- 服务化会选择service=true的机器  
- 不同项目的任务会选择对应org=xx的机器。默认为org=public 
  
管理标注哪些机器属于哪些项目后，可通过项目组的expand字段控制项目组的调度机器
	
	{
		"org": "public"
	}

管理该项目自动挂载什么pvc，可通过项目组的expand字段控制项目组的调度机器
	{
		"volume_mount": "kubeflow-user-workspace(pvc):/mnt"
	}
	
# 在线开发->notebook

定时任务中配置了notebook的定时清理和提醒用户续期，没有及时续期的notebook会被删除。容器消失后环境丢失

# 在线开发->镜像调试

在线调试镜像，保存按钮会自动commit/push到仓库，当晚也会自动释放

# 训练->仓库

仓库中的k8s hubsecret 以及账号密码，在保存时会更新到配置的所有集群中的各个命名空间下。并且发生用户拉起pod的情况也会附带上用户配置的k8s hubsecret以及config.py中配置的公共k8s hubsecret。

建议仓库拉取秘钥均在config.py中全局配置

# 训练->任务模板

### 版本：

模板的release版本才能被用户看到

### 工作目录/启动命令：

如果配置，则会覆盖镜像的启动目录和启动命令

### 挂载目录：

会在task创建时添加模板的挂载项，一般用户需要固定挂载的模板会配置该参数(比如docker in docker 需要 docker.socket)。目前支持hostpath/pvc/memory/configmap几种挂载的配置方式

```bash
kubeflow-user-workspace(pvc):/mnt          pvc的挂载方式，会自动在pvc下挂载个人子目录
/data/k8s/kubeflow/pipeline/workspace/xxxx(hostpath):/mnt    挂载主机目录的方式
4G(memory):/dev/shm                        内存挂载为磁盘的书写方式
kubernetes-config(configmap):/root/.kube   挂载configmap成文件夹
```

### 环境变量：

会附加给每个使用该模板的task

模板中特殊的环境变量
```bash
NO_RESOURCE_CHECK=true    使用该模板的task不会进行资源配置的自动校验
TASK_RESOURCE_CPU=4       使用该模板的task 忽略用户的资源配置，cpu固定配置资源为4核
TASK_RESOURCE_MEMORY=4G   使用该模板的task 忽略用户的资源配置，mem固定配置资源为4G
TASK_RESOURCE_GPU=0       使用该模板的task 忽略用户的资源配置，gpu固定配置资源为0卡
```
### k8s账号

为模板配置k8s账号，主要为那些需要操控k8s集群的模板而设定的。比如临时分布式训练集群的launcher端

### 扩展

```bash
{
    "index": 7,     index控制在pipeline编排界面同一个模板分组中每个模板的显示顺序
    "help_url": "http://xx.xx.xx.xx/xx"     help_url 为pipeline编排界面每个模板的帮助文档的地址显示
}
```


# 训练->任务流

 - 先调试单独的task，再调试pipeline的原则。   
 - task包含debug/run/log/clear等功能，通过run-id串联对应的pod和crd  
 - pipeline包含run/log/运行实例/定时调度记录/pod等功能。使用run-id串联所有的task  
 - 手动运行pipeline会先清空之前手动运行的pipeline实例（workflow）
 - myapp/task/scheduler.py中会定时清理运行结束的pod，避免pod堆积过多
 - 在pieline正常运行10次以后，定时任务会取其中真实使用的资源最大值，再预留一定的空间，动态的修改用户配置的资源项，防止资源配置不合理而浪费算力


# 训练->demo任务流

首页会显示所有的demo pipeline。即 pipeline 扩展字段expand（dict）中包含
```
{
    "demo": "true",
    "img": "https://xx.xx.xx.xx/xx.png"
}
```

# 训练->定时调度记录

手动运行和定时运行同一个pipeline相互之间不干扰。






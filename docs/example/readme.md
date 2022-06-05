# 文档目录
	example
	- notebook     	# 在线notebook使用
	- job-template  # 模板制作与使用
	- pipeline: 	# 任务流使用
	- images        # 镜像制作 && 常用基础镜像
	
# 基础认识：
	
__容器__：对于建模人员，可以认为容器是一个轻量的虚拟机，一个机器可以同时启动多个容器，容器之间资源是隔离的。在平台上运行的任务，最终都会以容器内进程的形式运行在实体机上。所有容器的/mnt目录下都是自己的个人工作目录，容器重启后环境消失。

__镜像__：可以将容器整个操作系统的文件和目录打包成镜像，只需要解压打包好的容器镜像，容器就被重现出来了。所以我们可以用这种方式打包运行ml任务的复杂环境。

__任务模板__：当我们要在平台例行化运行一个ml任务（例如xgb分类），可以先定义好这个任务用到的镜像，需要什么参数，分配多少资源等信息，形成一个任务模板。以后使用的时候只需选择模板并填写相应参数即可。

__任务流(pipeline)__: 多个任务及其依赖关系形成的DAG图。任务的构建产物可以在任务间传递。

__运行实例__：任务流的一次运行。

__notebook__：开启一个jupyter-notebook，自动挂载个人工作目录。有jupyter、vscode两种模式
	
# 加入项目组

### 申请加入项目组

路径：项目组->项目分组，oa联系creator将自己加入到项目组中

![image](https://user-images.githubusercontent.com/20157705/167538387-0119b48a-cd7e-48ca-b89a-efd7c2cae657.png)

备注：

- 每个人自动都在public项目组中
- 可自行创建项目组

### creator添加用户

路径：项目组->项目分组。编辑项目组，添加组成员。creator角色的组成员可以添加其他用户进组

### 项目组控制调度集群cluster（管理员学习）

平台支持跨集群调度。需要管理员创建集群并配置到系统配置后，可通过项目组的expand字段控制项目组的调度集群
	
	{
		"cluster": "dev"
	}

### 项目组控制调度机器node_selector（管理员学习）

平台支持单集群中划分资源组。需要管理员配置不同机器所属的资源组后，可通过项目组的expand字段控制项目组的调度机器。

调度机器可以是不同项目组，不同机型，不同区域等划分方式
	
	{
		"node_selector": "org=public"
	}

### 项目组控制挂载volume_mount（管理员学习）

平台支持单集群中划分挂载。可以配置项目组下成员将自有分布式存储挂载到平台，以及项目组内共享目录等功能，可通过项目组的expand字段控制项目组的挂载.

- 挂载pvc，会自动挂载pvc下面的个人用户子目录。默认挂载kubeflow-user-workspace的pvc
- 挂载hostpath，不会自动挂载个人子目录，可以用来控制多人共享同一个目录
- 挂载memory，主要用来控制k8s中共享内存的挂载

```
	{
		"volume_mount": "kubeflow-user-workspace(pvc):/mnt/;data/aidata(hostpath):/aidata;4G(memory):/dev/shm"
	}
```

### 项目组控制服务service代理ip（管理员学习）

平台支持单集群中划分服务的代理ip，多用于边缘集群，或多网关情况下。可通过项目组的expand字段控制项目组的服务的代理ip.

```
	{
		"SERVICE_EXTERNAL_IP":"xx.xx.xx.xx"
	}
```




# 在线notebook开发

### notebook支持类型

1. Jupyter （cpu/gpu）
2. vscode（cpu/gpu）

### 支持功能

1. 代码开发/调试，上传/下载，命令行，git工蜂/github，内网/外网，tensorboard，自主安装插件

### 添加notebook

路径：在线开发->notebook->添加

![image](https://user-images.githubusercontent.com/20157705/167538439-d921aeb9-635f-4d0a-8a59-726d21b04e5e.png)

备注：
1. Running状态下，方可进入
2. 无法进去时，reset重新激活
3. notebook会自动挂载一下目录
 - a）个人工作目录到容器 /mnt/$username
 - b）个人归档目录到容器/archives/$username

### jupyter示例：

<img width="70%" alt="167874734-5b1629e0-c3bb-41b0-871d-ffa43d914066" src="https://user-images.githubusercontent.com/20157705/167538488-cba41bf6-ba66-4150-b17e-f31f5cc5013d.png">


### vscode示例：

![image](https://user-images.githubusercontent.com/20157705/167538518-d9c05758-b808-481a-be4e-dc42477f98c5.png)

### 切换归档目录示例：

![image](https://user-images.githubusercontent.com/20157705/167538586-7ce638da-72a9-4c4a-904d-7d76d4356c82.png)

### tensorboard示例：

进入到对应的日志目录，再打开tensorboard按钮

![image](https://user-images.githubusercontent.com/20157705/167538606-e6ba0559-3238-41cb-aa37-d2c61632085d.png)


# 在线构建镜像

![image](https://user-images.githubusercontent.com/20157705/167538625-39c19c33-a63d-44fa-a16a-2aaa7b480190.png)

扩展字段高级配置：
```bash
{
  "volume_mount":"kubeflow-user-workspace(pvc):/mnt,kubeflow-archives(pvc):/archives",
  "resource_memory":"8G",
  "resource_cpu": "4"
}
```
[基础镜像和封装方法参考](https://github.com/tencentmusic/cube-studio/tree/master/images)

# 配置/调试/定时运行pipeline

### 创建pipeline

路径：训练->任务流->新建

![image](https://user-images.githubusercontent.com/20157705/167538656-34631373-a055-4659-acbe-33594397efbe.png)

主要配置： 参考每个配置参数的描述

### 编排pipeline

![image](https://user-images.githubusercontent.com/20157705/167538672-db0f4e3f-d59f-48d5-a712-6655df5f6b4b.png)

 - task公共参数：参考每个配置的描述  
 - task的模板参数：参考每个模板的链接教程文档

### 运行调试

##### task运行调试：

使用task的run按钮和log按钮可单独调试一个task

![image](https://user-images.githubusercontent.com/20157705/167538698-d08078c0-c399-457a-9f2b-32cb002b4b06.png)

##### pipeline运行调试：

pipeline的运行按钮发起调度

![image](https://user-images.githubusercontent.com/20157705/167538718-e07e1144-2cfb-4fec-8dad-1522e957592b.png)

##### pipeline日志效果：

![image](https://user-images.githubusercontent.com/20157705/167538739-1c079d43-8922-4254-ad5e-edff5d670e2a.png)

### pod查看示意图

![image](https://user-images.githubusercontent.com/20157705/167538762-4c65bf8a-0599-4fbd-8fc0-43ce89146a80.png)

pod效果：

![image](https://user-images.githubusercontent.com/20157705/167538775-77a71603-7e73-4e2a-8913-cef753a51c3a.png)

### 实例记录

![image](https://user-images.githubusercontent.com/20157705/167538787-52ee881b-b151-49cd-a904-76516525a043.png)

调度实例记录。停止可以清除调度容器

![image](https://user-images.githubusercontent.com/20157705/167538802-3c292d93-fdbd-4145-b58c-6518969b0ac4.png)

### 定时调度

配置定时：pipeline编辑界面

<img width="50%" alt="167874734-5b1629e0-c3bb-41b0-871d-ffa43d914066" src="https://user-images.githubusercontent.com/20157705/167538811-3644c420-5b00-4c13-af75-c672aef899b2.png">


查看路径：训练-定时调度记录

![image](https://user-images.githubusercontent.com/20157705/167538824-60bf1d3d-1739-4820-b083-fcc72314ec6d.png)

字段说明：执行时间为这个pipeline本次调度该发起的时间点  
状态：comed，为调度配置已经产生。created为调度已经发起。

##### 操作说明

	1、平台会根据pipeline的配置决定是否发起调度。
	2、状态链接中可以看到本地调度发起的workflow的运行情况
	3、日志链接中可以看到本地调度发起的日志

# nni超参搜索

可以参考[nni官网](https://github.com/microsoft/nni)的书写方式

## 超参空间
必须是标准的json。示例
```
{
    "batch_size": {"_type":"choice", "_value": [16, 32, 64, 128]},
    "hidden_size":{"_type":"choice","_value":[128, 256, 512, 1024]},
    "lr":{"_type":"choice","_value":[0.0001, 0.001, 0.01, 0.1]},
    "momentum":{"_type":"uniform","_value":[0, 1]}
}
```
不同超参算法支持不同的超参空间

|choice |choice(nested) |randint |uniform |quniform |loguniform |qloguniform |normal |qnormal |lognormal |qlognormal |
| :----- | :----- | :----- | :----- | :----- | :----- | :----- | :----- | :----- | :----- | :----- | 
|TPE Tuner |✓ |✓ |✓ |✓ |✓ |✓ |✓ |✓ |✓ |✓ |✓ |
|Random Search Tuner |✓ |✓ |✓ |✓ |✓ |✓ |✓ |✓ |✓ |✓ |✓ |
|Anneal Tuner |✓ |✓ |✓ |✓ |✓ |✓ |✓ |✓ |✓ |✓ |✓ |
|Evolution Tuner |✓ |✓ |✓ |✓ |✓ |✓ |✓ |✓ |✓ |✓ |✓ |
|SMAC Tuner |✓ | |✓ |✓ |✓ |✓ | | | | | |
|Batch Tuner |✓ | | | | | | | | | | |
|Grid Search Tuner |✓ | |✓ | |✓ | | | | | | |
|Hyperband Advisor |✓ | |✓ |✓ |✓ |✓ |✓ |✓ |✓ |✓ |✓ |
|Metis Tuner |✓ | |✓ |✓ |✓ | | | | | | |
|GP Tuner |✓ | |✓ |✓ |✓ |✓ |✓ | | | |
  
## 代码要求

### 参数接收
启动超参搜索，会根据用户配置的超参搜索算法，选择好超参的可选值，并将选择值传递给用户的容器。例如上面的超参定义会在用户docker运行时传递下面的参数。所以用户不需要在启动命令或参数中添加这些变量，系统会自动添加，用户只需要在自己的业务代码中接收这些参数，并根据这些参数输出值就可以了。

```
--lr=0.021593113434583065 --num-layers=5 --optimizer=ftrl
```

### 结果上报
业务方容器和代码启动接收超参进行迭代计算，通过主动上报结果来进行迭代。
示例如下，用户代码需要能接受超参可取值为输入参数，同时每次迭代通过nni.report_intermediate_result上报每次epoch的结果值，并使用nni.report_final_result上报每次实例的结果值。 
```
import os
import argparse
import logging,random,time
import nni
from nni.utils import merge_parameter

logger = logging.getLogger('mnist_AutoML')

def main(args):
    test_acc=random.randint(30,50)
    for epoch in range(1, 11):
        test_acc_epoch= random.randint(3,5)
        time.sleep(3)
        test_acc+=test_acc_epoch
        # 上报当前迭代目标值
        nni.report_intermediate_result(test_acc)
    # 上报最总目标值
    nni.report_final_result(test_acc)


def get_params():
    # 必须接收超参数为输入参数
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')

    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        params = vars(merge_parameter(get_params(), tuner_params))
        print(tuner_params,params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
```

## web发起一个搜索实验

![image](https://user-images.githubusercontent.com/20157705/169943169-6fb72bdf-0913-4873-92be-6702b11084c7.png)

## web查看搜索效果

可以参考：https://nni.readthedocs.io/zh/stable/Tutorial/WebUI.html

总览界面可以看到实验的id，和当前示例运行的状态

![image](https://user-images.githubusercontent.com/20157705/169943044-65efa03d-6023-4675-978e-e2b10570dc54.png)

![image](https://user-images.githubusercontent.com/20157705/169943083-9eef65fd-dd1f-4a75-8100-c794be9a236b.png)

可以看每次trial的运行情况，计算出来的目标值


![image](https://user-images.githubusercontent.com/20157705/169943117-43a19fc7-7598-44d6-82bf-af32ca618d12.png)

也可以看某次trial中每次epoch得到的结果值

# 内部服务

##  普通服务

### 开发注册

1、开发你的服务化镜像，push到docker仓库内

2、注册你的服务

![image](https://user-images.githubusercontent.com/20157705/169932303-0ec981cc-09ca-423c-96f9-da164ed309da.png)

## mysql web服务

镜像：ai.tencentmusic.com/tme-public/phpmyadmin

环境变量：
```
PMA_HOST=xx.xx.xx.xx
PMA_PORT=xx
PMA_USER=xx
PMA_PASSWORD=xx
```
端口：80

## mongo web服务
镜像：mongo-express:0.54.0

环境变量：
```
ME_CONFIG_MONGODB_SERVER=xx.xx.xx.xx
ME_CONFIG_MONGODB_PORT=xx
ME_CONFIG_MONGODB_ENABLE_ADMIN=true
ME_CONFIG_MONGODB_ADMINUSERNAME=xx
ME_CONFIG_MONGODB_ADMINPASSWORD=xx
ME_CONFIG_MONGODB_AUTH_DATABASE=xx
VCAP_APP_HOST=0.0.0.0
VCAP_APP_PORT=8081
ME_CONFIG_OPTIONS_EDITORTHEME=ambiance
```
端口：8081

## redis web
镜像：ai.tencentmusic.com/tme-public/patrikx3:latest

环境变量
```
REDIS_NAME=xx
REDIS_HOST=xx
REDIS_PORT=xx
REDIS_PASSWORD=xx
```
端口：7843

## 图数据库neo4j

镜像：ai.tencentmusic.com/tme-public/neo4j:4.4

环境变量
```
NEO4J_AUTH=neo4j/admin
```
端口：7474,7687

## jaeger链路追踪

镜像：jaegertracing/all-in-one:1.29

端口：5775,16686


## 服务暴露：

 - 1、域名暴露需要平台配置泛域名，SERVICE_DOMAIN
 - 2、ip暴露需要平台配置SERVICE_EXTERNAL_IP=[xx.xx.xx.xx]，或者项目中配置SERVICE_EXTERNAL_IP=xx.xx.xx.xx  ip需为集群中节点的ip或者clb的ip



# 推理服务

## 版本/域名/pod的关系
`$服务名=$服务类型-$模型名-$模型版本(只取版本中的数字)`

![image](https://user-images.githubusercontent.com/20157705/169943323-0849f8fd-b20e-4036-9ce5-33892a5bb643.png)

`$k8s-deploymnet-name=$服务名`

![image](https://user-images.githubusercontent.com/20157705/169943360-b7883e39-f070-4dbb-af16-caf021e3b7fa.png)

`$k8s-hpa-name=$服务名`  

在最大最小副本数不一致时创建hpa  

![image](https://user-images.githubusercontent.com/20157705/169943401-6e7abef7-29e2-4986-a4c9-cb3d5da4a7f0.png)

`$k8s-service-name=$服务名`  用于域名的代理  

`$k8s-service-name=$服务名-external`   用户ip/L5的代理  

![image](https://user-images.githubusercontent.com/20157705/169943472-34b161c2-b487-4aab-a335-f45465bda33b.png)


## 系统自带域名

自动配置域名需要泛域名支持。例如泛域名为domain = *.kfserving.woa.com

生产域名

http://$服务名.service.$domain  

测试环境域名  

http://test.$服务名.service.$domain  
http://debug.$服务名.service.$domain  

## 自定义域名

用户可通过host字段配置服务的访问域名，但是必须与泛域名结尾

多个服务可以配置相同的域名

## 流量复制和分流

多个服务（可以是相同模型或者不同模型间）配置相同的域名  
1、分流属性字段控制分配多少流量到其他服务上，剩余流量归属于当前服务  
2、流量镜像字段控制复制多少流量到其他服务上。但只会将当前服务的响应返回给客户端  

![image](https://user-images.githubusercontent.com/20157705/169944196-bd98064d-124f-4233-af24-5b226ab38831.png)

## 灰度升级

1、同一个服务灰度升级，只需要修改服务的配置，重新部署，服务会自动滚动升级pod  
2、不同服务进行灰度升级。比如同一个模型的不同版本之间，那么多个服务使用相同的域名，新部署的服务上线正常后，会自动下线同域名的旧服务。  

## 弹性伸缩容

弹性伸缩容的触发条件：可以使用自定义指标，可以使用其中一个指标或者多个指标，示例：cpu:50%,mem:%50,gpu:50%  

## 环境变量

系统携带的环境变量
```bash
KUBEFLOW_ENV=test
KUBEFLOW_MODEL_PATH=
KUBEFLOW_MODEL_VERSION=
KUBEFLOW_MODEL_IMAGES=
KUBEFLOW_MODEL_NAME=
KUBEFLOW_AREA=shanghai/guangzhou

K8S_NODE_NAME=
K8S_POD_NAMESPACE=
K8S_POD_IP=
K8S_HOST_IP=
K8S_POD_NAME=
```


## 服务暴露：

 - 1、域名暴露需要平台配置泛域名，SERVICE_DOMAIN
 - 2、ip暴露需要平台配置SERVICE_EXTERNAL_IP=[xx.xx.xx.xx]，或者项目中配置SERVICE_EXTERNAL_IP=xx.xx.xx.xx  ip需为集群中节点的ip或者clb的ip



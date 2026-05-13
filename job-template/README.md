# 视频教程

[job模板制作教程](https://www.bilibili.com/video/BV15B4y197nm)

# 1.job模板规范

参考demo任务模板，job-template/job/demo/

### 1.1.模板目录结构
```
  
 -  job-template            # job模板合计
   - $job_template_name     # 自定义模板
     - build.sh        # job镜像构建过程
     - Dockerfile      # 构建所需的Dockerfile
     - readme.md       # 使用方法，规定格式定义
     - xxx             # 项目代码
```

### 1.2.关于代码：

不关注代码的实现，只要最终形成docker即可，镜像的输入参数统一为字符串。

1、代码中可以从环境变量中读取任务实例的信息。
```bash
creator = os.getenv('KFJ_CREATOR', 'admin')                  # 任务流的创建者
runner = os.getenv('KFJ_RUNNER', 'admin')                    # 任务流的运行者
pipeline_name = os.getenv('KFJ_PIPELINE_NAME','test')        # 任务流的名称，或id
task_name = os.getenv('KFJ_TASK_NAME','test')                # 任务的名称或id
cpu = os.getenv('KFJ_TASK_RESOURCE_CPU','2')                 # 申请的cpu资源或内存资源
gpu = os.getenv('KFJ_TASK_RESOURCE_GPU','0')                 # 申请的gpu资源
host_ip = os.getenv('K8S_HOST_IP','xx.xx.xx.xx')             # 任务调度的主机ip
run_id = os.getenv('KFJ_RUN_ID','xx')                        # 任务流运行实例id
cache = redis.Redis.from_url(os.getenv('KFJ_CACHE_URL',''))  # 缓存地址，可以在上下有传递数据
```
2、代码中中可以读取输入参数中包含的输入地址和输出地址，做任何逻辑计算，读取输入地址文件，输出结果到输出地址。注意要想持久化，输入输出要在容器的/mnt/{{creator}}/目录下

```bash
def do_something(args):
    input_file_path = args.input_file_path
    output_file_path = args.output_file_path

    # 示例：你的任务逻辑，从目录读取输入，进行计算，保存输出
    input = pandas.read_csv(input_file_path)        # 读取输入数据
    result = input*3                                  # 根据输入数据和输入参数，做任何你想要的逻辑
    result.to_csv(output_file_path, index=False)    # 保存结果数据
```
3、我们也可以从缓存中读取输入数据，写数据到缓存中。
```bash
@pysnooper.snoop()
def do_something_with_cache():

    # 示例：你的任务逻辑，从目录读取输入，进行计算，保存输出
    input = cache.hget(run_id, "output")
    if not input:
        raise ValueError("上游输出不存在")
    input = pickle.loads(input)
    result = input*3                                      # 根据输入数据和输入参数，做任何你想要的逻辑
    cache.hset(run_id, "output", pickle.dumps(result))    # 保存结果数据

```
4、指标可视化，格式要求参考 使用/任务流.md中的“结果可视化”环节

构建任务模板时，在代码中按照规范格式在容器/metrics中输出可视化结果，包括文字、图片和echart源码，实现在web界面上的可视化展示。

/metric.json文件格式为
```bash
[
  {
    "metric_type":"image",
    "describe":"指标描述",
    "image":"图片地址"
  },
  {
    "metric_type":"text",
    "describe":"指标描述",
    "text":"文本内容"
  },
  {
    "metric_type":"html",
    "describe":"指标描述",
    "html":"html源码内容"
  },
  {
    "metric_type":"iframe",
    "describe":"指标描述",
    "url":"链接的地址"
  },
  {
    "metric_type":"table", 
    "describe":"指标描述",
    "file_path":"csv文件地址"
  },
  {
    "metric_type":"echart-xx",  # 其中xx可以为line，bar，pie，scatter，radar，candlestick，heatmap，parallel，tree，sunburst 其中tree，sunburst 时数据为json，其他类型数据为csv
    "describe":"指标描述",
    "file_path":"csv或者json数据文件地址"
  },
  {
     ...或者是echart的option源码
  }
]
```
5、要接收用户输入，如果输入为json的，接收后要进行反序列化，cube-studio提供过来的输入均为字符串，要自己手动转为相应格式。

### 1.3.关于镜像

建议镜像的tag使用日期

`ccr.ccs.tencentyun.com/cube-studio/$image_name:$image_tag`

可以使用自己的镜像仓库，自己构建后推送到仓库，前面的过程都可以不在cube-studio上完成

# 2.注册模板

### 2.1.注册仓库和镜像

如果你的镜像public，可以不用添加仓库

如果你的镜像是private的，需要 在 Cube Studio页面上，训练->仓库，先添加仓库的账号密码，才能在使用的时候拉取镜像。

然后 在 Cube Studio页面上，训练->镜像 菜单里面，添加自己的构建的镜像名。

### 2.2.模板注册入口

在 Cube Studio页面上，训练->任务模板->添加按钮

### 2.3 模板配置

1、模板的名称和描述：会显示在pipeline编排界面

2、模板版本： release版本的模板才会出现在pipeline编排界面

3、目录和启动命令：会覆盖Dockerfile中的启动目录和命令，比如Dockerfile中未定义启动命令，或多个模板使用同一个镜像时为了实现不同的功能配置不同的启动命令

4、挂载目录：会为使用此模板的任务自动添加此挂载配置。比如

```bash
kubeflow-user-workspace(pvc):/mnt          pvc的挂载方式，会自动在pvc下挂载个人子目录
/data/k8s/kubeflow/pipeline/workspace/xxxx(hostpath):/mnt    挂载主机目录的方式
```

5、启动参数编写实例
```bash
{
    "group1":{               # 属性分组，仅做web显示使用
       "attr1":{             # 属性名
        "type":"str",        # str,text,json,int,float,bool,list
        "item_type": "",     # 在type为text，item_type可为python，java，json，sql
        "label":"属性1",      # 中文名
        "require":1,         # 是否必须
        "choice":[],         # type为enum/multiple时，可选值，可以分别设置key value，例如"choice":{"key1":"value1","key2":"value2",},  
        "range":"$min,$max", # 最小最大取值，在int,float时使用，包含$min，$max
        "default":"",        # 默认值
        "placeholder":"",    # 输入提示内容
        "describe":"这里是这个字段的描述和备注",
        "editable":1        # 是否可修改
      },
      "attr2":{
       ...
      }
    },
    "group2":{
    }
}
```
6、环境变量：会为基于此模板的任务运行时均添加该环境变量。 同时可以通过一些特殊的环境变量来控制任务配置。

模板中特殊的环境变量
```bash
NO_RESOURCE_CHECK=true    使用该模板的task不会进行资源配置的自动校验
TASK_RESOURCE_CPU=4       使用该模板的task 忽略用户的资源配置，cpu固定配置资源为4核
TASK_RESOURCE_MEMORY=4G   使用该模板的task 忽略用户的资源配置，mem固定配置资源为4G
TASK_RESOURCE_GPU=0       使用该模板的task 忽略用户的资源配置，gpu固定配置资源为0卡
```


7、k8s账号：表示此类任务运行时所附带的k8s账号，主要是此类任务要启动一个分布式的pod集群，用来进行分布式数据处理或训练。  

8、扩展：扩展是json格式，index用来控制在同一分组中，排序的位置，help_url用来表示此类模板的帮助文档  

```
{
    "index": 1,
    "help_url": "https://github.com/data-infra/cube-studio/tree/main/job-template/job/pytorch"
}
```

# 3.模板使用

注册模板以后，在pipeline编排界面，点击 模板列表顶部的刷新小按钮，然后就可以在模板列表中搜索到自己创建的模板了。
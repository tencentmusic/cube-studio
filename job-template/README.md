
# 1.模板注册流程

1、编写代码，打包镜像，推送远程仓库。 

2、在 Cube Stdio页面上填写信息，注册模板。

# 2.job模板规范

### 2.1.模板目录结构
```
  
 -  job-template            # job模板合计
   - $job_template_name     # 自定义模板
     - src             # 项目代码
     - build.sh        # job镜像构建过程
     - Dockerfile      # 构建所需的Dockerfile
     - readme.md       # 使用方法，规定格式定义
```
可参照ray目录下的模板格式

### 2.2.关于构建：

1、 统一的构建脚本 `sh job/$job_template_name/build.sh`

2、 Dcokerfile文件定义镜像构建过程，构建路径为当前路径

### 2.3.关于代码：

不关注代码的实现，只要最终形成docker即可，镜像的输入参数统一为字符串

### 2.4.关于镜像
建议镜像的tag使用日期

`ccr.ccs.tencentyun.com/cube-studio/$image_name:$image_tag`

# 3.注册模板
### 3.1.模板注册入口
在 Cube Stdio页面上，训练->任务模板->添加按钮

### 3.2.注册仓库和镜像
在 Cube Stdio页面上，训练->仓库、镜像。先注册完仓库和镜像，再注册任务模板。

### 3.3 启动参数
启动参数编写实例
```bash
{
    "group1":{               # 属性分组，仅做web显示使用
       "attr1":{             # 属性名
        "type":"str",        # int,str,text,bool,enum,float,multiple,date,datetime,file,dict,list
        "item_type": "",     # 在type为enum,multiple,list时每个子属性的类型
        "label":"属性1",      # 中文名
        "require":1,         # 是否必须
        "choice":[],         # type为enum/multiple时，可选值
        "range":"$min,$max", # 最小最大取值，在int,float时使用，包含$min，但是不包含$max
        "default":"",        # 默认值
        "placeholder":"",    # 输入提示内容
        "describe":"这里是这个字段的描述和备注",
        "editable":1,        # 是否可修改
        "condition":"",      # 显示的条件
        "sub_args": {        # 如果type是dict或者list对应下面的参数
        }
      },
      "attr2":{
       ...
      }
    },
    "group2":{
    }
}
```
### 3.4. 其他注册参数
参照页面上的说明

### 公共魔法变量

为了便于使用，在配置中支持几个公共的魔法变量，类似占位符，在实际运行中会被展开成实际值，魔法变量的格式为`${PARAM_NAME}$`，目前支持的有如下几个：

`__${PACK_PATH}$__`：包目录，即用户自己代码数据等所在目录，例如/mnt/lionpeng/ai_radio_v2。这个目录是分布式存储挂载到集群worker docker中的目录，该目录会挂载到pipeline中每一个job对应worker docker中。
	
`__${DATA_PATH}$__`: 数据目录，表示pipeline一次运行的目录，这里面会存放本次运行中各job产生的数据，包括用户自己代码所产生的数据都放在这里。每次运行目录是不一样的，便于每次运行之间隔离，另外也是便于同一次运行中上下游job进行数据交互。例如/mnt/lionpeng/ai_radio_v2_runs/20201021-141656.624784。同样该目录也会挂载到pipeline中每一个job对应worker docker中。

`__${DATE[(-|+numd|w|h|m|s][:format]}$__`: 日期变量，例如${DATE}$表示任务运行时的时间。该变量还支持偏移，偏移单位支持d(天)，w(星期)，h(小时)，m(分钟)，s(秒)，y(年)，M(月)。例如`${DATE-1d}$`，表示运行日的前一天，例如今天是20201021，则${DATE-1d}$展开后就是20201020，而${DATE+2d}$则表示运行日的后两天，即20201023。另外支持指定日期的格式化格式，默认格式是%Y%m%d，格式化符号与python datetime格式化符号一致，可参考说明。例如当前时间是2020年10月21日早上10点5分35秒，`${DATE-1d:%Y-%m-%d %H:%M:%S}$`的展开结果就是"2020-10-20 10:05:35"

`__${ONLINE_MODEL}$__`：线上模型，用于在评估任务方便用户拉取线上模型进行指标对比，关于评估任务见后面详述。
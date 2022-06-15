
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

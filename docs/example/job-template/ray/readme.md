# 模板说明
模板镜像内置了python分布式任务ray框架。基于ray框架编写代码，可以实现在集群内启动多个pod分布式执行任务。

# 模板镜像

`ai.tencentmusic.com/tme-public/ray:gpu-20210601`

# 模板注册
参考上级目录的readme.md，注册时填写以下配置。

1、挂载目录：`2G(memory):/dev/shm`

2、启动参数：
```
{
    "shell": {
        "-n": {
            "type": "int",
            "item_type": "",
            "label": "分布式任务worker的数量",
            "require": 1,
            "choice": [],
            "range": "$min,$max",
            "default": "3",
            "placeholder": "",
            "describe": "分布式任务worker的数量",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "-i": {
            "type": "str",
            "item_type": "str",
            "label": "每个worker的初始化脚本文件地址，用来安装环境",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "每个worker的初始化脚本文件地址，用来安装环境",
            "describe": "每个worker的初始化脚本文件地址，用来安装环境",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "-f": {
            "type": "str",
            "item_type": "str",
            "label": "python启动命令，例如 python3 /mnt/xx/xx.py",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "python启动命令，例如 python3 /mnt/xx/xx.py",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        }
    }
}
```
3、k8s账号： `kubeflow-pipeline`

# 使用方法

原有代码
```
# import ray

def fun1(index):
    # 这里是耗时的任务
    return 'back_data'

def main():
    for index in [...]:
         fun1(index)    # 直接执行任务

if __name__=="__main__":
    main()
```

启用ray框架的代码

```
import ray

@ray.remote
def fun1(index):
    # 这里是耗时的任务
    return 'back_data'

def main():
    tasks=[]
    for index in [...]:
         tasks.append(fun1.remote(index))   # 建立远程函数
    result = ray.get(tasks)   #  获取任务结果

if __name__=="__main__":

    head_service_ip = os.getenv('RAY_HOST','')
    if head_service_ip:
        # 集群模式
        head_host = head_service_ip+".pipeline"+":10001"
        ray.util.connect(head_host)
    else:
        # 本地模式
        ray.init()

    main()
```

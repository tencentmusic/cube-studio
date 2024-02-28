# ray 模板
镜像：ccr.ccs.tencentyun.com/cube-studio/ray:gpu-20230801

环境变量：
```bash
NO_RESOURCE_CHECK=true
TASK_RESOURCE_CPU=2
TASK_RESOURCE_MEMORY=4G
TASK_RESOURCE_GPU=0
```

账号：kubeflow-pipeline

启动参数：
```bash
{
    "参数": {
        "images": {
          "type": "str",
          "item_type": "str",
          "label": "启动镜像",
          "require": 0,
          "choice": [],
          "range": "",
          "default": "ccr.ccs.tencentyun.com/cube-studio/ray:gpu-20240101",
          "placeholder": "",
          "describe": "启动镜像，基础镜像需为该镜像，可自行添加封装自己的环境",
          "editable": 1
        },
        "--workdir": {
            "type": "str",
            "item_type": "",
            "label": "工作目录",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "/mnt/admin/pipeline/example/ray/",
            "placeholder": "",
            "describe": "分布式任务worker的数量",
            "editable": 1
        },
        "--init": {
            "type": "str",
            "item_type": "str",
            "label": "每个worker的初始化脚本文件地址，用来安装环境",
            "require": 0,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "每个worker的初始化脚本文件地址，用来安装环境",
            "describe": "每个worker的初始化脚本文件地址，用来安装环境",
            "editable": 1
        },
        "--command": {
            "type": "str",
            "item_type": "str",
            "label": "python启动命令，例如 python3 /mnt/xx/xx.py",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "python demo.py",
            "placeholder": "",
            "describe": "python启动命令，例如 python3 /mnt/xx/xx.py",
            "editable": 1
        },
        "--num_worker": {
            "type": "str",
            "item_type": "",
            "label": "分布式任务worker的数量",
            "require": 1,
            "choice": [],
            "range": "$min,$max",
            "default": "3",
            "placeholder": "",
            "describe": "分布式任务worker的数量",
            "editable": 1
        }
    }
}
```
# 安装包 
pip install ray

# 使用
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

启用ray框架的代码，示例demo.py
```
import ray,os,time


@ray.remote
def fun1(arg):
    # 这里是耗时的任务，函数内不能引用全局变量，只能使用函数内的局部变量。
    print(arg)
    time.sleep(1)
    return 'back_data'


def main():
    tasks=[]
    tasks_args = range(100)
    for arg in tasks_args:
        tasks.append(fun1.remote(arg))  # 建立远程函数
    result = ray.get(tasks)  # 获取任务结果


if __name__ == "__main__":

    head_service_ip = os.getenv('RAY_HOST', '')
    if head_service_ip:
        # 集群模式
        head_host = head_service_ip + ".pipeline" + ":10001"
        ray.util.connect(head_host)
    else:
        # 本地模式
        ray.init()

    main()
```

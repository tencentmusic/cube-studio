# volcanojob 模板
镜像：ai.tencentmusic.com/tme-public/volcano:20211001
挂载：kubernetes-config(configmap):/root/.kube
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
    "shell": {
        "--working_dir": {
            "type": "str",
            "item_type": "str",
            "label": "启动目录",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "/mnt/xx",
            "placeholder": "",
            "describe": "启动目录",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--command": {
            "type": "str",
            "item_type": "str",
            "label": "启动命令",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "echo aa",
            "placeholder": "",
            "describe": "启动命令",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--num_worker": {
            "type": "str",
            "item_type": "str",
            "label": "占用机器个数",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "3",
            "placeholder": "",
            "describe": "占用机器个数",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--image": {
            "type": "str",
            "item_type": "str",
            "label": "",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "ai.tencentmusic.com/tme-public/ubuntu-gpu:cuda10.1-cudnn7-python3.6",
            "placeholder": "",
            "describe": "worker镜像，直接运行你代码的环境镜像<a href='https://docs.qq.com/doc/DU0ptZEpiSmtMY1JT'>基础镜像</a>",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        }
    }
}
```

# 用户代码示例

保留单机的代码，添加识别集群信息的代码（多少个worker，当前worker是第几个），添加分工（只处理归属于当前worker的任务），

完成。

worker示例：
```
import time, datetime, json, requests, io, os
from multiprocessing import Pool
from functools import partial
import os, random

WORLD_SIZE = int(os.getenv('VC_WORKER_NUM', ''))  # 总worker的数目
RANK = int(os.getenv("VC_TASK_INDEX", '0'))     # 当前是第几个worker 从0开始

print(WORLD_SIZE, RANK)


# 子进程要执行的代码
def task(key):
    print('worker:',RANK,', task:',key,flush=True)
    time.sleep(1)


if __name__ == '__main__':

    input = range(30000)    # 所有要处理的数据
    local_task = []         # 当前worker需要处理的任务
    for index in input:
        if index%WORLD_SIZE==RANK:
            local_task.append(index)     # 要处理的数据均匀分配到每个worker

	    # 每个worker内部还可以用多进程，线程池之类的并发操作。
    pool = Pool(10)  # 开辟包含指定数目线程的线程池
    pool.map(partial(task), local_task)  # 当前worker，只处理分配给当前worker的任务
    pool.close()
    pool.join()
```
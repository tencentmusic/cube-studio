
镜像：ccr.ccs.tencentyun.com/cube-studio/mxnet:20221010
环境变量
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
        "--image": {
            "type": "str",
            "item_type": "str",
            "label": "worker镜像，直接运行你代码的环境镜像",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "ccr.ccs.tencentyun.com/cube-studio/ubuntu-gpu:cuda10.1-cudnn7-python3.6",
            "placeholder": "",
            "describe": "worker镜像，直接运行你代码的环境镜像 <a target='_blank' href='https://github.com/tencentmusic/cube-studio/tree/master/images'>基础镜像</a>",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--working_dir": {
            "type": "str",
            "item_type": "str",
            "label": "命令的启动目录",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "/mnt/xxx/mxnet/",
            "placeholder": "",
            "describe": "命令的启动目录",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--command": {
            "type": "str",
            "item_type": "str",
            "label": "启动命令，例如 python3 xxx.py",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "启动命令，例如 python3 xxx.py",
            "describe": "启动命令，例如 python3 xxx.py",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--num_worker": {
            "type": "str",
            "item_type": "str",
            "label": "分布式训练worker的数目",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "3",
            "placeholder": "分布式训练worker的数目",
            "describe": "分布式训练worker的数目",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--num_ps": {
            "type": "str",
            "item_type": "str",
            "label": "分布式训练ps的数目",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "0",
            "placeholder": "分布式训练ps的数目",
            "describe": "分布式训练ps的数目",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        }
    }
}
```
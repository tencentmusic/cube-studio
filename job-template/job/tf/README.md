镜像：ccr.ccs.tencentyun.com/cube-studio/tf:20230801

账号：kubeflow-pipeline

参数：
```bash
{
    "shell": {
        "--image": {
            "type": "str",
            "item_type": "str",
            "label": "worker镜像，直接运行你代码的环境镜像",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "ccr.ccs.tencentyun.com/cube-studio/ubuntu-gpu:cuda11.8.0-cudnn8-python3.9",
            "placeholder": "",
            "describe": "worker镜像，直接运行你代码的环境镜像 <a href='https://github.com/tencentmusic/cube-studio/tree/master/images'>基础镜像</a>",
            "editable": 1
        },
        "--working_dir": {
            "type": "str",
            "item_type": "str",
            "label": "命令的启动目录",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "/mnt/xxx/pytorchjob/",
            "placeholder": "",
            "describe": "命令的启动目录",
            "editable": 1
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
            "editable": 1
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
            "editable": 1
        }
    }
}
```
环境变量
```bash
NO_RESOURCE_CHECK=true
TASK_RESOURCE_CPU=2
TASK_RESOURCE_MEMORY=4G
TASK_RESOURCE_GPU=0
```

示例代码

启动脚本
```
python demo.py
```
示例代码 [demo.py](https://github.com/kubeflow/training-operator/blob/master/examples/tensorflow/mnist_with_summaries/mnist_with_summaries.py)
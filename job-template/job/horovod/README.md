# horovod 模板
镜像：ccr.ccs.tencentyun.com/cube-studio/horovod:20210401
k8s账号: kubeflow-pipeline
启动参数：
```bash
{
    "参数": {
        "--num_worker": {
            "type": "str",
            "item_type": "str",
            "label": "分布式worker的数目",
            "require": 1,
            "choice": [],
            "range": "",
            "default": 2,
            "placeholder": "",
            "describe": "分布式worker的数目",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--command": {
            "type": "str",
            "item_type": "str",
            "label": "训练启动命令",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "python /horovod/examples/tensorflow2/tensorflow2_mnist.py",
            "placeholder": "",
            "describe": "训练启动命令",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--work_images": {
            "type": "str",
            "item_type": "str",
            "label": "worker的运行镜像",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "ccr.ccs.tencentyun.com/cube-studio/horovod:20210401",
            "placeholder": "",
            "describe": "worker的运行镜像",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        }
    }
}
```


# mnsit训练 模板
镜像：ccr.ccs.tencentyun.com/cube-studio/mnist:20220814
启动参数：
```bash
{
    "参数分组1": {
        "--modelpath": {
            "type": "str",
            "item_type": "str",
            "label": "参数1",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "/mnt/admin/pytorch/model",
            "placeholder": "",
            "describe": "模型保存路径",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--datapath": {
            "type": "str",
            "item_type": "str",
            "label": "参数1",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "/mnt/admin/mnist",
            "placeholder": "",
            "describe": "数据读取路径",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        }
    }
}
```
# dataset 模板
镜像：ccr.ccs.tencentyun.com/cube-studio/dataset

参数
```bash
{
    "参数": {
      "--src_type": {
            "type": "str",
            "item_type": "str",
            "label": "数据集的来源",
            "require": 1,
            "choice": ["当前平台","huggingface","魔塔"],
            "range": "",
            "default": "当前平台",
            "placeholder": "",
            "describe": "数据集的来源",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--name": {
            "type": "str",
            "item_type": "str",
            "label": "数据集的名称",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "数据集的名称",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--version": {
            "type": "str",
            "item_type": "str",
            "label": "数据集的版本",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "latest",
            "placeholder": "",
            "describe": "数据集的版本",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--partition": {
            "type": "str",
            "item_type": "str",
            "label": "数据集的分区，或者子数据集",
            "require": 0,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "数据集的分区，或者子数据集",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--save_dir": {
            "type": "str",
            "item_type": "str",
            "label": "数据集的保存地址",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "数据集的保存地址",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        }
    }
}
```

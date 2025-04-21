# model_download 模板
镜像：ccr.ccs.tencentyun.com/cube-studio/model_download:20240501
启动参数：
```bash
{
    "参数": {
        "--from": {
            "type": "str",
            "item_type": "str",
            "label": "模型来源地",
            "require": 1,
            "choice": ["模型管理","推理服务","huggingface"],
            "range": "",
            "default": "模型管理",
            "placeholder": "",
            "describe": "模型来源地",
            "editable": 1
        },
        "--model_name": {
            "type": "str",
            "item_type": "str",
            "label": "模型名(a-z0-9-字符组成，最长54个字符)",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "模型名",
            "editable": 1
        },
        "--sub_model_name": {
            "type": "str",
            "item_type": "str",
            "label": "子模型名(a-z0-9-字符组成，最长54个字符)",
            "require": 0,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "子模型名，对于包含多个子模型的用户填写",
            "editable": 1
        },
        "--model_version": {
            "type": "str",
            "item_type": "str",
            "label": "模型版本号",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "v2022.10.01.1",
            "placeholder": "",
            "describe": "模型版本号",
            "editable": 1
        },
        "--model_status": {
            "type": "str",
            "item_type": "str",
            "label": "模型状态",
            "require": 1,
            "choice": ["online","offline","test"],
            "range": "",
            "default": "online",
            "placeholder": "",
            "describe": "模型状态，模型来自推理服务时有效",
            "editable": 1
        },
        "--save_path": {
            "type": "str",
            "item_type": "str",
            "label": "下载目的目录",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "/mnt/xx/download/model/",
            "placeholder": "",
            "describe": "下载目的目录",
            "editable": 1
        }
    }
}
```
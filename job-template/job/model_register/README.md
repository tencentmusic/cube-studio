# register_model 模板
镜像：ccr.ccs.tencentyun.com/cube-studio/model_register:20230501
启动参数：
```bash
{
    "参数": {
        "--project_name": {
            "type": "str",
            "item_type": "str",
            "label": "部署项目名",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "public",
            "placeholder": "",
            "describe": "部署项目名",
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
        "--model_metric": {
            "type": "str",
            "item_type": "str",
            "label": "模型指标",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "模型指标",
            "editable": 1
        },
        "--model_path": {
            "type": "str",
            "item_type": "str",
            "label": "模型地址",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "模型地址",
            "editable": 1
        },
        "--describe": {
            "type": "str",
            "item_type": "str",
            "label": "模型描述",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "模型描述",
            "editable": 1
        },
        "--framework": {
            "type": "str",
            "item_type": "str",
            "label": "模型框架",
            "require": 1,
            "choice": [
                "lr",
                "xgb",
                "tf",
                "pytorch",
                "onnx",
                "tensorrt",
                "aihub"
            ],
            "range": "",
            "default": "tf",
            "placeholder": "",
            "describe": "模型框架",
            "editable": 1
        },
        "--inference_framework": {
            "type": "str",
            "item_type": "str",
            "label": "推理框架",
            "require": 1,
            "choice": [
                "serving",
                "ml-server",
                "tfserving",
                "torch-server",
                "onnxruntime",
                "triton-server",
                "aihub"
            ],
            "range": "",
            "default": "tfserving",
            "placeholder": "",
            "describe": "推理框架",
            "editable": 1
        }
    }
}
```
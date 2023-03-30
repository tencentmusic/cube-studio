# register_model 模板
镜像：ccr.ccs.tencentyun.com/cube-studio/model:20221001
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
            "editable": 1,
            "condition": "",
            "sub_args": {}
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
            "editable": 1,
            "condition": "",
            "sub_args": {}
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
            "editable": 1,
            "condition": "",
            "sub_args": {}
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
            "editable": 1,
            "condition": "",
            "sub_args": {}
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
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--framework": {
            "type": "str",
            "item_type": "str",
            "label": "模型框架",
            "require": 1,
            "choice": [
                "xgb",
                "tf",
                "pytorch",
                "onnx",
                "tensorrt"
            ],
            "range": "",
            "default": "tf",
            "placeholder": "",
            "describe": "模型框架",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--inference_framework": {
            "type": "str",
            "item_type": "str",
            "label": "推理框架",
            "require": 1,
            "choice": [
                "tfserving",
                "torch-server",
                "onnxruntime",
                "triton-server"
            ],
            "range": "",
            "default": "tfserving",
            "placeholder": "",
            "describe": "推理框架",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        }
    }
}
```
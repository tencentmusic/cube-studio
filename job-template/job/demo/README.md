# demo 模板
镜像：ccr.ccs.tencentyun.com/cube-studio/demo:20230505

参数
```bash
{
    "训练相关参数": {
        "--input_file_path": {
            "type": "str",
            "item_type": "str",
            "label": "输入csv文件地址",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "data.csv",
            "describe": "输入csv文件地址",
            "editable": 1
        },
        "--output_file_path": {
            "type": "str",
            "item_type": "str",
            "label": "输出csv文件地址",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "data-result.csv",
            "describe": "输出csv文件地址",
            "editable": 1
        },
        "--kwargs": {
            "type": "json",
            "item_type": "str",
            "label": "其他参数",
            "require": 1,
            "choice": [],
            "range": "",
            "default": {
              "args1":"value1",
              "args2":"value2"
            },
            "describe": "其他参数",
            "editable": 1
        }
    }
}
```

镜像：ccr.ccs.tencentyun.com/cube-studio/spark:20221010
账号：kubeflow-pipeline
参数
```bash
{
    "参数": {
        "--image": {
            "type": "str",
            "item_type": "str",
            "label": "执行镜像",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "ccr.ccs.tencentyun.com/cube-studio/spark-operator:spark-v3.1.1",
            "placeholder": "",
            "describe": "执行镜像",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--num_worker": {
            "type": "str",
            "item_type": "str",
            "label": "executor 数目",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "3",
            "placeholder": "",
            "describe": "executor 数目",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--code_type": {
            "type": "str",
            "item_type": "str",
            "label": "语言类型",
            "require": 1,
            "choice": [
                "Java",
                "Python",
                "Scala",
                "R"
            ],
            "range": "",
            "default": "Python",
            "placeholder": "",
            "describe": "语言类型",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--code_class": {
            "type": "str",
            "item_type": "str",
            "label": "Java/Scala类名",
            "require": 0,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "Java/Scala类名，其他语言下不填",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--code_file": {
            "type": "str",
            "item_type": "str",
            "label": "代码文件地址",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "local:///opt/spark/examples/src/main/python/pi.py",
            "placeholder": "",
            "describe": "代码文件地址，支持local://,http://,hdfs://,s3a://,gcs://",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--code_arguments": {
            "type": "str",
            "item_type": "str",
            "label": "代码参数",
            "require": 0,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "代码参数",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--sparkConf": {
            "type": "text",
            "item_type": "str",
            "label": "spark配置",
            "require": 0,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "spark配置，每行一个配置，xx=yy",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--hadoopConf": {
            "type": "text",
            "item_type": "str",
            "label": "hadoop配置，每行一个配置，xx=yy",
            "require": 0,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "hadoop配置",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        }
    }
}
```
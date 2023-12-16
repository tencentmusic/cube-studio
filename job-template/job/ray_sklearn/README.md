# ray-sklearn 模板
描述：基于ray的分布式能力，实现sklearn机器学习模型的分布式训练。  

镜像：ccr.ccs.tencentyun.com/cube-studio/sklearn_estimator:v1  

环境变量：  
```bash
NO_RESOURCE_CHECK=true
TASK_RESOURCE_CPU=2
TASK_RESOURCE_MEMORY=4G
TASK_RESOURCE_GPU=0
```

启动参数：  
```bash
{
    "shell": {
        "--train_csv_file_path": {
            "type": "str",
            "item_type": "str",
            "label": "训练集csv",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "训练集csv，|分割符，首行是列名",
            "editable": 1
        },
        "--predict_csv_file_path": {
            "type": "str",
            "item_type": "str",
            "label": "预测数据集csv",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "预测数据集csv，格式和训练集一致，默认为空，需要predict时填",
            "editable": 1
        },
        "--label_name": {
            "type": "str",
            "item_type": "str",
            "label": "label的列名，必填",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "label的列名，必填",
            "editable": 1
        },
        "--model_name": {
            "type": "str",
            "item_type": "str",
            "label": "模型名称，必填",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "训练用到的模型名称，如LogisticRegression，必填。常用的都支持，要加联系管理员",
            "editable": 1
        },
        "--model_args_dict": {
            "type": "str",
            "item_type": "str",
            "label": "模型参数",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "模型参数，json格式，默认为空",
            "editable": 1
        },
        "--model_file_path": {
            "type": "str",
            "item_type": "str",
            "label": "模型文件保存文件名，必填",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "模型文件保存文件名，必填",
            "editable": 1
        },
        "--predict_result_path": {
            "type": "str",
            "item_type": "str",
            "label": "预测结果保存文件名，默认为空，需要predict时填",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "预测结果保存文件名，默认为空，需要predict时填",
            "editable": 1
        },
        "--worker_num": {
            "type": "str",
            "item_type": "str",
            "label": "ray worker数量",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "ray worker数量",
            "editable": 1
        }
    }
}
```

